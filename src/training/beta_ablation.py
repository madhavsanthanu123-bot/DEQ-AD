import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

from src.data.oasis_dataset import OASISDataset
from src.models.deq_ad import DEQADModel

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_cdr_class(cdr_values):
    mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
    classes = [mapping[float(val.item())] for val in cdr_values]
    return torch.tensor(classes, dtype=torch.long)

def safe_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0
    return acc, prec, rec, f1, auc

def run_safety_check(model, dataloader, device, optimizer, criterion_cls, criterion_cdr):
    model.train()
    batch = next(iter(dataloader))
    
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    cdrs = batch["cdr"]
    cdr_class = get_cdr_class(cdrs).to(device)
    
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images)
    
    classification_logits = outputs["classification_logits"].squeeze(1)
    cdr_logits = outputs["cdr_logits"]
    
    loss_cls = criterion_cls(classification_logits, labels.float())
    loss_cdr = criterion_cdr(cdr_logits, cdr_class)
    loss = loss_cls + 0.5 * loss_cdr
    
    loss.backward()
    
    has_nan_inf = not torch.isfinite(loss)
    deq_converged = model.deq.last_converged
    
    grad_finite = True
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                grad_finite = False
                break
                
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    return not has_nan_inf, deq_converged, grad_finite, mem_alloc

def extract_features(model, dataloader, device):
    features = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = model(images)
            eq_feat = outputs["equilibrium_features"].cpu().numpy()
            features.append(eq_feat)
            
    features = np.vstack(features)
    return features

def run_experiment(beta, train_loader, val_loader, test_loader, device, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = DEQADModel(beta=beta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_cdr = nn.CrossEntropyLoss()
    
    print(f"\n--- Safety check for beta={beta} ---")
    loss_finite, deq_conv, grad_finite, mem = run_safety_check(model, train_loader, device, optimizer, criterion_cls, criterion_cdr)
    print(f"Loss finite: {loss_finite}, DEQ conv: {deq_conv}, Grad finite: {grad_finite}, GPU Mem: {mem:.2f}MB")
    
    if not (loss_finite and grad_finite):
        print(f"WARNING: beta={beta} is UNSTABLE. Aborting this beta.")
        return None
        
    epochs = 20
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            cdr_class = get_cdr_class(batch["cdr"]).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            
            logits_cls = outputs["classification_logits"].squeeze(1)
            logits_cdr = outputs["cdr_logits"]
            
            loss_cls = criterion_cls(logits_cls, labels.float())
            loss_cdr = criterion_cdr(logits_cdr, cdr_class)
            loss = loss_cls + 0.5 * loss_cdr
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                cdr_class = get_cdr_class(batch["cdr"]).to(device)
                
                outputs = model(images)
                logits_cls = outputs["classification_logits"].squeeze(1)
                logits_cdr = outputs["cdr_logits"]
                
                loss_cls = criterion_cls(logits_cls, labels.float())
                loss_cdr = criterion_cdr(logits_cdr, cdr_class)
                loss = loss_cls + 0.5 * loss_cdr
                
                val_loss += loss.item() * images.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping beta={beta} at epoch {epoch}")
            break

    # Evaluation on Test Set
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=False))
    model.eval()
    
    cls_labels, cls_probs, cls_preds = [], [], []
    cdr_true, cdr_preds = [], []
    deq_iters = []
    non_converged = 0
    max_iters = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            cdrs = get_cdr_class(batch["cdr"]).numpy()
            
            outputs = model(images)
            logits_cls = outputs["classification_logits"].squeeze(1)
            logits_cdr = outputs["cdr_logits"]
            
            probs = torch.sigmoid(logits_cls).cpu().numpy()
            preds = (probs >= 0.5).astype(int) # Standard 0.5 threshold
            
            cdr_p = torch.argmax(logits_cdr, dim=1).cpu().numpy()
            
            cls_labels.extend(labels)
            cls_probs.extend(probs)
            cls_preds.extend(preds)
            
            cdr_true.extend(cdrs)
            cdr_preds.extend(cdr_p)
            
            iters = model.deq.last_iterations
            deq_iters.append(iters)
            max_iters = max(max_iters, iters)
            if not model.deq.last_converged:
                non_converged += 1
                
    acc, prec, rec, f1, auc = safe_metrics(cls_labels, cls_preds, cls_probs)
    
    cm_bin = confusion_matrix(cls_labels, cls_preds, labels=[0, 1])
    if cm_bin.shape == (2, 2):
        tn, fp, fn, tp = cm_bin.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0
        
    cdr_acc = accuracy_score(cdr_true, cdr_preds)
    cdr_macro_f1 = f1_score(cdr_true, cdr_preds, average="macro", zero_division=0)
    
    avg_deq_iters = np.mean(deq_iters)
    conv_perc = 100.0 * (len(deq_iters) - non_converged) / len(deq_iters)
    
    # Feature extraction & Diagnostics
    tr_f = extract_features(model, train_loader, device)
    
    std_devs = np.std(tr_f, axis=0)
    mean_std = np.mean(std_devs)
    p_1e3 = np.mean(std_devs < 1e-3) * 100
    p_1e2 = np.mean(std_devs < 1e-2) * 100
    
    if len(tr_f) > 1:
        dists = pdist(tr_f, metric='euclidean')
        avg_pdist = np.mean(dists)
        min_pdist = np.min(dists)
        max_pdist = np.max(dists)
    else:
        avg_pdist = min_pdist = max_pdist = 0.0
        
    # PCA
    pca = PCA(n_components=2, random_state=42)
    tr_pca = pca.fit_transform(tr_f)
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    
    # Plot PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(tr_pca[:, 0], tr_pca[:, 1], alpha=0.7)
    plt.title(f"PCA - Train Set (beta={beta})")
    plt.xlabel(f"PC1 ({pc1_var*100:.1f}%)")
    plt.ylabel(f"PC2 ({pc2_var*100:.1f}%)")
    plt.grid(True)
    plt.savefig(out_dir / f"pca_train_beta_{beta}.png")
    plt.close()
    
    return {
        "beta": beta,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "test_roc_auc": auc,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "cdr_accuracy": cdr_acc,
        "cdr_macro_f1": cdr_macro_f1,
        "avg_deq_iterations": avg_deq_iters,
        "max_deq_iterations": max_iters,
        "non_converged_solves": non_converged,
        "convergence_percentage": conv_perc,
        "feature_mean_std": mean_std,
        "dims_std_below_1e3": p_1e3,
        "dims_std_below_1e2": p_1e2,
        "avg_pairwise_distance": avg_pdist,
        "min_pairwise_distance": min_pdist,
        "max_pairwise_distance": max_pdist,
        "pc1_variance": pc1_var,
        "pc2_variance": pc2_var,
        "pc1_pc2_variance": pc1_var + pc2_var
    }

def main():
    set_seed(42)
    
    base_results_dir = Path(r"D:\DEQ-AD\results")
    ablation_dir = base_results_dir / "beta_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 60)
    print("BETA ABLATION STUDY")
    print("=" * 60)
    
    train_dataset = OASISDataset(base_results_dir / "train_manifest.csv")
    val_dataset = OASISDataset(base_results_dir / "val_manifest.csv")
    test_dataset = OASISDataset(base_results_dir / "test_manifest.csv")
    
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=pin_mem)
    # for feature extraction, we want no shuffle
    train_loader_no_shuf = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    results = []
    
    for beta in [0.5, 0.7, 0.9]:
        print(f"\n>>> Running Experiment B={beta} <<<")
        out_dir = ablation_dir / f"beta_{beta}"
        res = run_experiment(beta, train_loader, val_loader, test_loader, device, out_dir)
        
        # Override feature extraction with unshuffled dataloader
        if res is not None:
            model = DEQADModel(beta=beta).to(device)
            model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=False))
            model.eval()
            tr_f = extract_features(model, train_loader_no_shuf, device)
            std_devs = np.std(tr_f, axis=0)
            res["feature_mean_std"] = np.mean(std_devs)
            res["dims_std_below_1e3"] = np.mean(std_devs < 1e-3) * 100
            res["dims_std_below_1e2"] = np.mean(std_devs < 1e-2) * 100
            if len(tr_f) > 1:
                dists = pdist(tr_f, metric='euclidean')
                res["avg_pairwise_distance"] = np.mean(dists)
                res["min_pairwise_distance"] = np.min(dists)
                res["max_pairwise_distance"] = np.max(dists)
            results.append(res)
        else:
            results.append({
                "beta": beta,
                "test_accuracy": None,
                "test_roc_auc": None,
                "avg_deq_iterations": None,
                "convergence_percentage": 0.0,
                "feature_mean_std": None,
                "avg_pairwise_distance": None,
                "status": "UNSTABLE"
            })
            
    df = pd.DataFrame(results)
    df.to_csv(ablation_dir / "beta_comparison.csv", index=False)
    
    # Ensure JSON serializability
    json_results = []
    for r in results:
        json_r = {}
        for k, v in r.items():
            if isinstance(v, (np.float32, np.float64, np.float16)):
                json_r[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                json_r[k] = int(v)
            else:
                json_r[k] = v
        json_results.append(json_r)
        
    with open(ablation_dir / "ablation_summary.json", "w") as f:
        json.dump(json_results, f, indent=4)
        
    # Generate Plots
    valid_df = df.dropna(subset=["test_roc_auc"])
    if len(valid_df) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(valid_df["beta"], valid_df["test_roc_auc"], marker='o')
        plt.title("Test ROC-AUC vs Beta")
        plt.xlabel("Beta")
        plt.ylabel("ROC-AUC")
        plt.grid(True)
        plt.savefig(ablation_dir / "plot_roc_auc.png")
        plt.close()
        
        plt.figure(figsize=(6, 4))
        plt.plot(valid_df["beta"], valid_df["test_f1"], marker='o')
        plt.title("Test F1 vs Beta")
        plt.xlabel("Beta")
        plt.ylabel("F1 Score")
        plt.grid(True)
        plt.savefig(ablation_dir / "plot_f1.png")
        plt.close()
        
        plt.figure(figsize=(6, 4))
        plt.plot(valid_df["beta"], valid_df["feature_mean_std"], marker='o', color='green')
        plt.title("Avg Feature STD vs Beta")
        plt.xlabel("Beta")
        plt.ylabel("Mean STD")
        plt.grid(True)
        plt.savefig(ablation_dir / "plot_feature_std.png")
        plt.close()
        
        plt.figure(figsize=(6, 4))
        plt.plot(valid_df["beta"], valid_df["avg_pairwise_distance"], marker='o', color='purple')
        plt.title("Avg Pairwise Feature Dist vs Beta")
        plt.xlabel("Beta")
        plt.ylabel("Avg Pairwise Distance")
        plt.grid(True)
        plt.savefig(ablation_dir / "plot_feature_dist.png")
        plt.close()
        
        plt.figure(figsize=(6, 4))
        plt.plot(valid_df["beta"], valid_df["avg_deq_iterations"], marker='o', color='orange')
        plt.title("Avg DEQ Iterations vs Beta")
        plt.xlabel("Beta")
        plt.ylabel("Iterations")
        plt.grid(True)
        plt.savefig(ablation_dir / "plot_deq_iters.png")
        plt.close()

    print("\nBeta Ablation Complete!")
    print(f"Results saved to: {ablation_dir}")

if __name__ == "__main__":
    main()
