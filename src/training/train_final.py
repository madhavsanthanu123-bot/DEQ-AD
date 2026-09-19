import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.oasis_dataset import OASISDataset
from src.models.deq_ad import DEQADModel
from src.training.evaluate_final import evaluate_test_set, get_cdr_class

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def find_best_threshold(y_true, y_prob):
    best_f1 = -1.0
    best_thresh = 0.50
    
    thresholds = np.linspace(0.10, 0.90, 81)
    
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
        elif np.isclose(f1, best_f1, atol=1e-5):
            if abs(th - 0.5) < abs(best_thresh - 0.5):
                best_thresh = th
                
    return best_thresh

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

def train_and_eval_seed(seed, base_results_dir, final_dir, device):
    set_seed(seed)
    
    print("=" * 60)
    print(f"RUNNING FINAL EXPERIMENT - SEED {seed}")
    print("=" * 60)
    
    out_dir = final_dir / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = OASISDataset(base_results_dir / "train_manifest.csv")
    val_dataset = OASISDataset(base_results_dir / "val_manifest.csv")
    test_dataset = OASISDataset(base_results_dir / "test_manifest.csv")
    
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=pin_mem)
    train_loader_no_shuf = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_mem)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    model = DEQADModel(beta=0.7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_cdr = nn.CrossEntropyLoss()
    
    print("\nSafety check...")
    loss_finite, deq_conv, grad_finite, mem = run_safety_check(model, train_loader, device, optimizer, criterion_cls, criterion_cdr)
    print(f"Loss finite: {loss_finite}, DEQ conv: {deq_conv}, Grad finite: {grad_finite}, GPU Mem: {mem:.2f}MB")
    
    epochs = 20
    patience = 5
    best_val_auc = -1.0
    best_epoch = -1
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Training
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
        val_labels, val_probs = [], []
        
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
                
                probs = torch.sigmoid(logits_cls).cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except:
            val_auc = None
            
        # Checkpoint Selection
        improved = False
        if val_auc is not None:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_loss = val_loss
                improved = True
        else:
            # fallback to loss if AUC is uncalculatable
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                print(f"Epoch {epoch}: AUC uncalculatable, improved by loss.")
                
        if improved:
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            epochs_no_improve = 0
            print(f"Epoch {epoch}: New best model! (AUC={val_auc}, Loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch}: No improvement. (AUC={val_auc}, Loss={val_loss:.4f})")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    
    # -------------------------------------------------------------
    # THRESHOLD SELECTION (ON VALIDATION SET USING BEST MODEL)
    # -------------------------------------------------------------
    print("\nSelecting threshold on Validation Set...")
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=False))
    model.eval()
    
    val_labels, val_probs = [], []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            logits_cls = outputs["classification_logits"].squeeze(1)
            probs = torch.sigmoid(logits_cls).cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(labels.cpu().numpy())
            
    selected_threshold = find_best_threshold(np.array(val_labels), np.array(val_probs))
    print(f"Selected Threshold: {selected_threshold:.4f}")
    
    with open(out_dir / "selected_threshold.txt", "w") as f:
        f.write(str(selected_threshold))
        
    # -------------------------------------------------------------
    # FINAL TEST & REPRESENTATION EVALUATION
    # -------------------------------------------------------------
    print("\nEvaluating Test Set...")
    res = evaluate_test_set(out_dir / "best_model.pt", selected_threshold, train_loader_no_shuf, test_loader, device, out_dir)
    
    # Add metadata
    res["seed"] = seed
    res["best_epoch"] = best_epoch
    res["best_val_roc_auc"] = best_val_auc
    res["best_val_loss"] = best_val_loss
    res["selected_threshold"] = selected_threshold
    res["total_training_time"] = total_time
    if torch.cuda.is_available():
        res["gpu_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        res["gpu_memory_mb"] = 0.0
        
    with open(out_dir / "seed_results.json", "w") as f:
        json.dump(res, f, indent=4)
        
    return res

def main():
    base_results_dir = Path(r"D:\DEQ-AD\results")
    final_dir = base_results_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    seeds = [42, 123, 2026]
    all_results = []
    
    for seed in seeds:
        res = train_and_eval_seed(seed, base_results_dir, final_dir, device)
        all_results.append(res)
        
    # Summarize
    df = pd.DataFrame(all_results)
    
    cols_to_export = [
        "seed", "best_epoch", "best_val_roc_auc", "selected_threshold", 
        "test_accuracy", "test_precision", "test_recall", "test_f1", 
        "test_roc_auc", "test_sensitivity", "test_specificity", 
        "cdr_accuracy", "cdr_macro_f1", "avg_deq_iterations", "max_deq_iterations", 
        "non_converged_solves", "convergence_percentage", "feature_mean_std", 
        "avg_pairwise_distance", "pc1_variance", "pc2_variance"
    ]
    df_export = df[cols_to_export]
    df_export.to_csv(final_dir / "final_seed_comparison.csv", index=False)
    
    # Calculate stats for multi_seed_summary.csv
    metrics = ["test_accuracy", "test_f1", "test_roc_auc", "cdr_macro_f1", "avg_deq_iterations"]
    summary_stats = []
    
    for m in metrics:
        vals = df[m].values
        summary_stats.append({
            "metric": m,
            "mean": np.mean(vals),
            "std": np.std(vals, ddof=1),
            "min": np.min(vals),
            "max": np.max(vals)
        })
        
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv(final_dir / "multi_seed_summary.csv", index=False)
    
    print("\n" + "=" * 60)
    print("ALL SEEDS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Metrics Summary:")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()
