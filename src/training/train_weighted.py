import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.oasis_dataset import OASISDataset
from src.models.deq_ad import DEQADModel

def set_seed(seed=42):
    random.seed(seed)
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

def find_best_threshold(y_true, y_prob):
    best_f1 = -1.0
    best_thresh = 0.5
    
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
                
    return best_thresh, best_f1

def run_safety_check(model, dataloader, device, optimizer, criterion_cls, criterion_cdr):
    print("=" * 60)
    print("FIRST RUN SAFETY CHECK (WEIGHTED)")
    print("=" * 60)
    
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
    
    # Check gradients
    grad_finite = True
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                grad_finite = False
                break
                
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
    
    print(f"Loss finite: {not has_nan_inf}")
    print(f"DEQ converged: {deq_converged}")
    print(f"Gradients finite: {grad_finite}")
    print(f"GPU memory allocated: {mem_alloc:.2f} MB")
    print("[PASS] Safety check completed successfully.\n")

def plot_curves(history_df, results_dir):
    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Total Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'loss_curve.png')
    plt.close()
    
    # F1 curve
    plt.figure(figsize=(8, 6))
    plt.plot(history_df['epoch'], history_df['train_f1'], label='Train F1')
    plt.plot(history_df['epoch'], history_df['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Binary F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'classification_f1_curve.png')
    plt.close()

def main():
    set_seed(42)
    
    base_results_dir = Path(r"D:\DEQ-AD\results")
    results_dir = base_results_dir / "weighted"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # DataLoaders (using exact existing split)
    train_dataset = OASISDataset(base_results_dir / "train_manifest.csv")
    val_dataset = OASISDataset(base_results_dir / "val_manifest.csv")
    
    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=pin_mem)
    
    # Model
    model = DEQADModel().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Class-weighted loss
    pos_weight = torch.tensor([94.0 / 70.0]).to(device)
    criterion_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_cdr = nn.CrossEntropyLoss()
    
    run_safety_check(model, train_loader, device, optimizer, criterion_cls, criterion_cdr)
    
    epochs = 20
    patience = 5
    best_val_f1 = -1.0
    epochs_no_improve = 0
    
    history = []
    
    for epoch in range(1, epochs + 1):
        print("=" * 60)
        print(f"EPOCH {epoch}/{epochs}")
        print("=" * 60)
        
        # Training
        model.train()
        train_loss, train_cls_loss, train_cdr_loss = 0.0, 0.0, 0.0
        train_labels, train_probs = [], []
        
        deq_iters = []
        non_converged = 0
        
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
            train_cls_loss += loss_cls.item() * images.size(0)
            train_cdr_loss += loss_cdr.item() * images.size(0)
            
            probs = torch.sigmoid(logits_cls).detach().cpu().numpy()
            train_probs.extend(probs)
            train_labels.extend(labels.cpu().numpy())
            
            deq_iters.append(model.deq.last_iterations)
            if not model.deq.last_converged:
                non_converged += 1
                
        n_train = len(train_dataset)
        train_loss /= n_train
        train_cls_loss /= n_train
        train_cdr_loss /= n_train
        
        # Validation
        model.eval()
        val_loss, val_cls_loss, val_cdr_loss = 0.0, 0.0, 0.0
        val_labels, val_probs = [], []
        
        val_deq_iters = []
        val_non_converged = 0
        
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
                val_cls_loss += loss_cls.item() * images.size(0)
                val_cdr_loss += loss_cdr.item() * images.size(0)
                
                probs = torch.sigmoid(logits_cls).cpu().numpy()
                val_probs.extend(probs)
                val_labels.extend(labels.cpu().numpy())
                
                val_deq_iters.append(model.deq.last_iterations)
                if not model.deq.last_converged:
                    val_non_converged += 1
                    
        n_val = len(val_dataset)
        val_loss /= n_val
        val_cls_loss /= n_val
        val_cdr_loss /= n_val
        
        # Find best threshold on validation
        best_val_thresh, v_f1 = find_best_threshold(np.array(val_labels), np.array(val_probs))
        
        # Calculate metrics for train/val using the validation threshold
        train_preds = (np.array(train_probs) >= best_val_thresh).astype(int)
        val_preds = (np.array(val_probs) >= best_val_thresh).astype(int)
        
        t_acc, t_prec, t_rec, t_f1, _ = safe_metrics(train_labels, train_preds)
        v_acc, v_prec, v_rec, _, v_auc = safe_metrics(val_labels, val_preds, val_probs)
        
        avg_deq_iters = np.mean(deq_iters)
        avg_val_deq_iters = np.mean(val_deq_iters)
        
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"TRAIN: Loss={train_loss:.4f} (Cls:{train_cls_loss:.4f}, CDR:{train_cdr_loss:.4f})")
        print(f"       Acc={t_acc:.4f} Prec={t_prec:.4f} Rec={t_rec:.4f} F1={t_f1:.4f}")
        print(f"VAL  : Loss={val_loss:.4f} (Cls:{val_cls_loss:.4f}, CDR:{val_cdr_loss:.4f})")
        print(f"       Acc={v_acc:.4f} Prec={v_prec:.4f} Rec={v_rec:.4f} F1={v_f1:.4f} AUC={v_auc if v_auc else 0:.4f}")
        print(f"       Selected Threshold={best_val_thresh:.4f}")
        print(f"DEQ  : Train Avg Iters={avg_deq_iters:.2f} (Non-converged={non_converged})")
        print(f"       Val Avg Iters={avg_val_deq_iters:.2f} (Non-converged={val_non_converged})")
        
        if torch.cuda.is_available():
            print(f"GPU Mem: Alloc={torch.cuda.memory_allocated() / (1024**2):.1f}MB, Reserved={torch.cuda.memory_reserved() / (1024**2):.1f}MB")
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_cls_loss": train_cls_loss,
            "train_cdr_loss": train_cdr_loss,
            "train_accuracy": t_acc,
            "train_precision": t_prec,
            "train_recall": t_rec,
            "train_f1": t_f1,
            "val_loss": val_loss,
            "val_cls_loss": val_cls_loss,
            "val_cdr_loss": val_cdr_loss,
            "val_accuracy": v_acc,
            "val_precision": v_prec,
            "val_recall": v_rec,
            "val_f1": v_f1,
            "val_auc": v_auc if v_auc else 0.0,
            "val_threshold": best_val_thresh,
            "avg_deq_iterations": avg_deq_iters,
            "nonconverged_deq": non_converged
        })
        
        # Checkpointing
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_metrics": {"f1": v_f1, "loss": val_loss, "auc": v_auc},
            "selected_threshold": best_val_thresh,
            "random_seed": 42,
            "configuration": {"batch_size": 2, "lr": 1e-4, "pos_weight": 94.0/70.0}
        }
        
        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            torch.save(checkpoint_data, results_dir / "best_model.pt")
            with open(results_dir / "selected_threshold.txt", "w") as f:
                f.write(str(best_val_thresh))
            print(">>> New best model saved based on Val F1! <<<")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs (patience={patience}).")
            break
            
    # Save history and plots
    df_history = pd.DataFrame(history)
    df_history.to_csv(results_dir / "training_history.csv", index=False)
    plot_curves(df_history, results_dir)
    print("\nTraining complete. History and plots saved.")

if __name__ == "__main__":
    main()
