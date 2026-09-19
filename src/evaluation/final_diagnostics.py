import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve, auc)
from scipy.spatial.distance import pdist
from scipy.stats import entropy

from src.data.oasis_dataset import OASISDataset
from src.models.deq_ad import DEQADModel

def get_cdr_class(cdr_values):
    mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
    classes = [mapping[float(val.item())] for val in cdr_values]
    return torch.tensor(classes, dtype=torch.long)

def extract_features(model, dataloader, device):
    features = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = model(images)
            features.append(outputs["equilibrium_features"].cpu().numpy())
    return np.vstack(features)

def plot_confusion_matrix(cm, classes, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
           
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(filepath)
    plt.close()

def main():
    base_dir = Path(r"D:\DEQ-AD")
    results_dir = base_dir / "results"
    final_dir = results_dir / "final"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataloaders
    train_ds = OASISDataset(results_dir / "train_manifest.csv")
    val_ds = OASISDataset(results_dir / "val_manifest.csv")
    test_ds = OASISDataset(results_dir / "test_manifest.csv")
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
    
    seeds = [42, 123, 2026]
    diagnostic_rows = []
    full_report = {"seeds": {}}
    
    roc_data = {}
    pr_data = {}
    
    for seed in seeds:
        print(f"\n>>> Running diagnostics for Seed {seed} <<<")
        seed_dir = final_dir / f"seed_{seed}"
        
        # Load Selected Threshold
        with open(seed_dir / "selected_threshold.txt", "r") as f:
            threshold = float(f.read().strip())
            
        model = DEQADModel(beta=0.7).to(device)
        model.load_state_dict(torch.load(seed_dir / "best_model.pt", weights_only=False))
        model.eval()
        
        # ---------------------------------------------------------
        # Validation Calibrations (Threshold curve & Brier)
        # ---------------------------------------------------------
        val_labels, val_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].cpu().numpy()
                logits = model(images)["classification_logits"].squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_labels.extend(labels)
                val_probs.extend(probs)
                
        val_labels = np.array(val_labels)
        val_probs = np.array(val_probs)
        
        val_brier = brier_score_loss(val_labels, val_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        
        thresholds = np.linspace(0.10, 0.90, 81)
        f1_scores = []
        for t in thresholds:
            p = (val_probs >= t).astype(int)
            f1_scores.append(f1_score(val_labels, p, zero_division=0))
            
        df_val_th = pd.DataFrame({"threshold": thresholds, "f1_score": f1_scores})
        df_val_th.to_csv(seed_dir / "validation_thresholds.csv", index=False)
        
        plt.figure(figsize=(6, 4))
        plt.plot(thresholds, f1_scores, marker='.', color='b')
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Selected ({threshold:.4f})')
        plt.title(f"Validation F1 vs Threshold (Seed {seed})")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(seed_dir / "validation_threshold_curve.png")
        plt.close()
        
        # ---------------------------------------------------------
        # Test Inference & Diagnostics
        # ---------------------------------------------------------
        sub_ids, true_bin, pred_bin, prob_bin = [], [], [], []
        true_cdr, pred_cdr_cls = [], []
        prob_cdrs = []
        deq_iters = []
        non_conv = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].cpu().numpy()
                cdrs_val = batch["cdr"].numpy()
                cdrs_class = get_cdr_class(batch["cdr"]).numpy()
                subs = batch["subject_id"]
                
                outputs = model(images)
                
                logits_cls = outputs["classification_logits"].squeeze(1)
                probs = torch.sigmoid(logits_cls).cpu().numpy()
                preds = (probs >= threshold).astype(int)
                
                logits_cdr = outputs["cdr_logits"]
                cdr_probs_batch = torch.softmax(logits_cdr, dim=1).cpu().numpy()
                cdr_preds_batch = np.argmax(cdr_probs_batch, axis=1)
                
                sub_ids.extend(subs)
                true_bin.extend(labels)
                pred_bin.extend(preds)
                prob_bin.extend(probs)
                true_cdr.extend(cdrs_val)
                pred_cdr_cls.extend(cdr_preds_batch)
                prob_cdrs.extend(cdr_probs_batch)
                
                iters = model.deq.last_iterations
                deq_iters.append(iters)
                if not model.deq.last_converged:
                    non_conv += 1
                    
        prob_cdrs = np.array(prob_cdrs)
        
        # Save Test Predictions
        df_preds = pd.DataFrame({
            "subject_id": sub_ids,
            "true_binary_label": true_bin,
            "predicted_binary_label": pred_bin,
            "binary_probability": prob_bin,
            "true_CDR": true_cdr,
            "predicted_CDR_class": pred_cdr_cls,
            "prob_CDR_0": prob_cdrs[:, 0],
            "prob_CDR_0.5": prob_cdrs[:, 1],
            "prob_CDR_1.0": prob_cdrs[:, 2],
            "prob_CDR_2.0": prob_cdrs[:, 3]
        })
        df_preds.to_csv(seed_dir / "test_predictions.csv", index=False)
        
        # Test Binary Metrics
        acc = accuracy_score(true_bin, pred_bin)
        prec = precision_score(true_bin, pred_bin, zero_division=0)
        rec = recall_score(true_bin, pred_bin, zero_division=0)
        f1 = f1_score(true_bin, pred_bin, zero_division=0)
        auc_val = roc_auc_score(true_bin, prob_bin)
        
        cm_bin = confusion_matrix(true_bin, pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm_bin.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Test ROC & PR Curves
        fpr, tpr, _ = roc_curve(true_bin, prob_bin)
        roc_data[seed] = (fpr, tpr, auc_val)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {auc_val:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - Test (Seed {seed})")
        plt.legend()
        plt.grid(True)
        plt.savefig(seed_dir / f"roc_seed_{seed}.png")
        plt.close()
        
        precision_arr, recall_arr, _ = precision_recall_curve(true_bin, prob_bin)
        pr_auc = auc(recall_arr, precision_arr)
        pr_data[seed] = (recall_arr, precision_arr, pr_auc)
        plt.figure(figsize=(6, 5))
        plt.plot(recall_arr, precision_arr, label=f'PR (AUC = {pr_auc:.3f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - Test (Seed {seed})")
        plt.legend()
        plt.grid(True)
        plt.savefig(seed_dir / f"pr_seed_{seed}.png")
        plt.close()
        
        # CDR Metrics & Confusion Matrix
        # Map true_cdr values to classes
        mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
        true_cdr_cls = [mapping[c] for c in true_cdr]
        
        cdr_acc = accuracy_score(true_cdr_cls, pred_cdr_cls)
        cdr_mac = f1_score(true_cdr_cls, pred_cdr_cls, average="macro", zero_division=0)
        
        cm_cdr = confusion_matrix(true_cdr_cls, pred_cdr_cls, labels=[0, 1, 2, 3])
        df_cm_cdr = pd.DataFrame(cm_cdr, index=["True_0.0", "True_0.5", "True_1.0", "True_2.0"], 
                                 columns=["Pred_0.0", "Pred_0.5", "Pred_1.0", "Pred_2.0"])
        df_cm_cdr.to_csv(seed_dir / "cdr_confusion_matrix.csv")
        plot_confusion_matrix(cm_cdr, ['0.0', '0.5', '1.0', '2.0'], f'CDR Confusion Matrix (Seed {seed})', seed_dir / "cdr_confusion_matrix.png")
        
        # CDR Collapse check
        unique_cdrs = np.unique(pred_cdr_cls)
        pct_cdrs = [np.mean(np.array(pred_cdr_cls) == i) * 100 for i in range(4)]
        dom_idx = np.argmax(pct_cdrs)
        dom_pct = pct_cdrs[dom_idx]
        ent = entropy([p/100.0 for p in pct_cdrs if p > 0])
        
        avg_max_prob = np.mean(np.max(prob_cdrs, axis=1))
        avg_prob_vec = np.mean(prob_cdrs, axis=0)
        
        true_counts = [np.sum(np.array(true_cdr_cls) == i) for i in range(4)]
        pred_counts = [np.sum(np.array(pred_cdr_cls) == i) for i in range(4)]
        
        cdr_collapse_status = "Collapsed" if len(unique_cdrs) == 1 else ("Partial" if dom_pct > 80 else "Diverse")
        
        # DEQ Diagnostics
        avg_iter = np.mean(deq_iters)
        med_iter = np.median(deq_iters)
        max_iter = np.max(deq_iters)
        conv_pct = 100.0 * (len(deq_iters) - non_conv) / len(deq_iters)
        
        # Representation Statistics (TRAIN ONLY)
        tr_f = extract_features(model, train_loader, device)
        std_devs = np.std(tr_f, axis=0)
        mean_std = np.mean(std_devs)
        med_std = np.median(std_devs)
        min_std = np.min(std_devs)
        max_std = np.max(std_devs)
        p_1e3 = np.mean(std_devs < 1e-3) * 100
        p_1e2 = np.mean(std_devs < 1e-2) * 100
        
        if len(tr_f) > 1:
            avg_pdist = np.mean(pdist(tr_f, metric='euclidean'))
        else:
            avg_pdist = 0.0
            
        row = {
            "seed": f"seed_{seed}",
            "test_roc_auc": float(auc_val),
            "test_f1": float(f1),
            "test_accuracy": float(acc),
            "test_precision": float(prec),
            "test_recall": float(rec),
            "test_specificity": float(spec),
            "cdr_accuracy": float(cdr_acc),
            "cdr_macro_f1": float(cdr_mac),
            "unique_predicted_cdr_classes": int(len(unique_cdrs)),
            "dominant_predicted_cdr_class": int(dom_idx),
            "dominant_predicted_cdr_percentage": float(dom_pct),
            "brier_score": float(val_brier),
            "mean_feature_std": float(mean_std),
            "median_feature_std": float(med_std),
            "avg_pairwise_distance": float(avg_pdist),
            "avg_deq_iterations": float(avg_iter),
            "max_deq_iterations": int(max_iter),
            "convergence_percentage": float(conv_pct)
        }
        diagnostic_rows.append(row)
        
        full_report["seeds"][f"seed_{seed}"] = {
            "binary_metrics": {
                "test_roc_auc": float(auc_val), "test_f1": float(f1), "test_accuracy": float(acc),
                "test_precision": float(prec), "test_recall": float(rec), "test_specificity": float(spec)
            },
            "cdr_metrics": {
                "test_accuracy": float(cdr_acc), "test_macro_f1": float(cdr_mac)
            },
            "cdr_distributions": {
                "true_counts": [int(x) for x in true_counts],
                "pred_counts": [int(x) for x in pred_counts],
                "unique_classes": int(len(unique_cdrs)),
                "entropy": float(ent),
                "average_max_probability": float(avg_max_prob),
                "average_probability_vector": [float(x) for x in avg_prob_vec],
                "collapse_status": cdr_collapse_status
            },
            "calibration": {
                "val_brier_score": float(val_brier),
                "val_roc_auc": float(val_auc)
            },
            "representation": {
                "mean_std": float(mean_std),
                "median_std": float(med_std),
                "min_std": float(min_std),
                "max_std": float(max_std),
                "pct_std_lt_1e3": float(p_1e3),
                "pct_std_lt_1e2": float(p_1e2),
                "avg_pairwise_distance": float(avg_pdist)
            },
            "deq": {
                "avg_iter": float(avg_iter),
                "median_iter": float(med_iter),
                "max_iter": int(max_iter),
                "conv_pct": float(conv_pct),
                "non_conv_count": int(non_conv)
            },
            "cdr_confusion_matrix": cm_cdr.tolist()
        }
        
    # Combined ROC
    plt.figure(figsize=(7, 6))
    for seed in seeds:
        fpr, tpr, a = roc_data[seed]
        plt.plot(fpr, tpr, label=f'Seed {seed} (AUC={a:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Combined ROC Curves - Final Models")
    plt.legend()
    plt.grid(True)
    plt.savefig(final_dir / "roc_all_seeds.png")
    plt.close()
    
    # Combined PR
    plt.figure(figsize=(7, 6))
    for seed in seeds:
        rec_arr, prec_arr, a = pr_data[seed]
        plt.plot(rec_arr, prec_arr, label=f'Seed {seed} (AUC={a:.3f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Combined Precision-Recall Curves - Final Models")
    plt.legend()
    plt.grid(True)
    plt.savefig(final_dir / "pr_all_seeds.png")
    plt.close()
    
    # Final Summary CSV & JSON
    df_diag = pd.DataFrame(diagnostic_rows)
    df_diag.to_csv(final_dir / "final_diagnostics.csv", index=False)
    
    # Aggregate stats for JSON
    agg_stats = {}
    for col in df_diag.select_dtypes(include=[np.number]).columns:
        vals = df_diag[col].values
        agg_stats[col] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        }
        
    full_report["summary_statistics"] = agg_stats
    
    with open(final_dir / "final_report.json", "w") as f:
        json.dump(full_report, f, indent=4)
        
    print("\nFinal Diagnostics Completed Successfully!")
    print(f"Results saved to: {final_dir}")

if __name__ == "__main__":
    main()
