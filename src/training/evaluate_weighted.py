import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

from src.data.oasis_dataset import OASISDataset
from src.models.deq_ad import DEQADModel

def get_cdr_class(cdr_values):
    mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
    classes = [mapping[float(val.item())] for val in cdr_values]
    return torch.tensor(classes, dtype=torch.long)

def main():
    base_results_dir = Path(r"D:\DEQ-AD\results")
    results_dir = base_results_dir / "weighted"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("CLASS-WEIGHTED DEQ-AD EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"Using device: {device}")
    
    # Load dataset
    test_dataset = OASISDataset(base_results_dir / "test_manifest.csv")
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Load model & threshold
    model = DEQADModel().to(device)
    checkpoint = torch.load(results_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    threshold = checkpoint.get("selected_threshold", 0.5)
    print(f"\nLoaded validation-selected threshold: {threshold:.4f}")
    
    cls_labels, cls_probs, cls_preds = [], [], []
    cdr_true, cdr_preds = [], []
    
    deq_iters = []
    non_converged = 0
    
    print("\nRunning inference...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()
            cdrs = get_cdr_class(batch["cdr"]).numpy()
            
            outputs = model(images)
            logits_cls = outputs["classification_logits"].squeeze(1)
            logits_cdr = outputs["cdr_logits"]
            
            probs = torch.sigmoid(logits_cls).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            
            cdr_p = torch.argmax(logits_cdr, dim=1).cpu().numpy()
            
            cls_labels.extend(labels)
            cls_probs.extend(probs)
            cls_preds.extend(preds)
            
            cdr_true.extend(cdrs)
            cdr_preds.extend(cdr_p)
            
            deq_iters.append(model.deq.last_iterations)
            if not model.deq.last_converged:
                non_converged += 1
                
    # Binary metrics
    acc = accuracy_score(cls_labels, cls_preds)
    prec = precision_score(cls_labels, cls_preds, zero_division=0)
    rec = recall_score(cls_labels, cls_preds, zero_division=0)
    f1 = f1_score(cls_labels, cls_preds, zero_division=0)
    try:
        auc = roc_auc_score(cls_labels, cls_probs)
    except:
        auc = 0.0
        
    cm_bin = confusion_matrix(cls_labels, cls_preds, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    print("\n--- Binary Classification Results ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {prec:.4f}")
    print(f"Recall:      {rec:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    print("\nBinary Confusion Matrix:")
    print(cm_bin)
    
    # CDR metrics
    cdr_acc = accuracy_score(cdr_true, cdr_preds)
    cdr_macro_f1 = f1_score(cdr_true, cdr_preds, average="macro", zero_division=0)
    cm_cdr = confusion_matrix(cdr_true, cdr_preds, labels=[0, 1, 2, 3])
    
    print("\n--- CDR Severity Results ---")
    print(f"Accuracy: {cdr_acc:.4f}")
    print(f"Macro-F1: {cdr_macro_f1:.4f}")
    print("CDR Confusion Matrix:")
    print(cm_cdr)
    
    avg_deq = np.mean(deq_iters)
    print(f"\nDEQ Avg Iterations: {avg_deq:.2f} (Non-converged: {non_converged})")
    
    # Save Results
    results = {
        "binary_classification": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "selected_threshold": threshold
        },
        "cdr_classification": {
            "accuracy": cdr_acc,
            "macro_f1": cdr_macro_f1
        },
        "deq": {
            "average_iterations": avg_deq,
            "non_converged_solves": non_converged
        }
    }
    
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    df_cm_bin = pd.DataFrame(cm_bin, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    df_cm_bin.to_csv(results_dir / "binary_confusion_matrix.csv")
    
    df_cm_cdr = pd.DataFrame(cm_cdr, index=["True_0.0", "True_0.5", "True_1.0", "True_2.0"], 
                             columns=["Pred_0.0", "Pred_0.5", "Pred_1.0", "Pred_2.0"])
    df_cm_cdr.to_csv(results_dir / "cdr_confusion_matrix.csv")
    
    # ROC Curve Plot
    if len(np.unique(cls_labels)) > 1:
        fpr, tpr, _ = roc_curve(cls_labels, cls_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"Weighted DEQ-AD (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve on Test Set")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(results_dir / "roc_curve.png")
        plt.close()
        
    # Experiment Comparison
    comparison_file = base_results_dir / "experiment_comparison.csv"
    
    new_row = {
        "Experiment": "Class-Weighted DEQ",
        "Test Accuracy": acc,
        "Test Precision": prec,
        "Test Recall": rec,
        "Test F1": f1,
        "Test ROC-AUC": auc,
        "CDR Accuracy": cdr_acc,
        "CDR Macro-F1": cdr_macro_f1,
        "Selected Threshold": threshold,
        "Avg DEQ Iters": avg_deq,
        "Non-converged Solves": non_converged
    }
    
    if not comparison_file.exists():
        # Create with baseline as well
        baseline_row = {
            "Experiment": "Baseline Unweighted DEQ",
            "Test Accuracy": 0.5833,
            "Test Precision": 0.0,
            "Test Recall": 0.0,
            "Test F1": 0.0,
            "Test ROC-AUC": 0.6222,
            "CDR Accuracy": 0.5833,
            "CDR Macro-F1": 0.2456,
            "Selected Threshold": 0.5,
            "Avg DEQ Iters": 9.0,
            "Non-converged Solves": 0
        }
        df_comp = pd.DataFrame([baseline_row, new_row])
    else:
        df_comp = pd.read_csv(comparison_file)
        # Check if already exists, else append
        if not (df_comp["Experiment"] == "Class-Weighted DEQ").any():
            df_comp = pd.concat([df_comp, pd.DataFrame([new_row])], ignore_index=True)
            
    df_comp.to_csv(comparison_file, index=False)
    
    print(f"\nResults saved to {results_dir}")
    print(f"Comparison saved to {comparison_file}")
    print("============================================================")

if __name__ == "__main__":
    main()
