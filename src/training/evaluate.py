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
    results_dir = Path(r"D:\DEQ-AD\results")
    chkpt_dir = Path(r"D:\DEQ-AD\checkpoints")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DEQ-AD EVALUATION ON TEST SET")
    print("=" * 60)
    print(f"Using device: {device}")
    
    # Load dataset
    test_dataset = OASISDataset(results_dir / "test_manifest.csv")
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Print dataset info
    df_test = pd.read_csv(results_dir / "test_manifest.csv")
    n_total = len(df_test)
    n_control = len(df_test[df_test["Label"] == 0])
    n_impaired = len(df_test[df_test["Label"] == 1])
    
    print(f"\nTest subjects: {n_total}")
    print(f"Control: {n_control}")
    print(f"Impaired: {n_impaired}")
    print("CDR Distribution:")
    print(df_test['CDR'].value_counts().sort_index().to_string())
    
    # Load model
    model = DEQADModel().to(device)
    checkpoint = torch.load(chkpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    cls_labels, cls_probs, cls_preds = [], [], []
    cdr_true, cdr_preds = [], []
    
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
            preds = (probs > 0.5).astype(int)
            
            cdr_p = torch.argmax(logits_cdr, dim=1).cpu().numpy()
            
            cls_labels.extend(labels)
            cls_probs.extend(probs)
            cls_preds.extend(preds)
            
            cdr_true.extend(cdrs)
            cdr_preds.extend(cdr_p)
            
    # Binary metrics
    acc = accuracy_score(cls_labels, cls_preds)
    prec = precision_score(cls_labels, cls_preds, zero_division=0)
    rec = recall_score(cls_labels, cls_preds, zero_division=0)
    f1 = f1_score(cls_labels, cls_preds, zero_division=0)
    try:
        auc = roc_auc_score(cls_labels, cls_probs)
    except:
        auc = 0.0
        
    print("\n--- Binary Classification Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # CDR metrics
    cdr_acc = accuracy_score(cdr_true, cdr_preds)
    cdr_macro_f1 = f1_score(cdr_true, cdr_preds, average="macro", zero_division=0)
    cm = confusion_matrix(cdr_true, cdr_preds, labels=[0, 1, 2, 3])
    
    print("\n--- CDR Severity Results ---")
    print(f"Accuracy: {cdr_acc:.4f}")
    print(f"Macro-F1: {cdr_macro_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Save Results
    results = {
        "binary_classification": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        },
        "cdr_classification": {
            "accuracy": cdr_acc,
            "macro_f1": cdr_macro_f1
        }
    }
    
    with open(results_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    df_cm = pd.DataFrame(cm, index=["True_0.0", "True_0.5", "True_1.0", "True_2.0"], 
                             columns=["Pred_0.0", "Pred_0.5", "Pred_1.0", "Pred_2.0"])
    df_cm.to_csv(results_dir / "cdr_confusion_matrix.csv")
    
    # ROC Curve Plot
    if len(np.unique(cls_labels)) > 1:
        fpr, tpr, _ = roc_curve(cls_labels, cls_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"DEQ-AD (AUC = {auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve on Test Set")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(results_dir / "roc_curve.png")
        plt.close()
        
    print(f"\nResults saved to {results_dir}")
    print("============================================================")

if __name__ == "__main__":
    main()
