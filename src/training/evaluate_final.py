import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

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

def evaluate_test_set(model_path, threshold, train_loader, test_loader, device, out_dir):
    model = DEQADModel(beta=0.7).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=False))
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
            preds = (probs >= threshold).astype(int)
            
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
                
    acc = accuracy_score(cls_labels, cls_preds)
    prec = precision_score(cls_labels, cls_preds, zero_division=0)
    rec = recall_score(cls_labels, cls_preds, zero_division=0)
    f1 = f1_score(cls_labels, cls_preds, zero_division=0)
    try:
        auc = roc_auc_score(cls_labels, cls_probs)
    except:
        auc = 0.0
        
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
    cm_cdr = confusion_matrix(cdr_true, cdr_preds, labels=[0, 1, 2, 3])
    
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
    else:
        avg_pdist = 0.0
        
    # PCA
    pca = PCA(n_components=2, random_state=42)
    tr_pca = pca.fit_transform(tr_f)
    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    
    # Dump CMs
    df_cm_bin = pd.DataFrame(cm_bin, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
    df_cm_bin.to_csv(out_dir / "binary_confusion_matrix.csv")
    
    df_cm_cdr = pd.DataFrame(cm_cdr, index=["True_0.0", "True_0.5", "True_1.0", "True_2.0"], 
                             columns=["Pred_0.0", "Pred_0.5", "Pred_1.0", "Pred_2.0"])
    df_cm_cdr.to_csv(out_dir / "cdr_confusion_matrix.csv")
    
    res = {
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(auc),
        "test_sensitivity": float(sensitivity),
        "test_specificity": float(specificity),
        "cdr_accuracy": float(cdr_acc),
        "cdr_macro_f1": float(cdr_macro_f1),
        "avg_deq_iterations": float(avg_deq_iters),
        "max_deq_iterations": int(max_iters),
        "non_converged_solves": int(non_converged),
        "convergence_percentage": float(conv_perc),
        "feature_mean_std": float(mean_std),
        "dims_std_below_1e3": float(p_1e3),
        "dims_std_below_1e2": float(p_1e2),
        "avg_pairwise_distance": float(avg_pdist),
        "pc1_variance": float(pc1_var),
        "pc2_variance": float(pc2_var),
        "pc1_pc2_variance": float(pc1_var + pc2_var)
    }
    
    return res
