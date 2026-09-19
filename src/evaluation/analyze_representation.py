import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

from src.data.oasis_dataset import OASISDataset
from src.models.deq_ad import DEQADModel

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_features(model, dataloader, device):
    features = []
    labels = []
    cdrs = []
    subjects = []
    deq_iters = []
    non_converged = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            lab = batch["label"].cpu().numpy()
            cdr = batch["cdr"].cpu().numpy()
            sub = batch["subject_id"]
            
            outputs = model(images)
            eq_feat = outputs["equilibrium_features"].cpu().numpy()
            
            features.append(eq_feat)
            labels.extend(lab)
            cdrs.extend(cdr)
            subjects.extend(sub)
            
            deq_iters.append(model.deq.last_iterations)
            if not model.deq.last_converged:
                non_converged += 1
                
    features = np.vstack(features)
    labels = np.array(labels)
    cdrs = np.array(cdrs)
    subjects = np.array(subjects)
    
    return features, labels, cdrs, subjects, deq_iters, non_converged

def save_features_csv(features, labels, cdrs, subjects, filepath):
    cols = ["subject_id", "binary_label", "CDR"] + [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(np.column_stack((subjects, labels, cdrs, features)), columns=cols)
    # Convert numerical types back properly
    df["binary_label"] = df["binary_label"].astype(int)
    df["CDR"] = df["CDR"].astype(float)
    for i in range(features.shape[1]):
        df[f"feature_{i}"] = df[f"feature_{i}"].astype(float)
    df.to_csv(filepath, index=False)
    return df

def analyze_collapse(features):
    std_devs = np.std(features, axis=0)
    mean_std = np.mean(std_devs)
    
    perc_lt_1e3 = np.mean(std_devs < 1e-3) * 100
    perc_lt_1e2 = np.mean(std_devs < 1e-2) * 100
    
    if len(features) > 1:
        avg_pdist = np.mean(pdist(features, metric='euclidean'))
    else:
        avg_pdist = 0.0
        
    # unique rows
    unique_rows = np.unique(features, axis=0)
    num_unique = len(unique_rows)
    
    return mean_std, perc_lt_1e3, perc_lt_1e2, avg_pdist, num_unique

def main():
    set_seed(42)
    
    base_dir = Path(r"D:\DEQ-AD")
    results_dir = base_dir / "results"
    chkpt_dir = base_dir / "checkpoints"
    out_dir = results_dir / "representation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = DEQADModel().to(device)
    checkpoint = torch.load(chkpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Dataloaders
    train_ds = OASISDataset(results_dir / "train_manifest.csv")
    val_ds = OASISDataset(results_dir / "val_manifest.csv")
    test_ds = OASISDataset(results_dir / "test_manifest.csv")
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)
    
    # 2. Extract features
    print("Extracting features from Train set...")
    tr_f, tr_y, tr_cdr, tr_sub, tr_iters, tr_nc = extract_features(model, train_loader, device)
    
    print("Extracting features from Val set...")
    va_f, va_y, va_cdr, va_sub, va_iters, va_nc = extract_features(model, val_loader, device)
    
    print("Extracting features from Test set...")
    te_f, te_y, te_cdr, te_sub, te_iters, te_nc = extract_features(model, test_loader, device)
    
    all_iters = tr_iters + va_iters + te_iters
    total_nc = tr_nc + va_nc + te_nc
    avg_iters = np.mean(all_iters)
    
    # 4. Save CSVs
    tr_df = save_features_csv(tr_f, tr_y, tr_cdr, tr_sub, out_dir / "train_features.csv")
    va_df = save_features_csv(va_f, va_y, va_cdr, va_sub, out_dir / "val_features.csv")
    te_df = save_features_csv(te_f, te_y, te_cdr, te_sub, out_dir / "test_features.csv")
    
    # 5 & 6. Feature Statistics and Collapse Detection (on train set as representative)
    mean_std, p_1e3, p_1e2, avg_pdist, num_unique = analyze_collapse(tr_f)
    
    print("\nFeature Collapse Statistics (Training Set):")
    print(f"Average Feature STD: {mean_std:.6f}")
    print(f"% Dims with STD < 1e-3: {p_1e3:.2f}%")
    print(f"% Dims with STD < 1e-2: {p_1e2:.2f}%")
    print(f"Average Pairwise Distance: {avg_pdist:.6f}")
    print(f"Unique Feature Vectors: {num_unique} / {len(tr_f)}")
    
    collapse_status = "No strong evidence of feature collapse"
    if p_1e3 > 50 or mean_std < 1e-2:
        collapse_status = "Evidence of feature collapse"
    elif p_1e2 > 30 or mean_std < 0.1:
        collapse_status = "Partial feature collapse"
        
    print(f"\nConclusion: {collapse_status}")
    
    # 7. PCA
    print("\nRunning PCA...")
    pca = PCA(n_components=2, random_state=42)
    pca.fit(tr_f) # Fit strictly on train
    evr = pca.explained_variance_ratio_
    print(f"PCA Explained Variance Ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}")
    
    tr_pca = pca.transform(tr_f)
    va_pca = pca.transform(va_f)
    te_pca = pca.transform(te_f)
    
    # Build pca_features.csv
    splits_list = ["Train"]*len(tr_pca) + ["Validation"]*len(va_pca) + ["Test"]*len(te_pca)
    pca_df = pd.DataFrame({
        "subject_id": np.concatenate([tr_sub, va_sub, te_sub]),
        "split": splits_list,
        "binary_label": np.concatenate([tr_y, va_y, te_y]),
        "CDR": np.concatenate([tr_cdr, va_cdr, te_cdr]),
        "PC1": np.concatenate([tr_pca[:, 0], va_pca[:, 0], te_pca[:, 0]]),
        "PC2": np.concatenate([tr_pca[:, 1], va_pca[:, 1], te_pca[:, 1]])
    })
    pca_df.to_csv(out_dir / "pca_features.csv", index=False)
    
    # Plot PCAs
    def plot_pca(df, title, filepath):
        plt.figure(figsize=(8, 6))
        c0 = df[df["binary_label"] == 0]
        c1 = df[df["binary_label"] == 1]
        plt.scatter(c0["PC1"], c0["PC2"], label="Control (0)", alpha=0.7)
        plt.scatter(c1["PC1"], c1["PC2"], label="Impaired (1)", alpha=0.7)
        plt.title(title)
        plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
        plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
        plt.legend()
        plt.grid(True)
        plt.savefig(filepath)
        plt.close()

    plot_pca(pca_df[pca_df["split"] == "Train"], "PCA - Training Set", out_dir / "pca_train.png")
    plot_pca(pca_df[pca_df["split"] == "Validation"], "PCA - Validation Set", out_dir / "pca_validation.png")
    plot_pca(pca_df[pca_df["split"] == "Test"], "PCA - Test Set", out_dir / "pca_test.png")
    
    # 8. PCA combined
    plt.figure(figsize=(10, 8))
    for sp, marker in zip(["Train", "Validation", "Test"], ['o', 's', '^']):
        sub_df = pca_df[pca_df["split"] == sp]
        c0 = sub_df[sub_df["binary_label"] == 0]
        c1 = sub_df[sub_df["binary_label"] == 1]
        plt.scatter(c0["PC1"], c0["PC2"], marker=marker, color='blue', label=f'{sp} Control', alpha=0.6)
        plt.scatter(c1["PC1"], c1["PC2"], marker=marker, color='red', label=f'{sp} Impaired', alpha=0.6)
    plt.title("PCA - All Splits")
    plt.xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "pca_all_splits.png")
    plt.close()
    
    # 10. PCA CDR
    plt.figure(figsize=(8, 6))
    for cdr_val in [0.0, 0.5, 1.0, 2.0]:
        sub_df = pca_df[pca_df["CDR"] == cdr_val]
        if len(sub_df) > 0:
            plt.scatter(sub_df["PC1"], sub_df["PC2"], label=f'CDR {cdr_val}', alpha=0.7)
    plt.title("PCA - By CDR Severity")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "pca_cdr.png")
    plt.close()
    
    # 9. t-SNE
    print("Running t-SNE...")
    all_f = np.vstack([tr_f, va_f, te_f])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    all_tsne = tsne.fit_transform(all_f)
    
    pca_df["tSNE1"] = all_tsne[:, 0]
    pca_df["tSNE2"] = all_tsne[:, 1]
    
    plt.figure(figsize=(10, 8))
    for sp, marker in zip(["Train", "Validation", "Test"], ['o', 's', '^']):
        sub_df = pca_df[pca_df["split"] == sp]
        c0 = sub_df[sub_df["binary_label"] == 0]
        c1 = sub_df[sub_df["binary_label"] == 1]
        plt.scatter(c0["tSNE1"], c0["tSNE2"], marker=marker, color='blue', label=f'{sp} Control', alpha=0.6)
        plt.scatter(c1["tSNE1"], c1["tSNE2"], marker=marker, color='red', label=f'{sp} Impaired', alpha=0.6)
    plt.title("t-SNE - All Splits")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_all_splits.png")
    plt.close()
    
    # 11. Feature Separation Statistics (Cohen's d)
    print("Calculating Cohen's d...")
    tr_0 = tr_f[tr_y == 0]
    tr_1 = tr_f[tr_y == 1]
    
    mean_0 = np.mean(tr_0, axis=0)
    mean_1 = np.mean(tr_1, axis=0)
    std_0 = np.std(tr_0, axis=0, ddof=1)
    std_1 = np.std(tr_1, axis=0, ddof=1)
    
    n_0, n_1 = len(tr_0), len(tr_1)
    pooled_std = np.sqrt(((n_0 - 1) * std_0**2 + (n_1 - 1) * std_1**2) / (n_0 + n_1 - 2))
    
    # avoid div by zero
    pooled_std = np.where(pooled_std == 0, 1e-8, pooled_std)
    cohens_d = (mean_1 - mean_0) / pooled_std
    
    feat_stats = []
    for i in range(tr_f.shape[1]):
        feat_stats.append({
            "Feature": i,
            "Cohen_d": cohens_d[i],
            "Abs_Cohen_d": abs(cohens_d[i]),
            "Mean_Control": mean_0[i],
            "Mean_Impaired": mean_1[i],
            "Std_Control": std_0[i],
            "Std_Impaired": std_1[i]
        })
        
    df_sep = pd.DataFrame(feat_stats)
    df_sep = df_sep.sort_values(by="Abs_Cohen_d", ascending=False)
    df_sep.to_csv(out_dir / "feature_separation.csv", index=False)
    
    # 12. Summary JSON
    summary = {
        "dataset": {
            "total_subjects": len(all_f),
            "train_subjects": len(tr_f),
            "val_subjects": len(va_f),
            "test_subjects": len(te_f)
        },
        "representation": {
            "feature_dimensionality": tr_f.shape[1],
            "mean_feature_std": float(mean_std),
            "collapsed_dimensions_pct_1e3": float(p_1e3),
            "collapsed_dimensions_pct_1e2": float(p_1e2),
            "average_pairwise_distance": float(avg_pdist),
            "pca_explained_variance_ratio": [float(evr[0]), float(evr[1])],
            "conclusion": collapse_status
        },
        "deq_performance": {
            "average_iterations": float(avg_iters),
            "non_converged_count": int(total_nc)
        }
    }
    
    with open(out_dir / "representation_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\nRepresentation Analysis Complete!")
    print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
