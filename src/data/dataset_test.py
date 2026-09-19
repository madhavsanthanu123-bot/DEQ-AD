import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from src.data.oasis_dataset import create_oasis_splits, OASISDataset

def main():
    print("=" * 60)
    print("OASIS-1 DATASET & DATALOADER TEST")
    print("=" * 60)
    
    # Paths
    manifest_path = r"D:\DEQ-AD\results\oasis_manifest.csv"
    output_dir = r"D:\DEQ-AD\results"
    
    print("\nCreating splits...")
    train_df, val_df, test_df = create_oasis_splits(manifest_path, output_dir, seed=42)
    
    splits = {
        "Train": train_df,
        "Validation": val_df,
        "Test": test_df
    }
    
    # Print statistics
    for name, df in splits.items():
        print(f"\n--- {name} Split ---")
        print(f"Total samples: {len(df)}")
        print(f"Control (Label 0): {len(df[df['Label'] == 0])}")
        print(f"Impaired (Label 1): {len(df[df['Label'] == 1])}")
        
        print("CDR Distribution:")
        print(df['CDR'].value_counts().sort_index().to_string())

    # Verify leakage again
    train_subs = set(train_df['Subject_ID'])
    val_subs = set(val_df['Subject_ID'])
    test_subs = set(test_df['Subject_ID'])
    
    assert len(train_subs.intersection(val_subs)) == 0, "Leakage found between train and val!"
    assert len(train_subs.intersection(test_subs)) == 0, "Leakage found between train and test!"
    assert len(val_subs.intersection(test_subs)) == 0, "Leakage found between val and test!"
    
    print("\n[PASS] Leakage check passed: No overlapping subjects between splits.")
    
    # Test PyTorch Dataset and DataLoader
    print("\nInitializing PyTorch Datasets...")
    train_dataset = OASISDataset(Path(output_dir) / "train_manifest.csv")
    
    print("Creating DataLoader (batch_size=2)...")
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Load one batch
    print("Loading one batch...")
    batch = next(iter(train_loader))
    
    images = batch["image"]
    labels = batch["label"]
    cdrs = batch["cdr"]
    subject_ids = batch["subject_id"]
    
    print("\n--- Batch Information ---")
    print(f"Image shape: {images.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Labels: {labels.tolist()}")
    print(f"CDRs: {cdrs.tolist()}")
    print(f"Subject IDs: {subject_ids}")
    
    # Verify values
    print("\nVerifying image values...")
    has_nan = torch.isnan(images).any().item()
    has_inf = torch.isinf(images).any().item()
    
    print(f"Contains NaN: {has_nan}")
    print(f"Contains Inf: {has_inf}")
    
    assert not has_nan, "Images contain NaN values!"
    assert not has_inf, "Images contain Inf values!"
    
    print("[PASS] Value check passed: All values are finite.")
    
    # Test CUDA transfer
    print("\nTesting CUDA transfer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target device: {device}")
    
    try:
        images_gpu = images.to(device)
        labels_gpu = labels.to(device)
        print(f"Transferred image shape on device: {images_gpu.shape}")
        print(f"Transferred image device: {images_gpu.device}")
        print("[PASS] CUDA transfer successful!")
    except Exception as e:
        print(f"[FAIL] CUDA transfer failed: {e}")
        
    print("\n=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()
