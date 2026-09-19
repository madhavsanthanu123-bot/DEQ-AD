import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from .preprocessing import preprocess_mri

def create_oasis_splits(manifest_path, output_dir, seed=42):
    """
    Creates subject-level train/val/test splits from the OASIS manifest.
    
    Args:
        manifest_path (str or Path): Path to oasis_manifest.csv
        output_dir (str or Path): Directory to save the split manifests
        seed (int): Random seed for reproducibility
    """
    df = pd.read_csv(manifest_path)
    
    # Use only samples with valid CDR and valid MRI paths
    df = df[df['CDR'].notna() & df['MRI_Path'].notna()]
    
    # Ensure MRI_Found is True if the column exists
    if 'MRI_Found' in df.columns:
        # Convert to boolean if it's not already
        if df['MRI_Found'].dtype == object:
            df = df[df['MRI_Found'].astype(str).str.lower() == 'true']
        else:
            df = df[df['MRI_Found'] == True]
            
    # Verify uniqueness of Subject_ID to ensure subject-level splitting
    # since we have 1 labeled MRI per session/subject in the current subset.
    assert df['Subject_ID'].nunique() == len(df), "Duplicate Subject_IDs found in filtered data!"
    
    # Stratified split: 70% train, 15% val, 15% test
    # First split: 70% train, 30% temp (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        random_state=seed, 
        stratify=df['Label']
    )
    
    # Second split: 50% of temp for val, 50% of temp for test (15% / 15% overall)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=seed, 
        stratify=temp_df['Label']
    )
    
    # Verify no leakage
    train_subs = set(train_df['Subject_ID'])
    val_subs = set(val_df['Subject_ID'])
    test_subs = set(test_df['Subject_ID'])
    
    assert len(train_subs.intersection(val_subs)) == 0, "Leakage between train and val splits!"
    assert len(train_subs.intersection(test_subs)) == 0, "Leakage between train and test splits!"
    assert len(val_subs.intersection(test_subs)) == 0, "Leakage between val and test splits!"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train_manifest.csv", index=False)
    val_df.to_csv(output_dir / "val_manifest.csv", index=False)
    test_df.to_csv(output_dir / "test_manifest.csv", index=False)
    
    return train_df, val_df, test_df


class OASISDataset(Dataset):
    def __init__(self, manifest_path):
        """
        PyTorch Dataset for OASIS-1 MRI data.
        
        Args:
            manifest_path (str or Path): Path to a split manifest csv (e.g. train_manifest.csv)
        """
        self.df = pd.read_csv(manifest_path)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        mri_path = row['MRI_Path']
        label = int(row['Label'])
        cdr = float(row['CDR'])
        subject_id = str(row['Subject_ID'])
        
        # Lazy load and preprocess the MRI volume
        image = preprocess_mri(mri_path) # Returns shape [1, 96, 112, 96] float32 tensor
        
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "cdr": torch.tensor(cdr, dtype=torch.float32),
            "subject_id": subject_id,
            "mri_path": mri_path
        }
