from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

OASIS_ROOT = Path(r"D:\OASIS-1")
MRI_ROOT = OASIS_ROOT / "extracted"

EXCEL_FILE = (
    OASIS_ROOT /
    "oasis_cross-sectional-5708aa0a98d82080.xlsx"
)


# ============================================================
# LOAD CLINICAL DATA
# ============================================================

print("=" * 70)
print("OASIS-1 DATASET AUDIT")
print("=" * 70)

print("\nLoading clinical spreadsheet...")

df = pd.read_excel(EXCEL_FILE)

print(f"Clinical records: {len(df)}")

print("\nClinical columns:")
print(list(df.columns))


# ============================================================
# BASIC CLINICAL AUDIT
# ============================================================

print("\n" + "=" * 70)
print("CLINICAL DATA")
print("=" * 70)

print(f"Total records: {len(df)}")

print(
    f"Unique subject/session IDs: "
    f"{df['ID'].nunique()}"
)

print(
    f"Missing CDR: "
    f"{df['CDR'].isna().sum()}"
)

print(
    f"Valid CDR: "
    f"{df['CDR'].notna().sum()}"
)


# ============================================================
# CDR DISTRIBUTION
# ============================================================

print("\nCDR distribution:")

cdr_counts = (
    df["CDR"]
    .value_counts(dropna=False)
    .sort_index()
)

print(cdr_counts)


# ============================================================
# MRI SEARCH
# ============================================================

print("\n" + "=" * 70)
print("MRI FILE AUDIT")
print("=" * 70)

print("\nSearching for masked GFC MRI files...")

mri_files = list(
    MRI_ROOT.rglob("*_masked_gfc.img")
)

print(
    f"Masked MRI files found: "
    f"{len(mri_files)}"
)


# ============================================================
# CREATE MRI ID MAP
# ============================================================

mri_map = {}

for path in mri_files:

    # Example filename:
    #
    # OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.img
    #
    # Extract:
    #
    # OAS1_0001_MR1

    name = path.name

    subject_id = "_".join(
        name.split("_")[:3]
    )

    mri_map[subject_id] = str(path)


print(
    f"Unique MRI IDs: "
    f"{len(mri_map)}"
)


# ============================================================
# MATCH MRI WITH CLINICAL DATA
# ============================================================

df["MRI_Path"] = df["ID"].map(mri_map)

df["MRI_Found"] = df["MRI_Path"].notna()

print("\n" + "=" * 70)
print("MRI ↔ CLINICAL MATCHING")
print("=" * 70)

print(
    f"Clinical records with MRI: "
    f"{df['MRI_Found'].sum()}"
)

print(
    f"Clinical records without MRI: "
    f"{(~df['MRI_Found']).sum()}"
)


# ============================================================
# VALID LABELED DATA
# ============================================================

valid = df[
    df["CDR"].notna() &
    df["MRI_Found"]
].copy()

print("\n" + "=" * 70)
print("VALID LABELED DATA")
print("=" * 70)

print(
    f"Records with MRI + valid CDR: "
    f"{len(valid)}"
)

print(
    f"Unique subjects with MRI + CDR: "
    f"{valid['ID'].nunique()}"
)


# ============================================================
# BINARY LABEL
# ============================================================

valid["Label"] = (
    valid["CDR"] > 0
).astype(int)

valid["Label_Name"] = valid["Label"].map(
    {
        0: "Control",
        1: "Impaired"
    }
)


print("\nBinary classification distribution:")

print(
    valid["Label_Name"]
    .value_counts()
)


# ============================================================
# CDR SEVERITY
# ============================================================

print("\nCDR severity distribution:")

print(
    valid["CDR"]
    .value_counts()
    .sort_index()
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE / LEAKAGE CHECK")
print("=" * 70)

duplicate_ids = (
    valid["ID"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate MRI/clinical IDs: "
    f"{duplicate_ids}"
)


# ============================================================
# SUBJECT ID EXTRACTION
# ============================================================

valid["Subject_ID"] = (
    valid["ID"]
    .str.extract(
        r"(OAS1_\d+)"
    )[0]
)


print(
    f"Unique subjects: "
    f"{valid['Subject_ID'].nunique()}"
)


# ============================================================
# SUBJECT-LEVEL CLASS DISTRIBUTION
# ============================================================

subject_labels = (
    valid
    .groupby("Subject_ID")["CDR"]
    .max()
    .reset_index()
)

subject_labels["Label"] = (
    subject_labels["CDR"] > 0
).astype(int)

print("\nSubject-level label distribution:")

print(
    subject_labels["Label"]
    .value_counts()
    .rename(
        {
            0: "Control",
            1: "Impaired"
        }
    )
)


# ============================================================
# MISSING DATA SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MISSING DATA")
print("=" * 70)

for column in [
    "Age",
    "Educ",
    "SES",
    "MMSE",
    "CDR",
    "eTIV",
    "nWBV",
    "ASF"
]:

    if column in df.columns:

        missing = df[column].isna().sum()

        print(
            f"{column:8s}: "
            f"{missing} missing"
        )


# ============================================================
# SAVE MANIFEST
# ============================================================

output_dir = Path(r"D:\DEQ-AD\results")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

manifest_path = (
    output_dir /
    "oasis_manifest.csv"
)

valid.to_csv(
    manifest_path,
    index=False
)

print("\n" + "=" * 70)
print("MANIFEST SAVED")
print("=" * 70)

print(manifest_path)

print("\nAudit completed successfully.")
print("=" * 70)