from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


def load_mri(path):
    """
    Load an OASIS-1 Analyze 7.5 MRI volume.

    Returns:
        numpy array with shape (D, H, W)
    """

    path = Path(path)

    image = nib.load(str(path))

    volume = image.get_fdata(
        dtype=np.float32
    )

    # Remove singleton dimension:
    # (176, 208, 176, 1)
    # ->
    # (176, 208, 176)
    volume = np.squeeze(volume)

    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3-D MRI volume, "
            f"got shape {volume.shape}"
        )

    return volume


def normalize_mri(volume):
    """
    Normalize MRI intensities using only
    non-zero brain voxels.
    """

    volume = volume.astype(
        np.float32,
        copy=False
    )

    brain_mask = volume > 0

    if not np.any(brain_mask):
        raise ValueError(
            "MRI contains no non-zero voxels."
        )

    brain_values = volume[brain_mask]

    # Robust intensity clipping.
    low = np.percentile(
        brain_values,
        1
    )

    high = np.percentile(
        brain_values,
        99
    )

    volume = np.clip(
        volume,
        low,
        high
    )

    # Z-score normalization using brain voxels.
    brain_values = volume[brain_mask]

    mean = brain_values.mean()
    std = brain_values.std()

    if std < 1e-8:
        std = 1.0

    volume = (
        volume - mean
    ) / std

    # Restore background to zero.
    volume[~brain_mask] = 0.0

    return volume


def resize_mri(
    volume,
    target_shape=(96, 112, 96)
):
    """
    Resize a 3-D MRI volume.

    target_shape:
        (D, H, W)
    """

    tensor = torch.from_numpy(
        volume
    ).float()

    # Add batch and channel dimensions:
    #
    # (D,H,W)
    # ->
    # (1,1,D,H,W)

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = F.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False
    )

    tensor = tensor.squeeze(
        0,
        1
    )

    return tensor


def preprocess_mri(
    path,
    target_shape=(96, 112, 96)
):
    """
    Complete OASIS-1 preprocessing pipeline.

    Returns:
        torch.Tensor
        Shape: (1, D, H, W)
    """

    volume = load_mri(path)

    volume = normalize_mri(
        volume
    )

    volume = resize_mri(
        volume,
        target_shape=target_shape
    )

    # Add channel dimension.
    volume = volume.unsqueeze(0)

    return volume


if __name__ == "__main__":

    # Test with the first MRI.
    from pathlib import Path

    root = Path(
        r"D:\OASIS-1\extracted"
    )

    mri_file = next(
        root.rglob("*_masked_gfc.img")
    )

    print("=" * 60)
    print("OASIS-1 PREPROCESSING TEST")
    print("=" * 60)

    print("\nMRI:")
    print(mri_file)

    volume = load_mri(
        mri_file
    )

    print(
        "\nOriginal shape:",
        volume.shape
    )

    print(
        "Original dtype:",
        volume.dtype
    )

    normalized = normalize_mri(
        volume
    )

    print(
        "\nNormalized min:",
        normalized.min()
    )

    print(
        "Normalized max:",
        normalized.max()
    )

    print(
        "Normalized mean:",
        normalized[normalized != 0].mean()
    )

    print(
        "Normalized std:",
        normalized[normalized != 0].std()
    )

    processed = preprocess_mri(
        mri_file
    )

    print(
        "\nFinal tensor shape:",
        processed.shape
    )

    print(
        "Final dtype:",
        processed.dtype
    )

    print(
        "Final min:",
        processed.min().item()
    )

    print(
        "Final max:",
        processed.max().item()
    )

    print(
        "Final mean:",
        processed.mean().item()
    )

    print(
        "\nPreprocessing test completed."
    )