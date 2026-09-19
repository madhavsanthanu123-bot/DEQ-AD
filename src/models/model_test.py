import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.oasis_dataset import OASISDataset
from src.models.deq_ad import DEQADModel

def get_cdr_class(cdr_values):
    """
    Convert raw CDR floats to class indices.
    CDR 0.0 -> 0
    CDR 0.5 -> 1
    CDR 1.0 -> 2
    CDR 2.0 -> 3
    """
    mapping = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
    classes = [mapping[float(val.item())] for val in cdr_values]
    return torch.tensor(classes, dtype=torch.long)

def main():
    # Setup
    manifest_path = Path(r"D:\DEQ-AD\results\train_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("DEQ-AD MODEL FORWARD TEST")
    print("=" * 60)
    
    print(f"\nDevice:\n{device}")
    if torch.cuda.is_available():
        print(f"GPU:\n{torch.cuda.get_device_name(0)}")
        
    # Dataset and DataLoader
    print("\nLoading dataset...")
    dataset = OASISDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    
    batch = next(iter(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)
    cdrs = batch["cdr"]
    
    # Instantiate Model
    print("\nInstantiating model...")
    model = DEQADModel().to(device)
    
    # Forward Pass
    print("Running forward pass...")
    outputs = model(images)
    
    classification_logits = outputs["classification_logits"]
    cdr_logits = outputs["cdr_logits"]
    equilibrium_features = outputs["equilibrium_features"]
    
    # Capture DEQ state
    deq_iterations = model.deq.last_iterations
    deq_converged = model.deq.last_converged
    
    print("\nInput shape:\n", images.shape)
    
    # We can get encoder output size by passing it separately to see (though we already know it's [2, 64])
    # But let's just print what we have to match the expected format.
    encoder_out = model.encoder(images)
    print("\nEncoder output shape:\n", encoder_out.shape)
    
    print("\nEquilibrium output shape:\n", equilibrium_features.shape)
    print("\nClassification logits shape:\n", classification_logits.shape)
    print("\nCDR logits shape:\n", cdr_logits.shape)
    print("\nDEQ iterations:\n", deq_iterations)
    print("\nDEQ converged:\n", deq_converged)
    
    if torch.cuda.is_available():
        print("\nGPU memory allocated:\n", torch.cuda.memory_allocated() / (1024**2), "MB")
        print("\nGPU memory reserved:\n", torch.cuda.memory_reserved() / (1024**2), "MB")
        
    print("\n" + "=" * 60)
    
    # Verifications
    print("\nVerifying outputs...")
    assert torch.isfinite(classification_logits).all(), "Classification logits contain NaN/Inf!"
    assert torch.isfinite(cdr_logits).all(), "CDR logits contain NaN/Inf!"
    assert torch.isfinite(equilibrium_features).all(), "Equilibrium features contain NaN/Inf!"
    
    assert classification_logits.shape == (2, 1), f"Expected classification logits shape (2,1), got {classification_logits.shape}"
    assert cdr_logits.shape == (2, 4), f"Expected CDR logits shape (2,4), got {cdr_logits.shape}"
    assert equilibrium_features.shape == (2, 64), f"Expected equilibrium features shape (2,64), got {equilibrium_features.shape}"
    assert deq_converged, "DEQ did not converge!"
    
    print("[PASS] Forward verifications completed successfully.")
    
    # Loss Calculation
    print("\nRunning dummy backward pass...")
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    
    classification_loss = bce_loss_fn(classification_logits.squeeze(1), labels.float())
    
    cdr_class = get_cdr_class(cdrs).to(device)
    cdr_loss = ce_loss_fn(cdr_logits, cdr_class)
    
    combined_loss = classification_loss + 0.5 * cdr_loss
    
    # Backward
    combined_loss.backward()
    
    print("\nClassification loss:\n", classification_loss.item())
    print("\nCDR loss:\n", cdr_loss.item())
    print("\nCombined loss:\n", combined_loss.item())
    
    print("\nGradient verification:")
    
    # Check encoder gradient (take one parameter from conv1)
    has_enc_grad = model.encoder.conv1.weight.grad is not None and model.encoder.conv1.weight.grad.norm().item() > 0
    print("Encoder gradient:\n", has_enc_grad)
    
    # Check DEQ gradient (hidden layer weight)
    # The hidden_layer has spectral norm, so the raw parameter is hidden_layer.parametrizations.weight.original
    has_deq_grad = False
    for param in model.deq.parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            has_deq_grad = True
            break
    print("DEQ gradient:\n", has_deq_grad)
    
    # Check classification head gradient
    has_cls_grad = model.classification_head.weight.grad is not None and model.classification_head.weight.grad.norm().item() > 0
    print("Classification head gradient:\n", has_cls_grad)
    
    # Check CDR head gradient
    has_cdr_grad = model.cdr_head.weight.grad is not None and model.cdr_head.weight.grad.norm().item() > 0
    print("CDR head gradient:\n", has_cdr_grad)
    
    assert has_enc_grad, "Encoder has no gradient!"
    assert has_deq_grad, "DEQ has no gradient!"
    assert has_cls_grad, "Classification head has no gradient!"
    assert has_cdr_grad, "CDR head has no gradient!"
    
    if torch.cuda.is_available():
        print("\nFinal GPU memory allocated:\n", torch.cuda.memory_allocated() / (1024**2), "MB")
        
    print("\n[PASS] Backward verification completed successfully.")
    
if __name__ == "__main__":
    main()
