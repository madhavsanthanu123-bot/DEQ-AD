import torch
import torch.nn as nn
import torch.optim as optim

from src.deq.equilibrium import DEQLayer


# =========================================================
# Configuration
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INPUT_DIM = 64
HIDDEN_DIM = 64
NUM_CLASSES = 2

SAMPLES = 256
EPOCHS = 10
LEARNING_RATE = 0.001


# =========================================================
# GPU
# =========================================================

print("=" * 60)
print("TRAINABLE DEQ TEST")
print("=" * 60)

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0)
            .total_memory / (1024 ** 3),
            2
        ),
        "GB"
    )


# =========================================================
# Synthetic dataset
# =========================================================

torch.manual_seed(42)

X = torch.randn(
    SAMPLES,
    INPUT_DIM,
    device=DEVICE
)

signal = X[:, :10].sum(dim=1)

y = (
    signal > 0
).long()


# =========================================================
# Model
# =========================================================

class DEQClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                INPUT_DIM,
                HIDDEN_DIM
            ),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_DIM,
                HIDDEN_DIM
            )
        )

        self.deq = DEQLayer(
            input_dim=HIDDEN_DIM,
            hidden_dim=HIDDEN_DIM,
            max_iter=30,
            tolerance=1e-5
        )

        self.classifier = nn.Linear(
            HIDDEN_DIM,
            NUM_CLASSES
        )

    def forward(self, x):

        features = self.encoder(x)

        z_star = self.deq(
            features
        )

        return self.classifier(
            z_star
        )


model = DEQClassifier().to(DEVICE)


# =========================================================
# Optimizer
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# Training
# =========================================================

print()
print("Starting training...")
print()

for epoch in range(EPOCHS):

    optimizer.zero_grad()

    logits = model(X)

    loss = criterion(
        logits,
        y
    )

    loss.backward()

    optimizer.step()

    predictions = torch.argmax(
        logits,
        dim=1
    )

    accuracy = (
        predictions == y
    ).float().mean().item()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Loss: {loss.item():.6f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Forward iterations: "
        f"{model.deq.last_iterations} | "
        f"Converged: "
        f"{model.deq.last_converged}"
    )


# =========================================================
# Gradient verification
# =========================================================

print()
print("=" * 60)
print("DEQ GRADIENT VERIFICATION")
print("=" * 60)

for name, parameter in model.deq.named_parameters():

    if parameter.grad is None:

        print(
            f"{name}: NO GRADIENT"
        )

    else:

        print(
            f"{name}: "
            f"gradient norm = "
            f"{parameter.grad.norm().item():.8f}"
        )


# =========================================================
# GPU memory
# =========================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / (1024 ** 2)
    )

    reserved = (
        torch.cuda.memory_reserved()
        / (1024 ** 2)
    )

    print()
    print(
        f"GPU memory allocated: "
        f"{allocated:.2f} MB"
    )

    print(
        f"GPU memory reserved: "
        f"{reserved:.2f} MB"
    )


print()
print("=" * 60)
print("TRAINABLE DEQ TEST COMPLETED")
print("=" * 60)