import torch
import torch.nn as nn
from src.deq.equilibrium import DEQLayer

class MRIEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [B, 1, 96, 112, 96]
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(8)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(16)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(32)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(32, 64)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.pool(x)
        x = self.flatten(x)
        
        x = self.fc(x)
        return x

class DEQADModel(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        
        self.encoder = MRIEncoder()
        
        self.deq = DEQLayer(
            input_dim=64,
            hidden_dim=64,
            beta=beta,
            max_iter=50,
            tolerance=1e-5
        )
        
        # Multi-task heads
        self.classification_head = nn.Linear(64, 1)  # Binary
        self.cdr_head = nn.Linear(64, 4)             # 4-class severity
        
    def forward(self, x):
        # x shape: [B, 1, 96, 112, 96]
        
        # 1. Extract feature vector
        features = self.encoder(x)  # [B, 64]
        
        # 2. Compute equilibrium state
        z_star = self.deq(features) # [B, 64]
        
        # 3. Predict targets from the equilibrium representation
        classification_logits = self.classification_head(z_star)
        cdr_logits = self.cdr_head(z_star)
        
        return {
            "classification_logits": classification_logits,
            "cdr_logits": cdr_logits,
            "equilibrium_features": z_star
        }
