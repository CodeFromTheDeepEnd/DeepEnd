import torch
import torch.nn as nn
import torch.nn.functional as F

# ============== MODELS ==============
class CyclicBottleneck(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 2)
    
    def forward(self, x):
        raw = self.fc(x)
        normalized = F.normalize(raw, p=2, dim=-1)
        return normalized

class TypedAE(nn.Module):
    def __init__(self, latent_dim=5):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder_backbone = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.head_rot = CyclicBottleneck(128)  # Rotation (2D)
        self.head_latent = nn.Linear(128, latent_dim)  # Other latent dims

        # Decoder takes in: 2 (rot) + latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(2 + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder_backbone(x.view(x.size(0), -1))
        
        z_rot = self.head_rot(h)
        z_latent = self.head_latent(h)
        
        z_combined = torch.cat([z_rot, z_latent], dim=-1)
        recon = self.decoder(z_combined).view(x.size(0), 1, 28, 28)
        
        return recon, z_rot, z_latent
