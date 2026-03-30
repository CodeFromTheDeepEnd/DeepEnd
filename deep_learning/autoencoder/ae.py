# deep_learning/autoencoder/ae.py
import torch.nn as nn
import torch.nn.functional as F
import torch

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))
    
    def forward(self, x):
        z = self.encode(x.view(-1, 784))
        return self.decode(z)