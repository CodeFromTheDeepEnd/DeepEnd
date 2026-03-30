# deep_learning/autoencoder/ae_denoise.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap
import numpy as np
from deep_learning.autoencoder.ae import Autoencoder

# Function to add noise
def add_noise(data, noise_factor=0.3):
    noisy_data = data + noise_factor * torch.randn_like(data)
    return torch.clamp(noisy_data, 0., 1.)

# Hyperparameters
latent_dim = 10  
batch_size = 128
epochs = 15
learning_rate = 1e-3
noise_factor = 0.3

# Data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Train the model
model = Autoencoder(latent_dim=latent_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        # Add noise to input
        noisy_data = add_noise(data, noise_factor)
        
        optimizer.zero_grad()
        recon_batch = model(noisy_data)
        # Loss: compare reconstruction to CLEAN data
        loss = F.mse_loss(recon_batch, data.view(-1, 784), reduction='sum')
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset):.4f}')

# Visualize 1: Latent space clustering
print("\nGenerating latent space visualization...")
with torch.no_grad():
    z_list, labels_list = [], []
    for data, labels in test_loader:
        z = model.encode(data.view(-1, 784))
        z_list.append(z)
        labels_list.append(labels)
    
    z = torch.cat(z_list).numpy()
    labels = torch.cat(labels_list).numpy()
    
    plt.figure(figsize=(10, 8))
    
    # Use UMAP for dimensionality reduction if latent_dim > 2
    if latent_dim > 2:
        print(f"Reducing {latent_dim}D latent space to 2D using UMAP...")
        reducer = umap.UMAP(random_state=42)
        z_2d = reducer.fit_transform(z)
    else:
        z_2d = z
    
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', alpha=0.5, s=1)
    plt.colorbar(scatter)
    plt.title(f'Latent Space Clustering ({latent_dim}D → 2D projection)' if latent_dim > 2 else 'Latent Space Clustering')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.show()

# Visualize 2: Reconstruction (without noise)
print("\nGenerating reconstruction visualization...")
with torch.no_grad():
    data, _ = next(iter(test_loader))
    recon = model(data[:8])
    
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', size=12)
    axes[1, 0].set_ylabel('Reconstructed', size=12)
    plt.suptitle('Clean Reconstruction')
    plt.tight_layout()
    plt.show()

# Visualize 3: Denoising demonstration
print("\nGenerating denoising visualization...")
model.eval()
with torch.no_grad():
    data, _ = next(iter(test_loader))
    data = data[:8]
    noisy_data = add_noise(data, noise_factor)
    recon = model(noisy_data)
    
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    for i in range(8):
        # Original
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        # Noisy
        axes[1, i].imshow(noisy_data[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        # Denoised
        axes[2, i].imshow(recon[i].view(28, 28), cmap='gray')
        axes[2, i].axis('off')
    
    axes[0, 0].set_ylabel('Original', size=12)
    axes[1, 0].set_ylabel('Noisy', size=12)
    axes[2, 0].set_ylabel('Denoised', size=12)
    plt.suptitle('Denoising Autoencoder Performance')
    plt.tight_layout()
    plt.show()

# Visualize 4: Different noise levels
print("\nGenerating noise level comparison...")
with torch.no_grad():
    data, _ = next(iter(test_loader))
    data = data[:4]
    
    noise_levels = [0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(len(noise_levels) * 2 + 1, 4, figsize=(8, 12))
    
    # Original images
    for i in range(4):
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel('Original', size=10)
    
    # Different noise levels - show both noisy and denoised
    for idx, noise_level in enumerate(noise_levels):
        noisy_data = add_noise(data, noise_level)
        recon = model(noisy_data)
        
        noisy_row = idx * 2 + 1
        recon_row = idx * 2 + 2
        
        for i in range(4):
            # Noisy image
            axes[noisy_row, i].imshow(noisy_data[i].squeeze(), cmap='gray')
            axes[noisy_row, i].axis('off')
            # Denoised image
            axes[recon_row, i].imshow(recon[i].view(28, 28), cmap='gray')
            axes[recon_row, i].axis('off')
        
        axes[noisy_row, 0].set_ylabel(f'Noisy {noise_level}', size=10)
        axes[recon_row, 0].set_ylabel(f'Denoised', size=10)
    
    plt.suptitle('Denoising at Different Noise Levels')
    plt.tight_layout()
    plt.show()

# Visualize 5: Random generation from decoder
print("\nGenerating random samples from decoder...")
with torch.no_grad():
    # Generate 6 random latent vectors
    random_z = torch.randn(6, latent_dim)
    generated = model.decode(random_z)
    
    fig, axes = plt.subplots(2, 3, figsize=(6, 4))
    for i in range(6):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(generated[i].view(28, 28), cmap='gray')
        axes[row, col].axis('off')
    
    plt.suptitle('Random Samples from Decoder')
    plt.tight_layout()
    plt.savefig('random_generation.png', dpi=300, bbox_inches='tight')
    plt.show()



print("\nAll visualizations complete!")
