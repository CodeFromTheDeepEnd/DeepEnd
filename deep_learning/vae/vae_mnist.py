# deep_learning/vae/vae_mnist.py
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import umap
from deep_learning.vae.vae import VAE 
import numpy as np
from PIL import Image

# Hyperparameters
latent_dim = 4 
batch_size = 128
epochs = 15
learning_rate = 1e-3
beta = 1.0

# Data
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loss-function
def loss_function(recon_x, x, mu, logvar):
    # Reconstruction loss (BCE)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD

# Train the model
model = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
for epoch in range(epochs):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset):.4f}')

# Visualize 1: Latent space
model.eval()
with torch.no_grad():
    z_list, labels_list = [], []
    for data, labels in train_loader:
        data = data.to(device)
        mu, _ = model.encode(data.view(-1, 784))
        z_list.append(mu)
        labels_list.append(labels)
    
    z = torch.cat(z_list).numpy()
    labels = torch.cat(labels_list).numpy()
    
    plt.figure(figsize=(10, 8))
    reducer = umap.UMAP(random_state=42)
    z_2d = reducer.fit_transform(z)
    scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels, cmap='tab10', alpha=0.5)
    plt.colorbar(scatter)
    plt.title('Latent Space (2D/projection)')
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.show()

# Visualize 2: Generate numbers
with torch.no_grad():
    # Sample z ~ N(0,I)
    z = torch.randn(16, latent_dim)
    samples = model.decode(z).view(-1, 28, 28)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    plt.suptitle('Generated Digits')
    plt.show()

# Use the same batch for both reconstructions
data, _ = next(iter(train_loader))
data = data.to(device)

# Visualize 3: Reconstructed images - stochastic
with torch.no_grad():
    recon, _, _ = model(data[:4])
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 10))
    for i in range(4):
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', size=12)
    axes[1, 0].set_ylabel('Reconstructed', size=12)
    plt.show()

# Visualize 4: Reconstructed images - deterministic
with torch.no_grad():
    mu, logvar = model.encode(data[:4].view(-1, 784))
    z = mu  # Deterministic
    recon = model.decode(z)
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 10))
    for i in range(4):  # ← 4 kuvaa
        axes[0, i].imshow(data[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', size=12)
    axes[1, 0].set_ylabel('Reconstructed', size=12)
    plt.show()

# Animation: 4 -> 9 -> 7
model.eval()
with torch.no_grad():
    # Find example digits
    digits_to_find = [4, 9, 7]
    examples = {}
    
    for data, labels in train_loader:
        for digit in digits_to_find:
            if digit not in examples:
                mask = labels == digit
                if mask.any():
                    idx = mask.nonzero()[0].item()
                    examples[digit] = data[idx].to(device)
        if len(examples) == 3:
            break
    
    # Latent representation
    z_4 = model.encode(examples[4].view(1, 784))[0]
    z_9 = model.encode(examples[9].view(1, 784))[0]
    z_7 = model.encode(examples[7].view(1, 784))[0]
    
    # Interpolation
    steps = 30  # Frames per move
    frames = []
    
    # 4 -> 9
    for t in np.linspace(0, 1, steps):
        z_interp = (1 - t) * z_4 + t * z_9
        img = model.decode(z_interp).view(28, 28).cpu().numpy()
        frames.append(img)
    
    # 9 -> 7
    for t in np.linspace(0, 1, steps):
        z_interp = (1 - t) * z_9 + t * z_7
        img = model.decode(z_interp).view(28, 28).cpu().numpy()
        frames.append(img)
    
    # Convert to PIL and save GIF
    pil_frames = []
    for frame in frames:
        img_uint8 = (frame * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, mode='L')
        # scale up (28x28 -> 140x140)
        img = img.resize((140, 140), Image.LANCZOS)  # NEAREST = pixels
        pil_frames.append(img)

    # Add frames backwards, exclude first and last
    pil_frames_reverse = pil_frames[-2:0:-1]
    all_frames = pil_frames + pil_frames_reverse

    # Save  GIF
    all_frames[0].save(
        'vae_morph_4_9_7.gif',
        save_all=True,
        append_images=all_frames[1:],
        duration=50,  # ms per frame
        loop=0  # Loop
    )
    
    print("GIF-animation saved: vae_morph_4_9_7.gif")
