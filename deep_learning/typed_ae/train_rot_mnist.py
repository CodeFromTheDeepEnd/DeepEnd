import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from deep_learning.typed_ae.tae_rot import TypedAE

# ============================================================================
# Phase 1: Train plain autoencoder (no rotation component)
# ============================================================================
def phase1_train_basic_ae(epochs=20):
    print("=" * 80)
    print("Phase 1: Train AE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TypedAE().to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                               download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - NO rotation
            h = model.encoder_backbone(images.view(images.size(0), -1))
            z_latent = model.head_latent(h)
            
            # z_rot = zero (no rotation)
            z_rot = torch.zeros(images.size(0), 2).to(device)
            z_rot[:, 0] = 1.0  # cos(0) = 1
            
            z_combined = torch.cat([z_rot, z_latent], dim=-1)
            recon = model.decoder(z_combined).view(images.size(0), 1, 28, 28)
            
            # Loss
            loss = nn.functional.mse_loss(recon, images)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save phase 1
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 1
    }, 'phase1_basic_ae.pth')
    
    print("Phase 1 ready! Model saved: phase1_basic_ae.pth\n")
    return model

# ============================================================================
# VAIHE 2: Jäädytä encoder, opeta decoder pyörittämään
# ============================================================================
def rotate_image_batch(images, angles_rad):
    """Pyörittää kuvia annetuilla kulmilla"""
    batch_size = images.size(0)
    rotated = []
    
    for i in range(batch_size):
        angle_deg = angles_rad[i].item() * 180 / np.pi
        img = images[i].cpu()
        
        # Käytä torchvision.transforms.functional
        from torchvision.transforms import functional as F
        rotated_img = F.rotate(img, angle_deg, fill=0)
        rotated.append(rotated_img)
    
    return torch.stack(rotated).to(images.device)

def phase2_train_decoder_rotation(model=None, epochs=20):
    print("=" * 80)
    print("VAIHE 2: Opeta decoder pyörittämään (encoder jäädytetty)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        model = TypedAE().to(device)
        checkpoint = torch.load('phase1_basic_ae.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze encoder
    for param in model.encoder_backbone.parameters():
        param.requires_grad = False
    for param in model.head_latent.parameters():
        param.requires_grad = False
    for param in model.head_rot.parameters():
        param.requires_grad = False
    
    # Train only decoder
    optimizer = optim.Adam(model.decoder.parameters(), lr=1e-3)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                               download=True, transform=transform)
 #   train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
 #                                                  download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                # Encode the original image
                h = model.encoder_backbone(images.view(images.size(0), -1))
                z_latent = model.head_latent(h)
            
            # Random angle for each image
            angles_rad = (torch.rand(images.size(0)) * 2 * np.pi).to(device)  # 0-360°
            z_rot = torch.stack([torch.cos(angles_rad), torch.sin(angles_rad)], dim=1)
            
            # Decode with the given angle
            z_combined = torch.cat([z_rot, z_latent], dim=-1)
            recon = model.decoder(z_combined).view(images.size(0), 1, 28, 28)
            
            # Target: manually rotated image
            target = rotate_image_batch(images, angles_rad)
            
            # Loss: decoder must produce the rotated image
            loss = nn.functional.mse_loss(recon, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    # Save phase 2
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 2
    }, 'phase2_decoder_rotation.pth')
    
    print("Phase 2 ready! Model saved in phase2_decoder_rotation.pth\n")
    return model

# ============================================================================
# Phase 3: Feeze decoder, train rotation head
# ============================================================================
def phase3_train_rotation_head(model=None, epochs=20):
    print("=" * 80)
    print("Phase 3: Train rotation head (freeze decoder)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        model = TypedAE().to(device)
        checkpoint = torch.load('phase2_decoder_rotation.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Freeze decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # Release encoder (esp. rotation head)
    for param in model.encoder_backbone.parameters():
        param.requires_grad = True # Try false to see the effect
    for param in model.head_latent.parameters():
        param.requires_grad = True # Try false to see the effect
    for param in model.head_rot.parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam([
        {'params': model.encoder_backbone.parameters()},
        {'params': model.head_latent.parameters()},
        {'params': model.head_rot.parameters()},
    ], lr=1e-3)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                               download=True, transform=transform)
#    train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
#                                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_rot_loss = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = images.to(device)
            
            # Rotate with random angle
            angles_rad = (torch.rand(images.size(0)) * 2 * np.pi).to(device)
            images_rotated = rotate_image_batch(images, angles_rad)
            
            optimizer.zero_grad()
            
            # Encode rotated image
            h = model.encoder_backbone(images_rotated.view(images_rotated.size(0), -1))
            z_latent = model.head_latent(h)
            z_rot = model.head_rot(h)  # We train the angle!
            
            # Decode (frozen decoder)
            z_combined = torch.cat([z_rot, z_latent], dim=-1)
            recon = model.decoder(z_combined).view(images_rotated.size(0), 1, 28, 28)
            
            # Losses
            loss_recon = nn.functional.mse_loss(recon, images_rotated)
            
            # Rotation loss: z_rot must match the angle
            target_rot = torch.stack([torch.cos(angles_rad), torch.sin(angles_rad)], dim=1)
            loss_rot = nn.functional.mse_loss(z_rot, target_rot)
            
            loss = loss_recon + 1.0 * loss_rot
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_rot_loss += loss_rot.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_rot_loss = total_rot_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Rot Loss: {avg_rot_loss:.6f}")
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'phase': 3
    }, 'phase3_final_model.pth')
    
    print("Phase 3 ready! Model saved: phase3_final_model.pth\n")
    return model

# ============================================================================
# Test
# ============================================================================
def test_rotation_control():
    print("=" * 80)
    print("TEST: Can we control the angle?")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TypedAE().to(device)
    checkpoint = torch.load('phase3_final_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.ToTensor()
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    idx = np.random.randint(len(mnist))
    original_img, label = mnist[idx]
    original_img = original_img.to(device).unsqueeze(0)
    
    with torch.no_grad():
        # Encode original image
        h = model.encoder_backbone(original_img.view(1, -1))
        z_latent = model.head_latent(h)
        
        # Test different angles
        angles = torch.linspace(0, 360, 12)
        
        fig, axes = plt.subplots(2, 6, figsize=(12, 4))
        axes = axes.flatten()
        
        for i, angle_deg in enumerate(angles):
            angle_rad = angle_deg * np.pi / 180
            z_rot_new = torch.tensor([[torch.cos(torch.tensor(angle_rad)), 
                                       torch.sin(torch.tensor(angle_rad))]]).to(device)
            
            z_combined = torch.cat([z_rot_new, z_latent], dim=-1)
            recon = model.decoder(z_combined).view(1, 1, 28, 28)
            
            axes[i].imshow(recon.cpu().squeeze(), cmap='gray')
            axes[i].set_title(f'{angle_deg.item():.0f}°')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('final_rotation_test.png', dpi=150)
        plt.show()
        print("Test images saved in final_rotation_test.png")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Three training phases
    model = phase1_train_basic_ae(epochs=25)
    model = phase2_train_decoder_rotation(model, epochs=20)
    model = phase3_train_rotation_head(model, epochs=20)
    
    # Test
    test_rotation_control()
