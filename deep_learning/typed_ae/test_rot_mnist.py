import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from deep_learning.typed_ae.tae_rot import TypedAE

def create_rotation_animations():
    """
    Create animated GIFs for each digit 0-9.
    Each GIF shows the digit rotating 0° -> 360°
    """
    print("=" * 80)
    print("Creating animations")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TypedAE().to(device)
    
    # Lataa koulutettu malli
    checkpoint = torch.load('phase3_final_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load MNIST
    transform = transforms.ToTensor()
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Output-directory
    os.makedirs('rotation_animations', exist_ok=True)
    
    # Loop over the digits
    for digit in range(10):
        print(f"\nDigit {digit}...")
        
        # Find first instance of the digit in hand
        idx = None
        for i, (img, label) in enumerate(mnist):
            if label == digit:
                idx = i
                break
        
        if idx is None:
            print(f"  Can't find digit {digit}")
            continue
        
        original_img, label = mnist[idx]
        original_img = original_img.to(device).unsqueeze(0)
        
        with torch.no_grad():
            # Encode the original images
            h = model.encoder_backbone(original_img.view(1, -1))
            z_latent = model.head_latent(h)
            
            # Frame it
            frames = []
            angles = torch.linspace(0, 360, 36)
            
            for angle_deg in angles:
                angle_rad = angle_deg * np.pi / 180
                z_rot_new = torch.tensor([[torch.cos(torch.tensor(angle_rad)), 
                                           torch.sin(torch.tensor(angle_rad))]]).to(device)
                
                z_combined = torch.cat([z_rot_new, z_latent], dim=-1)
                recon = model.decoder(z_combined).view(1, 1, 28, 28)
                
                # From tensor to PIL image
                img_np = recon.cpu().squeeze().numpy()
                img_np = (img_np * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np, mode='L')
                
                # Scale up (28x28 -> 140x140)
                img_pil = img_pil.resize((140, 140), Image.NEAREST)
                
                frames.append(img_pil)
            
            # Save the GIF
            output_path = f'rotation_animations/digit_{digit}_rotation.gif'
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=100,  # 100ms per frame
                loop=0  # Loop forever
            )
            
            print(f"  ✓ Saved in: {output_path}")
    
    print("\n" + "=" * 80)
    print("All animations saved in directory: rotation_animations/")
    print("=" * 80)

def create_all_digits_grid():
    """
    Create an image with all digits 0-9 in different angles
    """
    print("\n" + "=" * 80)
    print("Creating the grid-image: all digits x all angles")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TypedAE().to(device)
    
    checkpoint = torch.load('phase3_final_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.ToTensor()
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Angles
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # Create the grid: 10 riws (digits) x 8 columns (angles)
    fig, axes = plt.subplots(10, 8, figsize=(16, 20))
    
    for digit in range(10):
        # Find an example
        idx = None
        for i, (img, label) in enumerate(mnist):
            if label == digit:
                idx = i
                break
        
        if idx is None:
            continue
        
        original_img, label = mnist[idx]
        original_img = original_img.to(device).unsqueeze(0)
        
        with torch.no_grad():
            h = model.encoder_backbone(original_img.view(1, -1))
            z_latent = model.head_latent(h)
            
            for col, angle_deg in enumerate(angles):
                angle_rad = angle_deg * np.pi / 180
                z_rot_new = torch.tensor([[torch.cos(torch.tensor(angle_rad)), 
                                           torch.sin(torch.tensor(angle_rad))]]).to(device)
                
                z_combined = torch.cat([z_rot_new, z_latent], dim=-1)
                recon = model.decoder(z_combined).view(1, 1, 28, 28)
                
                axes[digit, col].imshow(recon.cpu().squeeze(), cmap='gray')
                axes[digit, col].axis('off')
                
                # Header only on the top most line
                if digit == 0:
                    axes[digit, col].set_title(f'{angle_deg}°', fontsize=10)
        
        # Row label
        axes[digit, 0].set_ylabel(f'Digit {digit}', fontsize=12, rotation=0, 
                                   labelpad=30, va='center')
    
    plt.tight_layout()
    plt.savefig('rotation_animations/all_digits_grid.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: rotation_animations/all_digits_grid.png")
    plt.close()

def create_smooth_rotation_video(digit=1, fps=30, duration_seconds=3):
    """
    Create smooth animation (more frames)
    """
    print(f"\n" + "=" * 80)
    print(f"Creating smooth animation for digit {digit}")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TypedAE().to(device)
    
    checkpoint = torch.load('phase3_final_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    transform = transforms.ToTensor()
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Find the digit
    idx = None
    for i, (img, label) in enumerate(mnist):
        if label == digit:
            idx = i
            break
    
    if idx is None:
        print(f"Couldn't find digit {digit}")
        return
    
    original_img, label = mnist[idx]
    original_img = original_img.to(device).unsqueeze(0)
    
    with torch.no_grad():
        h = model.encoder_backbone(original_img.view(1, -1))
        z_latent = model.head_latent(h)
        
        # More frames for smoother animation
        num_frames = fps * duration_seconds
        frames = []
        angles = torch.linspace(0, 360, num_frames)
        
        print(f"Creating {num_frames} frames...")
        
        for i, angle_deg in enumerate(angles):
            if i % 10 == 0:
                print(f"  Frame {i}/{num_frames}")
            
            angle_rad = angle_deg * np.pi / 180
            z_rot_new = torch.tensor([[torch.cos(torch.tensor(angle_rad)), 
                                       torch.sin(torch.tensor(angle_rad))]]).to(device)
            
            z_combined = torch.cat([z_rot_new, z_latent], dim=-1)
            recon = model.decoder(z_combined).view(1, 1, 28, 28)
            
            img_np = recon.cpu().squeeze().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np, mode='L')
            img_pil = img_pil.resize((280, 280), Image.NEAREST)  # Larger
            
            frames.append(img_pil)
        
        # Save the GIF
        output_path = f'rotation_animations/digit_{digit}_smooth.gif'
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/fps),  # ms per frame
            loop=0
        )
        
        print(f"✓ Saved in: {output_path}")

if __name__ == "__main__":
    # 1. Create animations for all digits
    create_rotation_animations()
    
    # 2. Create grid
    create_all_digits_grid()
    
    # 3. Create smooth animation for chosen digits
    for digit in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        create_smooth_rotation_video(digit=digit, fps=30, duration_seconds=3)
    
    print("\n" + "=" * 80)
    print("Done! Check folder: rotation_animations/")
    print("=" * 80)
