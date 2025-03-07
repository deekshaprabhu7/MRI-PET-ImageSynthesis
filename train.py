import os
import torch
import torch.nn as nn
from dataset import train_loader, val_loader
from uNet import UNet3D
from discriminator import PatchGAN3D

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
generator = UNet3D().to(device)
discriminator = PatchGAN3D().to(device)

# Loss functions
criterion_GAN = nn.BCELoss()  # Adversarial loss
criterion_L1 = nn.L1Loss()  # Structural similarity loss

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Learning rate scheduler
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)  # Decay every 5 epochs
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)

# Training loop
epochs = 30  # Adjust as needed

for epoch in range(epochs):
    generator.train()
    discriminator.train()
    
    # Training phase
    for i, (mri, pet) in enumerate(train_loader):
        mri, pet = mri.to(device), pet.to(device)

        # Generate fake PET
        fake_pet = generator(mri)

        # Train Discriminator
        optimizer_D.zero_grad()
        real_validity = discriminator(torch.cat((mri, pet), 1))  # Real pairs
        fake_validity = discriminator(torch.cat((mri, fake_pet.detach()), 1))  # Fake pairs

        # Label smoothing for stability
        real_labels = torch.full_like(real_validity, 0.9, device=device)  
        fake_labels = torch.zeros_like(fake_validity)

        d_loss = (criterion_GAN(real_validity, real_labels) +
                  criterion_GAN(fake_validity, fake_labels)) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_validity = discriminator(torch.cat((mri, fake_pet), 1))
        
        # Balanced loss (adversarial + structural)
        g_loss = criterion_GAN(fake_validity, torch.ones_like(fake_validity)) + 100 * criterion_L1(fake_pet, pet)
        g_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        optimizer_G.step()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Step {i}/{len(train_loader)}, G Loss: {g_loss.item()}, D Loss: {d_loss.item()}")

    # Validation phase
    generator.eval()
    total_val_loss = 0
    with torch.no_grad():
        for mri, pet in val_loader:
            mri, pet = mri.to(device), pet.to(device)
            fake_pet = generator(mri)
            val_loss = criterion_L1(fake_pet, pet)  # L1 loss for validation
            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss}")

    # Step learning rate scheduler
    scheduler_G.step()
    scheduler_D.step()

print("Training complete!")

# Create a directory to store models
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)  # Create the folder if it doesn’t exist

# Save the trained models in the "models" folder
torch.save(generator.state_dict(), os.path.join(model_dir, "generator_3DUnet.pth"))
torch.save(discriminator.state_dict(), os.path.join(model_dir, "discriminator_3DUnet.pth"))

print(f"Models saved successfully in '{model_dir}/'")
