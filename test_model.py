import torch
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torch.utils.data import DataLoader
from dataset import MRIPETDataset  # Import dataset
from uNet import UNet3D  # Import trained generator model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load trained generator model
generator = UNet3D().to(device)
generator.load_state_dict(torch.load("models/generator_3DUnet.pth", map_location=device))
generator.eval()

# Define test dataset
test_root = "t1_flair_asl_fdg_preprocessed"
test_dirs = [os.path.join(test_root, d) for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))]
test_dataset = MRIPETDataset(test_dirs)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Define metric lists
ssim_scores = []
psnr_scores = []
mse_scores = []

# Create output directory
os.makedirs("outputs", exist_ok=True)

# Process test images
for i, (mri_sample, real_pet) in enumerate(test_loader):
    mri_sample, real_pet = mri_sample.to(device), real_pet.to(device)

    # Generate fake PET
    with torch.no_grad():
        fake_pet = generator(mri_sample)

    # Convert to NumPy arrays
    real_pet_np = real_pet.cpu().numpy().squeeze().astype(np.float32)
    fake_pet_np = fake_pet.cpu().numpy().squeeze().astype(np.float32)

    # Compute metrics with dynamic data range
    data_range = real_pet_np.max() - real_pet_np.min()
    mse_value = np.mean((real_pet_np - fake_pet_np) ** 2)
    psnr_value = psnr(real_pet_np, fake_pet_np, data_range=data_range)

    # Compute SSIM across slices and take mean
    ssim_values_per_slice = [ssim(real_pet_np[:, :, i], fake_pet_np[:, :, i], data_range=data_range) 
                             for i in range(real_pet_np.shape[-1])]
    ssim_value = np.mean(ssim_values_per_slice)

    # Store metrics
    mse_scores.append(mse_value)
    psnr_scores.append(psnr_value)
    ssim_scores.append(ssim_value)

    # Save every 10th test sample
    if i % 10 == 0:
        slice_idx = real_pet_np.shape[-1] // 2  # Middle slice

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(real_pet_np[:, :, slice_idx], cmap="gray")
        axes[0].set_title("Real PET")
        axes[1].imshow(fake_pet_np[:, :, slice_idx], cmap="gray")
        axes[1].set_title("Generated PET")
        axes[2].imshow(np.abs(real_pet_np[:, :, slice_idx] - fake_pet_np[:, :, slice_idx]), cmap="hot")
        axes[2].set_title("Difference (Error)")

        plt.savefig(f"outputs/test_subject_{i}.png")
        plt.close()

# Compute average scores
avg_mse = np.mean(mse_scores)
avg_psnr = np.mean(psnr_scores)
avg_ssim = np.mean(ssim_scores)

print("\n**Final Metrics on Test Set**")
print(f"Mean Squared Error (MSE): {avg_mse:.6f}")
print(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.2f} dB")
print(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")

# Save metric results
with open("outputs/test_results.txt", "w") as f:
    f.write(f"Mean Squared Error (MSE): {avg_mse:.6f}\n")
    f.write(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.2f} dB\n")
    f.write(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}\n")

print("\n**Testing Complete! Metrics saved to `test_results.txt`.** ")
