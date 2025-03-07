import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

# Function to load and normalize NIfTI images
def load_nifti(file_path):
    # print(f"Trying to load: {file_path}")  # Debugging step
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    img = nib.load(file_path)
    data = img.get_fdata()
    data = (data - data.min()) / (data.max() - data.min())  # Normalize to [0,1]
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, H, W, D)

# Custom dataset class for 3D MRI-to-PET
class MRIPETDataset(Dataset):
    def __init__(self, subject_dirs):
        self.subject_dirs = subject_dirs

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_path = self.subject_dirs[idx]

        # Define file paths
        t1_path = os.path.join(subject_path, "T1_MNI.nii.gz")
        flair_path = os.path.join(subject_path, "FLAIR_MNI.nii.gz")
        asl_path = os.path.join(subject_path, "ASL_MNI.nii.gz")
        fdg_path = os.path.join(subject_path, "FDG_MNI.nii.gz")  # PET target

        # # Print paths for debugging
        # print(f"Loading files from: {subject_path}")
        # print(f"T1 Path: {t1_path}")

        # Check if files exist
        for path in [t1_path, flair_path, asl_path, fdg_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")

        # Load images
        t1_img = load_nifti(t1_path)
        flair_img = load_nifti(flair_path)
        asl_img = load_nifti(asl_path)
        fdg_img = load_nifti(fdg_path)

        # Stack MRI images as 3-channel input: (3, H, W, D)
        input_mri = torch.cat([t1_img, flair_img, asl_img], dim=0)  # (3, H, W, D)
        target_pet = fdg_img  # (1, H, W, D)

        return input_mri, target_pet

# Define dataset path
dataset_root = "t1_flair_asl_fdg_preprocessed"

# Get subject directories
subject_dirs = [os.path.join(dataset_root, d) for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

# Split dataset: 80% Train, 10% Validation, 10% Test
train_dirs, test_dirs = train_test_split(subject_dirs, test_size=0.2, random_state=42)
val_dirs, test_dirs = train_test_split(test_dirs, test_size=0.5, random_state=42)

print(f"Training Samples: {len(train_dirs)}, Validation Samples: {len(val_dirs)}, Test Samples: {len(test_dirs)}")

# Create dataset objects
train_dataset = MRIPETDataset(train_dirs)
val_dataset = MRIPETDataset(val_dirs)
test_dataset = MRIPETDataset(test_dirs)

# Create dataloaders
# train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Example: Fetch a sample
mri_sample, pet_sample = next(iter(train_loader))
print("MRI Sample Shape:", mri_sample.shape)  # (batch, 3, H, W, D)
print("PET Sample Shape:", pet_sample.shape)  # (batch, 1, H, W, D)
