# MRI-to-PET Image Synthesis

## Overview
This project implements a deep learning-based Generative Adversarial Network (GAN) model for MRI-to-PET image synthesis. The goal is to generate PET-like images from MRI scans, providing a non-invasive and cost-effective alternative to PET imaging. The model utilizes a **3D U-Net Generator** and a **PatchGAN Discriminator**, trained using adversarial and L1 loss functions. The dataset consists of multi-modal MRI inputs (**T1, FLAIR, ASL**) and **FDG PET** images as ground truth.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/deekshaprabhu7/MRI-PET-ImageSynthesis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd MRI-PET-ImageSynthesis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Preparing Data
Ensure your MRI and PET data are placed inside the `t1_flair_asl_fdg_preprocessed/` directory.

### 2. Training the Model
To start training the GAN model, run:
   ```bash
   python train.py
   ```

### 3. Testing the Model
After training, you can evaluate the model using:
   ```bash
   python test_model.py
   ```

---

## Project Structure
```
MRI-PET-ImageSynthesis/
│── dataset.py            # Data loading and preprocessing (MRI & PET)
│── discriminator.py      # PatchGAN-based discriminator model
│── uNet.py               # 3D U-Net generator model
│── train.py              # Training script for GAN
│── test_model.py         # Script for evaluating trained model
│── requirements.txt      # Required dependencies
│── README.md             # Project documentation
│── outputs/              # Directory to store generated PET images
│── models/               # Directory to save trained models
│── t1_flair_asl_fdg_preprocessed/ # Directory for preprocessed MRI-PET dataset
```

---

## Model Details
- **Generator:** 3D U-Net architecture for PET image synthesis.
- **Discriminator:** PatchGAN-based discriminator for real vs. fake PET classification.
- **Loss Functions:**
  - **Adversarial Loss (GAN Loss)**
  - **Reconstruction Loss (L1 Loss)**

---

## Training Details
- **Dataset:** 176 MRI-PET pairs
- **Split:** Train (80%) - Validation (10%) - Test (10%)
- **Optimizer:** Adam (LR = 0.0002, Betas = (0.5, 0.999))
- **GPU:** Trained on NVIDIA RTX 3070 Ti

---

## Evaluation Metrics
- **MSE (Mean Squared Error)**
- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index)**

---

## References
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015.
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)*, CVPR 2017.

