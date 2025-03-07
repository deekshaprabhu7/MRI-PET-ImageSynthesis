import torch
import torch.nn as nn
# Instantiate discriminator

class PatchGAN3D(nn.Module):
    def __init__(self, in_channels=4):  
        super(PatchGAN3D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1),  # Patch output
            nn.Dropout3d(0.3),  # Dropout before Sigmoid
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
