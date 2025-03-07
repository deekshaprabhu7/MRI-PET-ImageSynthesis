import torch
import torch.nn as nn

class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet3D, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )

        def downsample(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )

        def upsample(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),  # Kernel 4 ensures proper sizing
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 64)
        self.down1 = downsample(64, 128)
        self.enc2 = conv_block(128, 128)
        self.down2 = downsample(128, 256)
        self.enc3 = conv_block(256, 256)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder (Fixed Spatial Mismatch)
        self.up1 = upsample(512, 256)
        self.dec1 = conv_block(256 + 256, 256)  # Ensuring correct concat dimensions

        self.up2 = upsample(256, 128)
        self.dec2 = conv_block(128 + 128, 128)

        self.up3 = upsample(128, 64)
        self.dec3 = conv_block(64 + 64, 64)

        # Output Layer
        self.final_conv = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        bottleneck = self.bottleneck(e3)

        d1 = self.up1(bottleneck)
        d1 = nn.functional.interpolate(d1, size=e3.shape[2:], mode='trilinear', align_corners=False)  # Fix mismatch
        d1 = self.dec1(torch.cat([d1, e3], dim=1))

        d2 = self.up2(d1)
        d2 = nn.functional.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d3 = self.up3(d2)
        d3 = nn.functional.interpolate(d3, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e1], dim=1))

        return self.final_conv(d3)

# Instantiate model
generator = UNet3D()
