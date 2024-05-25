import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = self.contracting_block(1, 64)
        self.enc2 = self.contracting_block(64, 128)
        self.enc3 = self.contracting_block(128, 256)
        self.enc4 = self.contracting_block(256, 512)
        self.bottleneck = self.contracting_block(512, 1024)
        self.dec4 = self.expansive_block(1024, 512)
        self.dec3 = self.expansive_block(512, 256)
        self.dec2 = self.expansive_block(256, 128)
        self.dec1 = self.expansive_block(128, 64)
        self.final_layer = nn.Conv2d(64, 1, kernel_size=1)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        return block

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))
        dec4 = self.dec4(F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True))
        dec3 = self.dec3(F.interpolate(dec4 + enc4, scale_factor=2, mode='bilinear', align_corners=True))
        dec2 = self.dec2(F.interpolate(dec3 + enc3, scale_factor=2, mode='bilinear', align_corners=True))
        dec1 = self.dec1(F.interpolate(dec2 + enc2, scale_factor=2, mode='bilinear', align_corners=True))
        final_layer = self.final_layer(dec1 + enc1)
        return final_layer

# Usage
model = UNet()
print(model)
