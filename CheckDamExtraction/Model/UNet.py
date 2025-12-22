#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: UNet
# @Time    : 2025/12/8 16:22
# @Author  : Kevin
# @Describe: This file implements a U-Net model for semantic segmentation tasks using PyTorch Lightning framework.

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- UNet Components ---
class DoubleConv(nn.Module):
    """(Conv2d => BatchNorm2d => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsample, we concatenate features from encoder, so in_channels becomes in_channels//2 + prev_out_channels
            # But since we pass in_channels which includes the concatenated features,
            # the DoubleConv will take in_channels and output out_channels
            # We need to adjust in_channels for DoubleConv accordingly if we want to reduce channels first.
            # Standard approach: keep in_channels as passed (includes skip connection), let DoubleConv handle it.
            # However, typical UNet halves channels after upsample if not concatenating.
            # Let's stick closer to standard: Upsample, then Conv to reduce channels, then DoubleConv.
            # But simpler way often done: just use DoubleConv with full in_channels.
            # Let's do the latter for simplicity unless channel mismatch occurs.
            # The passed in_channels should be the total after concatenation.
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # Use ConvTranspose2d
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# --- UNet 模型 ---

class UNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=2, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        # 定义网络层
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)  # x5 上采样并与 x4 跳跃连接
        x = self.up2(x, x3)  # x 上采样并与 x3 跳跃连接
        x = self.up3(x, x2)  # x 上采样并与 x2 跳跃连接
        x = self.up4(x, x1)  # x 上采样并与 x1 跳跃连接

        # Output Layer
        logits = self.outc(x)
        return logits


# Example usage remains the same if run directly
if __name__ == '__main__':
    model = UNet(n_channels=3, num_classes=2)
    print(model)
    dummy_input = torch.randn(2, 3, 256, 256)
    dummy_output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")