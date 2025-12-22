#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: CheckDamNet
# @Time    : 2025/12/8 17:22
# @Author  : Kevin
# @Describe:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import JaccardIndex

from .UNet import UNet

class ResidualBlock(nn.Module):
    """ResNet残差块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

        # 如果输入输出通道不一致，使用1x1卷积进行匹配
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        out = out + residual  # 残差连接
        return out

class ExtractCheckDam(nn.Module):

    def __init__(self, n_channels=3):
        super().__init__()

        self.down_conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=7, stride=4, padding=3)
        self.down_conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=7, stride=4, padding=3)
        self.down_conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=7, stride=4, padding=3)

        self.feature_1 = ResidualBlock(n_channels, 64)
        self.feature_2 = ResidualBlock(64, 64)
        self.feature_3 = ResidualBlock(64, 32)
        self.feature_4 = ResidualBlock(32, 1)
        self.feature = nn.Sequential(
            self.feature_1,
            self.feature_2,
            self.feature_3,
            self.feature_4
        )
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x):
        original_size = x.shape[2:]
        # 1. 多尺度下采样
        down_x1 = self.down_conv1(x)
        down_x2 = self.down_conv2(down_x1)
        down_x3 = self.down_conv3(down_x2)

        # 2. 先对每个尺度的特征应用 self.feature
        feat1 = self.feature(down_x1)  # [B, 1, 512, 512]
        feat2 = self.feature(down_x2)  # [B, 1, 256, 256]
        feat3 = self.feature(down_x3)  # [B, 1, 128, 128]

        # 3. 逐步上采样并融合 (类似 U-Net 解码器)
        # 从最低层开始向上融合
        fused_feat = feat3  # [B, 1, 128, 128]

        # 融合 feat3 和 feat2
        fused_feat = self.up(fused_feat)  # [B, 1, 256, 256]
        # 可选：在融合前应用 1x1 conv 调整通道数使其匹配 feat2
        fused_feat = fused_feat + feat2  # 或 torch.cat([fused_feat, feat2], dim=1) 然后加一个 conv

        # 融合 (fused_feat=feat3+feat2) 和 feat1
        fused_feat = self.up(fused_feat)  # [B, 1, 512, 512]
        # 可选：调整通道数匹配 feat1
        fused_feat = fused_feat + feat1  # 或 cat + conv

        # 4. 最后上采样到原始尺寸 (如果需要)
        if fused_feat.shape[2:] != original_size:
            final_result = F.interpolate(fused_feat, size=original_size, mode='bilinear', align_corners=True)
        else:
            final_result = fused_feat

        # 4. 应用 sigmoid
        prob_map = torch.sigmoid(final_result)
        return prob_map

class ExtractSiltedLand(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()

        self.feature = nn.Sequential(
            ResidualBlock(n_channels, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 32),
            ResidualBlock(32, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.feature(x))

class CheckDamNet(nn.Module):
    def __init__(self, n_channels=3, num_classes=2,  bilinear=True):
        super().__init__()

        self.extract_check_dam = ExtractCheckDam(n_channels)
        self.extract_silted_land = ExtractSiltedLand(n_channels)
        self.unet = UNet(n_channels, num_classes, bilinear)
        self.jaccard_index = JaccardIndex(num_classes=num_classes, average='micro', task='multiclass', )

    def forward(self, x):
        # 先处理RGB转灰度（如果需要）
        original_x = x
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

        # 提取淤地坝特征候选区和淤地候选区
        extract_check_dam = self.extract_check_dam(original_x)  # 使用原始RGB图像
        extract_silted_land = self.extract_silted_land(original_x)       # 使用原始RGB图像

        # 组合输入
        combined_input = torch.cat([extract_check_dam, extract_silted_land, x], dim=1)

        return self.unet(combined_input)


if __name__ == '__main__':
    model = CheckDamNet(n_channels=3, num_classes=2)
    test_input = torch.randn(2, 3, 1024, 1024)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")