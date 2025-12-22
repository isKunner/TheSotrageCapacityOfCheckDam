#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: FCN
# @Time    : 2025/12/3 09:09
# @Author  : Kevin
# @Describe: This file implements a Fully Convolutional Network (FCN) model for semantic segmentation tasks using PyTorch Lightning framework.

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class FCN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = self.classifier(features)
        out = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)
        return out