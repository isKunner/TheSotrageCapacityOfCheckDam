#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: loss
# @Time    : 2025/12/7 15:45
# @Author  : Kevin
# @Describe:

import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        # Register alpha as a buffer so it moves with the model (to GPU/CPU)
        # Handle if alpha is None, a scalar, or a list/tensor
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha)
            elif not isinstance(alpha, torch.Tensor):
                raise TypeError("alpha must be None, a scalar, a list, or a Tensor")
            # Ensure alpha is float
            if alpha.dtype != torch.float32:
                alpha = alpha.to(torch.float32)
            self.register_buffer('alpha', alpha)  # Stores alpha in state_dict, moves with model
        else:
            self.alpha = None  # If None, no alpha weighting applied beyond gamma

    def forward(self, inputs, targets):
        # Calculate Cross Entropy Loss without reduction and ignoring specified index
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)

        # Calculate pt (probability of the true class for each pixel)
        pt = torch.exp(-ce_loss)

        # Calculate the base focal loss component (without alpha)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply alpha weighting if alpha is provided
        if self.alpha is not None:
            # Gather the alpha value corresponding to each target class
            # targets shape: [N, H, W]
            # alpha shape: [C] (where C is num_classes)
            # We want alpha_per_pixel shape: [N, H, W], where each element is alpha[target_class]

            # Flatten targets for easier indexing (handle ignore_index later)
            flat_targets = targets.view(-1)  # Shape: [N*H*W]

            # Gather alpha values based on class indices in flat_targets
            # alpha.index_select(0, flat_targets) selects alpha[class_idx] for each class_idx in flat_targets
            alpha_per_pixel_flat = self.alpha.index_select(0, flat_targets)  # Shape: [N*H*W]

            # Reshape back to the original target shape
            alpha_per_pixel = alpha_per_pixel_flat.view(targets.shape)  # Shape: [N, H, W]

            # Apply the alpha weight to the focal loss
            focal_loss = alpha_per_pixel * focal_loss

        # Apply final reduction
        if self.reduction == 'mean':
            # Optionally, you might want to ignore the ignored pixels in the mean calculation
            # But since ce_loss was already calculated with ignore_index, those positions contribute 0 to sum.
            # Mean over all elements (including ignored if they exist post CE calculation)
            # More precise would be to mask them out explicitly for mean calc if needed.
            # For simplicity, we rely on ignored pixels contributing 0 loss.
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss