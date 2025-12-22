#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: SegmentationModel
# @Time    : 2025/12/9 08:26
# @Author  : Kevin
# @Describe:

import torch
import pytorch_lightning as pl
from torchmetrics import JaccardIndex
from .UNet import UNet
from .FCN import FCN
from .CheckDamNet import CheckDamNet
from .loss import FocalLoss

class SegmentationModel(pl.LightningModule):
    def __init__(self, model_name, num_classes=2, lr=1e-3, n_channels=3, bilinear=True):
        """
        Args:
            model_name (str): Name of the model architecture ('unet', 'fcn', 'checkdamnet').
            num_classes (int): Number of output classes for segmentation.
            lr (float): Learning rate.
        """
        super().__init__()
        self.save_hyperparameters() # Saves arguments passed to __init__
        self.lr = lr
        self.num_classes = num_classes

        # --- Model Instantiation ---
        if model_name.lower() == 'unet':
            self.model = UNet(num_classes=num_classes, n_channels=n_channels, bilinear=bilinear)
        elif model_name.lower() == 'fcn':
            # FCN usually fixes input channels based on backbone (e.g., ResNet outputs 2048 features)
            # Adjust num_classes via model_params if needed
            self.model = FCN(num_classes=num_classes)
        elif model_name.lower() == 'checkdamnet':
             # Ensure CheckDamNet expects n_channels in model_params
            self.model = CheckDamNet(num_classes=num_classes, n_channels=n_channels, bilinear=bilinear)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # --- Loss, Metrics, Optimizer Setup (part of LightningModule) ---
        # Note: Alpha might need adjustment based on your dataset's class distribution
        self.criterion = FocalLoss(alpha=[1.0 for _ in range(num_classes)], gamma=2) # Consider moving alpha config outside
        self.train_iou = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=255)
        self.val_iou = JaccardIndex(task='multiclass', num_classes=num_classes, ignore_index=255)

    def forward(self, x):
        """Forward pass delegates to the underlying model."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images) # Calls self.forward -> self.model.forward
        loss = self.criterion(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_iou.update(outputs, targets)
        self.log('train_iou', self.train_iou, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        train_iou_value = self.train_iou.compute()
        self.log('train_iou_epoch', train_iou_value, prog_bar=True)
        self.train_iou.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.val_iou.update(outputs, targets)
        self.log('val_iou', self.val_iou, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        val_iou_value = self.val_iou.compute()
        self.log('val_iou_epoch', val_iou_value, prog_bar=True)
        self.val_iou.reset()

    def configure_optimizers(self):
        # Access parameters from the underlying model
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) # Or self.model.parameters()?
        # Monitor validation IoU epoch for LR scheduling if desired
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_iou_epoch"}
        return optimizer # Simple Adam optimizer

# Example Usage (how you'd instantiate this in your training script)
# model = SegmentationModel(model_name='unet', model_params={'n_channels': 3, 'bilinear': True}, num_classes=2, lr=1e-3)
# model = SegmentationModel(model_name='fcn', model_params={}, num_classes=21, lr=1e-4) # Adjust params for FCN
# model = SegmentationModel(model_name='checkdamnet', model_params={'n_channels': 3}, num_classes=2, lr=1e-3)