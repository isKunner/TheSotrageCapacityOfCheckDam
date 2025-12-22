#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: main
# @Time    : 2025/12/3 16:05
# @Author  : Kevin
# @Describe: This file serves as the main entry point for training the check dam segmentation model. It initializes the data module with image and label directories, creates an FCN model instance, configures the PyTorch Lightning trainer with appropriate callbacks, and starts the training process.
import argparse
import os
import torch
import pytorch_lightning as pl
from lightning.pytorch.callbacks import TQDMProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from Data import CheckDamSegmentationDataModule, SegmentationTransform
from Model import SegmentationModel

torch.set_float32_matmul_precision('medium')

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='UNet', choices=['FCN', 'UNet', 'CheckDamNet'])
    parser.add_argument('--root_path', type=str, default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction")
    parser.add_argument('--image_dir', type=str, default=r"Google")
    parser.add_argument('--label_dir', type=str, default=r"Label")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--early_stopping', action='store_true', help="Whether to use early stopping.")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--early_stopping_monitor', type=str, default='val_loss', choices=['val_loss', 'val_iou_epoch'],
                        help="Metric to monitor for early stopping.")
    parser.add_argument('--early_stopping_mode', type=str, default='min', choices=['min', 'max'],
                        help="Mode for monitoring metric ('min' or 'max').")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--log_prefix', type=str, default='', help="Prefix for log directory name")


    return parser.parse_args()

if __name__ == '__main__':

    arg = args()

    pl.seed_everything(arg.seed)

    log_name = f"{arg.log_prefix}_{arg.model}" if arg.log_prefix else arg.model
    logger = TensorBoardLogger("lightning_logs", name=log_name)

    callbacks = [TQDMProgressBar(refresh_rate=1)]

    with open(os.path.join(arg.root_path, "labels.txt")) as f:
        class_names = [line.strip() for line in f.readlines()]

    dm = CheckDamSegmentationDataModule(
        transform=SegmentationTransform(),
        class_names=class_names,
        image_dir=os.path.join(arg.root_path, arg.image_dir),
        label_dir=os.path.join(arg.root_path, arg.label_dir),
        batch_size=arg.batch_size,
        num_workers=arg.num_workers,
        seed=arg.seed
    )

    model = SegmentationModel(arg.model, num_classes=dm.get_class_len(), lr=arg.learning_rate, n_channels=3, bilinear=True)

    if arg.early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor=arg.early_stopping_monitor,
            mode=arg.early_stopping_mode,
            patience=arg.early_stopping_patience,
            verbose=True
        )
        callbacks.append(early_stopping_callback)

    trainer = pl.Trainer(
        max_epochs=arg.max_epochs,
        accelerator='auto',
        devices='auto',
        log_every_n_steps=5,
        callbacks=callbacks,
        logger=logger
    )

    trainer.fit(model, dm)
