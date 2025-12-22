#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: SegmentationDataModule
# @Time    : 2025/12/3 11:03
# @Author  : Kevin
# @Describe: This file implements a CheckDamSegmentationDataModule class that manages data loading and preprocessing for check dam segmentation tasks.

import os
import glob
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split

from .SegmentationDataSet import CheckDamSegmentationDataSet

class CheckDamSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, class_names, image_dir, label_dir, test_image_dir=None, test_label_dir=None, batch_size=8, num_workers=4, pin_memory=False, transform=None, seed=42):
        super().__init__()
        self.class_names = class_names
        if "background" not in class_names:
            self.class_names.insert(0, "background")
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.test_image_dir = test_image_dir
        self.test_label_dir = test_label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.transform = transform

        self.use_test_set = (self.test_image_dir is not None) and (self.test_label_dir is not None)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def __len__(self):
        return len(self.train_dataset)

    def get_class_len(self):
        return len(self.class_names)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            all_label_files = glob.glob(os.path.join(self.label_dir, '**', '*.json'), recursive=True)
            total_len = len(all_label_files)
            if total_len == 0:
                raise ValueError('No label files found in the directory.')
            train_len = int(total_len * 0.8)

            indices = list(range(total_len))
            np.random.seed(self.seed)
            np.random.shuffle(indices)

            train_indices, val_indices = indices[:train_len], indices[train_len:]
            train_label_files, val_label_files = [all_label_files[i] for i in train_indices], [all_label_files[i] for i in val_indices]

            print(f"Total samples: {total_len}")
            print(f"Training samples: {len(train_label_files)}")
            print(f"Validation samples: {len(val_label_files)}")

            self.train_dataset = CheckDamSegmentationDataSet(train_label_files, class_names=self.class_names, img_dir=self.image_dir, lbl_dir=self.label_dir, transform=self.transform)
            self.val_dataset = CheckDamSegmentationDataSet(val_label_files, class_names=self.class_names, img_dir=self.image_dir, lbl_dir=self.label_dir, transform=self.transform)

        if self.use_test_set and (stage == 'test' or stage is None):
            test_label_files = glob.glob(os.path.join(self.test_label_dir, '**', '*.json'), recursive=True)
            if not test_label_files:
                raise ValueError('No label files found in the directory.')
            else:
                self.test_dataset = CheckDamSegmentationDataSet(test_label_files, class_names=self.class_names, img_dir=self.test_image_dir, lbl_dir=self.test_label_dir, transform=self.transform)

        if not self.use_test_set and (stage == 'test'):
            raise ValueError(
                "Test dataset requested (stage='test'), but 'test_image_dir' and/or "
                "'test_label_dir' were not provided during DataModule initialization."
            )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        if self.train_dataset is None:
            raise ValueError('Train dataset is not available. Call setup("fit") first.')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)

    def val_dataloader(self) -> TRAIN_DATALOADERS:
        if self.val_dataset is None:
            raise ValueError('Validation dataset is not available. Call setup("fit") first.')
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        if self.test_dataset is None:
            if self.use_test_set:
                raise RuntimeError('Test dataset is not available. Call setup("test") first.')
            else:
                raise RuntimeError(
                    "Test dataset requested (stage='test'), but 'test_image_dir' and/or "
                    "'test_label_dir' were not provided during DataModule initialization."
                )
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)