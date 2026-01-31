#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: SegmentationDataSet
# @Time    : 2025/12/3 10:48
# @Author  : Kevin
# @Describe: This file defines a PyTorch Dataset class for semantic segmentation of check dam images.
#           It loads LabelMe annotation files (.json) and corresponding images to create image-mask pairs for training segmentation models.

import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CheckDamSegmentationDataSet(Dataset):
    """
        Dataset for semantic segmentation of check dams and silted land from LabelMe annotations.
    """

    def __init__(self, label_files_list, class_names, img_dir, lbl_dir, transform=None):
        """
        Args:
            label_files_list (list): List of paths to .json label files.
            class_names(list): List of class labels for segmentation. The background class will be automatically inserted at index 0 if not present in the provided list. Example: ['background', 'check_dam', 'silted_land'].
            img_dir (str, optional): LocalPath to the directory containing images.
                                     If None, assumed to be same as lbl_dir or derived from json.
            lbl_dir (str, optional): LocalPath to the directory containing .json files.
                                     Defaults to LABEL_ROOT_PATH.
            transform (callable, optional): Optional transform to be applied on a sample. Transform is a callable that accepts two parameters (image, mask).
        """
        self.label_files = label_files_list
        self.num_classes = len(class_names)
        if "background" not in class_names:
            class_names.insert(0, "background")
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform

    def __len__(self):
        return 10
        # return len(self.label_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        json_path = self.label_files[idx]
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        # os.path.basename("..\Google\53770.tif") in Linux is error !!!

        if self.img_dir is None:
            img_path = os.path.join(os.path.dirname(json_path), label_data['imagePath'].replace('\\', '/'))
        else:
            img_path = os.path.join(self.img_dir, os.path.basename(label_data['imagePath'].replace('\\', '/')))

        img_path = os.path.normpath(img_path)

        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"img_path: {self.img_dir}")
            img_basename = os.path.basename(label_data['imagePath'].replace('\\', '/'))
            print(f"img: {img_basename}")
            print(f"file: {img_path}")
            print(f"Error loading image: {e}")

        h, w = image.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)

        shapes = label_data.get("shapes", [])
        for shape in shapes:
            label = shape.get("label", "")
            points = shape.get("points", [])
            if not points or len(points) < 3:
                continue
            if label in self.class_to_idx:
                class_id = self.class_to_idx[label]
                pts = np.array(points, np.int32)
                cv2.fillPoly(mask, [pts], color=class_id)

        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)

        if self.transform:
            image_pil, mask_pil = self.transform(image_pil, mask_pil)

        if not isinstance(image_pil, torch.Tensor):
            image_tensor = transforms.ToTensor()(image_pil)
        else:
            image_tensor = image_pil

        if not isinstance(mask_pil, torch.Tensor):
            mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
        else:
            mask_tensor = mask_pil

        return image_tensor, mask_tensor
