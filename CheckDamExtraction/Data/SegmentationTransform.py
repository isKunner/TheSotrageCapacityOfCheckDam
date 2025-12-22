#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: SegmentationTransform
# @Time    : 2025/12/7 22:28
# @Author  : Kevin
# @Describe:

import random
import torchvision.transforms.functional as TF

class SegmentationTransform:
    """
    Supports basic operations such as contrast, brightness, and rotation。
    """

    def __init__(self,
                 brightness_factor_range=(0.8, 1.2),
                 contrast_factor_range=(0.8, 1.2),
                 rotation_angle_range=(-30, 30),
                 horizontal_flip_prob=0.5,
                 apply_probability=0.3):
        """

        Args:
            brightness_factor_range: Brightness adjustment factor range
            contrast_factor_range: Contrast adjustment factor range
            rotation_angle_range: Rotation angle range (degrees)
            horizontal_flip_prob: Horizontal flip probability
            apply_probability: Apply the probability of each enhancement
        """
        self.brightness_factor_range = brightness_factor_range
        self.contrast_factor_range = contrast_factor_range
        self.rotation_angle_range = rotation_angle_range
        self.horizontal_flip_prob = horizontal_flip_prob
        self.apply_probability = apply_probability

    def __call__(self, image, mask):
        """

        Args:
            image: Input image of PIL image type
            mask: Split mask for PIL image types

        Returns:
            Enhanced images and masks (PIL.Image)
        """
        # Random brightness adjustment
        if random.random() < self.apply_probability:
            brightness_factor = random.uniform(*self.brightness_factor_range)
            image = TF.adjust_brightness(image, brightness_factor)

        # Random contrast adjustments
        if random.random() < self.apply_probability:
            contrast_factor = random.uniform(*self.contrast_factor_range)
            image = TF.adjust_contrast(image, contrast_factor)

        # Random horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random rotation
        if random.random() < self.apply_probability:
            angle = random.uniform(*self.rotation_angle_range)
            image = TF.rotate(image, angle, fill=0)
            mask = TF.rotate(mask, angle, fill=0)

        return image, mask