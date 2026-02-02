#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: augmentation.py
# @Time    : 2026/2/1 16:25
# @Author  : Kevin
# @Describe:

"""
数据增强模块

包含各种DEM数据增强技术
"""

import torch
import numpy as np


class RandomFlip:
    """随机翻转"""
    
    def __init__(self, p=0.5, direction='horizontal'):
        """
        Args:
            p: 翻转概率
            direction: 'horizontal' 或 'vertical'
        """
        self.p = p
        self.direction = direction
    
    def __call__(self, copernicus, google, usgs):
        if np.random.random() < self.p:
            if self.direction == 'horizontal':
                copernicus = torch.flip(copernicus, dims=[-1])
                google = torch.flip(google, dims=[-1])
                usgs = torch.flip(usgs, dims=[-1])
            else:  # vertical
                copernicus = torch.flip(copernicus, dims=[-2])
                google = torch.flip(google, dims=[-2])
                usgs = torch.flip(usgs, dims=[-2])
        return copernicus, google, usgs


class RandomRotate:
    """随机旋转（90度的倍数）"""
    
    def __init__(self, p=0.3):
        """
        Args:
            p: 旋转概率
        """
        self.p = p
    
    def __call__(self, copernicus, google, usgs):
        if np.random.random() < self.p:
            k = np.random.randint(1, 4)  # 90, 180, 270度
            copernicus = torch.rot90(copernicus, k, dims=[-2, -1])
            google = torch.rot90(google, k, dims=[-2, -1])
            usgs = torch.rot90(usgs, k, dims=[-2, -1])
        return copernicus, google, usgs


class RandomNoise:
    """随机添加高斯噪声（仅对DEM数据）"""
    
    def __init__(self, p=0.2, std_range=(0.001, 0.01)):
        """
        Args:
            p: 添加噪声的概率
            std_range: 噪声标准差范围
        """
        self.p = p
        self.std_range = std_range
    
    def __call__(self, copernicus, google, usgs):
        if np.random.random() < self.p:
            noise_std = np.random.uniform(*self.std_range)
            copernicus = copernicus + torch.randn_like(copernicus) * noise_std
            usgs = usgs + torch.randn_like(usgs) * noise_std
        return copernicus, google, usgs


class RandomBrightness:
    """随机亮度调整（仅对Google影像）"""
    
    def __init__(self, p=0.3, factor_range=(0.9, 1.1)):
        """
        Args:
            p: 调整概率
            factor_range: 亮度因子范围
        """
        self.p = p
        self.factor_range = factor_range
    
    def __call__(self, copernicus, google, usgs):
        if np.random.random() < self.p:
            brightness_factor = np.random.uniform(*self.factor_range)
            google = google * brightness_factor
        return copernicus, google, usgs


class DEMDataAugmentation:
    """
    DEM数据增强类
    
    组合多种数据增强技术
    """
    
    def __init__(
        self,
        p_flip=0.5,
        p_rotate=0.3,
        p_noise=0.2,
        p_brightness=0.3
    ):
        """
        Args:
            p_flip: 水平翻转概率
            p_rotate: 旋转概率
            p_noise: 添加噪声概率
            p_brightness: 亮度调整概率
        """
        self.augmentations = [
            RandomFlip(p=p_flip, direction='horizontal'),
            RandomFlip(p=p_flip, direction='vertical'),
            RandomRotate(p=p_rotate),
            RandomNoise(p=p_noise),
            RandomBrightness(p=p_brightness)
        ]
    
    def __call__(self, copernicus, google, usgs):
        """
        对输入数据进行增强
        
        Args:
            copernicus: (1, H, W)
            google: (3, H, W)
            usgs: (1, H, W)
        
        Returns:
            增强后的数据
        """
        for aug in self.augmentations:
            copernicus, google, usgs = aug(copernicus, google, usgs)
        return copernicus, google, usgs
