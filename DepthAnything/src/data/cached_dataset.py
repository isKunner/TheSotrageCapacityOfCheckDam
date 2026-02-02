#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: cached_dataset.py
# @Time    : 2026/2/1 16:23
# @Author  : Kevin
# @Describe:

"""
带缓存的数据集模块

从pl_train.py中提取并改进的数据缓存机制
第一次运行时从tif文件读取并保存为numpy数组
后续直接从numpy数组加载，大幅提高速度
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple

from .dataset import collect_valid_samples
from .augmentation import DEMDataAugmentation


class CachedDEMDataset(Dataset):
    """
    带缓存的DEM数据集

    第一次运行时从tif文件读取并保存为numpy数组
    后续直接从numpy数组加载，大幅提高速度
    """

    def __init__(
        self,
        samples: List[Dict],
        cache_dir: str = "./data_cache",
        target_size: int = 1022,  # 14的倍数
        normalize: bool = True,
        augmentation: bool = True,
        force_rebuild_cache: bool = False
    ):
        """
        Args:
            samples: 样本列表
            cache_dir: 缓存目录，如果为None则使用默认路径
            target_size: 目标尺寸（14的倍数）
            normalize: 是否归一化
            augmentation: 是否进行数据增强
            force_rebuild_cache: 是否强制重建缓存
        """
        self.samples = samples
        self.target_size = target_size
        self.normalize = normalize
        self.augmentation = augmentation

        # 数据增强器
        if augmentation:
            self.aug = DEMDataAugmentation()
        else:
            self.aug = None

        # Google Earth影像的归一化参数
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        # 设置缓存目录
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 检查缓存状态
        self.cache_status = self._check_cache()

        if not self.cache_status['all_cached'] or force_rebuild_cache:
            print(f"需要构建缓存，目标目录: {self.cache_dir}")
            self._build_cache()
        else:
            print(f"缓存已存在，共 {len(self.samples)} 个样本")

    def _get_cache_path(self, sample_idx: int, data_type: str) -> str:
        """获取缓存文件路径"""
        sample = self.samples[sample_idx]
        filename = sample['filename']
        group = sample['group']
        return os.path.join(self.cache_dir, f"{group}_{filename}_{data_type}.npy")

    def _check_cache(self) -> Dict:
        """检查缓存状态"""
        status = {'all_cached': True, 'cached_count': 0}

        for i in range(len(self.samples)):
            copernicus_cache = os.path.exists(self._get_cache_path(i, 'copernicus'))
            google_cache = os.path.exists(self._get_cache_path(i, 'google'))
            usgs_cache = os.path.exists(self._get_cache_path(i, 'usgs'))

            if copernicus_cache and google_cache and usgs_cache:
                status['cached_count'] += 1
            else:
                status['all_cached'] = False

        return status

    def _build_cache(self):
        """构建缓存"""

        print("开始构建数据缓存...")

        for idx in tqdm(range(len(self.samples)), desc="Building cache"):
            sample = self.samples[idx]

            if (not os.path.exists(sample['copernicus_path'])) or (not os.path.exists(sample['google_path'])) or ((sample['usgs_path'] is not None and not os.path.exists(sample['usgs_path'])) ):
                continue

            usgs_path = self._get_cache_path(idx, 'usgs')
            cop_path = self._get_cache_path(idx, 'copernicus')
            google_path = self._get_cache_path(idx, 'google')
            stats_path = self._get_cache_path(idx, 'stats')

            if (sample['usgs_path'] is None or os.path.exists(usgs_path)) and os.path.exists(cop_path) and os.path.exists(google_path) and os.path.exists(stats_path):
                continue

            # 读取tif文件
            copernicus_data = self._read_tif(sample['copernicus_path'])
            google_data = self._read_tif(sample['google_path'])
            
            # 检查是否有USGS真值
            has_ground_truth = sample['usgs_path'] is not None and os.path.exists(sample['usgs_path'])
            
            if has_ground_truth:
                usgs_data = self._read_tif(sample['usgs_path'])
                # Z-score归一化并获取统计信息
                copernicus_norm, usgs_norm, cop_stats, usgs_stats = self._normalize_dem(copernicus_data, usgs_data)
                np.save(usgs_path, usgs_norm)
            else:
                copernicus_norm, _, cop_stats, _ = self._normalize_dem(copernicus_data, None)
            
            # 归一化Google影像
            if self.normalize:
                for i in range(google_data.shape[0]):
                    google_data[i] = (google_data[i] - self.image_mean[i]) / self.image_std[i]

            # 保存缓存
            np.save(cop_path, copernicus_norm)
            np.save(google_path, google_data)
            
            # 保存统计量 (cop_mean, cop_std, usgs_mean, usgs_std)
            cop_mean, cop_std = cop_stats
            if usgs_stats is not None:
                usgs_mean, usgs_std = usgs_stats
            else:
                usgs_mean, usgs_std = 0.0, 1.0
            np.save(stats_path, np.array([cop_mean, cop_std, usgs_mean, usgs_std], dtype=np.float32))

        print(f"缓存构建完成！共 {len(self.samples)} 个样本")

    def _read_tif(self, filepath: str) -> np.ndarray:
        """读取tif文件"""
        import rasterio
        with rasterio.open(filepath) as src:
            data = src.read()
            if data.ndim == 2:
                data = data[np.newaxis, ...]
        
        # 裁剪到1022x1022
        if data.shape[1] == 1024 and data.shape[2] == 1024:
            data = data[:, 1:-1, 1:-1]
            
        return data.astype(np.float32)

    def _zscore_normalize(self, data: np.ndarray, outlier_percentile: float = 1.0) -> Tuple:
        """
        Z-score归一化，含异常值处理
        
        Args:
            data: 输入数据 (C, H, W)
            outlier_percentile: 作为异常值裁剪的百分位数
            
        Returns:
            normalized: 归一化后的数据
            mean: 均值（用于反归一化）
            std: 标准差（用于反归一化）
        """
        # 处理nodata值
        nodata_value = -100
        data = data.copy()
        data[data <= nodata_value] = np.nan
        
        # 计算异常值边界
        lower = np.nanpercentile(data, outlier_percentile)
        upper = np.nanpercentile(data, 100 - outlier_percentile)
        
        # 裁剪异常值（仅用于计算统计量）
        data_clipped = np.clip(data, lower, upper)
        
        # 计算稳健统计量
        mean = np.nanmean(data_clipped)
        std = np.nanstd(data_clipped) + 1e-6
        
        # 对整个数据（含异常值）进行Z-score归一化
        normalized = (data - mean) / std
        
        # 将nan替换为0（应该是没有nan了，但保险起见）
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return normalized.astype(np.float32), float(mean), float(std)

    def _normalize_dem(self, copernicus_data: np.ndarray, usgs_data: np.ndarray = None) -> Tuple:
        """
        Z-score归一化DEM数据（每个样本独立归一化）
        
        Returns:
            copernicus_normalized: Z-score归一化后的Copernicus
            usgs_normalized: Z-score归一化后的USGS（如果有）
            cop_stats: (mean, std) for Copernicus
            usgs_stats: (mean, std) for USGS（如果有）
        """
        # 对Copernicus进行Z-score归一化
        copernicus_normalized, cop_mean, cop_std = self._zscore_normalize(copernicus_data)
        
        if usgs_data is not None:
            # 对USGS进行Z-score归一化（独立归一化，保留相对起伏）
            usgs_normalized, usgs_mean, usgs_std = self._zscore_normalize(usgs_data)
            return copernicus_normalized, usgs_normalized, (cop_mean, cop_std), (usgs_mean, usgs_std)
        else:
            return copernicus_normalized, None, (cop_mean, cop_std), None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取样本（从缓存加载）"""
        # 从缓存加载
        copernicus_data = np.load(self._get_cache_path(idx, 'copernicus'))
        google_data = np.load(self._get_cache_path(idx, 'google'))
        stats = np.load(self._get_cache_path(idx, 'stats'))
        
        # 尝试加载USGS数据
        usgs_path = self._get_cache_path(idx, 'usgs')
        if os.path.exists(usgs_path):
            usgs_data = np.load(usgs_path)
            has_ground_truth = True
        else:
            usgs_data = np.zeros_like(copernicus_data)
            has_ground_truth = False

        # 转换为tensor
        copernicus_tensor = torch.from_numpy(copernicus_data)
        google_tensor = torch.from_numpy(google_data)
        usgs_tensor = torch.from_numpy(usgs_data)

        # 数据增强（仅训练时）
        if self.aug is not None and has_ground_truth:
            copernicus_tensor, google_tensor, usgs_tensor = self.aug(
                copernicus_tensor, google_tensor, usgs_tensor
            )

        return {
            'copernicus': copernicus_tensor,
            'google': google_tensor,
            'usgs': usgs_tensor,
            'group': self.samples[idx]['group'],
            'filename': self.samples[idx]['filename'],
            # Z-score统计量 (mean, std) 用于反归一化
            'cop_mean': stats[0],
            'cop_std': stats[1],
            'usgs_mean': stats[2],
            'usgs_std': stats[3],
            'has_ground_truth': has_ground_truth
        }


def create_dataloaders_with_cache(
    base_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
    cache_dir: str = './data_cache',
    target_size: int = 1022,
    force_rebuild_cache: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    创建带缓存的数据加载器
    
    Args:
        dam_root_path
        base_dir: 数据基础目录
        batch_size: 批次大小
        num_workers: worker数量
        seed: 随机种子
        cache_dir: 缓存目录
        target_size: 目标尺寸
        force_rebuild_cache: 是否强制重建缓存
    
    Returns:
        train_loader, test_loader
    """
    # 收集有效样本
    train_samples, test_samples = collect_valid_samples(base_dir, seed=seed)
    
    # 创建数据集
    train_dataset = CachedDEMDataset(
        train_samples,
        cache_dir=cache_dir,
        target_size=target_size,
        normalize=True,
        augmentation=True,
        force_rebuild_cache=force_rebuild_cache
    )
    
    test_dataset = CachedDEMDataset(
        test_samples,
        cache_dir=cache_dir,
        target_size=target_size,
        normalize=True,
        augmentation=False,
        force_rebuild_cache=False  # 测试集不重建缓存
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, test_loader
