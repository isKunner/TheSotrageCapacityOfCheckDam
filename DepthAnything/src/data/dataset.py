#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: dataset.py
# @Time    : 2026/2/1 16:40
# @Author  : Kevin
# @Describe:

"""
数据集创建代码

用于读取CopernicusDEM、GoogleRemoteSensing和USGSDEM三个文件夹下的tif文件
创建训练集和测试集（8:2比例）
"""

import os
import glob
import random
from typing import List, Tuple, Dict
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import rasterio
from tqdm import tqdm


def collect_test_samples(
    test_dir,
    copernicus_folder: str = 'Copernicus_1.0m_1024pixel',
    image_folder: str = 'WMG_1.0m_1024pixel',
) -> List[Dict]:
    """
    收集Test路径下的样本（用于推理/预测，无真值）

    目录结构：
    Test/
        Copernicus_1.0m_1024pixel/   # 低分辨率DEM输入 (30m->1m)
            tile_001.tif
        WMG_1.0m_1024pixel/          # 高分辨率遥感影像 (1m)
            tile_001.tif

    Returns:
        样本列表，每个样本包含：
        {
            'copernicus_path': Copernicus DEM路径,
            'google_path': 高分辨率影像路径(WMG),
            'usgs_path': None,  # 预测模式下无真值
            'group': 'test',
            'filename': 文件名
        }
    """

    copernicus_dir = os.path.join(test_dir, copernicus_folder)
    image_dir = os.path.join(test_dir, image_folder)

    assert os.path.exists(copernicus_dir), f"Copernicus目录不存在: {copernicus_dir}"
    assert os.path.exists(image_dir), f"影像目录不存在: {image_dir}"

    test_samples = []
    cop_files = glob.glob(os.path.join(copernicus_dir, '*.tif'))

    print(f"在 {copernicus_folder} 中找到 {len(cop_files)} 个文件")

    for cop_path in tqdm(cop_files, desc="收集测试样本"):
        filename = os.path.splitext(os.path.basename(cop_path))[0]

        # 对应的高分辨率影像
        img_path = os.path.join(image_dir, f"{filename}.tif")

        if not os.path.exists(img_path):
            print(f"警告: 找不到对应的影像文件 {img_path}，跳过")
            continue

        test_samples.append({
            'copernicus_path': cop_path,
            'google_path': img_path,
            'usgs_path': None,  # 关键：无真值
            'group': 'test',
            'filename': filename
        })

    print(f"总共找到 {len(test_samples)} 个有效测试样本")
    return test_samples


def collect_valid_samples(
    base_dir,
    subfolders: List[str] = None,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    收集有效的样本，确保三个文件夹中对应的tif文件都存在
    
    Args:
        base_dir: 基础目录路径
        subfolders: 子文件夹列表
        seed: 随机种子
    
    Returns:
        train_samples: 训练集样本列表
        test_samples: 测试集样本列表
        每个样本是一个字典，包含：
        {
            'copernicus_path': CopernicusDEM文件路径,
            'google_path': GoogleRemoteSensing文件路径,
            'usgs_path': USGSDEM文件路径,
            'group': 所属组名,
            'filename': 文件名（不含扩展名）
        }
    """
    
    # 三个主文件夹的路径
    copernicus_dir = os.path.join(base_dir, 'CopernicusDEM')
    google_dir = os.path.join(base_dir, 'GoogleRemoteSensing')
    usgs_dir = os.path.join(base_dir, 'USGSDEM')

    if subfolders is None:
        subfolders = os.listdir(usgs_dir)
    
    valid_samples = []
    
    # 遍历每个子文件夹
    for subfolder in subfolders:
        copernicus_sub = os.path.join(copernicus_dir, subfolder)
        google_sub = os.path.join(google_dir, subfolder)
        usgs_sub = os.path.join(usgs_dir, subfolder)
        
        # 检查子文件夹是否存在
        if not os.path.exists(copernicus_sub):
            print(f"警告: {copernicus_sub} 不存在，跳过")
            continue
        if not os.path.exists(google_sub):
            print(f"警告: {google_sub} 不存在，跳过")
            continue
        if not os.path.exists(usgs_sub):
            print(f"警告: {usgs_sub} 不存在，跳过")
            continue
        
        # 获取CopernicusDEM文件夹中的所有tif文件
        usgs_files = glob.glob(os.path.join(usgs_sub, '*.tif'))
        
        for usgs_path in tqdm(usgs_files, desc=f"Processing {subfolder}", unit="file"):

            # 获取文件名（不含路径和扩展名）
            filename = os.path.splitext(os.path.basename(usgs_path))[0]
            
            # 构建对应的GoogleRemoteSensing和CopernicusDEM文件路径
            google_path = os.path.join(google_sub, f"{filename}.tif")
            copernicus_path = os.path.join(copernicus_sub, f"{filename}.tif")
            
            # 检查三个文件是否都存在
            if os.path.exists(google_path) and os.path.exists(copernicus_path):
                valid_samples.append({
                    'copernicus_path': usgs_path,
                    'google_path': google_path,
                    'usgs_path': copernicus_path,
                    'group': subfolder,
                    'filename': filename
                })
            else:
                print(f"信息: 跳过不完整的样本 {filename} (缺少对应文件)")
    
    print(f"总共找到 {len(valid_samples)} 个有效样本")
    
    # 随机打乱并划分训练集和测试集（8:2）
    random.seed(seed)
    random.shuffle(valid_samples)
    
    split_idx = int(len(valid_samples) * 0.8)
    train_samples = valid_samples[:split_idx]
    test_samples = valid_samples[split_idx:]
    
    print(f"训练集: {len(train_samples)} 个样本")
    print(f"测试集: {len(test_samples)} 个样本")
    
    return train_samples, test_samples


class DEMSuperResolutionDataset(Dataset):
    """
    DEM超分辨率数据集
    
    输入:
        - CopernicusDEM: 1通道，低分辨率DEM (30m -> 1m)
        - GoogleRemoteSensing: 3通道，高分辨率遥感影像 (1m)
    输出:
        - USGSDEM: 1通道，高分辨率DEM (1m)
    
    所有数据尺寸: 1024 x 1024 变换到1022 x 1022
    """
    
    def __init__(
        self,
        samples: List[Dict],
        target_size: int = 1024,
        normalize: bool = True
    ):
        """
        Args:
            samples: 样本列表，由collect_valid_samples函数生成
            target_size: 目标尺寸（会自动调整为14的倍数）
            normalize: 是否进行归一化
        """
        self.samples = samples
        self.target_size = target_size - target_size % 14  # 确保是14的倍数
        self.normalize = normalize
        
        # Google Earth影像的归一化参数（ImageNet预训练模型常用）
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _read_tif(self, filepath: str) -> np.ndarray:
        """
        读取tif文件
        
        Args:
            filepath: tif文件路径
        
        Returns:
            读取的数组 (C, H, W)
        """
        with rasterio.open(filepath) as src:
            data = src.read()  # 读取所有波段
            
            # 如果是单波段，添加通道维度
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            
            # 裁剪到1022x1022（如果是1024x1024）
            if data.shape[1] == 1024 and data.shape[2] == 1024:
                data = data[:, 1:-1, 1:-1]
            
            return data.astype(np.float32)
    
    def _zscore_normalize(self, data: np.ndarray, outlier_percentile: float = 1.0):
        """
        Z-score归一化，含异常值处理
        
        Args:
            data: 输入数据
            outlier_percentile: 作为异常值裁剪的百分位数
            
        Returns:
            normalized: 归一化后的数据，如果全NoData则返回None
            mean: 均值
            std: 标准差
        """
        nodata_value = -100
        data = data.copy()
        data[data <= nodata_value] = np.nan
        
        # 检查是否全是 NoData
        if np.all(np.isnan(data)):
            return None, 0.0, 1.0
        
        # 检查有效数据是否全为0
        valid_mask = ~np.isnan(data)
        valid_data = data[valid_mask]
        if np.all(np.abs(valid_data) < 1e-6):
            return None, 0.0, 1.0
        
        # 计算异常值边界（带异常处理）
        try:
            lower = np.nanpercentile(data, outlier_percentile)
            upper = np.nanpercentile(data, 100 - outlier_percentile)
        except:
            lower, upper = np.nanmin(data), np.nanmax(data)
        
        # 裁剪异常值（仅用于计算统计量）
        data_clipped = np.clip(data, lower, upper)
        
        # 计算稳健统计量
        mean = np.nanmean(data_clipped)
        std = np.nanstd(data_clipped)
        
        # 处理 std 为 0 或 NaN 的情况
        if std < 1e-6 or np.isnan(std) or np.isnan(mean):
            print(f"警告: 样本标准差={std}, 均值={mean}，使用默认值")
            std = 1.0
            mean = 0.0
        
        # 对整个数据进行Z-score归一化
        normalized = (data - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        return normalized.astype(np.float32), float(mean), float(std)

    def _normalize_dem(self, copernicus_data: np.ndarray, usgs_data: np.ndarray = None):
        """
        Z-score归一化DEM数据（每个样本独立归一化）
        
        Args:
            copernicus_data: Copernicus DEM数据
            usgs_data: USGS DEM数据（可选）
            
        Returns:
            copernicus_normalized: Z-score归一化后的Copernicus
            usgs_normalized: Z-score归一化后的USGS（如果有）
            cop_stats: (mean, std) for Copernicus
            usgs_stats: (mean, std) for USGS（如果有）
        """
        # 对Copernicus进行Z-score归一化
        copernicus_normalized, cop_mean, cop_std = self._zscore_normalize(copernicus_data)
        
        if usgs_data is not None:
            # 对USGS进行Z-score归一化
            usgs_normalized, usgs_mean, usgs_std = self._zscore_normalize(usgs_data)
            return copernicus_normalized, usgs_normalized, (cop_mean, cop_std), (usgs_mean, usgs_std)
        else:
            return copernicus_normalized, None, (cop_mean, cop_std), None
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        filename = sample_info['filename']

        # 读取输入数据（必须有）
        copernicus_data = self._read_tif(sample_info['copernicus_path'])
        google_data = self._read_tif(sample_info['google_path'])

        # 检查是否有USGS真值（验证模式 vs 预测模式）
        has_ground_truth = sample_info['usgs_path'] is not None and os.path.exists(sample_info['usgs_path'])

        if has_ground_truth:
            # 验证模式：有USGS真值
            usgs_data = self._read_tif(sample_info['usgs_path'])
            
            # Z-score归一化并检查是否全NoData
            copernicus_data, usgs_data, cop_stats, usgs_stats = self._normalize_dem(copernicus_data, usgs_data)
            
            # 检查Copernicus或USGS是否全NoData
            if copernicus_data is None:
                print(f"[丢弃] 文件 {filename}: Copernicus数据全为NoData")
                return None
            if usgs_data is None:
                print(f"[丢弃] 文件 {filename}: USGS数据全为NoData")
                return None
                
            cop_mean, cop_std = cop_stats
            usgs_mean, usgs_std = usgs_stats
            usgs_tensor = torch.from_numpy(usgs_data)
        else:
            # 预测模式：无真值
            copernicus_data, _, cop_stats, _ = self._normalize_dem(copernicus_data, None)
            
            # 检查Copernicus是否全NoData
            if copernicus_data is None:
                print(f"[丢弃] 文件 {filename}: Copernicus数据全为NoData")
                return None
                
            cop_mean, cop_std = cop_stats
            usgs_mean, usgs_std = 0.0, 1.0
            # 创建dummy usgs tensor（保持接口一致，但实际不用）
            usgs_tensor = torch.zeros(1, self.target_size, self.target_size)

        # 检查Google数据是否全为0（NoData）
        if np.all(np.abs(google_data) < 1e-6):
            print(f"[丢弃] 文件 {filename}: Google影像数据全为NoData")
            return None

        # 处理Google影像
        google_data = google_data.astype(np.float32)
        if self.normalize:
            # 根据数据范围判断是否需要 /255
            # 如果最大值 > 10，认为是 0-255 范围，需要归一化
            if google_data.max() > 1:
                google_data = google_data / 255.0
            for i in range(min(3, google_data.shape[0])):
                google_data[i] = (google_data[i] - self.image_mean[i]) / self.image_std[i]
            # 确保3通道
            if google_data.shape[0] == 1:
                google_data = np.repeat(google_data, 3, axis=0)
            elif google_data.shape[0] > 3:
                google_data = google_data[:3]

        # 转换为Tensor
        copernicus_tensor = torch.from_numpy(copernicus_data)
        google_tensor = torch.from_numpy(google_data)

        # 尺寸检查
        if copernicus_tensor.shape != (1, self.target_size, self.target_size):
            copernicus_tensor = torch.nn.functional.interpolate(
                copernicus_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        if google_tensor.shape != (3, self.target_size, self.target_size):
            if google_tensor.shape[0] != 3:
                if google_tensor.shape[0] == 1:
                    google_tensor = google_tensor.repeat(3, 1, 1)
                else:
                    google_tensor = google_tensor[:3]
            google_tensor = torch.nn.functional.interpolate(
                google_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        if usgs_tensor.shape != (1, self.target_size, self.target_size) and has_ground_truth:
            usgs_tensor = torch.nn.functional.interpolate(
                usgs_tensor.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        return {
            'copernicus': copernicus_tensor,
            'google': google_tensor,
            'usgs': usgs_tensor,
            'group': sample_info['group'],
            'filename': sample_info['filename'],
            # Z-score统计量 (mean, std) 用于反归一化
            'cop_mean': float(cop_mean),
            'cop_std': float(cop_std),
            'usgs_mean': float(usgs_mean),
            'usgs_std': float(usgs_std),
            'has_ground_truth': has_ground_truth
        }


def _collate_fn_filter_none(batch):
    """
    自定义collate函数，过滤掉None值（全NoData的样本）
    """
    # 过滤掉None值
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # 使用默认的collate函数
    return torch.utils.data.default_collate(batch)


def create_dataloaders(
    base_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
    target_size: int = 1024
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练集和测试集的DataLoader
    
    Args:
        base_dir: 基础目录路径
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        seed: 随机种子
        target_size: 目标尺寸
    
    Returns:
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
    """
    # 收集有效样本
    train_samples, test_samples = collect_valid_samples(base_dir, seed=seed)
    
    # 创建数据集
    train_dataset = DEMSuperResolutionDataset(train_samples, target_size=target_size)
    test_dataset = DEMSuperResolutionDataset(test_samples, target_size=target_size)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate_fn_filter_none
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn_filter_none
    )
    
    return train_loader, test_loader
