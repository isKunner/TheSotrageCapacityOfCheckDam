"""
数据集创建代码
用于读取CopernicusDEM、GoogleRemoteSensing和USGSDEM三个文件夹下的tif文件
创建训练集和测试集（8:2比例）
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import glob
import random
from typing import List, Tuple, Dict
import numpy as np
from torch.utils.data import Dataset
import torch
import rasterio
from rasterio.errors import RasterioIOError
from tqdm import tqdm

from LocalPath import dam_root_path


def collect_valid_samples(
    base_dir: str = dam_root_path,
    subfolders: List[str] = None,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    收集有效的样本，确保三个文件夹中对应的tif文件都存在
    
    Args:
        base_dir: 基础目录路径
        subfolders: 子文件夹列表，默认为 ['GeoDAR_v11_dams_of_USA_group1', 
                                         'GeoDAR_v11_dams_of_USA_group10',
                                         'GeoDAR_v11_dams_of_USA_group11',
                                         'GeoDAR_v11_dams_of_USA_group14']
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
    if subfolders is None:
        subfolders = [
            'GeoDAR_v11_dams_of_USA_group1',
            'GeoDAR_v11_dams_of_USA_group10',
            'GeoDAR_v11_dams_of_USA_group11',
            'GeoDAR_v11_dams_of_USA_group14'
        ]
    
    # 三个主文件夹的路径
    copernicus_dir = os.path.join(base_dir, 'CopernicusDEM')
    google_dir = os.path.join(base_dir, 'GoogleRemoteSensing')
    usgs_dir = os.path.join(base_dir, 'USGSDEM')
    
    valid_samples = []
    
    # 遍历每个子文件夹
    for subfolder in subfolders:
        copernicus_sub = os.path.join(copernicus_dir, subfolder+"_paired")
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
            
            # 构建对应的GoogleRemoteSensing和USGSDEM文件路径
            google_path = os.path.join(google_sub, f"{filename}.tif")
            copernicus_path = os.path.join(copernicus_sub, f"{filename}.tif")
            
            # 检查三个文件是否都存在
            if os.path.exists(google_path) and os.path.exists(copernicus_path):
                # 尝试打开文件验证有效性
                valid_samples.append({
                    'copernicus_path': usgs_path,
                    'google_path': google_path,
                    'usgs_path': copernicus_path,
                    'group': subfolder,
                    'filename': filename
                })
                # try:
                #     with rasterio.open(usgs_path) as src1, \
                #          rasterio.open(google_path) as src2, \
                #          rasterio.open(copernicus_path) as src3:
                #         # 验证三个文件都有有效的数据
                #         if src1.read(1).size > 0 and src2.read(1).size > 0 and src3.read(1).size > 0:
                #             valid_samples.append({
                #                 'copernicus_path': usgs_path,
                #                 'google_path': google_path,
                #                 'usgs_path': copernicus_path,
                #                 'group': subfolder,
                #                 'filename': filename
                #             })
                # except RasterioIOError as e:
                #     print(f"警告: 文件读取失败 {filename}: {e}")
                #     continue
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
        copernicus_size: int = 1024,
        google_size: int = 1024,
        usgs_size: int = 1024,
        normalize: bool = True
    ):
        """
        Args:
            samples: 样本列表，由collect_valid_samples函数生成
            copernicus_size: CopernicusDEM的目标尺寸
            google_size: GoogleRemoteSensing的目标尺寸
            usgs_size: USGSDEM的目标尺寸
            normalize: 是否进行归一化
        """
        self.samples = samples
        self.copernicus_size = copernicus_size - copernicus_size%14
        self.google_size = google_size - google_size%14
        self.usgs_size = usgs_size - usgs_size%14
        self.normalize = normalize
        
        # Google Earth影像的归一化参数（ImageNet预训练模型常用）
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _read_tif(self, filepath: str, target_size: int = None) -> np.ndarray:
        """
        读取tif文件并调整尺寸
        
        Args:
            filepath: tif文件路径
            target_size: 目标尺寸，如果为None则保持原始尺寸
        
        Returns:
            读取的数组
        """
        with rasterio.open(filepath) as src:
            data = src.read()  # 读取所有波段
            
            # 如果是单波段，添加通道维度
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            
            # 调整尺寸
            if target_size is not None and (data.shape[1] != target_size or data.shape[2] != target_size):
                from rasterio.warp import reproject, Resampling
                
                # 创建输出数组
                new_data = np.zeros((data.shape[0], target_size, target_size), dtype=data.dtype)
                
                for i in range(data.shape[0]):
                    reproject(
                        source=data[i],
                        destination=new_data[i],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=src.transform,
                        dst_crs=src.crs,
                        resampling=Resampling.bilinear
                    )
                data = new_data

            if data.shape[1] == 1024 and data.shape[2] == 1024:
                data = data[:, 1:-1, 1:-1]  # 去除第一行、最后一行、第一列、最后一列

            return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            包含以下键的字典:
            - 'copernicus': CopernicusDEM张量, shape (1, H, W)
            - 'google': GoogleRemoteSensing张量, shape (3, H, W)
            - 'usgs': USGSDEM张量, shape (1, H, W)
            - 'group': 所属组名
            - 'filename': 文件名
        """
        sample_info = self.samples[idx]
        
        # 读取三个文件
        copernicus_data = self._read_tif(
            sample_info['copernicus_path'],
            self.copernicus_size
        )  # (1, H, W)
        
        google_data = self._read_tif(
            sample_info['google_path'],
            self.google_size
        )  # (3, H, W) 或 (C, H, W)
        
        usgs_data = self._read_tif(
            sample_info['usgs_path'],
            self.usgs_size
        )  # (1, H, W)
        
        # 转换为float32
        copernicus_data = copernicus_data.astype(np.float32)
        google_data = google_data.astype(np.float32)
        usgs_data = usgs_data.astype(np.float32)

        # 1. 标记 NoData（-9999 或其他异常值）为 NaN
        nodata_value = -100  # 也可能是 -32767，根据你的数据调整

        copernicus_data[copernicus_data <= nodata_value] = np.nan
        usgs_data[usgs_data <= nodata_value] = np.nan

        # 2. 计算有效数据的联合范围（忽略 NaN）
        # 方法 A：用 nanpercentile（推荐，自动忽略 NaN）
        all_valid = np.concatenate([
            copernicus_data[~np.isnan(copernicus_data)],
            usgs_data[~np.isnan(usgs_data)]
        ])

        if len(all_valid) == 0:
            # 全是 NoData，返回零
            copernicus_data = np.zeros_like(copernicus_data)
            usgs_data = np.zeros_like(usgs_data)
            dem_min, dem_max = 0.0, 1.0
        else:
            # 取 1% 和 99% 分位数，防止极端异常值
            dem_min = np.percentile(all_valid, 1)  # 或 0，如果用0会被-9999影响
            dem_max = np.percentile(all_valid, 99)  # 或 100
            dem_range = dem_max - dem_min

            if dem_range > 1e-6:
                # 3. 归一化（NaN 会保持 NaN，需要后续填充）
                copernicus_data = (copernicus_data - dem_min) / dem_range
                usgs_data = (usgs_data - dem_min) / dem_range

                # 裁剪到 [0, 1]，超过范围的异常值（包括原来是-9999的）都会被截断
                copernicus_data = np.clip(copernicus_data, 0, 1)
                usgs_data = np.clip(usgs_data, 0, 1)
            else:
                copernicus_data.fill(0)
                usgs_data.fill(0)

        # 4. 填充 NaN（原来的 NoData 区域）为 0 或其他值
        copernicus_data = np.nan_to_num(copernicus_data, nan=0.0)
        usgs_data = np.nan_to_num(usgs_data, nan=0.0)

        # 归一化
        if self.normalize:
            
            # 遥感影像归一化 (ImageNet风格)
            for i in range(google_data.shape[0]):
                google_data[i] = (google_data[i] - self.image_mean[i]) / self.image_std[i]
        
        # 转换为torch张量
        copernicus_tensor = torch.from_numpy(copernicus_data)
        google_tensor = torch.from_numpy(google_data)
        usgs_tensor = torch.from_numpy(usgs_data)
        
        # 确保尺寸正确
        if copernicus_tensor.shape != (1, self.copernicus_size, self.copernicus_size):
            copernicus_tensor = torch.nn.functional.interpolate(
                copernicus_tensor.unsqueeze(0),
                size=(self.copernicus_size, self.copernicus_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        if google_tensor.shape != (3, self.google_size, self.google_size):
            # 如果通道数不对，调整通道数
            if google_tensor.shape[0] != 3:
                if google_tensor.shape[0] == 1:
                    # 单通道复制为三通道
                    google_tensor = google_tensor.repeat(3, 1, 1)
                else:
                    # 取前3个通道
                    google_tensor = google_tensor[:3]
            
            google_tensor = torch.nn.functional.interpolate(
                google_tensor.unsqueeze(0),
                size=(self.google_size, self.google_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        if usgs_tensor.shape != (1, self.usgs_size, self.usgs_size):
            usgs_tensor = torch.nn.functional.interpolate(
                usgs_tensor.unsqueeze(0),
                size=(self.usgs_size, self.usgs_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return {
            'copernicus': copernicus_tensor,
            'google': google_tensor,
            'usgs': usgs_tensor,
            'group': sample_info['group'],
            'filename': sample_info['filename'],
            'dem_min': dem_min,      # 新增：保存统计量用于反归一化
            'dem_max': dem_max       # 新增
        }


def create_dataloaders(
    base_dir: str = r"D:\研究文件\ResearchData\USA",
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建训练集和测试集的DataLoader
    
    Args:
        base_dir: 基础目录路径
        batch_size: 批次大小
        num_workers: 数据加载的worker数量
        seed: 随机种子
    
    Returns:
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
    """
    # 收集有效样本
    train_samples, test_samples = collect_valid_samples(base_dir, seed=seed)
    
    # 创建数据集
    train_dataset = DEMSuperResolutionDataset(train_samples)
    test_dataset = DEMSuperResolutionDataset(test_samples)
    
    # 创建DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader


# 用于测试的代码
if __name__ == "__main__":
    # 测试数据集创建
    print("开始收集样本...")
    train_samples, test_samples = collect_valid_samples()
    
    print(f"\n训练集样本数: {len(train_samples)}")
    print(f"测试集样本数: {len(test_samples)}")
    
    if len(train_samples) > 0:
        # 测试数据集类
        print("\n测试数据集类...")
        dataset = DEMSuperResolutionDataset(train_samples[:5])
        
        sample = dataset[0]
        print(f"Copernicus shape: {sample['copernicus'].shape}")
        print(f"Google shape: {sample['google'].shape}")
        print(f"USGS shape: {sample['usgs'].shape}")
        print(f"Group: {sample['group']}")
        print(f"Filename: {sample['filename']}")
    
    print("\n数据集创建代码测试完成！")
