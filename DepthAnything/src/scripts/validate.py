"""
DEM超分辨率模型验证脚本

支持：
1. 验证集验证
2. 指定目录的批量验证
3. 单张图像验证
4. 可视化结果保存
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import glob
import json
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

from src.data import DEMSuperResolutionDataset, collect_valid_samples
from src.models import create_dam_model, create_super_resolution_system
from src.training import calculate_metrics


class Validator:
    """验证器"""
    
    def __init__(
        self,
        model,
        device='cuda',
        save_visualizations=True,
        visualization_dir='./visualizations'
    ):
        self.model = model
        self.device = device
        self.save_visualizations = save_visualizations
        self.visualization_dir = visualization_dir
        
        if self.save_visualizations:
            os.makedirs(visualization_dir, exist_ok=True)
    
    @torch.no_grad()
    def validate_dataset(
        self,
        val_loader: DataLoader,
        save_predictions: bool = False,
        prediction_dir: str = './predictions'
    ) -> Dict[str, float]:
        """
        验证整个数据集
        
        Args:
            val_loader: 验证集DataLoader
            save_predictions: 是否保存预测结果
            prediction_dir: 预测结果保存目录
        
        Returns:
            metrics: 评估指标字典
        """
        self.model.eval()
        
        if save_predictions:
            os.makedirs(prediction_dir, exist_ok=True)
        
        # 评估指标
        rmse_list = []
        mae_list = []
        mape_list = []
        bias_list = []
        
        # 详细结果
        detailed_results = []
        
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            copernicus = batch['copernicus'].to(self.device)
            google = batch['google'].to(self.device)
            usgs = batch['usgs'].to(self.device)
            dem_mins = batch['dem_min']
            dem_maxs = batch['dem_max']
            groups = batch['group']
            filenames = batch['filename']
            
            # 前向传播
            output = self.model(google, copernicus)
            
            hrdem = output['hrdem']
            
            # 计算每个样本的指标
            for i in range(hrdem.shape[0]):
                pred = hrdem[i:i+1]
                target = usgs[i:i+1]
                
                # 计算指标
                sample_metrics = calculate_metrics(pred, target)
                
                rmse_list.append(sample_metrics['rmse'])
                mae_list.append(sample_metrics['mae'])
                mape_list.append(sample_metrics['mape'])
                bias_list.append(sample_metrics['bias'])
                
                # 保存详细结果
                detailed_results.append({
                    'group': groups[i],
                    'filename': filenames[i],
                    **sample_metrics
                })
                
                # 保存预测结果
                if save_predictions:
                    self._save_prediction(
                        hrdem[i],
                        usgs[i],
                        copernicus[i],
                        groups[i],
                        filenames[i],
                        prediction_dir,
                        dem_mins[i].item(),
                        dem_maxs[i].item()
                    )
        
        # 计算统计指标
        metrics = {
            'rmse_mean': np.mean(rmse_list),
            'rmse_std': np.std(rmse_list),
            'rmse_min': np.min(rmse_list),
            'rmse_max': np.max(rmse_list),
            'mae_mean': np.mean(mae_list),
            'mae_std': np.std(mae_list),
            'mae_min': np.min(mae_list),
            'mae_max': np.max(mae_list),
            'mape_mean': np.mean(mape_list),
            'mape_std': np.std(mape_list),
            'bias_mean': np.mean(bias_list),
            'bias_std': np.std(bias_list),
            'num_samples': len(rmse_list)
        }
        
        # 保存详细结果
        results_path = os.path.join(self.visualization_dir, 'detailed_results.json')
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return metrics
    
    def _save_prediction(
        self,
        hrdem: torch.Tensor,
        usgs: torch.Tensor,
        copernicus: torch.Tensor,
        group: str,
        filename: str,
        prediction_dir: str,
        dem_min: float,
        dem_max: float
    ):
        """保存预测结果"""
        # 创建组目录
        group_dir = os.path.join(prediction_dir, group)
        os.makedirs(group_dir, exist_ok=True)

        range_val = dem_max - dem_min
        hrdem_orig = hrdem * range_val + dem_min
        usgs_orig = usgs * range_val + dem_min
        copernicus_orig = copernicus * range_val + dem_min
        
        # 转换为numpy
        hrdem_np = hrdem_orig.squeeze().cpu().numpy()
        usgs_np = usgs_orig.squeeze().cpu().numpy()
        copernicus_np = copernicus_orig.squeeze().cpu().numpy()
        
        # 保存为tif
        output_path = os.path.join(group_dir, f"{filename}_pred.tif")
        
        # 创建简单的transform
        height, width = hrdem_np.shape
        transform = from_origin(0, 0, 1, 1)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=hrdem_np.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(hrdem_np, 1)
        
        # 保存可视化图像
        if self.save_visualizations:
            self._save_visualization(
                hrdem_np, usgs_np, copernicus_np,
                group, filename
            )

        stats = {
            'dem_min': float(dem_min),
            'dem_max': float(dem_max),
            'pred_range': [float(hrdem_np.min()), float(hrdem_np.max())],
            'gt_range': [float(usgs_np.min()), float(usgs_np.max())]
        }
        with open(os.path.join(group_dir, f"{filename}_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
    
    def _save_visualization(
        self,
        hrdem: np.ndarray,
        usgs: np.ndarray,
        copernicus: np.ndarray,
        group: str,
        filename: str
    ):
        """保存可视化图像"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Copernicus DEM
        im0 = axes[0, 0].imshow(copernicus, cmap='terrain')
        axes[0, 0].set_title('Copernicus DEM (Input)')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Predicted HRDEM
        im1 = axes[0, 1].imshow(hrdem, cmap='terrain')
        axes[0, 1].set_title('Predicted HRDEM')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # USGS DEM (Ground Truth)
        im2 = axes[0, 2].imshow(usgs, cmap='terrain')
        axes[0, 2].set_title('USGS DEM (Ground Truth)')
        plt.colorbar(im2, ax=axes[0, 2])
        
        # Error map
        error = hrdem - usgs
        im3 = axes[1, 0].imshow(error, cmap='RdBu_r', vmin=-np.abs(error).max(), vmax=np.abs(error).max())
        axes[1, 0].set_title('Error (Pred - GT)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Absolute error
        abs_error = np.abs(error)
        im4 = axes[1, 1].imshow(abs_error, cmap='Reds')
        axes[1, 1].set_title('Absolute Error')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Histogram of errors
        axes[1, 2].hist(error.flatten(), bins=50, color='blue', alpha=0.7)
        axes[1, 2].set_title('Error Distribution')
        axes[1, 2].set_xlabel('Error')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].axvline(0, color='red', linestyle='--')
        
        plt.suptitle(f'{group}/{filename}')
        plt.tight_layout()
        
        # 保存
        vis_path = os.path.join(self.visualization_dir, f'{group}_{filename}.png')
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @torch.no_grad()
    def validate_directory(
        self,
        input_dir: str,
        output_dir: str,
        copernicus_dir: Optional[str] = None
    ):
        """
        验证指定目录下的所有图像
        
        Args:
            input_dir: 输入图像目录（Google Earth影像）
            output_dir: 输出目录
            copernicus_dir: Copernicus DEM目录（如果为None，则假设与input_dir同级）
        """
        self.model.eval()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找所有tif文件
        image_files = glob.glob(os.path.join(input_dir, '**/*.tif'), recursive=True)
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for img_path in tqdm(image_files, desc="Processing"):
            # 获取相对路径
            rel_path = os.path.relpath(img_path, input_dir)
            filename = os.path.splitext(rel_path)[0]
            
            # 读取Google Earth影像
            with rasterio.open(img_path) as src:
                google_image = src.read()
                profile = src.profile
            
            # 转换为tensor
            google_tensor = torch.from_numpy(google_image.astype(np.float32)).unsqueeze(0)
            
            # 如果需要，调整通道数
            if google_tensor.shape[1] != 3:
                if google_tensor.shape[1] == 1:
                    google_tensor = google_tensor.repeat(1, 3, 1, 1)
                else:
                    google_tensor = google_tensor[:, :3, :, :]
            
            # 读取Copernicus DEM
            if copernicus_dir is None:
                copernicus_dir = input_dir.replace('GoogleRemoteSensing', 'CopernicusDEM')
            
            copernicus_path = os.path.join(copernicus_dir, rel_path)
            
            if os.path.exists(copernicus_path):
                with rasterio.open(copernicus_path) as src:
                    copernicus_dem = src.read(1)
                
                copernicus_tensor = torch.from_numpy(
                    copernicus_dem.astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)
            else:
                print(f"警告: 找不到对应的Copernicus DEM: {copernicus_path}")
                copernicus_tensor = torch.zeros(1, 1, google_tensor.shape[2], google_tensor.shape[3])
            
            # 移动到设备
            google_tensor = google_tensor.to(self.device)
            copernicus_tensor = copernicus_tensor.to(self.device)
            
            # 前向传播
            output = self.model(google_tensor, copernicus_tensor)
            hrdem = output['hrdem']
            
            # 保存结果
            hrdem_np = hrdem.squeeze().cpu().numpy()
            
            # 创建输出路径
            output_path = os.path.join(output_dir, f"{filename}_hrdem.tif")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 更新profile
            profile.update(
                dtype=hrdem_np.dtype,
                count=1
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(hrdem_np, 1)
        
        print(f"处理完成，结果保存在: {output_dir}")
    
    @torch.no_grad()
    def validate_single(
        self,
        google_image_path: str,
        copernicus_path: str,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        验证单张图像
        
        Args:
            google_image_path: Google Earth影像路径
            copernicus_path: Copernicus DEM路径
            output_path: 输出路径（可选）
        
        Returns:
            hrdem: 预测的高分辨率DEM
        """
        self.model.eval()
        
        # 读取图像
        with rasterio.open(google_image_path) as src:
            google_image = src.read()
            profile = src.profile
        
        with rasterio.open(copernicus_path) as src:
            copernicus_dem = src.read(1)
        
        # 转换为tensor
        google_tensor = torch.from_numpy(google_image.astype(np.float32)).unsqueeze(0)
        copernicus_tensor = torch.from_numpy(
            copernicus_dem.astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)
        
        # 调整通道数
        if google_tensor.shape[1] != 3:
            if google_tensor.shape[1] == 1:
                google_tensor = google_tensor.repeat(1, 3, 1, 1)
            else:
                google_tensor = google_tensor[:, :3, :, :]
        
        # 移动到设备
        google_tensor = google_tensor.to(self.device)
        copernicus_tensor = copernicus_tensor.to(self.device)
        
        # 前向传播
        output = self.model(google_tensor, copernicus_tensor)
        hrdem = output['hrdem']
        
        # 转换为numpy
        hrdem_np = hrdem.squeeze().cpu().numpy()
        
        # 保存结果
        if output_path is not None:
            profile.update(
                dtype=hrdem_np.dtype,
                count=1
            )
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(hrdem_np, 1)
            
            print(f"结果已保存到: {output_path}")
        
        return hrdem_np


def load_model_from_checkpoint(
    checkpoint_path: str,
    dam_encoder: str = 'vitl',
    num_instances: int = 64,
    device: str = 'cuda'
):
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点路径
        dam_encoder: DAM编码器类型
        num_instances: 实例数量
        device: 设备
    
    Returns:
        model: 加载好的模型
    """
    # 创建DAM模型
    dam_model = create_dam_model(
        encoder=dam_encoder,
        num_instances=num_instances,
        device=device
    )
    
    # 创建超分辨率系统
    model = create_super_resolution_system(
        dam_model=dam_model,
        device=device
    )
    
    # 加载检查点
    print(f"加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"检查点已加载 (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    model.eval()
    
    return model


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='DEM超分辨率验证')
    
    # 模式选择
    parser.add_argument('--mode', type=str, required=True,
                        choices=['dataset', 'directory', 'single'],
                        help='验证模式')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--dam_encoder', type=str, default='vitl')
    parser.add_argument('--num_instances', type=int, default=64)
    
    # 数据集验证参数
    parser.add_argument('--data_dir', type=str,
                        default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # 目录验证参数
    parser.add_argument('--input_dir', type=str,
                        help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='./validation_output',
                        help='输出目录')
    parser.add_argument('--copernicus_dir', type=str,
                        help='Copernicus DEM目录')
    
    # 单张验证参数
    parser.add_argument('--google_image', type=str,
                        help='Google Earth影像路径')
    parser.add_argument('--copernicus_image', type=str,
                        help='Copernicus DEM路径')
    
    # 其他参数
    parser.add_argument('--save_predictions', action='store_true',
                        help='是否保存预测结果')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='是否保存可视化结果')
    parser.add_argument('--visualization_dir', type=str, default='./visualizations')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 尝试导入LocalPath
    if args.data_dir is None:
        try:
            from LocalPath import dam_root_path
            args.data_dir = dam_root_path
        except ImportError:
            args.data_dir = r"D:\ResearchData\USA"
    
    # 创建设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    model = load_model_from_checkpoint(
        args.checkpoint,
        args.dam_encoder,
        args.num_instances,
        device
    )
    
    # 创建验证器
    validator = Validator(
        model=model,
        device=device,
        save_visualizations=args.save_visualizations,
        visualization_dir=args.visualization_dir
    )
    
    # 根据模式进行验证
    if args.mode == 'dataset':
        print("\n验证数据集...")
        
        # 收集验证样本
        _, val_samples = collect_valid_samples(args.data_dir)
        
        # 创建验证数据集
        val_dataset = DEMSuperResolutionDataset(val_samples)
        
        # 创建DataLoader
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        # 验证
        metrics = validator.validate_dataset(
            val_loader,
            save_predictions=args.save_predictions,
            prediction_dir=args.output_dir
        )
        
        # 打印结果
        print("\n验证结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # 保存结果
        results_path = os.path.join(args.output_dir, 'validation_metrics.json')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n结果已保存到: {results_path}")
    
    elif args.mode == 'directory':
        print("\n验证目录...")
        validator.validate_directory(
            args.input_dir,
            args.output_dir,
            args.copernicus_dir
        )
    
    elif args.mode == 'single':
        print("\n验证单张图像...")
        output_path = os.path.join(args.output_dir, 'output.tif')
        os.makedirs(args.output_dir, exist_ok=True)
        
        hrdem = validator.validate_single(
            args.google_image,
            args.copernicus_image,
            output_path
        )
        
        print(f"HRDEM形状: {hrdem.shape}")
        print(f"HRDEM范围: [{hrdem.min():.2f}, {hrdem.max():.2f}]")


if __name__ == "__main__":
    main()
