"""
DEM超分辨率推理脚本

用于在Test数据集上进行推理/预测
输入: CopernicusDEM + google影像
输出: HRDEM
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import glob
import json
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

from ..models import create_dam_model, create_super_resolution_system

# 尝试导入LocalPath
try:
    from LocalPath import dam_root_path
except ImportError:
    dam_root_path = r"D:\ResearchData\USA"


def load_model(checkpoint_path: str, encoder: str = 'vitl', device='cuda'):
    """从检查点加载模型"""
    print(f"加载模型: {checkpoint_path}")

    # 创建DAM模型
    dam_model = create_dam_model(
        encoder=encoder,
        num_instances=64,
        device=device
    )

    # 创建超分辨率系统
    model = create_super_resolution_system(
        dam_model=dam_model,
        sr_channels=64,
        sr_residual_blocks=8,
        mapper_base_channels=32,
        use_instance_guidance=True,
        device=device
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"模型加载完成 (Epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def collect_test_samples(
    test_dir: str,
    copernicus_folder: str = 'Copernicus_1.0m_1024pixel',
    google_folder: str = 'google_1.0m_1024pixel'
) -> List[Dict]:
    """收集Test目录下的样本"""
    cop_dir = os.path.join(test_dir, copernicus_folder)
    google_dir = os.path.join(test_dir, google_folder)

    assert os.path.exists(cop_dir), f"找不到Copernicus目录: {cop_dir}"
    assert os.path.exists(google_dir), f"找不到google目录: {google_dir}"

    samples = []
    cop_files = sorted(glob.glob(os.path.join(cop_dir, '*.tif')))

    print(f"找到 {len(cop_files)} 个Copernicus文件")

    for cop_path in cop_files:
        filename = os.path.splitext(os.path.basename(cop_path))[0]
        google_path = os.path.join(google_dir, f"{filename}.tif")

        if not os.path.exists(google_path):
            print(f"警告: 跳过 {filename} (找不到对应的google文件)")
            continue

        samples.append({
            'copernicus_path': cop_path,
            'google_path': google_path,
            'filename': filename
        })

    print(f"有效样本数: {len(samples)}")
    return samples


class InferenceDataset(Dataset):
    """推理数据集(无真值)"""

    def __init__(self, samples: List[Dict], target_size: int = 1022):
        self.samples = samples
        self.target_size = target_size
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.samples)

    def _read_and_resize(self, filepath: str) -> np.ndarray:
        """读取tif文件并调整尺寸"""
        with rasterio.open(filepath) as src:
            data = src.read()

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        # 裁剪到1022x1022
        if data.shape[1] == 1024 and data.shape[2] == 1024:
            data = data[:, 1:-1, 1:-1]

        return data.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # 读取数据 - 保持 (C, H, W) 格式
        cop_data = self._read_and_resize(sample['copernicus_path'])
        google_data = self._read_and_resize(sample['google_path'])

        # 处理 NoData
        nodata_val = -100
        cop_data[0][cop_data[0] <= nodata_val] = np.nan

        # 归一化Copernicus
        valid_mask = ~np.isnan(cop_data[0])
        if valid_mask.sum() == 0:
            dem_min, dem_max = 0.0, 1.0
        else:
            dem_min = np.percentile(cop_data[0][valid_mask], 1)
            dem_max = np.percentile(cop_data[0][valid_mask], 99)

        dem_range = dem_max - dem_min

        # 归一化 Copernicus - 保持 (1, H, W)
        if dem_range > 1e-6:
            cop_norm = (cop_data - dem_min) / dem_range
            cop_norm = np.clip(cop_norm, 0, 1)
        else:
            cop_norm = np.zeros_like(cop_data)
        cop_norm = np.nan_to_num(cop_norm, nan=0.0)

        # 归一化 google 影像 - 保持 3 通道
        if google_data.shape[0] >= 3:
            google_norm = np.zeros((3, google_data.shape[1], google_data.shape[2]), dtype=np.float32)
            for i in range(3):
                google_norm[i] = (google_data[i] - self.image_mean[i]) / self.image_std[i]
        else:
            # 如果少于3通道，复制第一通道
            google_norm = np.zeros((3, google_data.shape[1], google_data.shape[2]), dtype=np.float32)
            for i in range(3):
                google_norm[i] = (google_data[0] - self.image_mean[i]) / self.image_std[i]

        # 转为 Tensor
        cop_tensor = torch.from_numpy(cop_norm)
        google_tensor = torch.from_numpy(google_norm)

        return {
            'copernicus': cop_tensor,
            'google_path': google_tensor,
            'filename': sample['filename'],
            'dem_min': float(dem_min),
            'dem_max': float(dem_max),
            'cop_path': sample['copernicus_path']
        }


def save_prediction(
    hrdem_norm: torch.Tensor,
    copernicus_norm: torch.Tensor,
    dem_min: float,
    dem_max: float,
    cop_path: str,
    output_dir: str,
    filename: str,
    save_viz: bool = True,
    dam_dem_norm: torch.Tensor = None
):
    """保存预测结果(保留地理信息)"""
    range_val = dem_max - dem_min

    # 反归一化
    hrdem_orig = hrdem_norm * range_val + dem_min
    copernicus_orig = copernicus_norm * range_val + dem_min

    hrdem_np = hrdem_orig.squeeze().cpu().numpy()
    copernicus_np = copernicus_orig.squeeze().cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    # 读取原始地理信息
    with rasterio.open(cop_path) as src:
        profile = src.profile
        transform = src.transform

    # 调整 transform
    if transform is not None:
        new_transform = rasterio.Affine(
            transform.a, transform.b,
            transform.c + transform.a,
            transform.d, transform.e,
            transform.f + transform.e
        )
    else:
        new_transform = from_origin(0, 0, 1, 1)

    # 更新 profile
    out_profile = profile.copy()
    out_profile.update({
        'dtype': 'float32',
        'count': 1,
        'height': hrdem_np.shape[0],
        'width': hrdem_np.shape[1],
        'transform': new_transform,
        'compress': 'lzw',
        'nodata': -9999
    })

    # 保存 HRDEM
    pred_path = os.path.join(output_dir, f"{filename}_HRDEM.tif")
    with rasterio.open(pred_path, 'w', **out_profile) as dst:
        dst.write(hrdem_np.astype(np.float32), 1)

    # 保存 DAM 原始输出（如果有）
    if dam_dem_norm is not None:
        dam_orig = dam_dem_norm * range_val + dem_min
        dam_np = dam_orig.squeeze().cpu().numpy()

        dam_path = os.path.join(output_dir, f"{filename}_DAM_raw.tif")
        with rasterio.open(dam_path, 'w', **out_profile) as dst:
            dst.write(dam_np.astype(np.float32), 1)

    # 保存 Copernicus(对比用)
    cop_out_path = os.path.join(output_dir, f"{filename}_Copernicus_resampled.tif")
    with rasterio.open(cop_out_path, 'w', **out_profile) as dst:
        dst.write(copernicus_np.astype(np.float32), 1)

    # 保存统计信息
    stats = {
        'filename': filename,
        'dimensions': f"{hrdem_np.shape[0]}x{hrdem_np.shape[1]}",
        'normalization': {'min': float(dem_min), 'max': float(dem_max), 'range': float(range_val)},
        'predicted_hrdem': {
            'min': float(hrdem_np.min()), 'max': float(hrdem_np.max()),
            'mean': float(hrdem_np.mean()), 'std': float(hrdem_np.std())
        },
        'input_copernicus': {
            'min': float(copernicus_np.min()), 'max': float(copernicus_np.max()),
            'mean': float(copernicus_np.mean())
        }
    }

    json_path = os.path.join(output_dir, f"{filename}_stats.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # 保存可视化图像
    if save_viz:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        im0 = axes[0].imshow(copernicus_np, cmap='terrain', vmin=dem_min, vmax=dem_max)
        axes[0].set_title(f'Copernicus Input (1022x1022)\nRange: [{dem_min:.1f}, {dem_max:.1f}]m')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(hrdem_np, cmap='terrain', vmin=dem_min, vmax=dem_max)
        axes[1].set_title(f'Predicted HRDEM (1022x1022)\nRange: [{hrdem_np.min():.1f}, {hrdem_np.max():.1f}]m')
        plt.colorbar(im1, ax=axes[1])

        diff = hrdem_np - copernicus_np
        vmax = np.abs(diff).max()
        im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2].set_title('Detail Enhancement')
        plt.colorbar(im2, ax=axes[2])

        plt.suptitle(f'{filename} | Min={dem_min:.1f}m Max={dem_max:.1f}m')
        plt.tight_layout()

        viz_path = os.path.join(output_dir, f"{filename}_comparison.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

    return pred_path


@torch.no_grad()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    output_dir: str,
    device: str = 'cuda',
    save_viz: bool = True
):
    """运行推理(批量处理)"""
    model.eval()
    print(f"\n开始推理，共 {len(dataloader)} 个batch...")

    for batch in tqdm(dataloader, desc="Inferencing"):
        copernicus = batch['copernicus'].to(device)
        google = batch['google'].to(device)

        filenames = batch['filename']
        dem_mins = batch['dem_min']
        dem_maxs = batch['dem_max']
        cop_paths = batch['cop_path']

        # 前向传播
        output = model(google, copernicus)
        hrdem_norm = output['hrdem']
        dam_dem_norm = output.get('dam_dem', None)

        # 逐个保存结果
        for i in range(hrdem_norm.shape[0]):
            save_prediction(
                hrdem_norm=hrdem_norm[i],
                copernicus_norm=copernicus[i],
                dem_min=dem_mins[i].item(),
                dem_max=dem_maxs[i].item(),
                cop_path=cop_paths[i],
                output_dir=output_dir,
                filename=filenames[i],
                save_viz=save_viz,
                dam_dem_norm=dam_dem_norm[i] if dam_dem_norm is not None else None
            )

    print(f"\n推理完成! 结果保存在: {output_dir}")


def setup_arguments():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='DEM超分辨率推理')

    # 输入参数
    parser.add_argument('--test_dir', type=str,
                        default=os.path.join(dam_root_path, 'Test'),
                        help='Test目录路径')
    parser.add_argument('--copernicus_folder', type=str,
                        default='Copernicus_1.0m_1024pixel',
                        help='Copernicus DEM文件夹名')
    parser.add_argument('--google_folder', type=str,
                        default='google_1.0m_1024pixel',
                        help='google影像文件夹名')
    parser.add_argument('--checkpoint', type=str,
                        default="./checkpoints/best_checkpoint.pth",
                        help='模型检查点路径(.pth)')
    parser.add_argument('--output_dir', type=str,
                        default='./Test/inference_results',
                        help='输出目录')

    # 模型参数
    parser.add_argument('--encoder', type=str, default='vitg',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='DAM编码器类型')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小(建议设为1)')
    parser.add_argument('--no_viz', action='store_true',
                        help='不保存可视化图像(节省空间)')

    return parser


def run_dem_super_resolution(
    test_dir: str,
    copernicus_folder: str,
    google_folder: str,
    checkpoint: str,
    output_dir: str,
    encoder: str,
    device: str,
    batch_size: int,
    no_viz: bool
):
    """运行DEM超分辨率推理的主要函数

    Args:
        test_dir: 测试目录路径
        copernicus_folder: Copernicus DEM文件夹名
        google_folder: google影像文件夹名
        checkpoint: 模型检查点路径
        output_dir: 输出目录
        encoder: DAM编码器类型
        device: 计算设备
        batch_size: 批次大小
        no_viz: 是否不保存可视化图像
    """
    # 创建设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    model = load_model(checkpoint, encoder, device)

    # 收集测试样本
    print(f"\n扫描测试目录: {test_dir}")
    samples = collect_test_samples(
        test_dir,
        copernicus_folder,
        google_folder
    )

    if len(samples) == 0:
        print("错误: 没有找到有效的测试样本!")
        return

    # 创建数据加载器
    dataset = InferenceDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # 运行推理
    run_inference(
        model=model,
        dataloader=dataloader,
        output_dir=output_dir,
        device=device,
        save_viz=not no_viz
    )

    # 打印结果信息
    print(f"\n{'=' * 50}")
    print("推理结果说明:")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print("输出文件:")
    print("  - *_HRDEM.tif: 预测的高分辨率DEM结果")
    print("  - *_Copernicus_resampled.tif: 输入DEM")
    print("  - *_stats.json: 统计信息")
    print("  - *_comparison.png: 可视化对比")
    print(f"{'=' * 50}")


def main():
    """主函数"""
    parser = setup_arguments()
    args = parser.parse_args()

    run_dem_super_resolution(
        test_dir=args.test_dir,
        copernicus_folder=args.copernicus_folder,
        google_folder=args.google_folder,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        encoder=args.encoder,
        device=args.device,
        batch_size=args.batch_size,
        no_viz=args.no_viz
    )


if __name__ == "__main__":
    main()
