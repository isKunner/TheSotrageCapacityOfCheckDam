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
import json
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

from ..models import create_dam_model, create_super_resolution_system
from ..data import collect_test_samples, DEMSuperResolutionDataset, _collate_fn_filter_none


def load_model(checkpoint_path: str,
               device='cuda',
               # 模型参数 - 与train.py保持一致
               dam_encoder='vits',
               num_prototypes=128,
               embedding_dim=64,
               sr_channels=64,
               sr_residual_blocks=8,
               mapper_base_channels=32,
               mapper_scale_factor=30,
               use_instance_guidance=True,
               use_adaptive_fusion=True,
               ):
    """从检查点加载模型"""
    print(f"加载模型: {checkpoint_path}")
    print(f"  - DAM编码器: {dam_encoder}")
    print(f"  - 原型数量: {num_prototypes}")
    print(f"  - 嵌入维度: {embedding_dim}")
    print(f"  - SR通道数: {sr_channels}")
    print(f"  - SR残差块数: {sr_residual_blocks}")
    print(f"  - Mapper基础通道: {mapper_base_channels}")
    print(f"  - Mapper下采样倍率: {mapper_scale_factor}")
    print(f"  - 实例引导: {'启用' if use_instance_guidance else '禁用'}")
    print(f"  - 自适应融合: {'启用' if use_adaptive_fusion else '禁用'}")

    # 创建DAM模型
    dam_model = create_dam_model(
        encoder=dam_encoder,
        num_prototypes=num_prototypes,
        embedding_dim=embedding_dim,
        device=device
    )

    # 创建超分辨率系统
    model = create_super_resolution_system(
        dam_model=dam_model,
        sr_channels=sr_channels,
        sr_residual_blocks=sr_residual_blocks,
        mapper_scale_factor=mapper_scale_factor,
        mapper_base_channels=mapper_base_channels,
        use_instance_guidance=use_instance_guidance,
        use_adaptive_fusion=use_adaptive_fusion,
        device=device,
        use_cached_dam_encoder=False,  # 推理时不使用缓存
    )

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"模型加载完成 (Epoch {checkpoint.get('epoch', 'unknown')})")
    return model


def save_prediction(
    hrdem_norm: torch.Tensor,
    copernicus_norm: torch.Tensor,
    cop_mean: float,
    cop_std: float,
    usgs_mean: float,
    usgs_std: float,
    cop_path: str,
    output_dir: str,
    filename: str,
    save_viz: bool = True,
    dam_dem_norm: torch.Tensor = None,
    has_ground_truth: bool = False
):
    """保存预测结果(保留地理信息)"""
    # 如果无真值，使用 cop 的统计量作为后备（因为 usgs_mean/std 默认为 0/1）
    if not has_ground_truth:
        usgs_mean, usgs_std = cop_mean, cop_std

    # Z-score 反归一化: x = x_norm * std + mean
    hrdem_orig = hrdem_norm * usgs_std + usgs_mean
    copernicus_orig = copernicus_norm * cop_std + cop_mean

    hrdem_np = hrdem_orig.squeeze().cpu().numpy()
    copernicus_np = copernicus_orig.squeeze().cpu().numpy()

    os.makedirs(output_dir, exist_ok=True)

    # 读取原始地理信息
    with rasterio.open(cop_path) as src:
        profile = src.profile
        transform = src.transform

    # 调整 transform（1024->1022 裁剪后，原点在左上角，需偏移1像素）
    if transform is not None:
        new_transform = rasterio.Affine(
            transform.a, transform.b,
            transform.c + transform.a,  # x方向偏移1个像素
            transform.d, transform.e,
            transform.f + transform.e   # y方向偏移1个像素（y分辨率通常为负）
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
        dam_orig = dam_dem_norm * usgs_std + usgs_mean
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
        'normalization': {
            'cop_mean': float(cop_mean),
            'cop_std': float(cop_std),
            'usgs_mean': float(usgs_mean),
            'usgs_std': float(usgs_std),
            'has_ground_truth': bool(has_ground_truth)
        },
        'predicted_hrdem': {
            'min': float(hrdem_np.min()),
            'max': float(hrdem_np.max()),
            'mean': float(hrdem_np.mean()),
            'std': float(hrdem_np.std())
        },
        'input_copernicus': {
            'min': float(copernicus_np.min()),
            'max': float(copernicus_np.max()),
            'mean': float(copernicus_np.mean()),
            'std': float(copernicus_np.std())
        }
    }

    json_path = os.path.join(output_dir, f"{filename}_stats.json")
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # 保存可视化图像
    if save_viz:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 使用Copernicus范围保持可视化一致性
        vmin, vmax = copernicus_np.min(), copernicus_np.max()

        im0 = axes[0].imshow(copernicus_np, cmap='terrain', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Copernicus Input ({copernicus_np.shape[0]}x{copernicus_np.shape[1]})\nRange: [{vmin:.1f}, {vmax:.1f}]m')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(hrdem_np, cmap='terrain', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Predicted HRDEM ({hrdem_np.shape[0]}x{hrdem_np.shape[1]})\nRange: [{hrdem_np.min():.1f}, {hrdem_np.max():.1f}]m')
        plt.colorbar(im1, ax=axes[1])

        diff = hrdem_np - copernicus_np
        diff_max = np.abs(diff).max()
        im2 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-diff_max, vmax=diff_max)
        axes[2].set_title('Detail Enhancement (HRDEM - Copernicus)')
        plt.colorbar(im2, ax=axes[2])

        plt.suptitle(f'{filename}\nCop: μ={cop_mean:.1f}, σ={cop_std:.1f} | USGS: μ={usgs_mean:.1f}, σ={usgs_std:.1f}')
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
        if batch is None:
            continue

        copernicus = batch['copernicus'].to(device)
        google = batch['google'].to(device)

        filenames = batch['filename']
        cop_paths = batch['copernicus_path']
        cop_means = batch['cop_mean']
        cop_stds = batch['cop_std']
        usgs_means = batch['usgs_mean']
        usgs_stds = batch['usgs_std']
        has_ground_truths = batch['has_ground_truth']

        # 前向传播
        output = model(google, copernicus)
        hrdem_norm = output['hrdem']
        dam_dem_norm = output.get('dam_dem', None)

        # 逐个保存结果
        for i in range(hrdem_norm.shape[0]):
            save_prediction(
                hrdem_norm=hrdem_norm[i],
                copernicus_norm=copernicus[i],
                cop_mean=cop_means[i].item(),
                cop_std=cop_stds[i].item(),
                usgs_mean=usgs_means[i].item(),
                usgs_std=usgs_stds[i].item(),
                cop_path=cop_paths[i],
                output_dir=output_dir,
                filename=filenames[i],
                save_viz=save_viz,
                dam_dem_norm=dam_dem_norm[i] if dam_dem_norm is not None else None,
                has_ground_truth=has_ground_truths[i].item()
            )

    print(f"\n推理完成! 结果保存在: {output_dir}")


def setup_arguments():
    """设置命令行参数解析器 - 与train.py保持一致"""
    parser = argparse.ArgumentParser(description='DEM超分辨率推理')

    # 输入参数
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test目录路径')
    parser.add_argument('--copernicus_folder', type=str,
                        default='Copernicus_1.0m_1024pixel',
                        help='Copernicus DEM文件夹名 (默认: Copernicus_1.0m_1024pixel)')
    parser.add_argument('--google_folder', type=str,
                        default='WMG_1.0m_1024pixel',
                        help='google影像文件夹名 (默认: WMG_1.0m_1024pixel)')
    parser.add_argument('--checkpoint', type=str,
                        default="./checkpoints/best_checkpoint.pth",
                        help='模型检查点路径(.pth)')
    parser.add_argument('--output_dir', type=str,
                        default='./Test/inference_results',
                        help='输出目录')

    # 模型参数 - 与train.py保持一致
    parser.add_argument('--encoder', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='DAM编码器类型 (默认: vits)')
    parser.add_argument('--num_prototypes', type=int, default=64,
                        help='原型数量 (默认: 128)')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='嵌入维度 (默认: 64)')
    parser.add_argument('--sr_channels', type=int, default=64,
                        help='SR网络通道数 (默认: 64)')
    parser.add_argument('--sr_residual_blocks', type=int, default=8,
                        help='SR网络残差块数 (默认: 8)')
    parser.add_argument('--mapper_base_channels', type=int, default=32,
                        help='Mapper基础通道数 (默认: 32)')
    parser.add_argument('--mapper_scale_factor', type=int, default=30,
                        help='Mapper下采样倍率 (默认: 30)')
    parser.add_argument('--use_instance_guidance', action='store_true', default=True,
                        help='启用实例引导 (默认: True)')
    parser.add_argument('--no_instance_guidance', action='store_false', dest='use_instance_guidance',
                        help='禁用实例引导')
    parser.add_argument('--use_adaptive_fusion', action='store_true', default=False,
                        help='启用自适应融合 (默认: True)')
    parser.add_argument('--no_adaptive_fusion', action='store_false', dest='use_adaptive_fusion',
                        help='禁用自适应融合')

    # 推理参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备 (默认: cuda)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小，建议设为1 (默认: 1)')
    parser.add_argument('--target_size', type=int, default=1024,
                        help='目标尺寸，应为14的倍数 (默认: 1024，实际使用1022)')
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
    device: str = 'cuda',
    batch_size: int = 1,
    no_viz: bool = False,
    target_size: int = 1024,
    num_prototypes: int = 64,
    embedding_dim: int = 32,
    sr_channels: int = 64,
    sr_residual_blocks: int = 8,
    mapper_base_channels: int = 32,
    mapper_scale_factor: int = 30,
    use_instance_guidance: bool = True,
    use_adaptive_fusion: bool = False,
):
    """运行DEM超分辨率推理的主要函数"""
    # 创建设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型 - 传递所有模型参数
    model = load_model(
        checkpoint_path=checkpoint,
        device=device,
        dam_encoder=encoder,
        num_prototypes=num_prototypes,
        embedding_dim=embedding_dim,
        sr_channels=sr_channels,
        sr_residual_blocks=sr_residual_blocks,
        mapper_base_channels=mapper_base_channels,
        mapper_scale_factor=mapper_scale_factor,
        use_instance_guidance=use_instance_guidance,
        use_adaptive_fusion=use_adaptive_fusion,
    )

    # 收集测试样本（使用dataset.py中定义的方法）
    print(f"\n扫描测试目录: {test_dir}")
    samples = collect_test_samples(
        test_dir=test_dir,
        copernicus_folder=copernicus_folder,
        image_folder=google_folder  # 注意：dataset.py中使用的是image_folder
    )

    if len(samples) == 0:
        print("错误: 没有找到有效的测试样本!")
        return

    # 创建推理数据集：继承DEMSuperResolutionDataset并添加路径返回
    class InferenceDataset(DEMSuperResolutionDataset):
        def __getitem__(self, idx):
            result = super().__getitem__(idx)
            if result is None:
                return None
            # 添加路径信息用于保存地理信息
            sample_info = self.samples[idx]
            result['copernicus_path'] = sample_info['copernicus_path']
            return result

    dataset = InferenceDataset(samples, target_size=target_size, normalize=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=_collate_fn_filter_none  # 使用dataset.py中的过滤函数
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
    print("  - *_Copernicus_resampled.tif: 输入DEM（Z-score反归一化后）")
    print("  - *_stats.json: 统计信息（含归一化参数）")
    print("  - *_comparison.png: 可视化对比（使用--no_viz可禁用）")
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
        no_viz=args.no_viz,
        target_size=args.target_size,
        num_prototypes=args.num_prototypes,
        embedding_dim=args.embedding_dim,
        sr_channels=args.sr_channels,
        sr_residual_blocks=args.sr_residual_blocks,
        mapper_base_channels=args.mapper_base_channels,
        mapper_scale_factor=args.mapper_scale_factor,
        use_instance_guidance=args.use_instance_guidance,
        use_adaptive_fusion=args.use_adaptive_fusion,
    )


if __name__ == "__main__":
    main()