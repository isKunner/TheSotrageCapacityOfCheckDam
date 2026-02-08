#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: test_real_tiff.py
# @Time    : 2026/2/8 19:40
# @Author  : Kevin
# @Describe: Test DEMFusionLayer with real TIFF data

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
from temp import DEMFusionLayer
import warnings

warnings.filterwarnings('ignore')


def load_tiff_file(file_path):
    """加载TIFF文件并返回数据、元数据和投影信息"""
    with rasterio.open(file_path) as src:
        # 读取数据
        data = src.read(1)  # 读取第一个波段

        # 获取元数据
        metadata = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
            'driver': src.driver,
            'bounds': src.bounds
        }

        print(f"Loaded {file_path}:")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Min value: {np.nanmin(data):.2f}")
        print(f"  Max value: {np.nanmax(data):.2f}")
        print(f"  No data value: {src.nodata}")

        return data, metadata


def save_tiff_file(data, metadata, output_path):
    """保存数据为TIFF文件，保持原始元数据"""
    print(f"\nSaving to {output_path}...")

    # 准备写入参数
    write_kwargs = {
        'driver': metadata['driver'],
        'dtype': metadata['dtype'],
        'nodata': metadata['nodata'],
        'width': metadata['width'],
        'height': metadata['height'],
        'count': metadata['count'],
        'crs': metadata['crs'],
        'transform': metadata['transform']
    }

    # 保存文件
    with rasterio.open(output_path, 'w', **write_kwargs) as dst:
        dst.write(data, 1)

    print(f"Successfully saved {output_path}")

    # 验证保存结果
    with rasterio.open(output_path) as src:
        saved_data = src.read(1)
        print(f"Saved file shape: {saved_data.shape}")
        print(f"Saved file range: [{np.nanmin(saved_data):.2f}, {np.nanmax(saved_data):.2f}]")


def preprocess_data(lr_dem_data, dam_data):
    """预处理数据：处理NaN值、归一化等"""
    # 创建数据副本以避免修改原始数据
    lr_dem_processed = lr_dem_data.copy()
    dam_processed = dam_data.copy()

    # 检查NaN值
    lr_nan_mask = np.isnan(lr_dem_processed)
    dam_nan_mask = np.isnan(dam_processed)

    if np.any(lr_nan_mask):
        print(f"LR DEM has {np.sum(lr_nan_mask)} NaN values")
        # 用均值填充NaN值
        lr_mean = np.nanmean(lr_dem_processed)
        lr_dem_processed[lr_nan_mask] = lr_mean

    if np.any(dam_nan_mask):
        print(f"DAM has {np.sum(dam_nan_mask)} NaN values")
        # 用中值填充NaN值（DAM应该在0-1之间）
        dam_median = np.nanmedian(dam_processed)
        dam_processed[dam_nan_mask] = dam_median

    # 确保DAM在0-1范围内
    dam_min = np.min(dam_processed)
    dam_max = np.max(dam_processed)
    if dam_min < 0 or dam_max > 1:
        print(f"DAM range is [{dam_min:.3f}, {dam_max:.3f}], normalizing to [0, 1]")
        dam_processed = (dam_processed - dam_min) / (dam_max - dam_min)

    # 计算LR DEM的统计量（用于Z-score归一化）
    lr_mean = np.mean(lr_dem_processed)
    lr_std = np.std(lr_dem_processed)

    # Z-score归一化LR DEM
    lr_dem_normalized = (lr_dem_processed - lr_mean) / (lr_std + 1e-6)

    print(f"\nPreprocessing statistics:")
    print(f"  LR DEM mean: {lr_mean:.2f}, std: {lr_std:.2f}")
    print(f"  DAM range: [{np.min(dam_processed):.3f}, {np.max(dam_processed):.3f}]")

    return {
        'lr_dem_original': lr_dem_processed,
        'dam_original': dam_processed,
        'lr_dem_normalized': lr_dem_normalized,
        'lr_mean': lr_mean,
        'lr_std': lr_std,
        'lr_nan_mask': lr_nan_mask,
        'dam_nan_mask': dam_nan_mask
    }


def visualize_results(lr_dem, dam, fused_dem, mask=None):
    """可视化输入和输出结果"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # LR DEM
    im1 = axes[0, 0].imshow(lr_dem, cmap='terrain')
    axes[0, 0].set_title('LR DEM (Original)')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0])

    # DAM
    im2 = axes[0, 1].imshow(dam, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].set_title('DAM (0-1)')
    plt.colorbar(im2, ax=axes[0, 1])

    # Fused DEM
    im3 = axes[0, 2].imshow(fused_dem, cmap='terrain')
    axes[0, 2].set_title('Fused DEM')
    plt.colorbar(im3, ax=axes[0, 2])

    # 差异图
    diff = fused_dem - lr_dem
    vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
    im4 = axes[1, 0].imshow(diff, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[1, 0].set_title('Difference (Fused - LR)')
    plt.colorbar(im4, ax=axes[1, 0])

    # 直方图：差异分布
    axes[1, 1].hist(diff.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--')
    axes[1, 1].set_title('Difference Distribution')
    axes[1, 1].set_xlabel('Difference')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # 剖面图
    row_to_plot = lr_dem.shape[0] // 2
    if row_to_plot < lr_dem.shape[0]:
        x = np.arange(lr_dem.shape[1])
        axes[1, 2].plot(x, lr_dem[row_to_plot, :], 'b-', linewidth=2, alpha=0.7, label='LR DEM')
        axes[1, 2].plot(x, fused_dem[row_to_plot, :], 'r-', linewidth=2, label='Fused DEM')
        axes[1, 2].set_title(f'Profile Comparison (Row {row_to_plot})')
        axes[1, 2].set_xlabel('Column')
        axes[1, 2].set_ylabel('Elevation')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('DEM Fusion Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 打印统计信息
    print("\n" + "=" * 80)
    print("FUSION STATISTICS")
    print("=" * 80)
    print(f"LR DEM range: [{np.nanmin(lr_dem):.2f}, {np.nanmax(lr_dem):.2f}]")
    print(f"Fused DEM range: [{np.nanmin(fused_dem):.2f}, {np.nanmax(fused_dem):.2f}]")
    print(f"\nDifference Statistics:")
    print(f"  Mean difference: {np.nanmean(diff):.4f}")
    print(f"  Max difference: {np.nanmax(diff):.4f}")
    print(f"  Min difference: {np.nanmin(diff):.4f}")
    print(f"  Std of differences: {np.nanstd(diff):.4f}")
    print(f"  Mean absolute difference: {np.nanmean(np.abs(diff)):.4f}")


def test_real_tiff_data(lr_dem_path, dam_path, output_path=None):
    """
    测试真实TIFF数据

    Args:
        lr_dem_path: LR DEM TIFF文件路径
        dam_path: DAM TIFF文件路径
        output_path: 输出文件路径（可选）
    """
    print("=" * 80)
    print("DEM FUSION TEST WITH REAL TIFF DATA")
    print("=" * 80)

    # 1. 加载数据
    print("\n1. Loading TIFF files...")
    lr_dem_data, lr_metadata = load_tiff_file(lr_dem_path)
    dam_data, dam_metadata = load_tiff_file(dam_path)
    # 在加载 DAM 数据后进行 padding
    dam_data = np.pad(dam_data, pad_width=1, mode='edge')


    # 检查数据尺寸是否匹配
    if lr_dem_data.shape != dam_data.shape:
        raise ValueError(f"Shape mismatch: LR DEM {lr_dem_data.shape}, DAM {dam_data.shape}")

    # 2. 预处理数据
    print("\n2. Preprocessing data...")
    processed = preprocess_data(lr_dem_data, dam_data)

    # 3. 准备融合层
    print("\n3. Setting up DEMFusionLayer...")
    fusion_layer = DEMFusionLayer(
        window_size=33,
        alpha_base=20,  # 与测试代码中的参数一致
        beta=1.2,
        min_delta=0.01,
        max_delta=1,
        dam_significant_threshold=0.5,
        normalize_input=False,  # 输入已经是Z-score归一化的
        return_normalized=True,  # 输出也是Z-score归一化的
        downscale_factor=30  # 实际下采样倍率
    )

    # 设置归一化统计量
    fusion_layer.set_normalization_stats(
        torch.tensor(processed['lr_mean']),
        torch.tensor(processed['lr_std'])
    )

    # 4. 转换为PyTorch张量
    print("\n4. Converting to PyTorch tensors...")
    lr_tensor = torch.FloatTensor(processed['lr_dem_normalized']).unsqueeze(0).unsqueeze(0)
    dam_tensor = torch.FloatTensor(processed['dam_original']).unsqueeze(0).unsqueeze(0)

    print(f"Input tensor shapes:")
    print(f"  LR DEM: {lr_tensor.shape}")
    print(f"  DAM: {dam_tensor.shape}")

    # 5. 执行融合
    print("\n5. Performing fusion...")
    fused_normalized = fusion_layer(lr_tensor, dam_tensor)

    # 6. 反归一化
    print("\n6. Denormalizing fused DEM...")
    fused_dem = fused_normalized * processed['lr_std'] + processed['lr_mean']
    fused_dem_np = fused_dem.squeeze().detach().numpy()

    # 恢复NaN值（如果原始数据有NaN）
    if np.any(processed['lr_nan_mask']):
        fused_dem_np[processed['lr_nan_mask']] = np.nan
        print(f"Restored NaN values in fused DEM")

    # 7. 可视化结果
    print("\n7. Generating visualizations...")
    # 创建有效值掩码用于可视化
    valid_mask = ~np.isnan(fused_dem_np)
    if np.any(~valid_mask):
        print(f"Fused DEM has {np.sum(~valid_mask)} NaN values")

    # 使用有效值进行可视化
    lr_dem_valid = processed['lr_dem_original'].copy()
    dam_valid = processed['dam_original'].copy()
    fused_dem_valid = fused_dem_np.copy()

    # 如果有NaN值，用均值填充以便可视化
    if np.any(~valid_mask):
        lr_dem_valid[~valid_mask] = processed['lr_mean']
        dam_valid[~valid_mask] = np.nanmedian(dam_valid)
        fused_dem_valid[~valid_mask] = processed['lr_mean']

    visualize_results(lr_dem_valid, dam_valid, fused_dem_valid)

    # 8. 保存结果
    if output_path:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存融合后的DEM
        save_tiff_file(fused_dem_np, lr_metadata, output_path)

    # 9. 获取融合层统计信息
    print("\n8. Fusion layer statistics...")
    intermediate = fusion_layer.get_intermediate_results()

    if 'delta' in intermediate:
        delta_np = intermediate['delta'].squeeze().detach().numpy()
        delta_valid = delta_np[valid_mask] if np.any(~valid_mask) else delta_np

        print(f"Adjustment Statistics (valid pixels only):")
        print(f"  Mean |Δ|: {np.mean(np.abs(delta_valid)):.4f}")
        print(f"  Max |Δ|: {np.max(np.abs(delta_valid)):.4f}")
        print(f"  Std Δ: {np.std(delta_valid):.4f}")

    # 获取超参数
    hyperparams = fusion_layer.get_hyperparameters()
    print("\nFusion Layer Hyperparameters:")
    for key, value in hyperparams.items():
        if isinstance(value, (int, float, bool, str)):
            print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return {
        'lr_dem': processed['lr_dem_original'],
        'dam': processed['dam_original'],
        'fused_dem': fused_dem_np,
        'metadata': lr_metadata,
        'fusion_layer': fusion_layer
    }


def main():
    """主函数"""
    # 文件路径
    lr_dem_path = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Test\Copernicus_1.0m_1024pixel\375876_1103749_Google.tif"
    dam_path = r"C:\Users\Kevin\Desktop\20260205_SR_v1.2\375876_1103749_Google_DAM_original.tif"

    # 输出路径（可选）
    output_dir = r"C:\Users\Kevin\Desktop\DEM_Fusion_Output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "375876_1103749_Google_Fused.tif")

    # 执行测试
    try:
        results = test_real_tiff_data(
            lr_dem_path=lr_dem_path,
            dam_path=dam_path,
            output_path=output_path,
        )

        # 保存中间结果（可选）
        print(f"\nResults saved to: {output_path}")
        print(f"\nYou can open the fused DEM in GIS software like QGIS or ArcGIS.")

    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    main()