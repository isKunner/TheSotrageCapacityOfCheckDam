#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: visualization
# @Time    : 2026/2/4 20:01
# @Author  : Kevin
# @Describe:

"""
可视化工具

用于可视化融合过程，帮助理解模型行为
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def visualize_fusion_process(copernicus_dem, relative_map, prototype_activations,
                             detail_strength_map, fusion_info, save_path=None):
    """
    可视化融合过程

    Args:
        copernicus_dem: (B, 1, H, W) 或 (H, W)
        relative_map: (B, 1, H, W) 或 (H, W)
        prototype_activations: (B, P, H, W) 或 (P, H, W)
        detail_strength_map: (B, 1, H, W) 或 (H, W)
        fusion_info: dict 包含融合过程的中间结果
        save_path: 保存路径（可选）
    """
    # 处理batch维度
    if isinstance(copernicus_dem, torch.Tensor):
        copernicus_dem = copernicus_dem.detach().cpu().numpy()
    if isinstance(relative_map, torch.Tensor):
        relative_map = relative_map.detach().cpu().numpy()
    if isinstance(prototype_activations, torch.Tensor):
        prototype_activations = prototype_activations.detach().cpu().numpy()
    if isinstance(detail_strength_map, torch.Tensor):
        detail_strength_map = detail_strength_map.detach().cpu().numpy()

    # 取第一个batch
    if copernicus_dem.ndim == 4:
        copernicus_dem = copernicus_dem[0, 0]
    elif copernicus_dem.ndim == 3:
        copernicus_dem = copernicus_dem[0]

    if relative_map.ndim == 4:
        relative_map = relative_map[0, 0]
    elif relative_map.ndim == 3:
        relative_map = relative_map[0]

    if prototype_activations.ndim == 4:
        prototype_activations = prototype_activations[0]

    if detail_strength_map.ndim == 4:
        detail_strength_map = detail_strength_map[0, 0]
    elif detail_strength_map.ndim == 3:
        detail_strength_map = detail_strength_map[0]

    H, W = copernicus_dem.shape
    num_prototypes = prototype_activations.shape[0]

    # 创建子图
    num_rows = 4
    num_cols = max(4, num_prototypes // 4 + 1)

    fig = plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    # 第1行：输入
    ax1 = plt.subplot(num_rows, num_cols, 1)
    im1 = ax1.imshow(copernicus_dem, cmap='terrain')
    ax1.set_title('Copernicus DEM (LR)')
    plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(num_rows, num_cols, 2)
    im2 = ax2.imshow(relative_map, cmap='terrain')
    ax2.set_title('DAM Relative Map')
    plt.colorbar(im2, ax=ax2)

    # 第2行：融合过程
    if 'relative_aligned' in fusion_info:
        relative_aligned = fusion_info['relative_aligned']
        if isinstance(relative_aligned, torch.Tensor):
            relative_aligned = relative_aligned.detach().cpu().numpy()
        if relative_aligned.ndim == 4:
            relative_aligned = relative_aligned[0, 0]
        elif relative_aligned.ndim == 3:
            relative_aligned = relative_aligned[0]

        ax3 = plt.subplot(num_rows, num_cols, num_cols + 1)
        im3 = ax3.imshow(relative_aligned, cmap='terrain')
        ax3.set_title('Relative Aligned (Instance-wise)')
        plt.colorbar(im3, ax=ax3)

    if 'residual' in fusion_info:
        residual = fusion_info['residual']
        if isinstance(residual, torch.Tensor):
            residual = residual.detach().cpu().numpy()
        if residual.ndim == 4:
            residual = residual[0, 0]
        elif residual.ndim == 3:
            residual = residual[0]

        ax4 = plt.subplot(num_rows, num_cols, num_cols + 2)
        im4 = ax4.imshow(residual, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title('Residual (Relative - Copernicus)')
        plt.colorbar(im4, ax=ax4)

    ax5 = plt.subplot(num_rows, num_cols, num_cols + 3)
    im5 = ax5.imshow(detail_strength_map, cmap='hot', vmin=0, vmax=1)
    ax5.set_title('Detail Strength Map')
    plt.colorbar(im5, ax=ax5)

    if 'mixed' in fusion_info:
        mixed = fusion_info['mixed']
        if isinstance(mixed, torch.Tensor):
            mixed = mixed.detach().cpu().numpy()
        if mixed.ndim == 4:
            mixed = mixed[0, 0]
        elif mixed.ndim == 3:
            mixed = mixed[0]

        ax6 = plt.subplot(num_rows, num_cols, num_cols + 4)
        im6 = ax6.imshow(mixed, cmap='terrain')
        ax6.set_title('Mixed (Copernicus + alpha * Residual)')
        plt.colorbar(im6, ax=ax6)

    # 第3行：原型激活图（显示前几个主要原型）
    # 找出激活最强的几个原型
    proto_sums = prototype_activations.sum(axis=(1, 2))
    top_proto_indices = np.argsort(proto_sums)[-min(4, num_prototypes):][::-1]

    for i, proto_idx in enumerate(top_proto_indices):
        ax = plt.subplot(num_rows, num_cols, 2 * num_cols + i + 1)
        im = ax.imshow(prototype_activations[proto_idx], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'Prototype {proto_idx} Activation')
        plt.colorbar(im, ax=ax)

    # 第4行：最终输出
    if 'mixed' in fusion_info:
        ax_final1 = plt.subplot(num_rows, num_cols, 3 * num_cols + 1)
        im_final1 = ax_final1.imshow(mixed, cmap='terrain')
        ax_final1.set_title('Mixed (Before Refinement)')
        plt.colorbar(im_final1, ax=ax_final1)

    if 'delta' in fusion_info:
        delta = fusion_info['delta']
        if isinstance(delta, torch.Tensor):
            delta = delta.detach().cpu().numpy()
        if delta.ndim == 4:
            delta = delta[0, 0]
        elif delta.ndim == 3:
            delta = delta[0]

        ax_final2 = plt.subplot(num_rows, num_cols, 3 * num_cols + 2)
        im_final2 = ax_final2.imshow(delta, cmap='RdBu_r')
        ax_final2.set_title('Delta (Network Refinement)')
        plt.colorbar(im_final2, ax=ax_final2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")

    return fig


def visualize_bias_map_decomposition(instance_bias_map, base_bias, spatial_residual,
                                     prototype_activations, save_path=None):
    """
    可视化BiasMap的分解（类别偏置 + 空间残差）

    Args:
        instance_bias_map: (B, 1, H, W) 或 (H, W) - 总偏置
        base_bias: (B, 1, H, W) 或 (H, W) - 类别级偏置
        spatial_residual: (B, 1, H, W) 或 (H, W) - 空间残差
        prototype_activations: (B, P, H, W) 或 (P, H, W) - 原型激活
        save_path: 保存路径（可选）
    """

    # 处理batch维度
    def process_tensor(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        if t.ndim == 4:
            t = t[0, 0]
        elif t.ndim == 3:
            t = t[0]
        return t

    instance_bias_map = process_tensor(instance_bias_map)
    base_bias = process_tensor(base_bias)
    spatial_residual = process_tensor(spatial_residual)

    if isinstance(prototype_activations, torch.Tensor):
        prototype_activations = prototype_activations.detach().cpu().numpy()
    if prototype_activations.ndim == 4:
        prototype_activations = prototype_activations[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 总偏置
    ax1 = axes[0, 0]
    im1 = ax1.imshow(instance_bias_map, cmap='RdBu_r')
    ax1.set_title('Total Bias Map')
    plt.colorbar(im1, ax=ax1)

    # 类别级偏置
    ax2 = axes[0, 1]
    im2 = ax2.imshow(base_bias, cmap='RdBu_r')
    ax2.set_title('Base Bias (Category-level)')
    plt.colorbar(im2, ax=ax2)

    # 空间残差
    ax3 = axes[0, 2]
    im3 = ax3.imshow(spatial_residual, cmap='RdBu_r')
    ax3.set_title('Spatial Residual (Instance-level)')
    plt.colorbar(im3, ax=ax3)

    # 残差占比
    residual_ratio = np.abs(spatial_residual) / (np.abs(instance_bias_map) + 1e-6)
    ax4 = axes[1, 0]
    im4 = ax4.imshow(residual_ratio, cmap='hot', vmin=0, vmax=1)
    ax4.set_title('Residual Ratio (|residual| / |total|)')
    plt.colorbar(im4, ax=ax4)

    # 主要原型
    num_prototypes = prototype_activations.shape[0]
    proto_sums = prototype_activations.sum(axis=(1, 2))
    top_proto_idx = np.argmax(proto_sums)

    ax5 = axes[1, 1]
    im5 = ax5.imshow(prototype_activations[top_proto_idx], cmap='hot')
    ax5.set_title(f'Main Prototype {top_proto_idx} Activation')
    plt.colorbar(im5, ax=ax5)

    # 直方图：偏置分布
    ax6 = axes[1, 2]
    ax6.hist(instance_bias_map.flatten(), bins=50, alpha=0.5, label='Total')
    ax6.hist(base_bias.flatten(), bins=50, alpha=0.5, label='Base')
    ax6.hist(spatial_residual.flatten(), bins=50, alpha=0.5, label='Residual')
    ax6.set_title('Bias Distribution')
    ax6.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"BiasMap分解可视化已保存到: {save_path}")

    return fig


def visualize_detail_strength_by_prototype(prototype_detail_strength, prototype_activations,
                                           num_display=8, save_path=None):
    """
    可视化每个原型的细节强度

    Args:
        prototype_detail_strength: (num_prototypes,) - 每个原型的细节强度
        prototype_activations: (B, P, H, W) 或 (P, H, W) - 原型激活
        num_display: 显示的原型数量
        save_path: 保存路径（可选）
    """
    if isinstance(prototype_detail_strength, torch.Tensor):
        prototype_detail_strength = prototype_detail_strength.detach().cpu().numpy()
    if isinstance(prototype_activations, torch.Tensor):
        prototype_activations = prototype_activations.detach().cpu().numpy()

    if prototype_activations.ndim == 4:
        prototype_activations = prototype_activations[0]

    num_prototypes = len(prototype_detail_strength)

    # 按激活强度排序
    proto_sums = prototype_activations.sum(axis=(1, 2))
    sorted_indices = np.argsort(proto_sums)[::-1][:num_display]

    fig, axes = plt.subplots(2, num_display // 2, figsize=(3 * num_display // 2, 6))
    axes = axes.flatten()

    for i, proto_idx in enumerate(sorted_indices):
        ax = axes[i]
        im = ax.imshow(prototype_activations[proto_idx], cmap='hot')
        strength = prototype_detail_strength[proto_idx]
        ax.set_title(f'Proto {proto_idx}\nStrength: {strength:.3f}')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle('Prototype Detail Strengths', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"细节强度可视化已保存到: {save_path}")

    return fig


def compare_hr_lr(hrdem, mapped_lrdem, usgs_gt=None, copernicus=None, save_path=None):
    """
    对比HR、LR和真值

    Args:
        hrdem: (H, W) - 高分辨率DEM
        mapped_lrdem: (H', W') - 映射后的低分辨率DEM
        usgs_gt: (H, W) - USGS真值（可选）
        copernicus: (H, W) - Copernicus DEM（可选）
        save_path: 保存路径（可选）
    """

    def process(t):
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        if t.ndim == 4:
            t = t[0, 0]
        elif t.ndim == 3:
            t = t[0]
        return t

    hrdem = process(hrdem)
    mapped_lrdem = process(mapped_lrdem)

    num_plots = 2
    if usgs_gt is not None:
        usgs_gt = process(usgs_gt)
        num_plots += 1
    if copernicus is not None:
        copernicus = process(copernicus)
        num_plots += 1

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    idx = 0

    if copernicus is not None:
        im = axes[idx].imshow(copernicus, cmap='terrain')
        axes[idx].set_title('Copernicus DEM (Input)')
        plt.colorbar(im, ax=axes[idx])
        idx += 1

    im = axes[idx].imshow(hrdem, cmap='terrain')
    axes[idx].set_title('HRDEM (Output)')
    plt.colorbar(im, ax=axes[idx])
    idx += 1

    im = axes[idx].imshow(mapped_lrdem, cmap='terrain')
    axes[idx].set_title('Mapped LR (Downsampled)')
    plt.colorbar(im, ax=axes[idx])
    idx += 1

    if usgs_gt is not None:
        im = axes[idx].imshow(usgs_gt, cmap='terrain')
        axes[idx].set_title('USGS GT (Ground Truth)')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")

    return fig