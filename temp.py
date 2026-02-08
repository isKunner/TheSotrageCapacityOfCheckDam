#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: dem_fusion_layer.py
# @Time    : 2026/2/8 19:40
# @Author  : Kevin
# @Describe: DEM-DAM fusion layer for neural networks

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class DEMFusionLayer(nn.Module):
    """
    DEM-DAM Fusion Layer for neural networks

    This layer performs statistical-aware fusion of LR DEM and DAM to produce
    high-resolution DEM. Designed to work with Z-score normalized inputs.

    Parameters:
    -----------
    window_size : int, default=3
        Window size for computing local statistics
    alpha_base : float, default=0.5
        Base adjustment magnitude (normalized space)
    beta : float, default=1.0
        Global scaling factor
    min_delta : float, default=0.05
        Minimum adjustment for significant DAM values (normalized space)
    max_delta : float, default=1.0
        Maximum adjustment magnitude (normalized space)
    dam_significant_threshold : float, default=0.2
        Threshold for considering DAM values as significant (|DAM-0.5|)
    normalize_input : bool, default=False
        Whether to normalize input LR DEM (assume it's already Z-score normalized)
    return_normalized : bool, default=True
        Whether to return normalized output (for further processing)
    downscale_factor : int, default=30
        Downscale factor from HR to LR (for block variance calculation)
    """

    def __init__(self,
                 window_size=3,
                 alpha_base=0.5,
                 beta=1.0,
                 min_delta=0.05,
                 max_delta=1.0,
                 dam_significant_threshold=0.2,
                 normalize_input=False,  # Typically False for Z-score inputs
                 return_normalized=True,  # Typically True for Z-score outputs
                 adjust_size=True,
                 downscale_factor=30):
        super(DEMFusionLayer, self).__init__()

        # Hyperparameters (potentially learnable in future)
        self.window_size = window_size
        self.downscale_factor = downscale_factor
        self.adjust_size = adjust_size

        # Register as parameters (not requiring grad for now, but could be made learnable)
        self.alpha_base = nn.Parameter(torch.tensor(alpha_base, dtype=torch.float32),
                                       requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32),
                                 requires_grad=False)
        self.min_delta = nn.Parameter(torch.tensor(min_delta, dtype=torch.float32),
                                      requires_grad=False)
        self.max_delta = nn.Parameter(torch.tensor(max_delta, dtype=torch.float32),
                                      requires_grad=False)
        self.dam_significant_threshold = dam_significant_threshold

        # Configuration flags
        self.normalize_input = normalize_input
        self.return_normalized = return_normalized

        # Statistics for normalization (will be set externally)
        self.register_buffer('lr_mean', torch.tensor(0.0))
        self.register_buffer('lr_std', torch.tensor(1.0))

        # Placeholder for storing intermediate results
        self._intermediate_results = {}

    def _adjust_to_multiple(self, tensor, factor, mode='pad'):
        """
        调整张量尺寸到factor的倍数

        Args:
            tensor: 输入张量 [B, C, H, W]
            factor: 倍数因子
            mode: 'pad'或'crop'或'resize'

        Returns:
            调整后的张量，以及调整信息用于恢复
        """
        B, C, H, W = tensor.shape

        # 计算需要调整的尺寸
        target_h = int(np.ceil(H / factor)) * factor
        target_w = int(np.ceil(W / factor)) * factor

        if mode == 'pad':
            # 填充到目标尺寸
            pad_h = target_h - H
            pad_w = target_w - W

            if pad_h > 0 or pad_w > 0:
                # 使用反射填充（地形数据适合反射填充）
                tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
                return tensor, {'original_h': H, 'original_w': W, 'pad_h': pad_h, 'pad_w': pad_w}

        elif mode == 'crop':
            # 裁剪到目标尺寸
            crop_h = H - target_h
            crop_w = W - target_w

            if crop_h > 0 or crop_w > 0:
                tensor = tensor[:, :, :target_h, :target_w]
                return tensor, {'original_h': H, 'original_w': W, 'crop_h': crop_h, 'crop_w': crop_w}

        elif mode == 'resize':
            # 调整大小（使用双线性插值）
            if H != target_h or W != target_w:
                tensor = F.interpolate(tensor, size=(target_h, target_w),
                                       mode='bilinear', align_corners=False)
                return tensor, {'original_h': H, 'original_w': W, 'new_h': target_h, 'new_w': target_w}

        return tensor, {'original_h': H, 'original_w': W}

    def _restore_original_size(self, tensor, adjust_info, mode='pad'):
        """
        恢复原始尺寸

        Args:
            tensor: 调整后的张量
            adjust_info: 调整信息
            mode: 调整模式
        """
        B, C, H, W = tensor.shape

        if mode == 'pad':
            # 移除填充
            if 'pad_h' in adjust_info and adjust_info['pad_h'] > 0:
                tensor = tensor[:, :, :adjust_info['original_h'], :]
            if 'pad_w' in adjust_info and adjust_info['pad_w'] > 0:
                tensor = tensor[:, :, :, :adjust_info['original_w']]

        elif mode == 'crop':
            # 恢复裁剪（需要重新插值）
            if 'crop_h' in adjust_info and adjust_info['crop_h'] > 0:
                target_h = adjust_info['original_h']
                tensor = F.interpolate(tensor, size=(target_h, W),
                                       mode='bilinear', align_corners=False)
            if 'crop_w' in adjust_info and adjust_info['crop_w'] > 0:
                _, _, H, _ = tensor.shape
                target_w = adjust_info['original_w']
                tensor = F.interpolate(tensor, size=(H, target_w),
                                       mode='bilinear', align_corners=False)

        elif mode == 'resize':
            # 恢复原始大小
            if 'new_h' in adjust_info:
                target_h, target_w = adjust_info['original_h'], adjust_info['original_w']
                tensor = F.interpolate(tensor, size=(target_h, target_w),
                                       mode='bilinear', align_corners=False)

        return tensor

    def _compute_local_statistics(self, tensor):
        """Compute local mean and standard deviation with reflection padding"""
        # Ensure we have a 4D tensor [batch, channel, height, width]
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(1)  # Add channel dimension

        # Apply reflection padding
        pad = self.window_size // 2
        padded = F.pad(tensor, (pad, pad, pad, pad), mode='reflect')

        # Compute local statistics using average pooling
        local_mean = F.avg_pool2d(padded, kernel_size=self.window_size, stride=1, padding=0)
        local_mean_sq = F.avg_pool2d(padded ** 2, kernel_size=self.window_size, stride=1, padding=0)
        local_var = torch.clamp(local_mean_sq - local_mean ** 2, min=1e-6)
        local_std = torch.sqrt(local_var)

        return local_mean, local_std

    def _compute_fusion_weights(self, dam_tensor, lr_local_std):
        """
        Compute fusion weights based on DAM and local statistics

        Args:
            dam_tensor: DAM values in range [0, 1]
            lr_local_std: Local standard deviation of LR DEM

        Returns:
            Dictionary containing various weights and masks
        """
        # DAM centered to [-0.5, 0.5]
        dam_centered = dam_tensor - 0.5

        # DAM direction with smooth transition (tanh for smooth transition around 0.5)
        dam_direction = torch.tanh(dam_centered * 4.0)  # Smooth transition

        # DAM weight: higher for extreme values (close to 0 or 1)
        dam_weight = 2.0 * torch.abs(dam_centered)  # [0, 1]

        # Local std weight: higher for areas with more variation
        if lr_local_std.max() > lr_local_std.min():
            lr_std_norm = (lr_local_std - lr_local_std.min()) / \
                          (lr_local_std.max() - lr_local_std.min() + 1e-6)
        else:
            lr_std_norm = torch.zeros_like(lr_local_std)

        # Standard deviation weight: [0.3, 1.0]
        std_weight = 0.3 + 0.7 * lr_std_norm

        # Combined weight
        combined_weight = dam_weight * std_weight

        # Significant DAM mask
        dam_significant = torch.abs(dam_centered) > self.dam_significant_threshold

        return {
            'dam_weight': dam_weight,
            'std_weight': std_weight,
            'combined_weight': combined_weight,
            'dam_direction': dam_direction,
            'dam_significant': dam_significant,
            'dam_centered': dam_centered
        }

    def _compute_adjustment(self, weights):
        """
        Compute the adjustment (delta) to apply to LR DEM

        Args:
            weights: Dictionary containing fusion weights

        Returns:
            delta: Adjustment tensor
        """
        # Base delta calculation
        delta_base = self.alpha_base * weights['dam_direction'] * self.beta

        # Apply weight modulation
        delta = delta_base * weights['combined_weight']

        # Apply bounds with smooth constraints
        delta_sign = torch.sign(delta)
        delta_abs = torch.abs(delta)

        # Apply minimum threshold for significant DAM values
        mask = weights['dam_significant'] & (delta_abs < self.min_delta)
        delta_abs = torch.where(mask, self.min_delta, delta_abs)

        # Apply maximum bound with smooth transition (tanh for smooth clamping)
        delta_abs = self.max_delta * torch.tanh(delta_abs / self.max_delta)
        delta = delta_sign * delta_abs

        return delta

    def forward(self, lr_dem, dam, adjust_mode='pad'):
        """
        Forward pass: fuse LR DEM with DAM

        Parameters:
        -----------
        lr_dem : torch.Tensor
            Low-resolution DEM, typically Z-score normalized.
            Shape can be: [B, C, H, W], [B, H, W], or [H, W]

        dam : torch.Tensor
            DAM (Detail Amplification Map), same shape as lr_dem.
            Values should be in range [0, 1]

        Returns:
        --------
        fused_dem : torch.Tensor
            Fused DEM, same shape as input.
            If return_normalized=True, returns Z-score normalized result.
            Otherwise, returns denormalized to original scale.
        """
        # Store original shape and device
        original_shape = lr_dem.shape
        device = lr_dem.device

        # Ensure tensors have at least 3 dimensions [..., H, W]
        if lr_dem.dim() == 2:
            lr_dem = lr_dem.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            dam = dam.unsqueeze(0).unsqueeze(0) if dam.dim() == 2 else dam
        elif lr_dem.dim() == 3:
            # Assume shape is [B, H, W], add channel dimension
            lr_dem = lr_dem.unsqueeze(1)  # [B, 1, H, W]
            if dam.dim() == 3:
                dam = dam.unsqueeze(1)  # [B, 1, H, W]
        elif lr_dem.dim() != 4:
            raise ValueError(f"Unsupported tensor dimension: {lr_dem.dim()}")

        if self.adjust_size and adjust_mode != 'none':
            # 调整LR DEM和DAM的尺寸
            lr_dem, lr_adjust_info = self._adjust_to_multiple(lr_dem, self.downscale_factor, adjust_mode)
            dam, dam_adjust_info = self._adjust_to_multiple(dam, self.downscale_factor, adjust_mode)

        # Ensure DAM has same shape as LR DEM
        if lr_dem.shape != dam.shape:
            # Try to broadcast if possible
            if dam.shape[0] == 1 and dam.shape[1] == 1:
                dam = dam.expand_as(lr_dem)
            else:
                raise ValueError(f"Shape mismatch: lr_dem {lr_dem.shape}, dam {dam.shape}")

        # Step 1: Normalize LR DEM if needed (typically not needed for Z-score inputs)
        if self.normalize_input:
            # Compute mean and std if not already computed
            lr_mean = lr_dem.mean()
            lr_std = lr_dem.std()
            lr_normalized = (lr_dem - lr_mean) / (lr_std + 1e-6)
        else:
            # Assume input is already normalized
            lr_normalized = lr_dem
            lr_mean = self.lr_mean
            lr_std = self.lr_std

        # Step 2: Compute local statistics
        lr_local_mean, lr_local_std = self._compute_local_statistics(lr_normalized)

        # Step 3: Compute fusion weights
        weights = self._compute_fusion_weights(dam, lr_local_std)

        # Step 4: Compute and apply adjustment
        delta = self._compute_adjustment(weights)
        fused_normalized = lr_normalized + delta

        # Step 5: Denormalize if needed
        if not self.return_normalized:
            fused_dem = fused_normalized * lr_std + lr_mean
        else:
            fused_dem = fused_normalized

        # Store intermediate results for analysis
        self._intermediate_results = {
            'lr_normalized': lr_normalized,
            'lr_local_mean': lr_local_mean,
            'lr_local_std': lr_local_std,
            'weights': weights,
            'delta': delta,
            'fused_normalized': fused_normalized,
            'normalization_mean': lr_mean,
            'normalization_std': lr_std
        }

        if self.adjust_size and adjust_mode != 'none':
            # 恢复原始尺寸
            fused_dem = self._restore_original_size(fused_dem, lr_adjust_info, adjust_mode)

        # Restore original shape if needed
        if fused_dem.shape != original_shape:
            # Remove added dimensions
            if len(original_shape) == 2:  # [H, W]
                fused_dem = fused_dem.squeeze(0).squeeze(0)
            elif len(original_shape) == 3:  # [B, H, W]
                fused_dem = fused_dem.squeeze(1)

        return fused_dem

    def get_intermediate_results(self):
        """Get intermediate results for analysis and visualization"""
        return self._intermediate_results

    def set_normalization_stats(self, mean, std):
        """
        Set normalization statistics manually

        Args:
            mean: Mean value for denormalization
            std: Standard deviation for denormalization
        """
        if isinstance(mean, (int, float)):
            mean = torch.tensor(mean)
        if isinstance(std, (int, float)):
            std = torch.tensor(std)

        self.lr_mean = mean.to(self.lr_mean.device)
        self.lr_std = std.to(self.lr_std.device)

    def reset_normalization_stats(self):
        """Reset normalization statistics to default"""
        self.lr_mean = torch.tensor(0.0)
        self.lr_std = torch.tensor(1.0)

    def get_hyperparameters(self):
        """Get current hyperparameter values"""
        return {
            'window_size': self.window_size,
            'alpha_base': self.alpha_base.item(),
            'beta': self.beta.item(),
            'min_delta': self.min_delta.item(),
            'max_delta': self.max_delta.item(),
            'dam_significant_threshold': self.dam_significant_threshold,
            'normalize_input': self.normalize_input,
            'return_normalized': self.return_normalized,
            'downscale_factor': self.downscale_factor
        }

    def make_hyperparameters_learnable(self):
        """Make hyperparameters learnable (for future optimization)"""
        self.alpha_base.requires_grad = True
        self.beta.requires_grad = True
        self.min_delta.requires_grad = True
        self.max_delta.requires_grad = True

    def freeze_hyperparameters(self):
        """Freeze hyperparameters (default state)"""
        self.alpha_base.requires_grad = False
        self.beta.requires_grad = False
        self.min_delta.requires_grad = False
        self.max_delta.requires_grad = False


# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: test_fusion_layer_full.py
# @Time    : 2026/2/8 19:40
# @Author  : Kevin
# @Describe: Full test of DEMFusionLayer with original workflow

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# from dem_fusion_layer import DEMFusionLayer


# 重新实现原始的地形生成和DAM生成函数，保持与原始代码一致
def generate_realistic_terrain_with_features(rows=6, cols=9):
    """Generate realistic terrain with clear features for testing"""
    terrain = np.zeros((rows, cols))

    # Create distinct terrain regions
    # Region 1: Low flat area (valley)
    for i in range(2):
        for j in range(3):
            terrain[i, j] = 5.0 + 0.1 * i + 0.05 * j

    # Region 2: Hill area
    for i in range(2):
        for j in range(3, 6):
            base = 15.0
            # Create a hill shape
            dx = (j - 4.0) / 2.0
            dy = (i - 0.5) / 2.0
            terrain[i, j] = base + 3.0 * np.exp(-0.5 * (dx ** 2 + dy ** 2))

    # Region 3: Mountainous area
    for i in range(2, 4):
        for j in range(3):
            terrain[i, j] = 20.0 + 2.0 * np.sin(0.8 * i) * np.cos(0.6 * j)

    # Region 4: Ridge
    for i in range(2, 4):
        for j in range(3, 6):
            terrain[i, j] = 12.0 + 1.5 * np.abs(j - 4)

    # Region 5: Depression area
    for i in range(2, 4):
        for j in range(6, 9):
            # Create a depression
            dx = (j - 7.0) / 1.5
            dy = (i - 2.5) / 1.5
            terrain[i, j] = 8.0 - 2.0 * np.exp(-0.8 * (dx ** 2 + dy ** 2))

    # Region 6: Flat plateau
    for i in range(4, 6):
        for j in range(3):
            terrain[i, j] = 18.0 + 0.2 * (i - 4)

    # Region 7: Complex terrain
    for i in range(4, 6):
        for j in range(3, 6):
            terrain[i, j] = 14.0 + 1.0 * np.sin(0.7 * i) * np.cos(0.7 * j)

    # Region 8: Gentle slope
    for i in range(4, 6):
        for j in range(6, 9):
            terrain[i, j] = 10.0 + 0.5 * (i - 4) + 0.3 * (j - 6)

    return terrain


def downsample_terrain(terrain, factor=3):
    """Downsample terrain by averaging"""
    h, w = terrain.shape
    new_h, new_w = h // factor, w // factor

    # Simple averaging downsampling
    downsampled = np.zeros((new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            patch = terrain[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor]
            downsampled[i, j] = np.mean(patch)

    return downsampled


def upsample_dem(lr_dem, factor=3):
    """Upsample LR DEM by repeating values"""
    return np.repeat(np.repeat(lr_dem, factor, axis=0), factor, axis=1)


def generate_dam_from_lr(lr_dem_upsampled, noise_level=0.08, bias_level=0.15):
    """
    Generate DAM based on normalized LR DEM with added noise and bias
    Simulating prediction deviations
    """
    # Normalize LR DEM to [0, 1] range
    lr_min = lr_dem_upsampled.min()
    lr_max = lr_dem_upsampled.max()
    if lr_max > lr_min:
        dam_base = (lr_dem_upsampled - lr_min) / (lr_max - lr_min)
    else:
        dam_base = np.zeros_like(lr_dem_upsampled)

    # Add Gaussian noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, noise_level, dam_base.shape)
    dam_noisy = dam_base + noise

    # Add systematic bias in some regions (simulating prediction errors)
    rows, cols = dam_noisy.shape
    dam_with_bias = dam_noisy.copy()

    # Add positive bias (overestimation) in some areas
    for i in range(rows):
        for j in range(cols):
            # Systematic bias based on position
            if (i + j) % 4 == 0:
                dam_with_bias[i, j] += bias_level * 0.5
            elif (i + j) % 3 == 0:
                dam_with_bias[i, j] -= bias_level * 0.3

    # Add specific features (buildings, trees, etc.)
    # Building-like feature (protrusion)
    if rows >= 3 and cols >= 3:
        dam_with_bias[1, 1] = 0.85  # Building
        dam_with_bias[1, 2] = 0.82
        dam_with_bias[2, 1] = 0.82

    # Tree-like features
    dam_with_bias[0, 4] = 0.78
    dam_with_bias[4, 7] = 0.75

    # Ditch/depression
    dam_with_bias[2, 7] = 0.25
    dam_with_bias[2, 8] = 0.28

    # Apply smoothing
    dam_smoothed = gaussian_filter(dam_with_bias, sigma=0.8)

    # Ensure range [0, 1]
    dam_final = np.clip(dam_smoothed, 0.0, 1.0)

    return dam_final, dam_base


def calculate_terrain_metrics(original, fused):
    """Calculate terrain fusion metrics"""
    metrics = {}

    abs_error = np.abs(fused - original)
    metrics['mean_abs_error'] = np.mean(abs_error)
    metrics['max_abs_error'] = np.max(abs_error)
    metrics['std_abs_error'] = np.std(abs_error)

    relative_error = abs_error / (np.abs(original) + 1e-6)
    metrics['mean_relative_error'] = np.mean(relative_error)

    rows, cols = original.shape
    block_var_orig = []
    block_var_fused = []

    for i in range(0, rows, 3):
        for j in range(0, cols, 3):
            block_orig = original[i:i + 3, j:j + 3]
            block_fused = fused[i:i + 3, j:j + 3]

            if block_orig.size > 0 and block_fused.size > 0:
                block_var_orig.append(np.var(block_orig))
                block_var_fused.append(np.var(block_fused))

    metrics['avg_block_var_orig'] = np.mean(block_var_orig) if block_var_orig else 0
    metrics['avg_block_var_fused'] = np.mean(block_var_fused) if block_var_fused else 0

    if metrics['avg_block_var_orig'] > 0:
        detail_increase = metrics['avg_block_var_fused'] / metrics['avg_block_var_orig']
    else:
        detail_increase = float('inf')
    metrics['detail_increase_ratio'] = detail_increase

    # Additional metrics
    metrics['correlation'] = np.corrcoef(original.flatten(), fused.flatten())[0, 1]

    return metrics


def test_fusion_layer_with_original_workflow():
    """Test DEMFusionLayer using original workflow"""
    print("=" * 80)
    print("Testing DEMFusionLayer with Original Workflow")
    print("=" * 80)

    # Step 1: Generate HR terrain
    print("\n1. Generating HR terrain...")
    hr_terrain = generate_realistic_terrain_with_features()
    print(f"HR terrain shape: {hr_terrain.shape}")
    print(f"Elevation range: {hr_terrain.min():.2f} to {hr_terrain.max():.2f} m")

    # Step 2: Downsample to get LR DEM
    print("\n2. Downsampling HR terrain (3x)...")
    lr_dem = downsample_terrain(hr_terrain, factor=3)
    print(f"LR DEM shape: {lr_dem.shape}")

    # Step 3: Upsample LR DEM to original size
    print("\n3. Upsampling LR DEM to original size...")
    lr_dem_upsampled = upsample_dem(lr_dem, factor=3)
    print(f"Upsampled LR DEM shape: {lr_dem_upsampled.shape}")

    # Step 4: Generate DAM from normalized LR DEM
    print("\n4. Generating DAM from normalized LR DEM...")
    dam_output, dam_base = generate_dam_from_lr(lr_dem_upsampled, noise_level=0.08, bias_level=0.15)
    print(f"DAM shape: {dam_output.shape}")
    print(f"DAM range: {dam_output.min():.2f} to {dam_output.max():.2f}")

    # Step 5: Normalize LR DEM for fusion (Z-score normalization)
    print("\n5. Normalizing LR DEM for fusion (Z-score)...")
    lr_mean = np.mean(lr_dem_upsampled)
    lr_std = np.std(lr_dem_upsampled)
    lr_dem_normalized = (lr_dem_upsampled - lr_mean) / lr_std
    print(f"Normalization: mean={lr_mean:.4f}, std={lr_std:.4f}")

    # Step 6: Create fusion layer with parameters matching original
    print("\n6. Creating DEMFusionLayer...")
    fusion_layer = DEMFusionLayer(
        window_size=3,
        alpha_base=0.8,  # Matching original enhanced_fusion_with_adaptive_params
        beta=1.2,
        min_delta=0.08,
        max_delta=1.5,
        dam_significant_threshold=0.2,
        normalize_input=False,  # Input is already normalized
        return_normalized=True,  # Output will be normalized
        downscale_factor=3
    )

    # Manually set normalization stats (for denormalization later)
    fusion_layer.set_normalization_stats(
        torch.tensor(lr_mean),
        torch.tensor(lr_std)
    )

    # Step 7: Convert to tensors
    print("\n7. Converting to PyTorch tensors...")
    lr_tensor = torch.FloatTensor(lr_dem_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    dam_tensor = torch.FloatTensor(dam_output).unsqueeze(0).unsqueeze(0)

    # Step 8: Perform fusion using DEMFusionLayer
    print("\n8. Performing fusion with DEMFusionLayer...")
    fused_normalized = fusion_layer(lr_tensor, dam_tensor)

    # Step 9: Get intermediate results for analysis
    intermediate_results = fusion_layer.get_intermediate_results()
    delta_tensor = intermediate_results['delta']

    # Step 10: Denormalize fused DEM
    print("\n9. Denormalizing fused DEM...")
    fused_elevation = fused_normalized * lr_std + lr_mean
    fused_elevation_np = fused_elevation.squeeze().detach().numpy()

    # Step 11: Calculate metrics
    print("\n10. Calculating fusion metrics...")
    metrics = calculate_terrain_metrics(hr_terrain, fused_elevation_np)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Key points analysis (matching original)
    test_points = [
        (1, 1, "Building (high DAM)"),
        (0, 4, "Tree feature"),
        (2, 7, "Ditch (low DAM)"),
        (4, 7, "Gentle slope"),
        (3, 2, "Mountainous area"),
        (1, 5, "Hill top"),
    ]

    print(f"\n{'Position':<10} {'HR':<8} {'LR':<8} {'DAM':<6} {'Fused':<8} {'Δ':<8} {'|Δ|/HR':<10}")
    print("-" * 70)

    for i, j, description in test_points:
        if i < hr_terrain.shape[0] and j < hr_terrain.shape[1]:
            hr_val = hr_terrain[i, j]
            lr_val = lr_dem_upsampled[i, j]
            dam_val = dam_output[i, j]
            fused_val = fused_elevation_np[i, j]
            delta_val = fused_val - lr_val
            rel_delta = abs(delta_val) / (abs(hr_val) + 1e-6)

            print(f"[{i},{j}]   {hr_val:7.3f} {lr_val:7.3f} {dam_val:6.3f} "
                  f"{fused_val:7.3f} {delta_val:+7.3f} {rel_delta:9.2%}  {description}")

    # Print metrics summary
    print("\n" + "=" * 80)
    print("FUSION METRICS SUMMARY")
    print("=" * 80)
    print(f"Mean Absolute Error: {metrics['mean_abs_error']:.4f} m")
    print(f"Max Absolute Error: {metrics['max_abs_error']:.4f} m")
    print(f"Std of Errors: {metrics['std_abs_error']:.4f} m")
    print(f"Mean Relative Error: {metrics['mean_relative_error']:.2%}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"Block Variance (Original): {metrics['avg_block_var_orig']:.6f}")
    print(f"Block Variance (Fused): {metrics['avg_block_var_fused']:.6f}")
    print(f"Detail Increase Ratio: {metrics['detail_increase_ratio']:.2f}x")

    # Check intermediate results
    print("\n" + "=" * 80)
    print("INTERMEDIATE RESULTS CHECK")
    print("=" * 80)

    # Check weights
    if 'weights' in intermediate_results:
        weights = intermediate_results['weights']
        dam_weight_mean = weights['dam_weight'].mean().item()
        std_weight_mean = weights['std_weight'].mean().item()
        combined_weight_mean = weights['combined_weight'].mean().item()

        print(f"DAM Weight mean: {dam_weight_mean:.4f}")
        print(f"Std Weight mean: {std_weight_mean:.4f}")
        print(f"Combined Weight mean: {combined_weight_mean:.4f}")

    # Check delta statistics
    delta_np = delta_tensor.squeeze().detach().numpy()
    print(f"\nAdjustment Statistics:")
    print(f"  Mean |Δ|: {np.mean(np.abs(delta_np)):.4f}")
    print(f"  Max |Δ|: {np.max(np.abs(delta_np)):.4f}")
    print(f"  Std Δ: {np.std(delta_np):.4f}")

    # Get hyperparameters
    hyperparams = fusion_layer.get_hyperparameters()
    print("\nFusion Layer Hyperparameters:")
    for key, value in hyperparams.items():
        if isinstance(value, (int, float, bool)):
            print(f"  {key}: {value}")

    return {
        'hr_terrain': hr_terrain,
        'lr_dem': lr_dem_upsampled,
        'dam_output': dam_output,
        'fused_dem': fused_elevation_np,
        'delta': delta_tensor,
        'intermediate': intermediate_results,
        'metrics': metrics,
        'fusion_layer': fusion_layer
    }


def visualize_results(results):
    """Visualize results similar to original code"""
    hr_terrain = results['hr_terrain']
    lr_dem = results['lr_dem']
    dam_output = results['dam_output']
    fused_dem = results['fused_dem']
    delta_tensor = results['delta']
    intermediate = results['intermediate']
    metrics = results['metrics']

    fig = plt.figure(figsize=(20, 12))

    # 1. Original HR Terrain
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.imshow(hr_terrain, cmap='terrain', aspect='auto')
    ax1.set_title('(a) Original HR Terrain')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 2. LR DEM (Upsampled)
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.imshow(lr_dem, cmap='terrain', aspect='auto')
    ax2.set_title('(b) LR DEM (Upsampled)')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. DAM Output
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.imshow(dam_output, cmap='viridis', vmin=0, vmax=1, aspect='auto')
    ax3.set_title('(c) DAM Output (0-1)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Fused DEM
    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.imshow(fused_dem, cmap='terrain', aspect='auto')
    ax4.set_title('(d) Fused DEM')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    # 5. Adjustment Delta
    ax5 = plt.subplot(3, 4, 5)
    delta_np = delta_tensor.squeeze().detach().numpy()
    vmax = max(abs(delta_np.min()), abs(delta_np.max()))
    im5 = ax5.imshow(delta_np, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    ax5.set_title('(e) Adjustment Δ')
    ax5.set_xlabel('Column')
    ax5.set_ylabel('Row')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    # 6. Detail Changes
    ax6 = plt.subplot(3, 4, 6)
    detail_change = fused_dem - lr_dem
    vmax = max(abs(detail_change.min()), abs(detail_change.max()))
    im6 = ax6.imshow(detail_change, cmap='coolwarm', vmin=-vmax, vmax=vmax, aspect='auto')
    ax6.set_title('(f) Detail Changes (Fused - LR)')
    ax6.set_xlabel('Column')
    ax6.set_ylabel('Row')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

    # 7. DAM Weight
    ax7 = plt.subplot(3, 4, 7)
    if 'weights' in intermediate:
        dam_weight_np = intermediate['weights']['dam_weight'].squeeze().detach().numpy()
        im7 = ax7.imshow(dam_weight_np, cmap='hot', vmin=0, vmax=1, aspect='auto')
        ax7.set_title('(g) DAM Weight')
        ax7.set_xlabel('Column')
        ax7.set_ylabel('Row')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    # 8. Std Weight
    ax8 = plt.subplot(3, 4, 8)
    if 'weights' in intermediate:
        std_weight_np = intermediate['weights']['std_weight'].squeeze().detach().numpy()
        im8 = ax8.imshow(std_weight_np, cmap='hot', vmin=0, vmax=1, aspect='auto')
        ax8.set_title('(h) Std Weight')
        ax8.set_xlabel('Column')
        ax8.set_ylabel('Row')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)

    # 9. Combined Weight
    ax9 = plt.subplot(3, 4, 9)
    if 'weights' in intermediate:
        combined_weight_np = intermediate['weights']['combined_weight'].squeeze().detach().numpy()
        im9 = ax9.imshow(combined_weight_np, cmap='hot', vmin=0, vmax=1, aspect='auto')
        ax9.set_title('(i) Combined Weight')
        ax9.set_xlabel('Column')
        ax9.set_ylabel('Row')
        plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.04)

    # 10. Profile Comparison
    ax10 = plt.subplot(3, 4, 10)
    row_to_plot = 2
    if row_to_plot < lr_dem.shape[0]:
        x = np.arange(lr_dem.shape[1])
        lr_profile = lr_dem[row_to_plot, :]
        fused_profile = fused_dem[row_to_plot, :]
        dam_profile = dam_output[row_to_plot, :]

        ax10.plot(x, lr_profile, 'b-', linewidth=2, alpha=0.7, label='LR DEM')
        ax10.plot(x, fused_profile, 'r-', linewidth=2, label='Fused DEM')

        # Highlight significant DAM points
        for col in range(len(x)):
            if dam_profile[col] < 0.3:  # Depression
                ax10.plot(col, fused_profile[col], 'v', color='red',
                          markersize=8, markeredgecolor='black')
            elif dam_profile[col] > 0.7:  # Protrusion
                ax10.plot(col, fused_profile[col], '^', color='red',
                          markersize=8, markeredgecolor='black')

        ax10.set_title('(j) Profile Comparison (Row 2)')
        ax10.set_xlabel('Column')
        ax10.set_ylabel('Elevation (m)')
        ax10.legend(loc='best')
        ax10.grid(True, alpha=0.3)

    # 11. DAM vs Adjustment Relationship
    ax11 = plt.subplot(3, 4, 11)
    dam_flat = dam_output.flatten()
    delta_flat = delta_np.flatten()

    scatter = ax11.scatter(dam_flat, delta_flat, c=np.abs(delta_flat),
                           cmap='viridis', alpha=0.7, s=30, edgecolor='black')

    # Add trend line
    if len(dam_flat) > 1:
        z = np.polyfit(dam_flat, delta_flat, 1)
        p = np.poly1d(z)
        dam_sorted = np.sort(dam_flat)
        ax11.plot(dam_sorted, p(dam_sorted), 'r-', linewidth=2, alpha=0.7, label='Trend')

    ax11.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='DAM=0.5')
    ax11.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax11.set_title('(k) DAM vs Adjustment')
    ax11.set_xlabel('DAM Value')
    ax11.set_ylabel('Adjustment Δ')
    ax11.grid(True, alpha=0.3)
    ax11.legend(loc='best')

    # 12. Metrics Display
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')

    metrics_text = f"""Fusion Quality Metrics:
Mean Absolute Error: {metrics['mean_abs_error']:.4f} m
Max Absolute Error: {metrics['max_abs_error']:.4f} m
Std of Errors: {metrics['std_abs_error']:.4f} m
Mean Relative Error: {metrics['mean_relative_error']:.2%}
Correlation: {metrics['correlation']:.4f}

Block Variance:
Original: {metrics['avg_block_var_orig']:.6f}
Fused: {metrics['avg_block_var_fused']:.6f}
Detail Increase: {metrics['detail_increase_ratio']:.2f}x

DEMFusionLayer Configuration:
- window_size: 3
- alpha_base: {results['fusion_layer'].alpha_base.item():.2f}
- beta: {results['fusion_layer'].beta.item():.2f}
- normalize_input: False
- return_normalized: True"""

    ax12.text(0.05, 0.95, metrics_text, transform=ax12.transAxes,
              fontsize=8, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('DEMFusionLayer Test Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def test_different_downsample_factors():
    """Test with different downsampling factors"""
    print("\n" + "=" * 80)
    print("Testing Different Downsample Factors")
    print("=" * 80)

    factors = [2, 3, 4]

    for factor in factors:
        print(f"\nTesting with downsampling factor = {factor}")

        # Generate HR terrain
        hr_terrain = generate_realistic_terrain_with_features(rows=12, cols=18)  # Larger terrain

        # Downsample and upsample
        lr_dem = downsample_terrain(hr_terrain, factor=factor)
        lr_dem_upsampled = upsample_dem(lr_dem, factor=factor)

        # Generate DAM
        dam_output, _ = generate_dam_from_lr(lr_dem_upsampled)

        # Normalize
        lr_mean = np.mean(lr_dem_upsampled)
        lr_std = np.std(lr_dem_upsampled)
        lr_dem_normalized = (lr_dem_upsampled - lr_mean) / lr_std

        # Create fusion layer
        fusion_layer = DEMFusionLayer(
            normalize_input=False,
            return_normalized=True,
            downscale_factor=factor
        )
        fusion_layer.set_normalization_stats(
            torch.tensor(lr_mean),
            torch.tensor(lr_std)
        )

        # Convert to tensors
        lr_tensor = torch.FloatTensor(lr_dem_normalized).unsqueeze(0).unsqueeze(0)
        dam_tensor = torch.FloatTensor(dam_output).unsqueeze(0).unsqueeze(0)

        # Fusion
        print(f"factor: {factor}, and lr_tensor.shape: {lr_tensor.shape}, dam_tensor: {dam_tensor.shape}")
        fused_normalized = fusion_layer(lr_tensor, dam_tensor)
        print(f"fused_normalized: {fused_normalized.shape}")
        fused_elevation = fused_normalized * lr_std + lr_mean
        print(f"fused_elevation: {fused_elevation.shape}")
        fused_elevation_np = fused_elevation.squeeze().detach().numpy()

        # Calculate metrics
        metrics = calculate_terrain_metrics(hr_terrain, fused_elevation_np)

        print(f"  Shape: HR={hr_terrain.shape}, LR={lr_dem.shape}")
        print(f"  MAE: {metrics['mean_abs_error']:.4f} m")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Detail Increase: {metrics['detail_increase_ratio']:.2f}x")


def main():
    """Main test function"""
    print("=" * 80)
    print("DEM Fusion Layer - Full Test Suite")
    print("=" * 80)

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Test with original workflow
    results = test_fusion_layer_with_original_workflow()

    # Visualize results
    print("\n" + "=" * 80)
    print("Generating Visualizations...")
    print("=" * 80)
    visualize_results(results)

    # Test with different downsampling factors
    test_different_downsample_factors()

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()