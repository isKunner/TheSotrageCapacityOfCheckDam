"""
损失函数模块（精简版，适配小batch训练）

保留的损失（适配小batch）：
1. RMSE（HRDEM / 映射）- 权重 = 1.0 - 最直接的深度数值拟合
2. 拉普拉斯高频损失 - 权重 = 0.3 - 捕捉山脊/山谷等高频地形结构
3. 梯度损失（Sobel）- 权重 = 0.2 - 辅助保留边缘
4. DAM Enhanced Loss - 权重 = 0.5 - 仅监督高频细节

移除的损失（小batch下不稳定或冗余）：
- SSIM / 多尺度 / 一致性损失
- 原型多样性 / 主导权 / 高斯正则
- 实例正则 / 激活熵
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RMSELoss(nn.Module):
    """
    RMSE损失（AMP友好的实现）
    
    使用torch.sqrt在FP16中可能导致数值不稳定，
    这里使用更稳定的实现方式
    """
    
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        # 计算MSE（在FP16中稳定）
        mse = torch.mean((pred - target) ** 2)
        # 添加eps防止sqrt(0)
        mse = mse + self.eps
        # 使用sqrt，但如果值太大可能溢出，所以先clamp
        # FP16最大约65504，sqrt(65504) ≈ 255.9，所以MSE需要小于65504
        if mse.dtype == torch.float16:
            mse = torch.clamp(mse, max=60000.0)
        return torch.sqrt(mse)


class GradientLoss(nn.Module):
    """
    梯度一致性损失（保留边缘）
    
    使用Sobel算子计算梯度，鼓励预测结果保留与目标相似的边缘结构
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        grad_loss = torch.mean(torch.abs(pred_grad_x - target_grad_x) + 
                               torch.abs(pred_grad_y - target_grad_y))
        return grad_loss


class LaplacianHFLoss(nn.Module):
    """
    拉普拉斯高频损失（保留地形结构线）

    使用拉普拉斯算子提取二阶梯度（曲率），强制DAM保留
    与USGS DEM一致的地形结构线（山脊、山谷、陡坎）
    """

    def __init__(self):
        super().__init__()
        # 拉普拉斯核（二阶梯度）
        self.register_buffer('laplacian_kernel', torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, pred, target):
        # 提取高频层（曲率）
        pred_hf = F.conv2d(pred, self.laplacian_kernel.to(pred.device), padding=1)
        target_hf = F.conv2d(target, self.laplacian_kernel.to(target.device), padding=1)

        # L1损失强制高频系数匹配
        return torch.mean(torch.abs(pred_hf - target_hf))


class DAMEnhancedHFLoss(nn.Module):
    """
    DAM增强深度图高频损失（改造版）

    不再监督整体数值，仅监督DAM输出的高频细节
    使用拉普拉斯算子提取高频成分进行监督
    """

    def __init__(self):
        super().__init__()
        # 拉普拉斯核（二阶梯度）
        self.register_buffer('laplacian_kernel', torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, enhanced_depth, target_depth):
        """
        Args:
            enhanced_depth: DAM增强后的深度图 (B, H, W) 或 (B, 1, H, W)
            target_depth: 目标深度图（如USGS DEM） (B, H, W) 或 (B, 1, H, W)

        Returns:
            loss: 高频L1损失
        """
        # 确保维度一致
        if enhanced_depth.dim() == 3:
            enhanced_depth = enhanced_depth.unsqueeze(1)
        if target_depth.dim() == 3:
            target_depth = target_depth.unsqueeze(1)

        # 提取高频成分（曲率/细节）
        enhanced_hf = F.conv2d(enhanced_depth, self.laplacian_kernel.to(enhanced_depth.device), padding=1)
        target_hf = F.conv2d(target_depth, self.laplacian_kernel.to(target_depth.device), padding=1)

        # 仅监督高频细节匹配
        return torch.mean(torch.abs(enhanced_hf - target_hf))


class CombinedLoss(nn.Module):
    """
    精简版组合损失函数（适配小batch训练）

    保留的损失：
    1. HRDEM与USGS DEM的RMSE损失（主要）- 权重 = 1.0
    2. Mapped LRDEM与Copernicus DEM的RMSE损失（辅助）- 权重 = 1.0
    3. 拉普拉斯高频损失 - 权重 = 0.3
    4. 梯度一致性损失（Sobel）- 权重 = 0.2
    5. DAM Enhanced 高频损失 - 权重 = 0.5（仅监督高频细节）

    分阶段训练策略：
    - 阶段1（pretrain_dam）：主要优化dam_enhanced_hf_loss
    - 阶段2（joint）：联合优化所有损失
    - 阶段3（finetune）：主要优化hrdem_loss
    """

    def __init__(
            self,
            hrdem_weight=1.0,
            mapping_weight=1.0,
            dam_enhanced_weight=0.5,
            grad_weight=0.2,
            laplacian_weight=0.3,
            training_stage='joint',
    ):
        super().__init__()

        self.hrdem_weight = hrdem_weight
        self.mapping_weight = mapping_weight
        self.dam_enhanced_weight = dam_enhanced_weight
        self.grad_weight = grad_weight
        self.laplacian_weight = laplacian_weight
        self.training_stage = training_stage

        # 核心损失函数
        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 高频损失函数
        self.laplacian_loss = LaplacianHFLoss()
        self.dam_enhanced_hf_loss = DAMEnhancedHFLoss()
        
        # 梯度损失（可选）
        self.grad_loss = GradientLoss() if grad_weight > 0 else None

    def forward(
            self,
            hrdem, usgs_dem,
            mapped_lrdem, copernicus_dem,
            dam_enhanced_depth=None,
            **kwargs  # 保留，用于兼容性
    ):
        """
        计算组合损失

        Args:
            hrdem: 预测的高分辨率DEM (B, 1, H, W)
            usgs_dem: USGS DEM真值 (B, 1, H, W)
            mapped_lrdem: 映射后的低分辨率DEM (B, 1, H', W')
            copernicus_dem: Copernicus DEM (B, 1, H', W')
            dam_enhanced_depth: DAM增强深度图 (B, H, W) 或 (B, 1, H, W)，可选

        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. HRDEM损失（主要损失）- 权重 = 1.0
        if self.hrdem_weight > 0:
            hrdem_loss = self.rmse_loss(hrdem, usgs_dem)
            total_loss += self.hrdem_weight * hrdem_loss
            loss_dict['hrdem'] = hrdem_loss.item()

        # 2. 映射损失 - 权重 = 1.0
        if self.mapping_weight > 0:
            mapping_loss = self.rmse_loss(mapped_lrdem, copernicus_dem)
            total_loss += self.mapping_weight * mapping_loss
            loss_dict['mapping'] = mapping_loss.item()

        # 3. 拉普拉斯高频损失 - 权重 = 0.3
        if self.laplacian_weight > 0:
            # HRDEM的高频损失
            hrdem_lap_loss = self.laplacian_loss(hrdem, usgs_dem)
            total_loss += self.laplacian_weight * hrdem_lap_loss
            loss_dict['laplacian'] = hrdem_lap_loss.item()

        # 4. 梯度一致性损失（Sobel）- 权重 = 0.2
        if self.grad_weight > 0 and self.grad_loss is not None:
            grad_loss = self.grad_loss(hrdem, usgs_dem)
            total_loss += self.grad_weight * grad_loss
            loss_dict['grad'] = grad_loss.item()

        # 5. DAM Enhanced 高频损失（改造版）- 权重 = 0.5
        # 不再监督整体数值，仅监督高频细节
        if self.dam_enhanced_weight > 0 and dam_enhanced_depth is not None:
            dam_hf_loss = self.dam_enhanced_hf_loss(dam_enhanced_depth, usgs_dem)
            total_loss += self.dam_enhanced_weight * dam_hf_loss
            loss_dict['dam_hf'] = dam_hf_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def set_stage(self, stage):
        """
        设置训练阶段

        Args:
            stage: 'pretrain_dam', 'joint', 'finetune'
        """
        self.training_stage = stage

        if stage == 'pretrain_dam':
            # 阶段1：重点优化DAM高频细节
            self.dam_enhanced_weight = 1.0
            self.hrdem_weight = 0.1
            self.mapping_weight = 0.0
            self.laplacian_weight = 0.5
            self.grad_weight = 0.0
            print("切换到阶段1：DAM预训练（重点优化高频细节）")
        elif stage == 'joint':
            # 阶段2：联合训练
            self.dam_enhanced_weight = 0.5
            self.hrdem_weight = 1.0
            self.mapping_weight = 1.0
            self.laplacian_weight = 0.3
            self.grad_weight = 0.2
            print("切换到阶段2：联合训练")
        elif stage == 'finetune':
            # 阶段3：SR微调
            self.dam_enhanced_weight = 0.0
            self.hrdem_weight = 1.0
            self.mapping_weight = 0.5
            self.laplacian_weight = 0.3
            self.grad_weight = 0.2
            print("切换到阶段3：SR微调")
