"""
损失函数模块

包含：
1. 添加DAMEnhancedLoss（直接监督DAM的enhanced_depth输出）
2. 添加MultiScaleLoss（多尺度损失）
3. 添加ConsistencyLoss（用于无监督区域的自监督）
4. 改进CombinedLoss支持分阶段训练
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


# 【修改】新增：拉普拉斯高频损失
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


class SSIMLoss(nn.Module):
    """
    SSIM损失（结构相似性）

    计算预测与目标之间的结构相似性指数，值越小表示越不相似
    """

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        # 使用高斯窗口
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - self.window_size//2)**2/float(2*sigma**2))
                              for x in range(self.window_size)])
        gauss = gauss / gauss.sum()

        window = gauss.unsqueeze(0).unsqueeze(0) * gauss.unsqueeze(0).unsqueeze(2)
        window = window.expand(1, 1, self.window_size, self.window_size).to(pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, padding=self.window_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred ** 2, window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim_map.mean()


class DAMEnhancedLoss(nn.Module):
    """
    DAM增强深度图损失

    直接监督DAM的enhanced_depth输出，确保实例分割头的有效性
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, enhanced_depth, target_depth):
        """
        Args:
            enhanced_depth: DAM增强后的深度图 (B, H, W) 或 (B, 1, H, W)
            target_depth: 目标深度图（如USGS DEM归一化后） (B, H, W) 或 (B, 1, H, W)

        Returns:
            loss: RMSE损失
        """
        # 确保维度一致
        if enhanced_depth.dim() == 3:
            enhanced_depth = enhanced_depth.unsqueeze(1)
        if target_depth.dim() == 3:
            target_depth = target_depth.unsqueeze(1)

        # 归一化目标到0~1范围（AMP友好的实现）
        B = target_depth.size(0)
        target_flat = target_depth.view(B, -1)
        target_min = target_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        target_max = target_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        target_range = torch.where(target_max - target_min < 1e-6,
                                   torch.ones_like(target_max),
                                   target_max - target_min)
        target_normalized = (target_depth - target_min) / target_range

        # 检查数值有效性
        if torch.isnan(target_normalized).any() or torch.isinf(target_normalized).any():
            print("警告: DAMEnhancedLoss中target_normalized包含NaN/Inf")
            return torch.tensor(0.0, device=enhanced_depth.device, requires_grad=True)

        # 计算RMSE（使用clamp防止FP16溢出）
        mse = torch.mean((enhanced_depth - target_normalized) ** 2)
        if mse.dtype == torch.float16:
            mse = torch.clamp(mse, max=60000.0)
        loss = torch.sqrt(mse + self.eps)

        return loss


class MultiScaleLoss(nn.Module):
    """
    多尺度损失

    在不同尺度上计算损失，鼓励模型在不同分辨率下都表现良好
    """

    def __init__(self, scales=[1, 2, 4], eps=1e-8):
        super().__init__()
        self.scales = scales
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred: 预测深度图 (B, 1, H, W)
            target: 目标深度图 (B, 1, H, W)

        Returns:
            loss: 多尺度RMSE损失
        """
        total_loss = 0.0

        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                # 下采样
                H, W = pred.shape[-2:]
                new_H, new_W = H // scale, W // scale
                pred_scaled = F.adaptive_avg_pool2d(pred, (new_H, new_W))
                target_scaled = F.adaptive_avg_pool2d(target, (new_H, new_W))

            # 计算RMSE
            scale_loss = torch.sqrt(torch.mean((pred_scaled - target_scaled) ** 2) + self.eps)
            total_loss += scale_loss / scale  # 小尺度的损失权重更大

        return total_loss / len(self.scales)


class ConsistencyLoss(nn.Module):
    """
    一致性损失（用于无监督区域的自监督）

    鼓励模型满足循环一致性：
    HRDEM -> Mapper -> LRDEM 应该与原始Copernicus DEM相似
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.rmse = RMSELoss(eps)

    def forward(self, mapped_lrdem, copernicus_dem, mask=None):
        """
        Args:
            mapped_lrdem: 映射后的低分辨率DEM (B, 1, H', W')
            copernicus_dem: 原始Copernicus DEM (B, 1, H', W')
            mask: 可选的掩码 (B, 1, H', W')，用于忽略某些区域

        Returns:
            loss: 一致性损失
        """
        if mask is not None:
            diff = (mapped_lrdem - copernicus_dem) * mask
            loss = torch.sqrt(torch.mean(diff ** 2) + self.eps)
        else:
            loss = self.rmse(mapped_lrdem, copernicus_dem)

        return loss


class PrototypeDiversityLoss(nn.Module):
    """
    原型多样性损失

    鼓励不同的原型学习不同的特征，避免所有原型趋同
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, prototypes):
        """
        Args:
            prototypes: (num_prototypes, embedding_dim)

        Returns:
            loss: 多样性损失（负的相似度，鼓励不相似）
        """
        # 归一化原型向量
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)

        # 计算余弦相似度矩阵
        similarity = torch.matmul(prototypes_norm, prototypes_norm.T)

        # 忽略对角线（自身相似度）
        num_prototypes = prototypes.size(0)
        mask = torch.eye(num_prototypes, device=prototypes.device).bool()
        similarity = similarity.masked_fill(mask, 0)

        # 鼓励相似度低（多样性高）
        # 使用hinge loss：如果相似度 > margin，则惩罚
        loss = F.relu(similarity - self.margin).mean()

        return loss


class CombinedLoss(nn.Module):
    """
    改进的组合损失函数（支持分阶段训练）

    包含：
    1. HRDEM与USGS DEM的RMSE损失（主要）
    2. Mapped LRDEM与Copernicus DEM的RMSE损失（辅助）
    3. DAM Enhanced Depth损失（可选，用于阶段1）
    4. 梯度一致性损失（可选）
    5. SSIM损失（可选）
    6. 实例偏置正则化损失
    7. 原型多样性损失（可选）
    8. 多尺度损失（可选）
    9. 【修改】拉普拉斯高频损失（保留地形结构）
    10. 【修改】激活熵正则（鼓励软分配）

    分阶段训练策略：
    - 阶段1（pretrain_dam）：主要优化dam_enhanced_loss
    - 阶段2（joint）：联合优化所有损失
    - 阶段3（finetune）：主要优化hrdem_loss
    """

    def __init__(
            self,
            hrdem_weight=1.0,
            mapping_weight=0.5,
            dam_enhanced_weight=0.0,  # 阶段1时设置较大值
            instance_weight=0.05,
            prototype_diversity_weight=0.01,
            grad_weight=0.0,
            ssim_weight=0.0,
            multiscale_weight=0.0,
            consistency_weight=0.0,
            dominance_weight=0.0,
            scales=[1, 2, 4],
            scale_factor=30,
            training_stage='joint',
            # 【修改】新增高频损失和熵正则权重
            laplacian_weight=0.5,  # 拉普拉斯高频损失权重
            entropy_weight=0.01,   # 激活熵正则权重（负值，最小化负熵=最大化熵）
    ):
        super().__init__()

        self.hrdem_weight = hrdem_weight
        self.mapping_weight = mapping_weight
        self.dam_enhanced_weight = dam_enhanced_weight
        self.instance_weight = instance_weight
        self.prototype_diversity_weight = prototype_diversity_weight
        self.grad_weight = grad_weight
        self.ssim_weight = ssim_weight
        self.multiscale_weight = multiscale_weight
        self.consistency_weight = consistency_weight
        self.dominance_weight = dominance_weight
        self.training_stage = training_stage

        # 【修改】保存高频损失权重
        self.laplacian_weight = laplacian_weight
        self.entropy_weight = entropy_weight

        self.rmse_loss = RMSELoss()
        self.l1_loss = nn.L1Loss()
        self.dam_enhanced_loss = DAMEnhancedLoss()

        # 【修改】初始化拉普拉斯高频损失
        if laplacian_weight > 0:
            self.laplacian_loss = LaplacianHFLoss()
        else:
            self.laplacian_loss = None

        # 多尺度主导权损失
        if dominance_weight > 0:
            self.dominance_loss = MultiScaleDominanceLoss(scale_factor=scale_factor)
        self.multiscale_loss = MultiScaleLoss(scales=scales) if multiscale_weight > 0 else None
        self.consistency_loss = ConsistencyLoss() if consistency_weight > 0 else None

        # 可选损失
        self.grad_loss = GradientLoss() if grad_weight > 0 else None
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None
        self.prototype_diversity_loss = PrototypeDiversityLoss() if prototype_diversity_weight > 0 else None

    def forward(
            self,
            hrdem, usgs_dem,
            mapped_lrdem, copernicus_dem,
            dam_enhanced_depth=None,
            instance_biases=None,
            prototypes=None,
            mask=None,
            dominance_map=None,
            # 【修改】新增参数
            activation_entropy=None,  # DAM返回的激活熵
    ):
        """
        计算组合损失

        Args:
            hrdem: 预测的高分辨率DEM (B, 1, H, W)
            usgs_dem: USGS DEM真值 (B, 1, H, W)
            mapped_lrdem: 映射后的低分辨率DEM (B, 1, H', W')
            copernicus_dem: Copernicus DEM (B, 1, H', W')
            dam_enhanced_depth: DAM增强深度图 (B, H, W) 或 (B, 1, H, W)，可选
            instance_biases: 实例偏置值（现在是增益值） (B, num_instances)，可选
            prototypes: 原型向量 (num_prototypes, embedding_dim)，可选
            mask: 掩码 (B, 1, H', W')，用于consistency loss，可选
            dominance_map: 主导权图 (B, 1, H, W)，可选
            activation_entropy: 原型激活熵，scalar，可选

        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失的字典
        """
        loss_dict = {}
        total_loss = 0.0

        # HRDEM损失（主要损失）
        if self.hrdem_weight > 0:
            hrdem_loss = self.rmse_loss(hrdem, usgs_dem)
            total_loss += self.hrdem_weight * hrdem_loss
            loss_dict['hrdem'] = hrdem_loss.item()

            # 【修改】拉普拉斯高频损失（强制保留地形结构线）
            if self.laplacian_weight > 0 and self.laplacian_loss is not None:
                lap_loss = self.laplacian_loss(hrdem, usgs_dem)
                total_loss += self.laplacian_weight * lap_loss
                loss_dict['laplacian'] = lap_loss.item()

        # 映射损失
        if self.mapping_weight > 0:
            mapping_loss = self.rmse_loss(mapped_lrdem, copernicus_dem)
            total_loss += self.mapping_weight * mapping_loss
            loss_dict['mapping'] = mapping_loss.item()

        # DAM Enhanced Depth损失（阶段1使用）
        if self.grad_weight > 0 and self.grad_loss is not None:
            dam_grad_loss = self.grad_loss(
                dam_enhanced_depth.unsqueeze(1) if dam_enhanced_depth.dim() == 3 else dam_enhanced_depth,
                usgs_dem
            )
            total_loss += self.grad_weight * dam_grad_loss * 0.5  # 权重可略低于SR阶段
            loss_dict['dam_grad'] = dam_grad_loss.item()

        if self.dam_enhanced_weight > 0 and dam_enhanced_depth is not None:
            dam_loss = self.dam_enhanced_loss(dam_enhanced_depth, usgs_dem)
            total_loss += self.dam_enhanced_weight * dam_loss
            loss_dict['dam_enhanced'] = dam_loss.item()

            # 【修改】DAM阶段也加入高频约束
            if self.laplacian_weight > 0 and self.laplacian_loss is not None:
                dam_lap_loss = self.laplacian_loss(dam_enhanced_depth.unsqueeze(1) if dam_enhanced_depth.dim()==3 else dam_enhanced_depth,
                                                   usgs_dem)
                total_loss += self.laplacian_weight * dam_lap_loss * 0.5  # DAM阶段权重可略低
                loss_dict['dam_laplacian'] = dam_lap_loss.item()

        # 多尺度损失
        if self.multiscale_weight > 0 and self.multiscale_loss is not None:
            ms_loss = self.multiscale_loss(hrdem, usgs_dem)
            total_loss += self.multiscale_weight * ms_loss
            loss_dict['multiscale'] = ms_loss.item()

        # 一致性损失（用于无监督区域）
        if self.consistency_weight > 0 and self.consistency_loss is not None:
            cons_loss = self.consistency_loss(mapped_lrdem, copernicus_dem, mask)
            total_loss += self.consistency_weight * cons_loss
            loss_dict['consistency'] = cons_loss.item()

        # 梯度一致性损失（Sobel边缘）
        if self.grad_weight > 0 and self.grad_loss is not None:
            grad_loss = self.grad_loss(hrdem, usgs_dem)
            total_loss += self.grad_weight * grad_loss
            loss_dict['grad'] = grad_loss.item()

        # SSIM损失
        if self.ssim_weight > 0 and self.ssim_loss is not None:
            ssim_loss_val = self.ssim_loss(hrdem, usgs_dem)
            total_loss += self.ssim_weight * ssim_loss_val
            loss_dict['ssim'] = ssim_loss_val.item()

        # 实例偏置正则化损失（现在是增益值，正则化鼓励接近1.0）
        instance_reg_loss = torch.tensor(0.0, device=hrdem.device)
        if instance_biases is not None and self.instance_weight > 0:
            # 【修改】乘性门控下，偏置接近1.0表示"不调整"，所以正则化 (bias - 1)^2
            instance_reg_loss = torch.mean((instance_biases - 1.0) ** 2)
            total_loss += self.instance_weight * instance_reg_loss
        loss_dict['instance_reg'] = instance_reg_loss.item()

        # 原型多样性损失
        prototype_div_loss = torch.tensor(0.0, device=hrdem.device)
        if prototypes is not None and self.prototype_diversity_weight > 0 and self.prototype_diversity_loss is not None:
            prototype_div_loss = self.prototype_diversity_loss(prototypes)
            total_loss += self.prototype_diversity_weight * prototype_div_loss
        loss_dict['prototype_div'] = prototype_div_loss.item()

        # 多尺度主导权损失
        dominance_loss_val = torch.tensor(0.0, device=hrdem.device)
        if self.dominance_weight > 0 and dominance_map is not None and hasattr(self, 'dominance_loss'):
            _, dom_loss_dict = self.dominance_loss(hrdem, usgs_dem, dominance_map, copernicus_dem)
            dominance_loss_val = torch.tensor(dom_loss_dict['total'], device=hrdem.device)
            total_loss += self.dominance_weight * dominance_loss_val
            loss_dict['dominance_global'] = dom_loss_dict['global']
            loss_dict['dominance_detail'] = dom_loss_dict['detail']
            loss_dict['dominance_alpha'] = dom_loss_dict['dominance']
        loss_dict['dominance'] = dominance_loss_val.item()

        # 【修改】激活熵正则（鼓励软分配，减少块状硬边界）
        if activation_entropy is not None and self.entropy_weight > 0:
            # 最大化熵 = 最小化负熵
            entropy_loss = -activation_entropy
            total_loss += self.entropy_weight * entropy_loss
            loss_dict['entropy'] = entropy_loss.item()

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
            # 阶段1：主要优化DAM，加入高频约束
            self.dam_enhanced_weight = 1.0
            self.hrdem_weight = 0.1
            self.mapping_weight = 0.0
            self.laplacian_weight = 1.0  # 【修改】DAM预训练时加强高频约束
            print("切换到阶段1：DAM预训练（乘性门控+高频保留）")
        elif stage == 'joint':
            # 阶段2：联合训练
            self.dam_enhanced_weight = 0.3
            self.hrdem_weight = 1.0
            self.mapping_weight = 0.5
            self.laplacian_weight = 0.5
            print("切换到阶段2：联合训练")
        elif stage == 'finetune':
            # 阶段3：微调SR
            self.dam_enhanced_weight = 0.0
            self.hrdem_weight = 1.0
            self.mapping_weight = 0.3
            self.laplacian_weight = 0.3
            self.grad_weight = 0.2  # 【修改】SR微调时启用Sobel边缘约束
            print("切换到阶段3：SR微调")


class MultiScaleDominanceLoss(nn.Module):
    """
    多尺度主导权损失

    目标：
    - 最大尺度（整幅图）：HRDEM必须与LR DEM统计对齐
    - 细节尺度（像素级）：允许自由发挥，只约束梯度
    - 中间尺度：监督alpha学习，使其与真实细节丰富度一致

    输入：
    - hrdem: 预测的高分辨率DEM
    - usgs: USGS真值
    - alpha: 主导权图（0=听LR DEM, 1=听Relative Map）
    - copernicus: LR DEM（用于计算真实细节对比）
    - scale_factor: 下采样倍率（定义"中间尺度"）
    """

    def __init__(self, scale_factor=30, global_weight=1.0, detail_weight=0.5, dominance_weight=0.3):
        super().__init__()
        self.scale_factor = scale_factor
        self.global_weight = global_weight
        self.detail_weight = detail_weight
        self.dominance_weight = dominance_weight

        # 拉普拉斯算子（提取高频细节）
        self.register_buffer('laplace_kernel', torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3))

    def forward(self, hrdem, usgs, alpha, copernicus):
        """
        Args:
            hrdem: (B, 1, H, W)
            usgs: (B, 1, H, W)
            alpha: (B, 1, H, W) - 主导权图
            copernicus: (B, 1, H, W) - LR DEM

        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失
        """
        loss_dict = {}

        # 1. 全局损失：整幅图统计量对齐（最大尺度）
        hrdem_mean = hrdem.mean(dim=(2, 3))
        usgs_mean = usgs.mean(dim=(2, 3))
        global_loss = F.mse_loss(hrdem_mean, usgs_mean)
        loss_dict['global'] = global_loss.item()

        # 2. 细节损失：高频成分（最小尺度）
        # 使用拉普拉斯算子提取细节
        hrdem_detail = F.conv2d(hrdem, self.laplacian_kernel.to(hrdem.device), padding=1)
        usgs_detail = F.conv2d(usgs, self.laplacian_kernel.to(usgs.device), padding=1)
        detail_loss = F.l1_loss(hrdem_detail, usgs_detail)
        loss_dict['detail'] = detail_loss.item()
        
        # 3. 主导权监督损失（中间尺度）
        # alpha应该与"USGS比Copernicus多多少细节"正相关
        
        # 3.1 计算中间尺度的平滑版本
        B, _, H, W = usgs.shape
        regional_h = max(1, H // self.scale_factor)
        regional_w = max(1, W // self.scale_factor)
        
        # USGS和Copernicus的中间尺度平滑
        usgs_pooled = F.adaptive_avg_pool2d(usgs, (regional_h, regional_w))
        cop_pooled = F.adaptive_avg_pool2d(copernicus, (regional_h, regional_w))
        
        # 上采样回原始尺寸
        usgs_smooth = F.interpolate(usgs_pooled, (H, W), mode='nearest')
        cop_smooth = F.interpolate(cop_pooled, (H, W), mode='nearest')
        
        # 3.2 计算每个像素的"细节丰富度"
        # USGS细节 = |原始 - 平滑|
        usgs_detail_map = torch.abs(usgs - usgs_smooth)
        cop_detail_map = torch.abs(copernicus - cop_smooth)
        
        # 3.3 USGS相对于Copernicus的细节优势
        detail_advantage = (usgs_detail_map - cop_detail_map).clamp(min=0)
        
        # 3.4 在中间尺度上计算平均细节优势
        detail_adv_pooled = F.adaptive_avg_pool2d(detail_advantage, (regional_h, regional_w))
        
        # 3.5 alpha也在同一尺度上计算
        alpha_pooled = F.adaptive_avg_pool2d(alpha, (regional_h, regional_w))
        
        # 3.6 目标：如果细节优势大，alpha应该大（听Relative Map的）
        # 归一化细节优势到[0,1]
        detail_max = detail_adv_pooled.max() + 1e-6
        alpha_target = torch.sigmoid(detail_adv_pooled * 10.0 / detail_max)
        
        dominance_loss = F.mse_loss(alpha_pooled, alpha_target)
        loss_dict['dominance'] = dominance_loss.item()
        
        # 4. 组合损失
        total_loss = (self.global_weight * global_loss + 
                     self.detail_weight * detail_loss + 
                     self.dominance_weight * dominance_loss)
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict