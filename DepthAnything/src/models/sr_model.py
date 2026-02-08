"""
改进的超分辨率重构模型 V2

主要改进：
1. 重命名 prototype_alphas 为 prototype_detail_strength，语义更清晰
2. 添加可视化功能
3. 改进融合公式注释
4. 适配DAM的乘性门控输出（instance_bias_map现为增益图）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class AdaptiveBiasFusion(nn.Module):
    """
    自适应偏置融合模块

    功能更新：
    - 输入的instance_bias_map现在是乘性增益图（来自DAM的乘性门控）
    - 范围大致在[0.5, 2.0]，中心在1.0（表示不调整）
    - 模块将其作为特征输入，学习如何利用增益信息调制融合过程

    输入：
    - copernicus_dem: 低分辨率DEM（绝对高程参考）
    - relative_map: DAM生成的相对深度图（已乘性调整）
    - instance_bias_map: (B, 1, H, W) 实例增益图，范围~[0.5, 2.0]

    输出：
    - fused_features: 融合后的特征
    - modulation_weights: 调制权重（用于可视化）
    """

    def __init__(self, channels):
        super(AdaptiveBiasFusion, self).__init__()

        # 差异特征提取
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(2, channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, channels // 4), num_channels=channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels // 2),
            nn.ReLU(inplace=True),
        )

        # 调制权重生成网络
        self.modulation_net = nn.Sequential(
            nn.Conv2d(channels // 2 + 1, channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出0~1的权重
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, copernicus_dem, relative_map, instance_bias_map):
        """
        Args:
            features: (B, C, H, W) - 输入特征
            copernicus_dem: (B, 1, H, W) - 低分辨率DEM
            relative_map: (B, 1, H, W) - 相对深度图（DAM已乘性调整）
            instance_bias_map: (B, 1, H, W) - 实例增益图（乘性门控，范围~[0.5,2.0]）

        Returns:
            fused_features: (B, C, H, W) - 融合后的特征
            modulation_weights: (B, 1, H, W) - 调制权重
        """
        B, C, H, W = features.shape

        # 归一化Copernicus DEM到0~1范围（与relative map一致）
        cop_norm = copernicus_dem.view(B, -1)
        cop_min = cop_norm.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cop_max = cop_norm.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cop_range = torch.where(cop_max - cop_min < 1e-6,
                                torch.ones_like(cop_max),
                                cop_max - cop_min)
        copernicus_normalized = (copernicus_dem - cop_min) / cop_range

        # 计算差异
        diff = torch.abs(relative_map - copernicus_normalized)  # (B, 1, H, W)

        # 编码差异特征
        diff_input = torch.cat([diff, copernicus_normalized], dim=1)
        diff_features = self.diff_encoder(diff_input)  # (B, C//2, H, W)

        # 生成调制权重：结合差异特征和实例增益图
        modulation_input = torch.cat([diff_features, instance_bias_map], dim=1)
        modulation_weights = self.modulation_net(modulation_input)  # (B, 1, H, W)

        # 调制instance bias：增益高的区域，relative map贡献增强
        modulated_features = features * (1 + modulation_weights * (instance_bias_map - 1.0))

        # 融合
        fusion_input = torch.cat([modulated_features, instance_bias_map], dim=1)
        fused_features = self.fusion_conv(fusion_input)

        return fused_features, modulation_weights



class InstanceGuidedAttention(nn.Module):
    """
    实例引导的注意力模块（改进版）

    功能更新：
    - 输入instance_bias_map现为乘性增益图（范围~[0.5, 2.0]）
    - 注意力机制学习：增益偏离1.0越大的区域（无论增强或减弱），越需要精细处理
    """

    def __init__(self, channels):
        super(InstanceGuidedAttention, self).__init__()

        # 从实例增益图生成注意力权重
        self.instance_conv = nn.Sequential(
            nn.Conv2d(1, channels // 4, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, channels // 4), num_channels=channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, instance_bias_map):
        """
        Args:
            features: (B, C, H, W) - 输入特征
            instance_bias_map: (B, 1, H, W) - 实例增益图（来自DAM乘性门控，范围~[0.5,2.0]）

        Returns:
            attended_features: (B, C, H, W) - 注意力加权后的特征
        """
        deviation = torch.abs(instance_bias_map - 1.0)
        attention = self.instance_conv(deviation)  # (B, C, H, W)

        # 应用注意力
        attended = features * attention

        # 特征变换
        output = self.feature_transform(attended)

        return output


class SuperResolutionNetwork(nn.Module):
    """
    改进的超分辨率重构网络 V2

    核心设计：
    ==========

    融合策略（每个实例）：

    1. 高程对齐（Elevation Alignment）
       - 目标：让Relative和Copernicus在实例级均值相同
       - 实现：offset = Copernicus_mean - Relative_mean
       - 输出：Relative_aligned = Relative + offset
       - 注意：输入的relative_map已是DAM乘性调整后的结果（enhanced_depth）

    2. 细节调制（Detail Modulation）
       - 目标：控制Relative的细节贡献程度
       - 实现：detail_strength（0=听Copernicus, 1=听Relative）
       - 输出：Mixed = Copernicus + detail_strength * (Relative_aligned - Copernicus)

    3. 空间精修（Spatial Refinement）
       - 目标：网络学习局部精修
       - 实现：delta = Network(Mixed)
       - 输出：HRDEM = Mixed + delta

    重要：instance_bias_map现在是乘性增益图（gain map），来自DAM的乘性门控
    其值范围大致在[0.5, 2.0]，中心为1.0（表示DAM未调整），用于指导SR网络的注意力
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        num_residual_blocks=8,
        out_channels=1,
        use_instance_guidance=True,
        use_adaptive_fusion=False,
        use_scale_dominance=True,
        scale_factor=30,
        num_prototypes=128,
    ):
        super().__init__()

        self.use_instance_guidance = use_instance_guidance
        self.use_scale_dominance = use_scale_dominance
        self.use_adaptive_fusion = use_adaptive_fusion
        self.scale_factor = scale_factor
        self.num_prototypes = num_prototypes

        # 实例级细节强度（每个原型学习一个细节强度）
        # 初始化为0，Sigmoid后=0.5，让网络自己学习
        if self.use_scale_dominance:
            self.prototype_detail_strength = nn.Parameter(torch.zeros(num_prototypes))

            # 融合后的特征提取
            self.fused_features = nn.Sequential(
                ConvBlock(1, base_channels, kernel_size=3, padding=1),
                ConvBlock(base_channels, base_channels, kernel_size=3, padding=1),
            )

        # 实例引导注意力模块
        if self.use_instance_guidance:
            self.instance_attention = InstanceGuidedAttention(base_channels)

        if self.use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveBiasFusion(base_channels)

        # 初始特征提取（备用方案）
        if not self.use_scale_dominance:
            self.initial_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_residual_blocks)
        ])

        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels)
        )

        # 上采样和重构层
        self.reconstruction = nn.Sequential(
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels // 2, out_channels, kernel_size=3, padding=1)
        )

        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # 实例感知的残差权重
        if self.use_instance_guidance:
            self.instance_adaptive_weight = nn.Sequential(
                nn.Conv2d(1, base_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 4, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, copernicus_dem, relative_map, instance_bias_map=None, prototype_activations=None):
        """
        前向传播

        Args:
            copernicus_dem: (B, 1, H, W) - 低分辨率DEM
            relative_map: (B, 1, H, W) - 相对深度图（DAM输出，已乘性调整）
            instance_bias_map: (B, 1, H, W) - 实例增益图（DAM乘性门控输出，范围~[0.5,2.0]）
            prototype_activations: (B, num_prototypes, H, W) - 原型激活图

        Returns:
            hrdem: (B, 1, H, W) - 高分辨率DEM
            detail_strength_map: (B, 1, H, W) - 细节强度图（用于可视化）
            fusion_info: dict - 融合过程的中间结果（用于可视化）
        """
        B, _, H, W = copernicus_dem.shape
        detail_strength_map = None
        fusion_info = {}

        # ========== 第1步：实例级高程融合 ==========
        if self.use_scale_dominance and prototype_activations is not None:
            num_proto = prototype_activations.shape[1]

            # 动态调整detail_strength到匹配的原型数量
            strengths_full = torch.sigmoid(self.prototype_detail_strength)
            if num_proto <= self.num_prototypes:
                strengths = strengths_full[:num_proto]
            else:
                mean_val = strengths_full.mean()
                strengths = torch.cat([strengths_full, mean_val.expand(num_proto - self.num_prototypes)], dim=0)

            # 计算每个像素的细节强度
            detail_strength = (strengths.view(1, -1, 1, 1) * prototype_activations).sum(dim=1, keepdim=True)
            detail_strength_map = detail_strength

            # 保存用于可视化
            fusion_info['detail_strength'] = detail_strength
            fusion_info['prototype_strengths'] = strengths

            # 实例级均值对齐
            cop_expanded = copernicus_dem.expand(B, num_proto, H, W)
            rel_expanded = relative_map.expand(B, num_proto, H, W)

            # 每个原型的掩码权重
            weights = prototype_activations / (prototype_activations.sum(dim=(2,3), keepdim=True) + 1e-6)

            # 实例级均值
            cop_means = (cop_expanded * weights).sum(dim=(2,3), keepdim=True)
            rel_means = (rel_expanded * weights).sum(dim=(2,3), keepdim=True)

            # 均值差异
            mean_offsets = cop_means - rel_means
            offsets = (mean_offsets * prototype_activations).sum(dim=1, keepdim=True)
            relative_aligned = relative_map + offsets

            fusion_info['mean_offsets'] = mean_offsets
            fusion_info['relative_aligned'] = relative_aligned

            # 残差融合
            residual = relative_aligned - copernicus_dem
            mixed = copernicus_dem + detail_strength * residual

            fusion_info['residual'] = residual
            fusion_info['mixed'] = mixed

        else:
            mixed = copernicus_dem
            fusion_info['mixed'] = mixed

        # ========== 第2步：提取特征 ==========
        if self.use_scale_dominance and prototype_activations is not None:
            x = self.fused_features(mixed)
        else:
            if instance_bias_map is not None and self.use_instance_guidance:
                x = torch.cat([copernicus_dem, relative_map, instance_bias_map], dim=1)
            else:
                x = torch.cat([copernicus_dem, relative_map], dim=1)
                if instance_bias_map is not None:
                    x = torch.cat([x, instance_bias_map], dim=1)
            x = self.initial_conv(x)

        # ========== 第3步：特征精修 ==========
        skip = x

        if self.use_adaptive_fusion and instance_bias_map is not None:
            x, _ = self.adaptive_fusion(x, copernicus_dem, relative_map, instance_bias_map)

        if instance_bias_map is not None and self.use_instance_guidance:
            x = self.instance_attention(x, instance_bias_map)

        for block in self.residual_blocks:
            x = block(x)

        x = self.fusion_conv(x)
        x = x + skip

        # 重构：预测精修量delta
        delta = self.reconstruction(x)

        # ========== 第4步：最终融合 ==========
        if instance_bias_map is not None and self.use_instance_guidance:
            deviation = torch.abs(instance_bias_map - 1.0)
            adaptive_weight = self.instance_adaptive_weight(deviation)
            effective_weight = self.residual_weight * (1 + adaptive_weight)
        else:
            effective_weight = self.residual_weight

        hrdem = mixed + effective_weight * delta

        fusion_info['delta'] = delta
        fusion_info['effective_weight'] = effective_weight

        return hrdem, detail_strength_map, fusion_info


class HRDEMToLRDEMMapper(nn.Module):
    """
    改进的HRDEM到LRDEM的映射网络 V2

    改进：
    1. 处理非整数倍下采样
    2. 添加可视化功能
    3. 适配乘性增益图输入（instance_bias_map作为增益特征）
    """

    def __init__(
        self,
        in_channels=2,
        base_channels=16,
        scale_factor=30,
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # 特征提取
        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=3, padding=1),
            ConvBlock(base_channels, base_channels, kernel_size=3, padding=1),
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 可学习降采样核
        self.downsample_conv = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * 2),
            nn.ReLU(inplace=True),

            ConvBlock(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, dilation=2, padding=2, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=base_channels * 2),
            nn.ReLU(inplace=True),

            ConvBlock(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1),
        )

        # 输出层
        self.output_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, hrdem, instance_bias_map, target_size=None):
        """
        前向传播

        Args:
            hrdem: (B, 1, H, W) - 高分辨率DEM
            instance_bias_map: (B, 1, H, W) - 实例增益图（DAM乘性门控，范围~[0.5,2.0]）
            target_size: tuple (H, W) - 目标输出尺寸（可选）

        Returns:
            lrdem: (B, 1, H', W') - 模拟的低分辨率DEM
        """
        x = torch.cat([hrdem, instance_bias_map], dim=1)

        features = self.feature_extractor(x)

        attention = self.spatial_attention(features)
        features = features * attention

        x = self.downsample_conv(features)

        # 精确调整到目标尺寸
        if target_size is not None:
            x = F.adaptive_avg_pool2d(x, target_size)
        else:
            H, W = hrdem.shape[-2:]
            # 使用round处理非整数倍下采样
            target_h = max(1, round(H / self.scale_factor))
            target_w = max(1, round(W / self.scale_factor))
            x = F.adaptive_avg_pool2d(x, (target_h, target_w))

        lrdem = self.output_conv(x)
        return lrdem


class DEMSuperResolutionSystem(nn.Module):
    """改进的DEM超分辨率系统 V2"""

    def __init__(
        self,
        dam_model,
        sr_channels=64,
        sr_residual_blocks=8,
        mapper_base_channels=32,
        mapper_scale_factor=30,
        use_instance_guidance=True,
        use_scale_dominance=True,
        use_cached_dam_encoder=False,
        use_adaptive_fusion=False,
        num_prototypes=128,
    ):
        super().__init__()

        self.dam_model = dam_model
        self.use_cached_dam_encoder = use_cached_dam_encoder
        self.num_prototypes = num_prototypes

        # 超分辨率重构网络
        self.sr_network = SuperResolutionNetwork(
            in_channels=3,
            base_channels=sr_channels,
            num_residual_blocks=sr_residual_blocks,
            out_channels=1,
            use_instance_guidance=use_instance_guidance,
            use_scale_dominance=use_scale_dominance,
            use_adaptive_fusion=use_adaptive_fusion,
            scale_factor=mapper_scale_factor,
            num_prototypes=num_prototypes,
        )

        # HR到LR映射
        self.mapper_network = HRDEMToLRDEMMapper(
            in_channels=2,
            base_channels=mapper_base_channels,
            scale_factor=mapper_scale_factor
        )

        self.mapper_scale_factor = mapper_scale_factor

    def forward(self, google_image, copernicus_dem, use_instance_guidance=True,
                return_fusion_info=False, dam_encoder_features=None):
        """
        前向传播

        Args:
            google_image: (B, 3, H, W) - Google Earth影像
            copernicus_dem: (B, 1, H, W) - Copernicus DEM
            use_instance_guidance: bool - 是否使用实例引导
            return_fusion_info: bool - 是否返回融合过程的中间结果
            dam_encoder_features: Dict - 预计算的DAM Encoder特征

        Returns:
            result: dict 包含以下键：
                - 'hrdem': 高分辨率DEM
                - 'mapped_lrdem': 映射后的低分辨率DEM
                - 'dam_output': DAM模型的完整输出
                - 'detail_strength_map': 细节强度图
                - 'fusion_info': 融合过程的中间结果（如果return_fusion_info=True）
                - 【修改】注意：dam_output中的instance_bias_map现在是乘性增益图
        """
        B, _, H, W = copernicus_dem.shape

        # DAM模型
        if self.use_cached_dam_encoder and dam_encoder_features is not None:
            if isinstance(dam_encoder_features, dict):
                features = dam_encoder_features.get('features', [])
                patch_h = dam_encoder_features.get('patch_h', H // 14)
                patch_w = dam_encoder_features.get('patch_w', W // 14)
            elif isinstance(dam_encoder_features, (list, tuple)):
                features = dam_encoder_features
                patch_h, patch_w = H // 14, W // 14

            dam_output = self.dam_model.forward_from_features(features, patch_h, patch_w)
        else:
            dam_output = self.dam_model(google_image)

        # 从DAM输出获取数据，维度保持 (B, H, W)
        enhanced_depth = dam_output['enhanced_depth']  # (B, H, W)
        instance_bias_map = dam_output['instance_bias_map']  # (B, H, W)
        prototype_activations = dam_output.get('prototype_activations', None)

        # 超分辨率重构 - SR网络需要 (B, 1, H, W)
        if use_instance_guidance and self.sr_network.use_instance_guidance:
            hrdem, detail_strength_map, fusion_info = self.sr_network(
                copernicus_dem, enhanced_depth.unsqueeze(1), instance_bias_map.unsqueeze(1), prototype_activations
            )
        else:
            hrdem, detail_strength_map, fusion_info = self.sr_network(
                copernicus_dem, enhanced_depth.unsqueeze(1), None, prototype_activations
            )

        # HR到LR映射 - mapper需要 (B, 1, H, W)
        target_h = max(1, round(H / self.mapper_scale_factor))
        target_w = max(1, round(W / self.mapper_scale_factor))
        mapped_lrdem = self.mapper_network(
            hrdem, instance_bias_map.unsqueeze(1), target_size=(target_h, target_w)
        )

        result = {
            'hrdem': hrdem,
            'mapped_lrdem': mapped_lrdem,
            'dam_output': {
                'original_depth': dam_output['original_depth'],  # (B, H, W) DAM原始输出
                'enhanced_depth': enhanced_depth,  # (B, H, W) DAM增强输出
                'instance_bias_map': instance_bias_map,  # (B, H, W)
                'prototype_activations': prototype_activations,
                'base_biases': dam_output.get('base_biases', None),
                'spatial_residual': dam_output.get('spatial_residual', None),
                'activation_entropy': dam_output.get('activation_entropy', None),
            },
            'detail_strength_map': detail_strength_map,
        }

        if return_fusion_info:
            result['fusion_info'] = fusion_info

        return result


def create_super_resolution_system(
    dam_model,
    sr_channels=64,
    sr_residual_blocks=8,
    mapper_base_channels=32,
    mapper_scale_factor=30,
    use_instance_guidance=True,
    use_scale_dominance=True,
    use_adaptive_fusion=False,
    device='cuda',
    use_cached_dam_encoder=False,
    num_prototypes=128,
):
    """创建改进的超分辨率系统 V2"""
    system = DEMSuperResolutionSystem(
        dam_model=dam_model,
        sr_channels=sr_channels,
        sr_residual_blocks=sr_residual_blocks,
        mapper_base_channels=mapper_base_channels,
        mapper_scale_factor=mapper_scale_factor,
        use_instance_guidance=use_instance_guidance,
        use_scale_dominance=use_scale_dominance,
        use_cached_dam_encoder=use_cached_dam_encoder,
        num_prototypes=num_prototypes,
        use_adaptive_fusion=use_adaptive_fusion,
    )
    system = system.to(device)
    return system