"""
超分辨率重构模型

包含两个主要组件：
1. 添加AdaptiveBiasFusion模块，根据LRDEM自适应调整偏置项
2. HRDEMToLRDEMMapper添加真正的下采样功能
3. 改进融合策略，更好地结合Copernicus DEM、relative map和instance bias
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ConvBlock(nn.Module):
    """卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AdaptiveBiasFusion(nn.Module):
    """
    自适应偏置融合模块

    功能：
    - 根据relative map与Copernicus DEM的差异，自适应调整instance bias的作用强度
    - 如果差异大（说明relative map有问题），增强偏置项的作用
    - 如果差异小，减弱偏置项的作用

    输入：
    - copernicus_dem: 低分辨率DEM（绝对高程参考）
    - relative_map: DAM生成的相对深度图
    - instance_bias_map: 实例偏置图

    输出：
    - fused_features: 融合后的特征
    - modulation_weights: 调制权重（用于可视化）
    """

    def __init__(self, channels):
        super(AdaptiveBiasFusion, self).__init__()

        # 差异特征提取
        self.diff_encoder = nn.Sequential(
            nn.Conv2d(2, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
        )

        # 调制权重生成网络
        self.modulation_net = nn.Sequential(
            nn.Conv2d(channels // 2 + 1, channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出0~1的权重
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, copernicus_dem, relative_map, instance_bias_map):
        """
        Args:
            features: (B, C, H, W) - 输入特征
            copernicus_dem: (B, 1, H, W) - 低分辨率DEM
            relative_map: (B, 1, H, W) - 相对深度图
            instance_bias_map: (B, 1, H, W) - 实例偏置图

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

        # 生成调制权重（基于差异和instance bias）
        modulation_input = torch.cat([diff_features, instance_bias_map], dim=1)
        modulation_weights = self.modulation_net(modulation_input)  # (B, 1, H, W)

        # 调制instance bias：差异大的地方，偏置作用增强
        modulated_features = features * (1 + modulation_weights * diff)

        # 融合
        fusion_input = torch.cat([modulated_features, instance_bias_map], dim=1)
        fused_features = self.fusion_conv(fusion_input)

        return fused_features, modulation_weights


class InstanceGuidedAttention(nn.Module):
    """
    实例引导的注意力模块（改进版）

    功能：
    - 根据实例偏置信息生成空间注意力图
    - 引导网络关注需要特殊处理的区域
    """

    def __init__(self, channels):
        super(InstanceGuidedAttention, self).__init__()

        # 从实例偏置图生成注意力权重
        self.instance_conv = nn.Sequential(
            nn.Conv2d(1, channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 特征变换
        self.feature_transform = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, instance_bias_map):
        """
        Args:
            features: (B, C, H, W) - 输入特征
            instance_bias_map: (B, 1, H, W) - 实例偏置图

        Returns:
            attended_features: (B, C, H, W) - 注意力加权后的特征
        """
        # 生成注意力权重
        attention = self.instance_conv(instance_bias_map)  # (B, C, H, W)

        # 应用注意力
        attended = features * attention

        # 特征变换
        output = self.feature_transform(attended)

        return output


class SuperResolutionNetwork(nn.Module):
    """
    改进的超分辨率重构网络

    主要改进：
    1. 添加AdaptiveBiasFusion模块实现自适应偏置融合
    2. 改进特征提取和融合策略
    3. 更好地利用LRDEM指导relative map和instance bias的融合

    输入：
    - copernicus_dem: (B, 1, H, W) - 低分辨率DEM（30m重采样到1m）
    - relative_map: (B, 1, H, W) - DAM生成的相对深度图
    - instance_bias_map: (B, 1, H, W) - 实例偏置图（可选，默认为None）

    输出：
    - hrdem: (B, 1, H, W) - 高分辨率DEM
    """

    def __init__(
            self,
            in_channels=3,  # Copernicus DEM + relative map + instance bias (可选)
            base_channels=64,
            num_residual_blocks=8,
            out_channels=1,
            use_instance_guidance=True,
            use_adaptive_fusion=True
    ):
        super(SuperResolutionNetwork, self).__init__()

        self.use_instance_guidance = use_instance_guidance
        self.use_adaptive_fusion = use_adaptive_fusion

        # 初始特征提取（3通道输入：DEM + relative + instance bias）
        self.initial_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)

        # 自适应偏置融合模块
        if self.use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveBiasFusion(base_channels)

        # 实例引导注意力模块
        if self.use_instance_guidance:
            self.instance_attention = InstanceGuidedAttention(base_channels)

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_residual_blocks)
        ])

        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels)
        )

        # 上采样和重构层
        self.reconstruction = nn.Sequential(
            ConvBlock(base_channels, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

        # 残差连接权重（针对不同区域的可学习权重）
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

        # 实例感知的残差权重
        if self.use_instance_guidance:
            self.instance_adaptive_weight = nn.Sequential(
                nn.Conv2d(1, base_channels // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 4, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

    def forward(self, copernicus_dem, relative_map, instance_bias_map=None):
        """
        前向传播

        Args:
            copernicus_dem: (B, 1, H, W) - 低分辨率DEM
            relative_map: (B, 1, H, W) - 相对深度图
            instance_bias_map: (B, 1, H, W) - 实例偏置图（可选）

        Returns:
            hrdem: (B, 1, H, W) - 高分辨率DEM
            modulation_weights: (B, 1, H, W) - 调制权重（用于可视化）
        """
        # 拼接输入
        if instance_bias_map is not None and self.use_instance_guidance:
            x = torch.cat([copernicus_dem, relative_map, instance_bias_map], dim=1)  # (B, 3, H, W)
        else:
            x = torch.cat([copernicus_dem, relative_map], dim=1)  # (B, 2, H, W)
            if instance_bias_map is not None:
                x = torch.cat([x, instance_bias_map], dim=1)

        # 初始特征提取
        x = self.initial_conv(x)

        # 保存残差
        residual = x

        modulation_weights = None

        # 应用自适应偏置融合
        if self.use_adaptive_fusion and instance_bias_map is not None:
            x, modulation_weights = self.adaptive_fusion(
                x, copernicus_dem, relative_map, instance_bias_map
            )

        # 应用实例引导注意力
        if instance_bias_map is not None and self.use_instance_guidance:
            x = self.instance_attention(x, instance_bias_map)

        # 残差块
        for block in self.residual_blocks:
            x = block(x)

        # 特征融合
        x = self.fusion_conv(x)

        # 添加残差连接
        x = x + residual

        # 重构
        x = self.reconstruction(x)

        # 实例感知的残差权重
        if instance_bias_map is not None and self.use_instance_guidance:
            # 根据instance bias生成自适应权重
            adaptive_weight = self.instance_adaptive_weight(instance_bias_map)
            # 结合全局权重和自适应权重
            effective_weight = self.residual_weight * (1 + adaptive_weight)
            hrdem = copernicus_dem + effective_weight * x
        else:
            # 使用全局残差权重
            hrdem = copernicus_dem + self.residual_weight * x

        return hrdem, modulation_weights


class HRDEMToLRDEMMapper(nn.Module):
    """
    改进的HRDEM到LRDEM的映射网络（可学习降采样核）

    主要改进：
    1. 使用可学习降采样核，结合HRDEM和Bias Map进行深度引导
    2. 空洞卷积扩大感受野，适应30倍下采样
    3. 空间注意力根据地形复杂度调整退化强度

    输入：
    - hrdem: (B, 1, H, W) - 高分辨率DEM（融合后的绝对高程）
    - instance_bias_map: (B, 1, H, W) - 实例偏置图（地形类型/复杂度）

    输出：
    - lrdem: (B, 1, H//scale_factor, W//scale_factor) - 模拟的低分辨率DEM
    """

    def __init__(
            self,
            in_channels=2,  # HRDEM + Bias Map
            base_channels=16,
            scale_factor=30,  # 下采样倍率
    ):
        super(HRDEMToLRDEMMapper, self).__init__()

        self.scale_factor = scale_factor

        # 1. 浅层特征提取
        self.feature_extractor = nn.Sequential(
            ConvBlock(in_channels, base_channels, kernel_size=3, padding=1),
            ConvBlock(base_channels, base_channels, kernel_size=3, padding=1),
        )

        # 2. 空间注意力（根据bias map决定关注区域）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 3. 可学习降采样核（使用空洞卷积捕获不同尺度的退化模式）
        # 1022 -> 511 -> 255 -> 127 -> 63 -> 32 (约30倍)
        self.downsample_conv = nn.Sequential(
            # 第一层：stride=2 下采样
            ConvBlock(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            # 空洞卷积扩大感受野
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, 
                     dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # 第二层：stride=2 下采样
            ConvBlock(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1),
            # 空洞卷积
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3,
                     dilation=2, padding=2, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # 第三层：stride=2 下采样
            ConvBlock(base_channels * 2, base_channels, kernel_size=3, stride=2, padding=1),
        )

        # 4. 自适应输出层
        self.output_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, hrdem, instance_bias_map, target_size=None):
        """
        前向传播

        Args:
            hrdem: (B, 1, H, W) - 高分辨率DEM
            instance_bias_map: (B, 1, H, W) - 实例偏置图
            target_size: tuple (H, W) - 目标输出尺寸（可选）

        Returns:
            lrdem: (B, 1, H', W') - 模拟的低分辨率DEM
        """
        # 拼接输入：HRDEM + Bias Map
        x = torch.cat([hrdem, instance_bias_map], dim=1)

        # 特征提取
        features = self.feature_extractor(x)

        # 空间注意力加权
        attention = self.spatial_attention(features)
        features = features * attention

        # 可学习降采样
        x = self.downsample_conv(features)

        # 精确调整到目标尺寸
        if target_size is not None:
            x = F.adaptive_avg_pool2d(x, target_size)
        else:
            # 计算目标尺寸
            H, W = hrdem.shape[-2:]
            target_h = H // self.scale_factor
            target_w = W // self.scale_factor
            x = F.adaptive_avg_pool2d(x, (target_h, target_w))

        # 输出
        lrdem = self.output_conv(x)
        return lrdem


class DEMSuperResolutionSystem(nn.Module):
    """
    改进的DEM超分辨率系统

    整合所有组件：
    1. DAM模型（带自适应实例分割）
    2. 超分辨率重构网络（带自适应偏置融合）
    3. HRDEM到LRDEM的映射网络（带真正下采样）
    """

    def __init__(
            self,
            dam_model,
            sr_channels=64,
            sr_residual_blocks=8,
            mapper_base_channels=32,
            mapper_scale_factor=30,
            use_instance_guidance=True,
            use_adaptive_fusion=True
    ):
        super(DEMSuperResolutionSystem, self).__init__()

        # DAM模型（带自适应实例分割）
        self.dam_model = dam_model

        # 超分辨率重构网络（支持自适应偏置融合）
        self.sr_network = SuperResolutionNetwork(
            in_channels=3,  # Copernicus + relative + instance bias
            base_channels=sr_channels,
            num_residual_blocks=sr_residual_blocks,
            out_channels=1,
            use_instance_guidance=use_instance_guidance,
            use_adaptive_fusion=use_adaptive_fusion
        )

        # HRDEM到LRDEM的映射网络（带真正下采样）
        self.mapper_network = HRDEMToLRDEMMapper(
            in_channels=2,  # HRDEM + Bias Map
            base_channels=mapper_base_channels,
            scale_factor=mapper_scale_factor
        )

        self.mapper_scale_factor = mapper_scale_factor

    def forward(self, google_image, copernicus_dem, use_instance_guidance=True, return_modulation=False):
        """
        前向传播

        Args:
            google_image: (B, 3, H, W) - Google Earth影像
            copernicus_dem: (B, 1, H, W) - Copernicus DEM
            use_instance_guidance: bool - 是否使用实例引导
            return_modulation: bool - 是否返回调制权重

        Returns:
            包含以下键的字典：
            - 'hrdem': 高分辨率DEM
            - 'mapped_lrdem': 映射后的低分辨率DEM
            - 'dam_output': DAM模型的完整输出
            - 'modulation_weights': 调制权重（如果return_modulation=True）
        """
        B, _, H, W = copernicus_dem.shape

        # DAM模型生成relative map（带实例分割增强）
        dam_output = self.dam_model(google_image)
        enhanced_depth = dam_output['enhanced_depth'].unsqueeze(1)  # (B, 1, H, W)
        instance_bias_map = dam_output['instance_bias_map'].unsqueeze(1)  # (B, 1, H, W)

        # 超分辨率重构网络融合Copernicus DEM、relative map和instance bias
        if use_instance_guidance and self.sr_network.use_instance_guidance:
            hrdem, modulation_weights = self.sr_network(
                copernicus_dem, enhanced_depth, instance_bias_map
            )
        else:
            hrdem, modulation_weights = self.sr_network(
                copernicus_dem, enhanced_depth, None
            )

        # 映射网络将HRDEM映射回LRDEM（下采样）
        # 输入：HRDEM + Instance Bias Map（深度引导的降采样）
        target_h = H // self.mapper_scale_factor
        target_w = W // self.mapper_scale_factor
        mapped_lrdem = self.mapper_network(
            hrdem, instance_bias_map, target_size=(target_h, target_w)
        )

        result = {
            'dam_dem': enhanced_depth,
            'hrdem': hrdem,
            'mapped_lrdem': mapped_lrdem,
            'dam_output': dam_output
        }

        if return_modulation and modulation_weights is not None:
            result['modulation_weights'] = modulation_weights

        return result

    def freeze_dam(self):
        """冻结DAM模型的参数（除了实例分割头）"""
        for param in self.dam_model.parameters():
            param.requires_grad = False

        # 解冻实例分割头
        for param in self.dam_model.instance_head.parameters():
            param.requires_grad = True

        # 解冻归一化参数
        self.dam_model.norm_min.requires_grad = True
        self.dam_model.norm_max.requires_grad = True

        print("DAM模型已冻结（除实例分割头外）")

    def unfreeze_dam(self):
        """解冻DAM模型的所有参数"""
        for param in self.dam_model.parameters():
            param.requires_grad = True
        print("DAM模型已解冻")


def create_super_resolution_system(
        dam_model,
        sr_channels=64,
        sr_residual_blocks=8,
        mapper_base_channels=32,
        mapper_scale_factor=30,
        use_instance_guidance=True,
        use_adaptive_fusion=True,
        device='cuda'
):
    """
    创建超分辨率系统

    Args:
        dam_model: DAM模型实例
        sr_channels: 超分辨率网络的基础通道数
        sr_residual_blocks: 超分辨率网络的残差块数量
        mapper_base_channels: 映射网络的基础通道数
        mapper_scale_factor: 映射网络的下采样倍率
        use_instance_guidance: 是否使用实例引导
        use_adaptive_fusion: 是否使用自适应偏置融合
        device: 设备

    Returns:
        system: DEM超分辨率系统
    """
    system = DEMSuperResolutionSystem(
        dam_model=dam_model,
        sr_channels=sr_channels,
        sr_residual_blocks=sr_residual_blocks,
        mapper_base_channels=mapper_base_channels,
        mapper_scale_factor=mapper_scale_factor,
        use_instance_guidance=use_instance_guidance,
        use_adaptive_fusion=use_adaptive_fusion,
    )

    system = system.to(device)

    return system