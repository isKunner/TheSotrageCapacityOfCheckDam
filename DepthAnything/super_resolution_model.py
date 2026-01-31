"""
超分辨率重构模型

包含两个主要组件：
1. SuperResolutionNetwork: 融合Copernicus DEM和relative maps生成HRDEM
2. HRDEMToLRDEMMapper: 学习HRDEM到Copernicus DEM的映射关系
"""

import torch
import torch.nn as nn


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


class InstanceGuidedAttention(nn.Module):
    """
    实例引导的注意力模块
    
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
    超分辨率重构网络（增强版，支持实例偏置信息）
    
    功能：
    - 融合Copernicus DEM（低分辨率，1通道）和DAM生成的relative maps（高分辨率细节，1通道）
    - 结合实例偏置信息，对不同实例区域采用不同的重构策略
    - 输出最终的HRDEM（高分辨率DEM，1通道）
    
    输入：
    - copernicus_dem: (B, 1, H, W) - 低分辨率DEM（30m重采样到1m）
    - relative_map: (B, 1, H, W) - DAM生成的相对深度图（包含细节信息）
    - instance_bias_map: (B, 1, H, W) - 实例偏置图（可选，默认为None）
    
    输出：
    - hrdem: (B, 1, H, W) - 高分辨率DEM
    
    网络结构：
    - 多分支特征提取：分别处理Copernicus DEM、relative map和instance bias
    - 实例引导注意力：根据实例偏置调整特征权重
    - 特征融合层：融合多种特征
    - 重构层：生成最终的HRDEM
    """
    
    def __init__(
        self,
        in_channels=3,  # Copernicus DEM + relative map + instance bias (可选)
        base_channels=64,
        num_residual_blocks=8,
        out_channels=1,
        use_instance_guidance=True
    ):
        super(SuperResolutionNetwork, self).__init__()
        
        self.use_instance_guidance = use_instance_guidance
        
        # 初始特征提取（3通道输入：DEM + relative + instance bias）
        self.initial_conv = ConvBlock(in_channels, base_channels, kernel_size=7, padding=3)
        
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
        
        # 实例感知的残差权重（根据instance bias自适应调整）
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
            attention_map: (B, 1, H, W) - 注意力图（用于可视化，可选）
        """
        # 拼接输入
        if instance_bias_map is not None and self.use_instance_guidance:
            x = torch.cat([copernicus_dem, relative_map, instance_bias_map], dim=1)  # (B, 3, H, W)
        else:
            x = torch.cat([copernicus_dem, relative_map], dim=1)  # (B, 2, H, W)
        
        # 初始特征提取
        x = self.initial_conv(x)
        
        # 保存残差
        residual = x
        
        # 应用实例引导注意力（如果提供了instance_bias_map）
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
        
        return hrdem


class HRDEMToLRDEMMapper(nn.Module):
    """
    HRDEM到LRDEM的映射网络
    
    功能：
    - 学习从HRDEM到Copernicus DEM的映射关系
    - 用于在没有USGS DEM真值的区域验证模型生成的HRDEM的准确性
    
    输入：
    - hrdem: (B, 1, H, W) - 高分辨率DEM
    
    输出：
    - lrdem: (B, 1, H, W) - 模拟的低分辨率DEM（应与Copernicus DEM相似）
    
    网络结构：
    - 简单的编码器-解码器结构
    - 模拟从HRDEM到LRDEM的退化过程
    """
    
    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        num_downsample=3
    ):
        super(HRDEMToLRDEMMapper, self).__init__()
        
        # 编码器（下采样）
        encoder_layers = []
        current_channels = in_channels
        
        for i in range(num_downsample):
            out_ch = base_channels * (2 ** i)
            encoder_layers.append(ConvBlock(current_channels, out_ch, stride=2))
            encoder_layers.append(ResidualBlock(out_ch))
            current_channels = out_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            ConvBlock(current_channels, current_channels),
            ResidualBlock(current_channels)
        )
        
        # 解码器（上采样）
        decoder_layers = []
        
        for i in range(num_downsample - 1, -1, -1):
            out_ch = base_channels * (2 ** i) if i > 0 else base_channels
            decoder_layers.append(nn.ConvTranspose2d(
                current_channels, out_ch,
                kernel_size=4, stride=2, padding=1
            ))
            decoder_layers.append(nn.BatchNorm2d(out_ch))
            decoder_layers.append(nn.ReLU(inplace=True))
            decoder_layers.append(ResidualBlock(out_ch))
            current_channels = out_ch
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 输出层
        self.output_conv = nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1)
    
    def forward(self, hrdem):
        """
        前向传播
        
        Args:
            hrdem: (B, 1, H, W) - 高分辨率DEM
        
        Returns:
            lrdem: (B, 1, H, W) - 模拟的低分辨率DEM
        """
        # 编码
        x = self.encoder(hrdem)
        
        # 瓶颈
        x = self.bottleneck(x)
        
        # 解码
        x = self.decoder(x)
        
        # 输出
        lrdem = self.output_conv(x)

        if lrdem.shape[-2:] != hrdem.shape[-2:]:
            h_out, w_out = lrdem.shape[-2:]
            h_in, w_in = hrdem.shape[-2:]

            # 计算裁剪量（假设 h_out >= h_in, w_out >= w_in）
            h_crop = h_out - h_in  # 2
            w_crop = w_out - w_in  # 2

            top = h_crop // 2  # 1
            bottom = h_crop - top  # 1
            left = w_crop // 2  # 1
            right = w_crop - left  # 1

            lrdem = lrdem[:, :, top:-bottom, left:-right]
        
        return lrdem


class DEMSuperResolutionSystem(nn.Module):
    """
    DEM超分辨率系统
    
    整合所有组件：
    1. DAM模型（带实例分割）
    2. 超分辨率重构网络
    3. HRDEM到LRDEM的映射网络
    """
    
    def __init__(
        self,
        dam_model,
        sr_channels=64,
        sr_residual_blocks=8,
        mapper_base_channels=32,
        use_instance_guidance=True
    ):
        super(DEMSuperResolutionSystem, self).__init__()
        
        # DAM模型（带实例分割）
        self.dam_model = dam_model
        
        # 超分辨率重构网络（支持实例引导）
        self.sr_network = SuperResolutionNetwork(
            in_channels=3,  # Copernicus + relative + instance bias
            base_channels=sr_channels,
            num_residual_blocks=sr_residual_blocks,
            out_channels=1,
            use_instance_guidance=use_instance_guidance
        )
        
        # HRDEM到LRDEM的映射网络
        self.mapper_network = HRDEMToLRDEMMapper(
            in_channels=1,
            base_channels=mapper_base_channels
        )
    
    def forward(self, google_image, copernicus_dem, use_instance_guidance=True):
        """
        前向传播
        
        Args:
            google_image: (B, 3, H, W) - Google Earth影像
            copernicus_dem: (B, 1, H, W) - Copernicus DEM
            use_instance_guidance: bool - 是否使用实例引导
        
        Returns:
            包含以下键的字典：
            - 'hrdem': 高分辨率DEM
            - 'mapped_lrdem': 映射后的低分辨率DEM
            - 'dam_output': DAM模型的完整输出
        """
        # DAM模型生成relative map（带实例分割增强）
        dam_output = self.dam_model(google_image)
        enhanced_depth = dam_output['enhanced_depth'].unsqueeze(1)  # (B, 1, H, W)
        instance_bias_map = dam_output['instance_bias_map'].unsqueeze(1)  # (B, 1, H, W)
        
        # 超分辨率重构网络融合Copernicus DEM、relative map和instance bias
        if use_instance_guidance and self.sr_network.use_instance_guidance:
            hrdem = self.sr_network(copernicus_dem, enhanced_depth, instance_bias_map)
        else:
            hrdem = self.sr_network(copernicus_dem, enhanced_depth, None)
        
        # 映射网络将HRDEM映射回LRDEM
        mapped_lrdem = self.mapper_network(hrdem)
        
        return {
            'hrdem': hrdem,
            'mapped_lrdem': mapped_lrdem,
            'dam_output': dam_output
        }
    
    def freeze_dam(self):
        """冻结DAM模型的参数（除了实例分割头）"""
        for param in self.dam_model.parameters():
            param.requires_grad = False
        
        # 解冻实例分割头
        for param in self.dam_model.instance_head.parameters():
            param.requires_grad = True
        
        # 解冻偏置归一化参数
        self.dam_model.bias_scale.requires_grad = True
        self.dam_model.bias_shift.requires_grad = True
        
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
    use_instance_guidance=True,
    device='cuda'
):
    """
    创建超分辨率系统
    
    Args:
        dam_model: DAM模型实例
        sr_channels: 超分辨率网络的基础通道数
        sr_residual_blocks: 超分辨率网络的残差块数量
        mapper_base_channels: 映射网络的基础通道数
        use_instance_guidance: 是否使用实例引导
        device: 设备
    
    Returns:
        system: DEM超分辨率系统
    """
    system = DEMSuperResolutionSystem(
        dam_model=dam_model,
        sr_channels=sr_channels,
        sr_residual_blocks=sr_residual_blocks,
        mapper_base_channels=mapper_base_channels,
        use_instance_guidance=use_instance_guidance
    )
    
    system = system.to(device)
    
    return system


if __name__ == "__main__":
    # 测试超分辨率网络
    print("测试超分辨率网络...")
    sr_net = SuperResolutionNetwork(in_channels=2, base_channels=64, num_residual_blocks=8)
    
    copernicus = torch.randn(2, 1, 1024, 1024)
    relative_map = torch.randn(2, 1, 1024, 1024)
    
    hrdem = sr_net(copernicus, relative_map)
    print(f"HRDEM形状: {hrdem.shape}")
    
    # 测试映射网络
    print("\n测试映射网络...")
    mapper = HRDEMToLRDEMMapper(in_channels=1, base_channels=32)
    
    mapped_lrdem = mapper(hrdem)
    print(f"Mapped LRDEM形状: {mapped_lrdem.shape}")
    
    # 统计参数
    sr_params = sum(p.numel() for p in sr_net.parameters())
    mapper_params = sum(p.numel() for p in mapper.parameters())
    
    print(f"\n超分辨率网络参数: {sr_params:,}")
    print(f"映射网络参数: {mapper_params:,}")
    
    print("\n模型测试完成！")
