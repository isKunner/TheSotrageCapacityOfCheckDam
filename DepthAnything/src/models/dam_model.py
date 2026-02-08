"""
改进的DAM模型 V1.2 - 高斯可变形原型版本

主要改进：
1. 可变形高斯原型：替代Softmax硬分配，消除矩形块状伪影
   - 每个原型有可学习的中心坐标(x,y)和标准差σ
   - 大σ自动覆盖大实例（如水库），小σ覆盖小实例（如建筑）
   - 高斯函数天然连续，边界平滑无硬角

2. 各向异性高斯：σ_x和σ_y独立，适应长条形地物（如河流、道路）

3. 完全兼容原接口：返回元组格式不变，SR无需修改
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .depth_anything_v2.dinov2 import DINOv2
from .depth_anything_v2.dpt import DPTHead, DepthAnythingV2
from .depth_anything_v2.util.blocks import FeatureFusionBlock


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features, nn.ReLU(False), deconv=False, bn=use_bn,
        expand=False, align_corners=True, size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class InstanceSegmentationHead(nn.Module):
    """
    改进的实例分割解码器头（高斯可变形原型版本）

    核心架构：可变形高斯分布替代Softmax硬分配
    总增益 = 类别级基础增益 × 实例级空间残差增益

    核心改进：
    1. 可变形高斯原型：每个原型有中心坐标(x,y)和标准差σ（可学习）
     - 大σ自动覆盖大实例（如水库），小σ覆盖小实例（如建筑）
     - 高斯函数天然连续，消除Softmax导致的块状伪影

    2. 各向异性：σ_x和σ_y独立，适应长条形地物（河流、道路）

    3. 乘性门控（保留）：使用增益而非加性偏置，避免覆盖DAM细节
    """

    def __init__(
        self,
        in_channels,  # DINOv2编码器输出维度（ViT-L为1024）
        features=256,  # 特征融合后的通道数
        use_bn=False,  # 是否使用BatchNorm（通常DEM数据不用）
        out_channels=[256, 512, 1024, 1024],  # DPT解码器4层输出通道
        num_prototypes=16,  # 原型数量
        embedding_dim=32,  # 嵌入维度
        use_clstoken=False,  # 是否使用CLS Token（DINOv2特性）
        bias_scale_init=0.3,  # 允许更大调整范围
        residual_scale_init=0.1,  # 残差增益初始范围（0.1表示±10%微调）
    ):
        super().__init__()

        self.use_clstoken = use_clstoken
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim
        self.bias_scale_init = bias_scale_init
        self.residual_scale_init = residual_scale_init

        # 高斯温度系数（<1使Softmax更软，消除硬边界）
        self.gaussian_temperature = 0.5

        # 特征投影层
        """
        【设计思路】
        DINOv2输出4个不同分辨率的特征层（类似FPN），需要统一处理：
        - 浅层（高分辨率）：细节丰富，适合边缘检测
        - 深层（低分辨率）：语义丰富，适合实例分类

        通过1×1卷积统一通道数，再通过转置卷积/插值对齐空间尺寸
        """
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1, stride=1, padding=0,
            ) for out_channel in out_channels
        ])

        """
        第一层（浅层特征）：用转置卷积（反卷积）将特征上采样 4 倍
        第二层（中等特征）：用转置卷积将特征上采样 2 倍
        第三层（深层特征）：不做任何操作
        第四层（最深层特征）：使用普通卷积将特征下采样 2 倍
        """
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        """
        CLS Token 是 Vision Transformer（ViT）中的一种特殊 token，通常用于聚合全局信息
        该列表包含多个线性变换层（nn.Linear）和激活函数（nn.GELU），数量等于 self.projects 的长度（即特征层数量）
        输入维度为 2 * in_channels 是因为会将 CLS Token 和图像 patch 特征拼接在一起
        CLS Token 提供全局语义信息，而 patch 特征提供局部细节信息
        通过将两者拼接并经过线性变换，可以实现全局与局部特征的有效融合
        由于 DINOv2 输出多层特征（如 4 层），每一层都需要独立处理 CLS Token，因此使用 ModuleList 来管理多个处理模块
        """
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

        """
        DINOv2 编码器输出多个层级的特征图（如浅层细节特征和深层语义特征） (B, features, H, W)
        通过 fusion_conv 将这些特征图在通道维度上拼接后进行融合，生成一个综合性的特征表示
        """
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(out_channels), features, kernel_size=1),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

        """
        用于生成像素级别的嵌入表示 (B, embedding_dim, H, W)
        生成的像素嵌入表示会被用于计算每个像素与原型向量之间的相似度（见 forward 方法中的 similarity 计算）
        这些相似度进一步用于实例分割任务，例如生成原型激活图和实例增益图。
        1. 计算与原型向量的相似度（软分配）
        2. 提取边缘信息（通过梯度计算edge_map）
        """
        self.pixel_embedding = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, embedding_dim, kernel_size=3, padding=1),
        )

        # ========== 第一层：类别级偏置（改进为高斯可变形原型） ==========
        """
        【核心改进说明】
        原方案使用Softmax硬分配（点积+Softmax），导致：
        - 硬边界（winner-takes-all）
        - 矩形块状伪影
        - 感受野受限（原型向量是全局的，无法自适应大小）
        
        新方案使用可变形高斯分布：
        - 每个原型有中心坐标(x,y)和标准差σ（可学习）
        - σ自动适应实例大小：大σ覆盖大实例（如水库），小σ覆盖小实例（如建筑）
        - 高斯函数天然连续，消除硬边界
        - 各向异性（σ_x, σ_y独立），适应长条形地物（河流、道路）
        
        数学公式：
        activation_i(x,y) = exp(-((x-cx_i)²/(2σ_x²) + (y-cy_i)²/(2σ_y²)))
        再经温度Softmax归一化（T=0.5使边界更软）
        """

        # 可学习参数：原型中心坐标，范围[0,1]
        # 初始化：均匀网格分布避免聚集
        grid_size = int(math.sqrt(num_prototypes))
        x = torch.linspace(0.2, 0.8, grid_size).repeat(grid_size)
        y = torch.linspace(0.2, 0.8, grid_size).repeat_interleave(grid_size)
        if len(x) < num_prototypes:
            x = torch.cat([x, torch.rand(num_prototypes - len(x))])
            y = torch.cat([y, torch.rand(num_prototypes - len(y))])
        self.prototype_centers = nn.Parameter(torch.stack([x[:num_prototypes], y[:num_prototypes]], dim=1))

        # 可学习参数：对数标准差（控制范围），各向异性(σ_x, σ_y)
        # 初始σ=exp(-2)≈0.13（覆盖约13%图像），训练后可自适应增大到0.5（覆盖半图）
        self.prototype_sigma = nn.Parameter(torch.ones(num_prototypes, 2) * -2.0)

        # 基础偏置值（每个原型一个）
        self.prototype_biases = nn.Parameter(torch.zeros(num_prototypes))

        # ========== 第二层：实例空间残差 ==========
        # 【修改】使用共享编码器减少参数量，避免每个原型独立网络导致的碎片化
        self.spatial_residual_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        # 每个原型独立的解码器（轻量）
        self.spatial_residual_decoder = nn.ModuleList([
            nn.Conv2d(8, 1, kernel_size=3, padding=1) for _ in range(num_prototypes)
        ])

        """
        残差缩放因子（控制空间变化的强度）
        限制残差的变化范围，防止过度扰动原始特征
        """
        self.spatial_residual_scale = nn.Parameter(torch.tensor(residual_scale_init))

        # 空间平滑卷积（鼓励相邻像素属于同一实例）
        self.spatial_smooth = nn.Conv2d(
            num_prototypes, num_prototypes,
            kernel_size=3, padding=1, groups=1
        )

    def forward(self, out_features, patch_h, patch_w):
        """
        前向传播

        Args:
            out_features: 编码器输出的特征列表
            patch_h, patch_w: patch的高度和宽度

        Returns:
            instance_bias_map: 实例增益图（乘性门控），shape (B, H, W)
            prototype_activations: 原型激活图，shape (B, num_prototypes, H, W)
            base_biases: 每个原型的基础增益值，shape (B, num_prototypes)
            spatial_residual_map: 空间残差图（用于可视化），shape (B, H, W)
            pixel_embeddings: 像素嵌入，shape (B, embedding_dim, H, W)
            activation_entropy: 占位符（保持接口兼容），scalar
        """
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]

            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        # 特征融合
        target_size = out[0].shape[-2:]
        resized_features = []
        for feat in out:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=True)
            resized_features.append(feat)

        fused = torch.cat(resized_features, dim=1)
        path_1 = self.fusion_conv(fused)

        # 上采样到最终分辨率
        H, W = int(patch_h * 14), int(patch_w * 14)
        path_1 = F.interpolate(path_1, size=(H, W), mode="bilinear", align_corners=True)

        # 生成像素嵌入
        pixel_embeddings = self.pixel_embedding(path_1)  # (B, embedding_dim, H, W)
        B, _, H, W = pixel_embeddings.shape

        # 【核心改进】计算高斯激活（替代原有点积+Softmax）
        device = pixel_embeddings.device

        # 1. 生成归一化坐标网格 [0,1]
        y_coords = torch.linspace(0, 1, H, device=device)
        x_coords = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).view(1, H*W, 2)  # (1, HW, 2)

        # 2. 计算到各原型的高斯距离（向量化）
        # centers: (num_prototypes, 2) -> (1, N, 1, 2)
        centers = self.prototype_centers.view(1, self.num_prototypes, 1, 2)
        coords_exp = coords.view(1, 1, H*W, 2)  # (1, 1, HW, 2)

        # 各向异性标准差 σ_x, σ_y
        sigma = torch.exp(self.prototype_sigma).view(1, self.num_prototypes, 1, 2)  # 确保>0

        # 马氏距离平方: (Δx/σ_x)² + (Δy/σ_y)²
        diff = coords_exp - centers  # 广播: (1, N, HW, 2)
        dist_sq = (diff ** 2) / (sigma ** 2 + 1e-6)
        dist_sq = dist_sq.sum(dim=-1)  # (1, N, HW)

        # 3. 高斯函数（连续激活，无硬边界）
        gaussian_sim = torch.exp(-dist_sq / 2.0)  # (1, N, HW)

        # 4. Softmax归一化（温度系数T=0.5使边界更软，消除块状）
        # T<1: 分布更均匀，减少winner-takes-all导致的硬分配
        similarity = gaussian_sim / math.sqrt(self.embedding_dim)
        prototype_activations = F.softmax(similarity / self.gaussian_temperature, dim=1)
        prototype_activations = prototype_activations.expand(B, -1, -1)  # (B, N, HW)
        prototype_activations = prototype_activations.view(B, self.num_prototypes, H, W)

        # 应用空间平滑卷积并reshape回图像
        similarity_map = prototype_activations
        similarity_map = self.spatial_smooth(similarity_map)
        # 重新归一化（softmax后经过卷积需要重新归一化）
        similarity_flat = similarity_map.reshape(B, self.num_prototypes, -1)  # (B, num_prototypes, H*W)
        similarity_flat = F.softmax(similarity_flat / self.gaussian_temperature, dim=1)  # 在num_prototypes维度做softmax
        prototype_activations = similarity_flat.view(B, self.num_prototypes, H, W)
        prototype_activations_img = prototype_activations

        # ========== 计算类别级增益（乘性门控）==========
        # 【修改】转换为增益系数：1.0 + tanh(bias) * scale，范围[1-scale, 1+scale]
        base_biases = 1.0 + torch.tanh(self.prototype_biases) * self.bias_scale_init  # (num_prototypes,)
        base_biases = base_biases.unsqueeze(0).expand(B, -1)  # (B, num_prototypes)

        # 加权得到基础增益图（加权平均，保持范围在[1-scale, 1+scale]）
        base_gain_flat = torch.einsum('bnp,bp->bn',
            prototype_activations.view(B, H*W, self.num_prototypes),  # (B, HW, N)
            base_biases
        )
        base_gain = base_gain_flat.reshape(B, 1, H, W)

        # ========== 计算实例空间残差（高频细节增益）==========
        # 将原型激活图reshape回图像形状
        spatial_residuals = []
        for i in range(self.num_prototypes):
            # 提取第i个原型的激活图
            proto_act = prototype_activations_img[:, i:i+1, :, :]  # (B, 1, H, W)
            # 生成空间残差（共享编码器+独立解码器）
            feat = self.spatial_residual_encoder(proto_act)
            residual = self.spatial_residual_decoder[i](feat)  # (B, 1, H, W)
            spatial_residuals.append(residual)

        spatial_residuals = torch.cat(spatial_residuals, dim=1)  # (B, num_prototypes, H, W)

        # 根据激活加权得到最终的空间残差图
        weighted_residual = (prototype_activations_img * spatial_residuals).sum(dim=1, keepdim=True)

        # 【修改】转换为微调增益：1.0 + tanh(residual) * scale，范围[1-scale, 1+scale]
        residual_gain = 1.0 + torch.tanh(weighted_residual) * self.spatial_residual_scale

        # ========== 乘性门控融合 ==========
        # 【修改】总增益 = 基础增益 * 残差增益，范围大致在[(1-s1)(1-s2), (1+s1)(1+s2)]
        instance_bias_map = base_gain * residual_gain  # (B, 1, H, W)
        # 限制范围防止极端值（如0.5~2.0）
        instance_bias_map = torch.clamp(instance_bias_map, 0.5, 2.0)

        # 保持接口兼容：返回元组（与原代码一致）
        return (
            instance_bias_map.squeeze(1),      # (B, H, W)
            prototype_activations_img,          # (B, N, H, W)
            base_biases,                        # (B, N)
            weighted_residual.squeeze(1),       # (B, H, W)
            pixel_embeddings,                   # (B, D, H, W)
            torch.tensor(0.0, device=device),    # 占位符（保持长度兼容）
            torch.exp(self.prototype_sigma).detach(),  # ← 第7个：gaussian_params
        )


class DepthAnythingV2WithInstance(DepthAnythingV2):
    """改进的Depth Anything V2模型（继承自原始DAM）"""

    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_prototypes=16,
        embedding_dim=32,
        freeze_encoder=True,
        freeze_original_decoder=True,
        bias_scale_init=0.3,
        residual_scale_init=0.1,
    ):
        # 调用父类初始化（原始DAM）
        super().__init__(
            encoder=encoder,
            features=features,
            out_channels=out_channels,
            use_bn=use_bn,
            use_clstoken=use_clstoken
        )

        # 冻结编码器权重
        if freeze_encoder:
            for param in self.pretrained.parameters():
                param.requires_grad = False
            print("编码器权重已冻结")

        # 冻结原始解码器权重
        if freeze_original_decoder:
            for param in self.depth_head.parameters():
                param.requires_grad = False
            print("原始解码器权重已冻结")

        # 改进的实例分割头（高斯可变形原型）
        self.instance_head = InstanceSegmentationHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            num_prototypes=num_prototypes,
            embedding_dim=embedding_dim,
            use_clstoken=use_clstoken,
            bias_scale_init=bias_scale_init,
            residual_scale_init=residual_scale_init,
        )

        # 偏置归一化参数（乘性门控下可能不需要，但保留以防万一）
        self.norm_min = nn.Parameter(torch.tensor(0.0))
        self.norm_max = nn.Parameter(torch.tensor(1.0))

    def get_encoder_features(self, x):
        """获取Encoder输出特征（用于缓存）"""
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        
        return features, patch_h, patch_w

    def forward_from_features(self, features, patch_h, patch_w):
        """
        从预计算的特征进行前向传播（兼容旧接口）
        
        Args:
            features: Encoder输出特征
            patch_h, patch_w: patch尺寸
        
        Returns:
            dict: 包含所有输出的字典
        """
        # 原始Decoder（与原始DAM完全一致）
        original_depth = self.depth_head(features, patch_h, patch_w)
        original_depth = F.relu(original_depth)
        
        # Instance Head
        outputs = self.instance_head(features, patch_h, patch_w)
        instance_bias_map = outputs[0].unsqueeze(1)
        prototype_activations = outputs[1]
        base_biases = outputs[2]
        spatial_residual = outputs[3]
        pixel_embeddings = outputs[4]
        
        # 归一化并融合
        B = original_depth.size(0)
        orig_min = original_depth.view(B, -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        orig_max = original_depth.view(B, -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        orig_range = torch.where(orig_max - orig_min < 1e-6, torch.ones_like(orig_max), orig_max - orig_min)
        original_depth_norm = (original_depth - orig_min) / orig_range
        original_depth_norm = torch.clamp(original_depth_norm, 0, 1)
        
        enhanced_depth = original_depth_norm * instance_bias_map
        enhanced_depth = torch.clamp(enhanced_depth, 0, 1)
        batch_min = enhanced_depth.view(enhanced_depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        batch_max = enhanced_depth.view(enhanced_depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        range_val = torch.where(batch_max - batch_min < 1e-6, torch.ones_like(batch_max), batch_max - batch_min)
        enhanced_depth = (enhanced_depth - batch_min) / range_val
        enhanced_depth = enhanced_depth * self.norm_max + self.norm_min
        enhanced_depth = torch.clamp(enhanced_depth, 0, 1)
        
        return {
            'original_depth': original_depth_norm.squeeze(1),  # 原始DAM输出
            'enhanced_depth': enhanced_depth.squeeze(1),
            'instance_bias_map': instance_bias_map.squeeze(1),
            'prototype_activations': prototype_activations,
            'base_biases': base_biases,
            'spatial_residual': spatial_residual,
            'pixel_embeddings': pixel_embeddings,
            'activation_entropy': torch.tensor(0.0, device=instance_bias_map.device),
            'gaussian_sigma': outputs[6] if len(outputs) > 6 else None,
        }
    
    def forward(self, x):
        """
        前向传播（与原始DAM接口一致，但返回字典）
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        # Encoder（与原始DAM完全一致）
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        
        return self.forward_from_features(features, patch_h, patch_w)


def create_dam_model(
    encoder='vitl',
    pretrained_path=None,
    num_prototypes=16,
    embedding_dim=32,
    device='cuda',
    bias_scale_init=0.3,
    residual_scale_init=0.1,
):
    """创建改进的DAM模型V2"""
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2WithInstance(
        encoder=model_configs[encoder]['encoder'],
        features=model_configs[encoder]['features'],
        out_channels=model_configs[encoder]['out_channels'],
        use_bn=False,
        use_clstoken=False,
        num_prototypes=num_prototypes,
        embedding_dim=embedding_dim,
        freeze_encoder=True,
        freeze_original_decoder=True,
        bias_scale_init=bias_scale_init,
        residual_scale_init=residual_scale_init,
    )

    if pretrained_path is not None:
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print("缺失的权重：", missing_keys)
        print("未使用的权重：", unexpected_keys)
        print("预训练权重加载完成")

    model = model.to(device)
    return model