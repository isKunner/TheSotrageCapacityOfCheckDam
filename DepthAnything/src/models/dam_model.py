"""
改进的DAM模型 V2

主要改进：
1. 双层BiasMap设计：类别偏置 + 实例空间残差
2. 支持渐变蒙版（如斜面校正）
3. 值域控制优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_anything_v2.dinov2 import DINOv2
from .depth_anything_v2.dpt import DPTHead
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
    改进的实例分割解码器头（双层BiasMap设计）

    核心架构：双层增益调制
    总增益 = 类别级基础增益 × 实例级空间残差增益

    1. 第一层（类别级）：每个原型学习一个全局缩放因子（类似类别偏置）
    2. 第二层（实例级）：每个原型学习一个空间变化图（类似实例蒙版）

    核心改进：
    1. 类别级偏置：每个原型有一个基础偏置值（全局） -> 乘性门控（Multiplicative Gating）：使用增益而非加性偏置，避免覆盖DAM细节
    2. 实例空间残差：每个原型学习一个轻量级空间变化生成器（局部）

    这样设计的好处：
    - 同一类别的实例共享相同的基础偏置
    - 每个实例内部可以有空间变化（如斜面校正）
    - 残差范围可控（Tanh限制在[-1, 1]）

    """

    def __init__(
        self,
        in_channels,  # DINOv2编码器输出维度（ViT-L为1024）
        features=256,  # 特征融合后的通道数
        use_bn=False,  # 是否使用BatchNorm（通常DEM数据不用）
        out_channels=[256, 512, 1024, 1024],  # DPT解码器4层输出通道
        num_prototypes=64,  # 原型数量（即最大实例类别数）
        embedding_dim=32,  # Pixel Embedding维度（用于计算原型相似度）
        use_clstoken=False,  # 是否使用CLS Token（DINOv2特性）
        bias_scale_init=0.2,  # 基础增益初始范围（0.2表示±20%调整）
        residual_scale_init=0.1,  # 残差增益初始范围（0.1表示±10%微调）
    ):
        super().__init__()

        self.use_clstoken = use_clstoken
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

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

        """
        这是一个可学习的参数矩阵，形状为 (num_prototypes, embedding_dim)
        pixel_flat 的形状是 (B, H*W, embedding_dim)
        torch.matmul(pixel_flat, self.prototypes.T) 
        结果是一个形状为 (B, H*W, num_prototypes) 的张量，表示每个像素与每个原型的相似度
        
        num_prototypes 定义了模型中可学习原型向量的数量，每个原型向量代表一个潜在的实例类别
        例如，如果 num_prototypes=64，则模型会学习 64 个原型向量，每个原型向量用于表示一种可能的实例模式
        每一行代表一个原型向量，用于与像素嵌入（pixel_embeddings）计算相似度，从而实现像素级别的实例分类
        通过计算像素嵌入与原型向量之间的相似度，确定每个像素属于哪个实例类别
        相似度越高，表示该像素越可能属于对应的原型类别
        prototypes 是模型中用于实例分割的关键组件，通过学习像素与原型之间的相似性关系，实现对图像中不同实例的识别和分割
        """
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        nn.init.normal_(self.prototypes, mean=0, std=0.01)

        # ========== 第一层：类别级偏置 ==========
        """
        每个原型拥有独立的基础增益值，同类实例共享相同的偏置
        这是一个可学习的参数向量，形状为 (num_prototypes,)
        每个元素对应一个原型（prototype）的基础偏置值，用于调节该原型的全局增益
        初始化为全零向量，通过 tanh 函数后会趋近于 0，使得初始增益接近 1
        """
        self.prototype_biases = nn.Parameter(torch.zeros(num_prototypes))

        """
        这是一个全局缩放因子，控制所有原型偏置的整体强度
        初始值由 bias_scale_init 指定（默认为 0.2），表示最大调整比例为 ±20%
        通过乘性门控机制，将偏置值映射到 [1 - bias_scale, 1 + bias_scale] 范围内，避免覆盖原始细节信息
        """
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale_init))

        # ========== 第二层：实例空间残差 ==========
        # 【修改】使用残差连接保留高频细节
        """
        这是一个 ModuleList，包含多个轻量级的卷积网络（每个原型对应一个）
        每个原型学习一个独立的空间变化模式（如斜面校正等）
        移除了 Tanh 激活函数，在 forward 中通过 torch.tanh 处理，便于动态控制残差范围
        """
        self.spatial_residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
                # 【修改】移除Tanh，在forward中通过torch.tanh处理，便于控制
            ) for _ in range(num_prototypes)
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
            instance_bias_map: 实例增益图（乘性门控），shape (B, 1, H, W)
            prototype_activations: 原型激活图，shape (B, num_prototypes, H, W)
            base_biases: 每个原型的基础增益值，shape (B, num_prototypes)
            spatial_residual_map: 空间残差图（用于可视化），shape (B, 1, H, W)
            pixel_embeddings: 像素嵌入，shape (B, embedding_dim, H, W)
            activation_entropy: 激活熵（用于正则化），scalar
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

        # 计算像素与原型向量的相似度
        pixel_flat = pixel_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, self.embedding_dim)
        similarity = torch.matmul(pixel_flat, self.prototypes.T)
        similarity = similarity / (self.embedding_dim ** 0.5)

        # 【修改】边缘感知平滑：基于pixel embeddings的梯度检测边缘
        similarity_map = similarity.permute(0, 2, 1).reshape(B, self.num_prototypes, H, W)

        # 计算pixel embeddings的梯度作为边缘指示器（近似canny边缘）
        grad_x = torch.abs(pixel_embeddings[:, :, :, :-1] - pixel_embeddings[:, :, :, 1:])
        grad_y = torch.abs(pixel_embeddings[:, :, :-1, :] - pixel_embeddings[:, :, 1:, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        edge_map = (grad_x.mean(dim=1, keepdim=True) + grad_y.mean(dim=1, keepdim=True)) / 2.0
        # 边缘掩码：高梯度区域（边缘）保持原值，低梯度区域（内部）平滑
        edge_mask = (edge_map > edge_map.mean()).float()

        # 双边滤波式平滑：非边缘区域进行空间平滑，边缘区域保持锐利的激活边界
        for _ in range(2):  # 迭代2次
            smooth = F.avg_pool2d(similarity_map, kernel_size=3, stride=1, padding=1)
            similarity_map = edge_mask * similarity_map + (1 - edge_mask) * smooth

        similarity = similarity_map.reshape(B, self.num_prototypes, H * W).permute(0, 2, 1)

        # Softmax获取每个像素的原型归属概率
        prototype_activations = F.softmax(similarity, dim=-1)  # (B, H*W, num_prototypes)

        # 【修改】计算激活熵（用于正则化损失，鼓励软分配）
        entropy = -torch.sum(prototype_activations * torch.log(prototype_activations + 1e-8), dim=-1)
        activation_entropy = entropy.mean()

        # 应用空间平滑卷积并reshape回图像
        similarity_map = similarity.permute(0, 2, 1).reshape(B, self.num_prototypes, H, W)
        similarity_map = self.spatial_smooth(similarity_map)
        # 重新归一化（softmax后经过卷积需要重新归一化）
        similarity_flat = similarity_map.reshape(B, self.num_prototypes, -1)  # (B, num_prototypes, H*W)
        similarity_flat = F.softmax(similarity_flat, dim=1)  # 在num_prototypes维度做softmax
        prototype_activations = similarity_flat.permute(0, 2, 1)  # (B, H*W, num_prototypes)
        prototype_activations_img = prototype_activations.permute(0, 2, 1).reshape(B, self.num_prototypes, H, W)

        # ========== 计算类别级增益（乘性门控）==========
        # 【修改】转换为增益系数：1.0 + tanh(bias) * scale，范围[1-scale, 1+scale]
        base_biases = 1.0 + torch.tanh(self.prototype_biases) * self.bias_scale  # (num_prototypes,)
        base_biases = base_biases.unsqueeze(0).expand(B, -1)  # (B, num_prototypes)

        # 加权得到基础增益图（加权平均，保持范围在[1-scale, 1+scale]）
        base_gain_flat = torch.einsum('bnp,bp->bn', prototype_activations, base_biases)
        base_gain = base_gain_flat.reshape(B, 1, H, W)

        # ========== 计算实例空间残差（高频细节增益）==========
        # 将原型激活图reshape回图像形状
        spatial_residuals = []
        for i in range(self.num_prototypes):
            # 提取第i个原型的激活图
            proto_act = prototype_activations_img[:, i:i+1, :, :]  # (B, 1, H, W)
            # 生成空间残差（未激活tanh）
            residual = self.spatial_residual_heads[i](proto_act)  # (B, 1, H, W)
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

        return instance_bias_map, prototype_activations_img, base_biases, weighted_residual, pixel_embeddings, activation_entropy


class DepthAnythingV2WithInstance(nn.Module):
    """改进的Depth Anything V2模型（V2版本）"""

    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_prototypes=128,
        embedding_dim=64,
        freeze_encoder=True,
        freeze_original_decoder=True,
        bias_scale_init=0.2,  # 【修改】配合乘性门控
        residual_scale_init=0.1,
    ):
        super().__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.encoder = encoder

        # 加载预训练的DINOv2编码器
        self.pretrained = DINOv2(model_name=encoder)

        # 冻结编码器权重
        if freeze_encoder:
            for param in self.pretrained.parameters():
                param.requires_grad = False
            print("编码器权重已冻结")

        # 原始DPT解码器
        self.depth_head = DPTHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken
        )

        # 冻结原始解码器权重
        if freeze_original_decoder:
            for param in self.depth_head.parameters():
                param.requires_grad = False
            print("原始解码器权重已冻结")

        # 改进的实例分割头
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
        """获取Encoder输出特征"""
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        with torch.no_grad():
            features = self.pretrained.get_intermediate_layers(
                x,
                self.intermediate_layer_idx[self.encoder],
                return_class_token=True
            )

        return features, patch_h, patch_w

    def forward_from_features(self, features, patch_h, patch_w):
        """从预计算的Encoder特征进行前向传播"""
        # 检查特征
        for i, feat in enumerate(features):
            if isinstance(feat, (list, tuple)):
                for j, f in enumerate(feat):
                    if torch.isnan(f).any():
                        print(f"警告: 编码器特征 [{i}][{j}] 包含NaN")
            else:
                if torch.isnan(feat).any():
                    print(f"警告: 编码器特征 [{i}] 包含NaN")

        # 原始解码器生成relative depth map
        with torch.cuda.amp.autocast(enabled=False):
            with torch.set_grad_enabled(False):
                features_fp32 = []
                for feat in features:
                    if isinstance(feat, (list, tuple)):
                        features_fp32.append([f.float() if f.dtype == torch.float16 else f for f in feat])
                    else:
                        features_fp32.append(feat.float() if feat.dtype == torch.float16 else feat)

                original_depth = self.depth_head(features_fp32, patch_h, patch_w)
                original_depth = F.relu(original_depth)

                if torch.isnan(original_depth).any():
                    print(f"警告: depth_head 输出包含NaN")
                    original_depth = torch.where(
                        torch.isnan(original_depth),
                        torch.zeros_like(original_depth),
                        original_depth
                    )

        # 实例分割头生成实例增益图（乘性门控）
        with torch.cuda.amp.autocast(enabled=False):
            instance_bias_map, prototype_activations, base_biases, spatial_residual, pixel_embeddings, activation_entropy = \
                self.instance_head(features_fp32, patch_h, patch_w)
            instance_bias_map = instance_bias_map.float()
            activation_entropy = activation_entropy.float()

        # 归一化并融合（乘性门控）
        with torch.cuda.amp.autocast(enabled=False):
            # 将original_depth归一化到0~1范围（AMP友好的实现）
            B = original_depth.size(0)
            orig_min = original_depth.view(B, -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            orig_max = original_depth.view(B, -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            orig_range = torch.where(orig_max - orig_min < 1e-6, torch.ones_like(orig_max), orig_max - orig_min)
            original_depth_norm = (original_depth - orig_min) / orig_range
            original_depth_norm = torch.clamp(original_depth_norm, 0, 1)

            # 【修改】乘性门控融合：细节增益调制（替代加性）
            enhanced_depth = original_depth_norm * instance_bias_map

            # 再次归一化（乘性后可能超界）
            enhanced_depth = torch.clamp(enhanced_depth, 0, 1)
            batch_min = enhanced_depth.view(enhanced_depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            batch_max = enhanced_depth.view(enhanced_depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            range_val = torch.where(batch_max - batch_min < 1e-6, torch.ones_like(batch_max), batch_max - batch_min)
            enhanced_depth = (enhanced_depth - batch_min) / range_val
            enhanced_depth = enhanced_depth * self.norm_max + self.norm_min
            enhanced_depth = torch.clamp(enhanced_depth, 0, 1)

        return {
            'enhanced_depth': enhanced_depth.squeeze(1),
            'original_depth': original_depth_norm.squeeze(1),
            'instance_bias_map': instance_bias_map.squeeze(1),  # 现在是增益图
            'prototype_activations': prototype_activations,
            'base_biases': base_biases,
            'spatial_residual': spatial_residual.squeeze(1),
            'pixel_embeddings': pixel_embeddings,
            'activation_entropy': activation_entropy,  # 【修改】返回熵用于正则
        }

    def forward(self, x):
        """前向传播"""
        features, patch_h, patch_w = self.get_encoder_features(x)
        return self.forward_from_features(features, patch_h, patch_w)


def create_dam_model(
    encoder='vitl',
    pretrained_path=None,
    num_prototypes=128,
    embedding_dim=64,
    device='cuda',
    bias_scale_init=0.2,  # 【修改】配合乘性门控
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
        model.load_state_dict(checkpoint, strict=False)
        print("预训练权重加载完成")

    model = model.to(device)
    return model