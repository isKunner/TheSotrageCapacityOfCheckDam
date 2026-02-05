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

    核心改进：
    1. 类别级偏置：每个原型有一个基础偏置值（全局）
    2. 实例空间残差：每个原型学习一个轻量级空间变化生成器（局部）

    这样设计的好处：
    - 同一类别的实例共享相同的基础偏置
    - 每个实例内部可以有空间变化（如斜面校正）
    - 残差范围可控（Tanh限制在[-1, 1]）
    """

    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        num_prototypes=128,
        embedding_dim=64,
        use_clstoken=False,
        bias_scale_init=1.0,  # 增大初始值
        residual_scale_init=0.1,
    ):
        super().__init__()

        self.use_clstoken = use_clstoken
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

        # 特征投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1, stride=1, padding=0,
            ) for out_channel in out_channels
        ])

        # 尺寸调整层
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU()))

        # 特征融合结构
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(out_channels), features, kernel_size=1),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

        # 像素嵌入网络
        self.pixel_embedding = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, embedding_dim, kernel_size=3, padding=1),
        )

        # 可学习的原型向量
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        nn.init.normal_(self.prototypes, mean=0, std=0.01)

        # ========== 第一层：类别级偏置 ==========
        # 每个原型（类别）有一个基础偏置值
        self.prototype_biases = nn.Parameter(torch.zeros(num_prototypes))

        # 全局缩放因子（控制整体偏置强度）
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale_init))

        # ========== 第二层：实例空间残差 ==========
        # 每个原型学习一个轻量级的空间残差生成器
        # 输入：该原型的激活图 (B, 1, H, W)
        # 输出：空间残差 (B, 1, H, W)，范围[-1, 1]
        self.spatial_residual_heads = nn.ModuleList([
            nn.Sequential(
                # 轻量级设计：少量参数，避免过拟合
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8, 1, kernel_size=3, padding=1),
                nn.Tanh(),  # 限制残差范围 [-1, 1]
            ) for _ in range(num_prototypes)
        ])

        # 残差缩放因子（控制空间变化的强度）
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
            instance_bias_map: 实例偏置图，shape (B, 1, H, W)
            prototype_activations: 原型激活图，shape (B, num_prototypes, H, W)
            base_biases: 每个原型的基础偏置值，shape (B, num_prototypes)
            spatial_residual_map: 空间残差图（用于可视化），shape (B, 1, H, W)
            pixel_embeddings: 像素嵌入，shape (B, embedding_dim, H, W)
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

        # 应用空间平滑
        similarity_map = similarity.permute(0, 2, 1).reshape(B, self.num_prototypes, H, W)
        similarity_map = self.spatial_smooth(similarity_map)
        similarity = similarity_map.reshape(B, self.num_prototypes, H * W).permute(0, 2, 1)

        # Softmax获取每个像素的原型归属概率
        prototype_activations = F.softmax(similarity, dim=-1)  # (B, H*W, num_prototypes)

        # ========== 计算类别级偏置 ==========
        # 每个原型的基础偏置
        base_biases = self.prototype_biases * self.bias_scale  # (num_prototypes,)
        base_biases = base_biases.unsqueeze(0).expand(B, -1)  # (B, num_prototypes)

        # 加权得到基础偏置图
        base_bias_flat = torch.einsum('bnp,bp->bn', prototype_activations, base_biases)
        base_bias = base_bias_flat.reshape(B, 1, H, W)

        # ========== 计算实例空间残差 ==========
        # 将原型激活图reshape回图像形状
        prototype_activations_img = prototype_activations.permute(0, 2, 1).reshape(B, self.num_prototypes, H, W)

        spatial_residuals = []
        for i in range(self.num_prototypes):
            # 提取第i个原型的激活图
            proto_act = prototype_activations_img[:, i:i+1, :, :]  # (B, 1, H, W)
            # 生成空间残差
            residual = self.spatial_residual_heads[i](proto_act)  # (B, 1, H, W)
            spatial_residuals.append(residual)

        spatial_residuals = torch.cat(spatial_residuals, dim=1)  # (B, num_prototypes, H, W)

        # 根据激活加权得到最终的空间残差图
        weighted_residual = (prototype_activations_img * spatial_residuals).sum(dim=1, keepdim=True)
        weighted_residual = weighted_residual * self.spatial_residual_scale

        # ========== 合并两层偏置 ==========
        instance_bias_map = base_bias + weighted_residual  # (B, 1, H, W)

        return instance_bias_map, prototype_activations_img, base_biases, weighted_residual, pixel_embeddings


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
        bias_scale_init=1.0,
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

        # 偏置归一化参数
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

        # 实例分割头生成实例偏置图
        with torch.cuda.amp.autocast(enabled=False):
            instance_bias_map, prototype_activations, base_biases, spatial_residual, pixel_embeddings = \
                self.instance_head(features_fp32, patch_h, patch_w)
            instance_bias_map = instance_bias_map.float()

        # 归一化并融合
        with torch.cuda.amp.autocast(enabled=False):
            # 将original_depth归一化到0~1
            orig_min = original_depth.view(original_depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            orig_max = original_depth.view(original_depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            orig_range = torch.where(orig_max - orig_min < 1e-6, torch.ones_like(orig_max), orig_max - orig_min)
            original_depth_norm = (original_depth - orig_min) / orig_range
            original_depth_norm = torch.clamp(original_depth_norm, 0, 1)

            # 在归一化空间相加
            enhanced_depth = original_depth_norm + instance_bias_map

            # 再次归一化
            enhanced_depth = F.relu(enhanced_depth)
            batch_min = enhanced_depth.view(enhanced_depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            batch_max = enhanced_depth.view(enhanced_depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            range_val = torch.where(batch_max - batch_min < 1e-6, torch.ones_like(batch_max), batch_max - batch_min)
            enhanced_depth = (enhanced_depth - batch_min) / range_val
            enhanced_depth = enhanced_depth * self.norm_max + self.norm_min
            enhanced_depth = torch.clamp(enhanced_depth, 0, 1)

        return {
            'enhanced_depth': enhanced_depth.squeeze(1),
            'original_depth': original_depth_norm.squeeze(1),
            'instance_bias_map': instance_bias_map.squeeze(1),
            'prototype_activations': prototype_activations,
            'base_biases': base_biases,
            'spatial_residual': spatial_residual.squeeze(1),  # 新增：用于可视化
            'pixel_embeddings': pixel_embeddings,
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
    bias_scale_init=1.0,
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