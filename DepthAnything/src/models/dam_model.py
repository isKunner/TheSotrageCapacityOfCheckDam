"""
修改后的DAM (Depth Anything Model) v2

主要修改：
1. 冻结原始编码器和解码器的权重
2. 添加一个新的实例分割解码器头
3. 实例分割头输出的是每个实例对应的偏置项
4. 将偏置项加到原始解码器的relative map上，进行归一化处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
import cv2
from torchvision.transforms import Compose

# 导入原始DAM v2的组件
from .depth_anything_v2.dinov2 import DINOv2
from .depth_anything_v2.dpt import DPTHead
from .depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from .depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    """创建特征融合块"""
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    """卷积块"""
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
    实例分割解码器头，加入聚类属性

    功能：
    1. 使用原型向量表示不同的地形模式
    2. 偏置项范围受控（tanh激活 + 可学习缩放）
    3. 支持空间连续性约束
    4. 真正的实例级偏置（相同实例区域共享相同偏置）

    结构：
    - 从编码器特征中提取像素嵌入
    - 计算像素与原型向量的相似度
    - 为每个原型预测一个偏置值
    - 根据相似度加权合成偏置图
    """

    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        num_prototypes=128,  # 最大实例数量
        embedding_dim=64,  # 嵌入维度
        use_clstoken=False,
        bias_scale_init=0.1  # 偏置缩放初始值
    ):
        super(InstanceSegmentationHead, self).__init__()

        self.use_clstoken = use_clstoken
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim

        # 特征投影层
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        # 尺寸调整层
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))

        # 特征融合结构
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(sum(out_channels), features, kernel_size=1),
            nn.BatchNorm2d(features) if use_bn else nn.Identity(),
            nn.ReLU(True)
        )

        # 像素嵌入网络（将特征转换为嵌入向量）
        self.pixel_embedding = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(features // 2),
            nn.ReLU(True),
            nn.Conv2d(features // 2, embedding_dim, kernel_size=3, padding=1),
        )

        # 可学习的原型向量
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        nn.init.normal_(self.prototypes, mean=0, std=0.01)

        # 偏置值预测网络（每个原型预测一个偏置值）
        # 使用tanh限制输出范围在[-1, 1]
        self.bias_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(True),
            nn.Linear(embedding_dim // 2, 1),
            nn.Tanh()  # 限制输出范围
        )

        # 可学习的偏置缩放因子（控制偏置强度）
        self.bias_scale = nn.Parameter(torch.tensor(bias_scale_init))

        # 空间连续性约束卷积（鼓励相邻像素属于同一实例）
        self.spatial_smooth = nn.Conv2d(num_prototypes, num_prototypes, kernel_size=3, padding=1, groups=1)


    def forward(self, out_features, patch_h, patch_w):
        """
        前向传播

        Args:
            out_features: 编码器输出的特征列表
            patch_h, patch_w: patch的高度和宽度

        Returns:
            instance_bias_map: 实例偏置图，shape (B, 1, H, W)
            instance_logits: 实例分割logits，shape (B, num_prototypes, H, W)
            instance_biases: 每个实例的偏置值，shape (B, num_prototypes)
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
        # 将所有特征上采样到相同尺寸
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
        # pixel_embeddings: (B, embedding_dim, H, W) -> (B, H*W, embedding_dim)
        pixel_flat = pixel_embeddings.permute(0, 2, 3, 1).reshape(B, H * W, self.embedding_dim)

        # prototypes: (num_prototypes, embedding_dim)
        # similarity: (B, H*W, num_prototypes)
        similarity = torch.matmul(pixel_flat, self.prototypes.T)
        similarity = similarity / (self.embedding_dim ** 0.5)  # 缩放

        # 应用空间平滑（鼓励连续性）
        similarity_map = similarity.permute(0, 2, 1).reshape(B, self.num_prototypes, H, W)
        similarity_map = self.spatial_smooth(similarity_map)
        similarity = similarity_map.reshape(B, self.num_prototypes, H * W).permute(0, 2, 1)

        # 使用softmax获取每个像素的原型归属概率
        prototype_activations = F.softmax(similarity, dim=-1)  # (B, H*W, num_prototypes)

        # 为每个原型预测偏置值
        # prototypes: (num_prototypes, embedding_dim)
        prototype_biases = self.bias_predictor(self.prototypes).squeeze(-1)  # (num_prototypes,)
        prototype_biases = prototype_biases.unsqueeze(0).expand(B, -1)  # (B, num_prototypes)

        # 应用可学习的缩放因子
        prototype_biases = prototype_biases * self.bias_scale

        # 加权求和得到最终的偏置图
        # prototype_activations: (B, H*W, num_prototypes)
        # prototype_biases: (B, num_prototypes)
        instance_bias_flat = torch.einsum('bnp,bp->bn', prototype_activations, prototype_biases)
        instance_bias_map = instance_bias_flat.reshape(B, 1, H, W)

        # 将原型激活图reshape回图像形状
        prototype_activations = prototype_activations.permute(0, 2, 1).reshape(B, self.num_prototypes, H, W)

        return instance_bias_map, prototype_activations, prototype_biases, pixel_embeddings


class DepthAnythingV2WithInstance(nn.Module):
    """
    修改后的Depth Anything V2模型

    特点：
    1. 使用预训练的DINOv2作为编码器（权重冻结）
    2. 原始DPT解码器用于生成relative depth map（权重冻结）
    3. 新增的实例分割解码器头（可训练）
    4. 将实例偏置加到relative map上，进行归一化
    """

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
            bias_scale_init=0.1
    ):
        super(DepthAnythingV2WithInstance, self).__init__()

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

        # 原始DPT解码器（用于生成relative depth map）
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

        self.instance_head = InstanceSegmentationHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            num_prototypes=num_prototypes,
            embedding_dim=embedding_dim,
            use_clstoken=use_clstoken,
            bias_scale_init=bias_scale_init
        )

        # 偏置归一化参数
        self.norm_min = nn.Parameter(torch.tensor(0.0))
        self.norm_max = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像，shape (B, 3, H, W)，应该是Google Earth影像
            original_depth: 可选的原始深度图（用于测试时绕过encoder）

        Returns:
           - enhanced_depth: 增强后的深度图（已归一化到0~1）
            - original_depth: 原始深度图（0~1范围）
            - instance_bias_map: 实例偏置图
            - prototype_activations: 原型激活图
            - prototype_biases: 每个原型的偏置值
            - pixel_embeddings: 像素嵌入
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14

        # 获取编码器特征
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        
        # 检查特征是否包含NaN
        for i, feat in enumerate(features):
            if isinstance(feat, (list, tuple)):
                for j, f in enumerate(feat):
                    if torch.isnan(f).any():
                        print(f"警告: 编码器特征 [{i}][{j}] 包含NaN")
            else:
                if torch.isnan(feat).any():
                    print(f"警告: 编码器特征 [{i}] 包含NaN")

        # 原始解码器生成relative depth map
        # 注意：在AMP模式下，depth_head可能在FP16中溢出，强制使用FP32
        with torch.cuda.amp.autocast(enabled=False):
            with torch.set_grad_enabled(not self.depth_head.scratch.output_conv1.weight.requires_grad):
                # 将特征转换为FP32（如果当前是FP16）
                features_fp32 = []
                for feat in features:
                    if isinstance(feat, (list, tuple)):
                        features_fp32.append([f.float() if f.dtype == torch.float16 else f for f in feat])
                    else:
                        features_fp32.append(feat.float() if feat.dtype == torch.float16 else feat)
                
                original_depth = self.depth_head(features_fp32, patch_h, patch_w)
                original_depth = F.relu(original_depth)  # (B, 1, H, W)
                
                # 检查是否有NaN
                if torch.isnan(original_depth).any():
                    print(f"警告: depth_head 输出包含NaN")
                    # 尝试用0替换NaN
                    original_depth = torch.where(torch.isnan(original_depth), 
                                                 torch.zeros_like(original_depth), 
                                                 original_depth)

        # 实例分割头生成实例偏置图（也使用FP32以确保数值稳定性）
        with torch.cuda.amp.autocast(enabled=False):
            instance_bias_map, prototype_activations, prototype_biases, pixel_embeddings = self.instance_head(
                features_fp32, patch_h, patch_w
            )
            
            # 确保instance_bias_map也是FP32
            instance_bias_map = instance_bias_map.float()

        # 将偏置加到原始深度图上（两者都是FP32）
        enhanced_depth = original_depth + instance_bias_map

        # 归一化操作也在FP32中进行
        with torch.cuda.amp.autocast(enabled=False):
            # 第一次归一化：确保非负
            enhanced_depth = F.relu(enhanced_depth)

            # 第二次归一化：min-max归一化到0~1
            batch_min = enhanced_depth.view(enhanced_depth.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
            batch_max = enhanced_depth.view(enhanced_depth.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)

            # 防止除零
            range_val = batch_max - batch_min
            range_val = torch.where(range_val < 1e-6, torch.ones_like(range_val), range_val)

            enhanced_depth = (enhanced_depth - batch_min) / range_val

            # 应用可学习的归一化参数
            enhanced_depth = enhanced_depth * self.norm_max + self.norm_min
            enhanced_depth = torch.clamp(enhanced_depth, 0, 1)

        return {
            'enhanced_depth': enhanced_depth.squeeze(1),  # (B, H, W)
            'original_depth': original_depth.squeeze(1),  # (B, H, W)
            'instance_bias_map': instance_bias_map.squeeze(1),  # (B, H, W)
            'prototype_activations': prototype_activations,  # (B, num_prototypes, H, W)
            'prototype_biases': prototype_biases,  # (B, num_prototypes)
            'pixel_embeddings': pixel_embeddings,  # (B, embedding_dim, H, W)
        }


    @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):
        """单张图像推理"""
        image, (h, w) = self.image2tensor(raw_image, input_size)

        result = self.forward(image)

        # 插值回原始尺寸
        enhanced_depth = F.interpolate(
            result['enhanced_depth'][:, None],
            (h, w),
            mode="bilinear",
            align_corners=True
        )[0, 0]

        original_depth = F.interpolate(
            result['original_depth'][:, None],
            (h, w),
            mode="bilinear",
            align_corners=True
        )[0, 0]

        return {
            'enhanced_depth': enhanced_depth.cpu().numpy(),
            'original_depth': original_depth.cpu().numpy()
        }

    def image2tensor(self, raw_image, input_size=518, is_unsqueeze=True):
        """将图像转换为张量"""
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = transform({'image': image})['image']

        image = torch.from_numpy(image)
        if is_unsqueeze:
            image = image.unsqueeze(0)

        device = next(self.parameters()).device
        image = image.to(device)

        return image, (h, w)


def create_dam_model(
    encoder='vitl',
    pretrained_path=None,
    num_prototypes=128,
    embedding_dim=64,
    device='cuda'
):
    """
    创建修改后的DAM模型

    Args:
        encoder: 编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
        pretrained_path: 预训练权重路径
        num_prototypes: 原型数量（最大实例数）
        embedding_dim: 嵌入维度
        device: 设备

    Returns:
        model: 修改后的DAM模型
    """

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
        bias_scale_init=0.1
    )

    # 加载预训练权重
    if pretrained_path is not None:
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        print("预训练权重加载完成")

    model = model.to(device)

    return model
