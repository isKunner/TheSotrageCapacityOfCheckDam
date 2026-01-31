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
import cv2
from torchvision.transforms import Compose

# 导入原始DAM v2的组件
from depth_anything_v2.dinov2 import DINOv2
from depth_anything_v2.dpt import DPTHead
from depth_anything_v2.util.blocks import FeatureFusionBlock, _make_scratch
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet


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
    实例分割解码器头
    
    功能：
    1. 从编码器特征中解码出实例分割掩码
    2. 每个实例区域对应一个偏置值
    3. 输出与relative map相同尺寸的单通道偏置图
    
    结构：
    - 使用与DPTHead类似的特征融合结构
    - 最终输出单通道的实例偏置图
    """
    
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        num_instances=64,  # 最大实例数量
        use_clstoken=False
    ):
        super(InstanceSegmentationHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        self.num_instances = num_instances
        
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
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        # 实例分割输出头
        head_features_1 = features
        head_features_2 = 64
        
        # 实例分割分支：输出实例ID图
        self.instance_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.instance_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, num_instances, kernel_size=1, stride=1, padding=0),
        )
        
        # 偏置值预测分支：为每个实例预测一个偏置值
        self.bias_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_instances, num_instances),
            nn.ReLU(True),
            nn.Linear(num_instances, num_instances),
        )
    
    def forward(self, out_features, patch_h, patch_w):
        """
        前向传播
        
        Args:
            out_features: 编码器输出的特征列表
            patch_h, patch_w: patch的高度和宽度
        
        Returns:
            instance_bias_map: 实例偏置图，shape (B, 1, H, W)
            instance_logits: 实例分割logits，shape (B, num_instances, H, W)
            instance_biases: 每个实例的偏置值，shape (B, num_instances)
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
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        # 实例分割分支
        instance_feat = self.instance_conv1(path_1)
        instance_feat = F.interpolate(
            instance_feat,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True
        )
        instance_logits = self.instance_conv2(instance_feat)  # (B, num_instances, H, W)
        
        # 预测每个实例的偏置值
        instance_biases = self.bias_predictor(instance_logits)  # (B, num_instances)
        
        # 将实例分割转换为偏置图
        # 使用softmax获取每个像素的实例归属概率
        instance_probs = F.softmax(instance_logits, dim=1)  # (B, num_instances, H, W)
        
        # 加权求和得到最终的偏置图
        # instance_biases: (B, num_instances)
        # instance_probs: (B, num_instances, H, W)
        instance_bias_map = torch.einsum('bn,bnhw->bhw', instance_biases, instance_probs)
        instance_bias_map = instance_bias_map.unsqueeze(1)  # (B, 1, H, W)
        
        return instance_bias_map, instance_logits, instance_biases


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
        num_instances=64,
        freeze_encoder=True,
        freeze_original_decoder=True
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
        
        # 新增的实例分割解码器头
        self.instance_head = InstanceSegmentationHead(
            self.pretrained.embed_dim,
            features,
            use_bn,
            out_channels=out_channels,
            num_instances=num_instances,
            use_clstoken=use_clstoken
        )
        
        # 偏置归一化参数
        self.bias_scale = nn.Parameter(torch.ones(1))
        self.bias_shift = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像，shape (B, 3, H, W)，应该是Google Earth影像
        
        Returns:
            enhanced_depth: 增强后的深度图（relative map + 实例偏置）
            original_depth: 原始深度图（relative map）
            instance_bias_map: 实例偏置图
            instance_logits: 实例分割logits
            instance_biases: 每个实例的偏置值
        """
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        # 获取编码器特征
        features = self.pretrained.get_intermediate_layers(
            x,
            self.intermediate_layer_idx[self.encoder],
            return_class_token=True
        )
        
        # 原始解码器生成relative depth map
        with torch.set_grad_enabled(not self.depth_head.scratch.output_conv1.weight.requires_grad):
            original_depth = self.depth_head(features, patch_h, patch_w)
            original_depth = F.relu(original_depth)  # (B, 1, H, W)
        
        # 实例分割头生成实例偏置图
        instance_bias_map, instance_logits, instance_biases = self.instance_head(
            features, patch_h, patch_w
        )
        
        # 对偏置进行缩放和平移
        instance_bias_map = instance_bias_map * self.bias_scale + self.bias_shift
        
        # 将偏置加到原始深度图上
        enhanced_depth = original_depth + instance_bias_map
        
        # 归一化到非负值
        enhanced_depth = F.relu(enhanced_depth)
        
        return {
            'enhanced_depth': enhanced_depth.squeeze(1),  # (B, H, W)
            'original_depth': original_depth.squeeze(1),  # (B, H, W)
            'instance_bias_map': instance_bias_map.squeeze(1),  # (B, H, W)
            'instance_logits': instance_logits,  # (B, num_instances, H, W)
            'instance_biases': instance_biases  # (B, num_instances)
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
    num_instances=64,
    device='cuda'
):
    """
    创建修改后的DAM模型
    
    Args:
        encoder: 编码器类型 ('vits', 'vitb', 'vitl', 'vitg')
        pretrained_path: 预训练权重路径
        num_instances: 实例数量
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
        num_instances=num_instances,
        freeze_encoder=True,
        freeze_original_decoder=True
    )
    
    # 加载预训练权重
    if pretrained_path is not None:
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 只加载编码器和原始解码器的权重
        model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(checkpoint, strict=False)
        print("预训练权重加载完成")
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # 测试模型
    print("创建修改后的DAM模型...")
    model = DepthAnythingV2WithInstance(
        encoder='vits',  # 使用小模型进行测试
        num_instances=64
    )
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {total_params - trainable_params:,}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    x = torch.randn(2, 3, 518, 518)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"增强深度图形状: {output['enhanced_depth'].shape}")
    print(f"原始深度图形状: {output['original_depth'].shape}")
    print(f"实例偏置图形状: {output['instance_bias_map'].shape}")
    print(f"实例分割logits形状: {output['instance_logits'].shape}")
    print(f"实例偏置值形状: {output['instance_biases'].shape}")
    
    print("\n模型测试完成！")
