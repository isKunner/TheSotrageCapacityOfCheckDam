"""
Models Package

包含所有模型定义：
- DAM模型（带实例分割）
- 超分辨率网络
- 映射网络
- 损失函数
"""

from .dam_model import (
    DepthAnythingV2WithInstance,
    InstanceSegmentationHead,
    create_dam_model
)

from .sr_model import (
    SuperResolutionNetwork,
    HRDEMToLRDEMMapper,
    DEMSuperResolutionSystem,
    create_super_resolution_system,
    ResidualBlock,
    ConvBlock,
    InstanceGuidedAttention
)

from .losses import (
    RMSELoss,
    CombinedLoss,
    GradientLoss,
    SSIMLoss
)

__all__ = [
    # DAM模型
    'DepthAnythingV2WithInstance',
    'InstanceSegmentationHead',
    'create_dam_model',
    # 超分辨率模型
    'SuperResolutionNetwork',
    'HRDEMToLRDEMMapper',
    'DEMSuperResolutionSystem',
    'create_super_resolution_system',
    'ResidualBlock',
    'ConvBlock',
    'InstanceGuidedAttention',
    # 损失函数
    'RMSELoss',
    'CombinedLoss',
    'GradientLoss',
    'SSIMLoss',
]
