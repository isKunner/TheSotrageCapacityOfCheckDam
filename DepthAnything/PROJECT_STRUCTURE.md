# 项目结构说明

## 文件组织

```
DEM-Super-Resolution/
│
├── 📁 dinov2_layers/              # DINOv2模型组件（用户提供）
│   ├── __init__.py
│   ├── attention.py
│   ├── block.py
│   ├── drop_path.py
│   ├── layer_scale.py
│   ├── mlp.py
│   ├── patch_embed.py
│   └── swiglu_ffn.py
│
├── 📁 util/                       # 工具模块（用户提供）
│   ├── blocks.py
│   └── transform.py
│
├── 📄 dinov2.py                   # DINOv2模型（用户提供）
├── 📄 dpt.py                      # DPT解码器（用户提供）
│
├── 📄 dataset.py                  # ⭐ 数据集创建代码
├── 📄 dam_model.py               # ⭐ 修改后的DAM模型
├── 📄 super_resolution_model.py  # ⭐ 超分辨率网络和映射网络
├── 📄 train.py                   # ⭐ 训练代码
├── 📄 validate.py                # ⭐ 验证代码
├── 📄 train_with_config.py       # 使用配置文件的训练脚本
│
├── 📄 test_models.py             # 模型测试脚本
├── 📄 example_usage.py           # 使用示例
│
├── 📄 config.yaml                # 配置文件
├── 📄 requirements.txt           # 依赖包列表
├── 📄 README.md                  # 项目说明文档
└── 📄 PROJECT_STRUCTURE.md       # 本文件
```

## 核心模块说明

### 1. 数据集模块 (dataset.py)

**功能**：
- 读取三个文件夹（CopernicusDEM、GoogleRemoteSensing、USGSDEM）的tif文件
- 检查文件完整性（确保三个文件夹中对应的文件都存在）
- 按8:2比例划分训练集和测试集
- 提供数据归一化和尺寸调整

**主要类**：
- `collect_valid_samples()`: 收集有效样本
- `DEMSuperResolutionDataset`: PyTorch Dataset类
- `create_dataloaders()`: 创建DataLoader

### 2. DAM模型模块 (dam_model.py)

**功能**：
- 基于DAM v2的修改版本
- 冻结原始编码器和解码器权重
- 添加实例分割解码器头
- 输出增强的relative depth map

**主要类**：
- `DPTHead`: 原始DPT解码器（冻结）
- `InstanceSegmentationHead`: 实例分割头（可训练）
- `DepthAnythingV2WithInstance`: 完整的DAM模型
- `create_dam_model()`: 模型创建函数

### 3. 超分辨率模块 (super_resolution_model.py)

**功能**：
- 融合Copernicus DEM和relative map生成HRDEM
- 学习HRDEM到Copernicus DEM的映射关系

**主要类**：
- `SuperResolutionNetwork`: 超分辨率重构网络
- `HRDEMToLRDEMMapper`: HRDEM到LRDEM的映射网络
- `DEMSuperResolutionSystem`: 完整的超分辨率系统

### 4. 训练模块 (train.py)

**功能**：
- 完整的训练流程
- 组合损失函数（HRDEM损失 + 映射损失 + 实例正则化）
- TensorBoard日志记录
- 模型检查点保存

**主要类**：
- `RMSELoss`: RMSE损失
- `CombinedLoss`: 组合损失
- `Trainer`: 训练器类
- `main()`: 主函数

### 5. 验证模块 (validate.py)

**功能**：
- 验证集验证
- 指定目录批量验证
- 单张图像验证
- 可视化结果保存

**主要类**：
- `Validator`: 验证器类
- `load_model_from_checkpoint()`: 从检查点加载模型
- `main()`: 主函数

## 数据流

```
输入数据
    │
    ├── Google Earth影像 (3通道, 1024×1024)
    │       │
    │       ▼
    │   DAM模型（带实例分割）
    │       │
    │       ├── 原始解码器 → relative map
    │       │
    │       └── 实例分割头 → 实例偏置
    │               │
    │               ▼
    │       增强的relative map
    │               │
    │               ▼
    └── Copernicus DEM (1通道, 1024×1024)
            │
            ▼
    超分辨率重构网络
            │
            ▼
        HRDEM (1通道, 1024×1024)
            │
            ├──► 与USGS DEM计算RMSE损失
            │
            └──► 映射网络 → Mapped LRDEM
                        │
                        ▼
                与Copernicus DEM计算映射损失
```

## 关键设计决策

### 1. 为什么冻结DAM的原始权重？

- DAM v2已经在大规模数据上预训练
- 冻结可以保持其强大的特征提取能力
- 只训练实例分割头，专注于解决relative map的偏置问题

### 2. 实例分割头的作用？

- 识别图像中的不同地形对象（如水坝、建筑物等）
- 为每个对象预测一个偏置值
- 解决relative map中某些对象整体偏高或偏低的问题

### 3. 为什么需要映射网络？

- 学习HRDEM到Copernicus DEM的映射关系
- 在没有USGS DEM真值的区域验证模型
- 提供额外的监督信号

### 4. 损失函数的设计？

- **HRDEM损失**（权重1.0）：主要目标，确保生成的DEM准确
- **映射损失**（权重0.5）：辅助目标，确保映射关系正确
- **实例正则化**（权重0.1）：防止偏置值过大

## 扩展建议

### 1. 添加数据增强

在`dataset.py`的`DEMSuperResolutionDataset`中添加：

```python
def __getitem__(self, idx):
    # ... 原有代码 ...
    
    # 随机水平翻转
    if random.random() > 0.5:
        copernicus = torch.flip(copernicus, dims=[-1])
        google = torch.flip(google, dims=[-1])
        usgs = torch.flip(usgs, dims=[-1])
    
    # 随机旋转
    if random.random() > 0.5:
        angle = random.choice([90, 180, 270])
        copernicus = torch.rot90(copernicus, angle // 90, dims=[-2, -1])
        google = torch.rot90(google, angle // 90, dims=[-2, -1])
        usgs = torch.rot90(usgs, angle // 90, dims=[-2, -1])
    
    return {...}
```

### 2. 添加更多的评估指标

在`validate.py`中添加：

```python
def calculate_metrics(pred, target):
    # RMSE
    rmse = torch.sqrt(nn.MSELoss()(pred, target))
    
    # MAE
    mae = nn.L1Loss()(pred, target)
    
    # PSNR
    mse = nn.MSELoss()(pred, target)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # SSIM（需要额外实现）
    ssim = calculate_ssim(pred, target)
    
    return {'rmse': rmse, 'mae': mae, 'psnr': psnr, 'ssim': ssim}
```

### 3. 支持多GPU训练

在`train.py`中修改：

```python
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU")
    model = nn.DataParallel(model)
```

## 调试技巧

### 1. 检查数据加载

```python
# 在dataset.py末尾添加
if __name__ == "__main__":
    train_samples, test_samples = collect_valid_samples()
    dataset = DEMSuperResolutionDataset(train_samples[:5])
    sample = dataset[0]
    
    print("Copernicus范围:", sample['copernicus'].min(), sample['copernicus'].max())
    print("Google范围:", sample['google'].min(), sample['google'].max())
    print("USGS范围:", sample['usgs'].min(), sample['usgs'].max())
```

### 2. 检查模型输出

```python
# 在训练循环中添加
if batch_idx == 0:
    print("HRDEM范围:", hrdem.min().item(), hrdem.max().item())
    print("Instance bias范围:", instance_bias_map.min().item(), instance_bias_map.max().item())
```

### 3. 可视化损失曲线

```bash
# 启动TensorBoard
tensorboard --logdir=./logs

# 在浏览器中打开
# http://localhost:6006
```

## 性能优化

### 1. 使用混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. 使用更快的数据加载

```python
# 在DataLoader中设置
DataLoader(
    dataset,
    batch_size=4,
    num_workers=8,        # 增加worker数量
    pin_memory=True,      # 使用固定内存
    persistent_workers=True,  # 保持worker进程
    prefetch_factor=2     # 预取因子
)
```

### 3. 使用xformers加速注意力

```bash
pip install xformers
```

代码中会自动检测并使用xformers。
