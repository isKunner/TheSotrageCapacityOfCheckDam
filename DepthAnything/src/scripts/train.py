#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: train.py
# @Time    : 2026/2/1 16:00
# @Author  : Kevin
# @Describe: 改进的训练脚本，支持分阶段训练

import os
import argparse
import torch.optim as optim

import torch

# 导入新的包 - 注意：实际使用时需要确保这些模块存在
from DepthAnything.src.models import create_dam_model, create_super_resolution_system, CombinedLoss
from DepthAnything.src.data import create_dataloaders_with_cache, create_dataloaders
from DepthAnything.src.training import MultiStageTrainer, EarlyStopping
from DepthAnything.src.utils import set_seed, count_parameters, get_device, clear_cuda_cache, print_gpu_memory_info


def train_sr_model(
        # 数据参数
        data_dir='./data',
        batch_size=2,
        num_workers=8,
        use_cache=True,
        cache_dir=None,
        target_size=1022,
        force_rebuild_cache=False,

        # 模型参数
        dam_encoder='vits',
        dam_pretrained_path=None,
        num_prototypes=128,
        embedding_dim=64,
        sr_channels=64,
        sr_residual_blocks=8,
        mapper_base_channels=32,
        mapper_scale_factor=30,
        use_instance_guidance=True,
        use_adaptive_fusion=False,

        # 训练参数
        num_epochs=20,
        lr=1e-4,
        weight_decay=1e-5,
        scheduler_step=4,
        scheduler_gamma=0.5,
        grad_clip=1.0,

        # 损失权重
        hrdem_weight=1.0,
        mapping_weight=0.5,
        dam_enhanced_weight=0.3,
        instance_weight=0.05,
        prototype_diversity_weight=0.01,
        grad_weight=0.0,
        ssim_weight=0.0,
        multiscale_weight=0.0,
        consistency_weight=0.0,
        dominance_weight=0.3,  # 新增：多尺度主导权损失权重

        # 其他参数
        checkpoints_dir='./checkpoints',
        save_dir='./checkpoints',
        log_dir='./logs',
        device='cuda',
        seed=42,

        # 早停参数
        use_early_stopping=False,
        early_stopping_patience=5,
        early_stopping_delta=0.0001,

        # 验证频率
        val_freq=2,
        save_freq=2,
        
        # AMP和显存优化
        use_amp=False,  # 启用混合精度训练
        clear_cache_freq=10,  # 每10个batch清理一次显存
        amp_mode='conservative',  # AMP模式：'conservative'更稳定，'aggressive'更快
        use_cached_dam_encoder=True,
        dam_batch_size=1,
):
    """
    主训练函数（改进版，支持分阶段训练）

    Args:
        详见参数列表
    """
    # 清理显存
    clear_cuda_cache()
    
    # 设置随机种子
    set_seed(seed)

    # 创建设备
    device = get_device(device)
    print(f"使用设备: {device}")
    print_gpu_memory_info("初始显存: ")

    # 设置缓存目录
    if cache_dir is None:
        cache_dir = os.path.join(data_dir, 'data_cache')

    # 自动查找DAM预训练权重
    if dam_pretrained_path is None:
        possible_weights = [
            os.path.join(checkpoints_dir, f'depth_anything_v2_{dam_encoder}.pth'),
            os.path.join(checkpoints_dir, f'depth_anything_v2_{dam_encoder}_encoder.pth'),
            os.path.join(checkpoints_dir, f'dam_v2_{dam_encoder}.pth'),
            os.path.join(checkpoints_dir, 'depth_anything_v2.pth'),
        ]

        for weight_path in possible_weights:
            if os.path.exists(weight_path):
                dam_pretrained_path = weight_path
                print(f"\n自动找到DAM预训练权重: {dam_pretrained_path}")
                break

        if dam_pretrained_path is None:
            print("\n警告: 未找到DAM预训练权重，将使用随机初始化的权重")
    else:
        print(f"\n使用指定的DAM预训练权重: {dam_pretrained_path}")

    # 创建DAM模型
    print("\n创建DAM模型...")
    dam_model = create_dam_model(
        encoder=dam_encoder,
        pretrained_path=dam_pretrained_path,
        num_prototypes=num_prototypes,
        embedding_dim=embedding_dim,
        device=device
    )


    # 创建数据加载器
    print("\n创建数据加载器...")
    if use_cache:
        train_loader, val_loader = create_dataloaders_with_cache(
            base_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            cache_dir=cache_dir,
            target_size=target_size,
            force_rebuild_cache=force_rebuild_cache,
            cache_dam_encoder=use_cached_dam_encoder,
            dam_model=dam_model,
            dam_batch_size=dam_batch_size,
        )
    else:
        train_loader, val_loader = create_dataloaders(
            base_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            target_size=target_size
        )

    if use_cached_dam_encoder:
        print("删除DAM Encoder前的显存占用：")
        print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"最大显存占用: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

        del dam_model.pretrained  # 删除Encoder
        torch.cuda.empty_cache()

        print("删除DAM Encoder后的显存占用：")
        print(f"当前显存占用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"最大显存占用: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")

    # 创建超分辨率系统
    print("\n创建超分辨率系统...")
    model = create_super_resolution_system(
        dam_model=dam_model,
        sr_channels=sr_channels,
        sr_residual_blocks=sr_residual_blocks,
        mapper_scale_factor=mapper_scale_factor,
        mapper_base_channels=mapper_base_channels,
        use_instance_guidance=use_instance_guidance,
        use_adaptive_fusion=use_adaptive_fusion,
        device=device,
        use_cached_dam_encoder=use_cached_dam_encoder,
    )

    print(f"实例引导: {'启用' if use_instance_guidance else '禁用'}")
    print(f"自适应融合: {'启用' if use_adaptive_fusion else '禁用'}")
    print(f"下采样倍率: {mapper_scale_factor}")

    # 统计可训练参数
    param_stats = count_parameters(model)
    print(f"\n总参数数量: {param_stats['total']:,}")
    print(f"可训练参数数量: {param_stats['trainable']:,}")
    print(f"冻结参数数量: {param_stats['frozen']:,}")

    # 创建优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    # 创建学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma
    )

    # 创建损失函数
    criterion = CombinedLoss(
        hrdem_weight=hrdem_weight,
        mapping_weight=mapping_weight,
        dam_enhanced_weight=dam_enhanced_weight,
        instance_weight=instance_weight,
        prototype_diversity_weight=prototype_diversity_weight,
        grad_weight=grad_weight,
        ssim_weight=ssim_weight,
        multiscale_weight=multiscale_weight,
        consistency_weight=consistency_weight,
        dominance_weight=dominance_weight,
        scale_factor=mapper_scale_factor,
        training_stage='joint',
    )

    # 创建早停机制
    early_stopping = None
    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_delta
        )

    # 创建训练器
    trainer = MultiStageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        num_epochs=num_epochs,
        save_freq=save_freq,
        val_freq=val_freq,
        grad_clip=grad_clip,
        early_stopping=early_stopping,
        stage_configs=None,  # 使用默认5阶段配置
        use_amp=use_amp,
        clear_cache_freq=clear_cache_freq,
        initial_lr=lr,  # 传递初始学习率用于阶段感知调度
        amp_mode=amp_mode,
        use_cached_dam_encoder=use_cached_dam_encoder
    )

    # 开始训练
    trainer.train()

    print("\n训练配置完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DEM超分辨率训练')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--use_cache', action='store_true', default=True,
                        help='使用缓存数据集')
    parser.add_argument('--no_cache', action='store_false', dest='use_cache',
                        help='不使用缓存')
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--target_size', type=int, default=1024,
                        help='目标尺寸（应为14的倍数）')
    parser.add_argument('--force_rebuild_cache', action='store_true',
                        help='强制重建缓存')

    # 模型参数
    parser.add_argument('--dam_encoder', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--dam_pretrained_path', type=str, default=None)
    parser.add_argument('--num_prototypes', type=int, default=128,
                        help='原型数量（替代num_instances）')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='嵌入维度')
    parser.add_argument('--sr_channels', type=int, default=64)
    parser.add_argument('--sr_residual_blocks', type=int, default=8)
    parser.add_argument('--mapper_base_channels', type=int, default=32)
    parser.add_argument('--mapper_scale_factor', type=int, default=30,
                        help='下采样倍率')
    parser.add_argument('--use_instance_guidance', action='store_true', default=True)
    parser.add_argument('--no_instance_guidance', action='store_false', dest='use_instance_guidance')
    parser.add_argument('--use_adaptive_fusion', action='store_true', default=False)
    parser.add_argument('--no_adaptive_fusion', action='store_false', dest='use_adaptive_fusion')

    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--scheduler_step', type=int, default=30)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # 阶段配置
    parser.add_argument('--use_multi_stage', action='store_true', default=True,
                        help='使用分阶段训练')
    parser.add_argument('--single_stage', action='store_false', dest='use_multi_stage',
                        help='单阶段训练')
    parser.add_argument('--pretrain_dam_epochs', type=int, default=None)
    parser.add_argument('--joint_epochs', type=int, default=None)
    parser.add_argument('--finetune_epochs', type=int, default=None)

    # 损失权重
    parser.add_argument('--hrdem_weight', type=float, default=1.0)
    parser.add_argument('--mapping_weight', type=float, default=0.5)
    parser.add_argument('--dam_enhanced_weight', type=float, default=0.3)
    parser.add_argument('--instance_weight', type=float, default=0.05)
    parser.add_argument('--prototype_diversity_weight', type=float, default=0.01)
    parser.add_argument('--grad_weight', type=float, default=0.0)
    parser.add_argument('--ssim_weight', type=float, default=0.0)
    parser.add_argument('--multiscale_weight', type=float, default=0.0)
    parser.add_argument('--consistency_weight', type=float, default=0.0)

    # 早停参数
    parser.add_argument('--use_early_stopping', action='store_true',
                        help='启用早停机制')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--early_stopping_delta', type=float, default=0.0001)

    # 显存优化参数
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='启用混合精度训练(AMP)节省显存(仅CUDA有效，CPU自动禁用)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp',
                        help='禁用混合精度训练')
    parser.add_argument('--clear_cache_freq', type=int, default=20,
                        help='每多少batch清理一次显存(0表示不清理)')

    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=10)

    args = parser.parse_args()

    # 运行主函数
    train_sr_model(**vars(args))