"""
训练器模块

包含：
- Trainer: 主训练器类
- EarlyStopping: 早停机制
1. 支持分阶段训练（pretrain_dam -> joint -> finetune）
2. 添加阶段切换逻辑
3. 改进损失计算，支持DAM enhanced loss
4. 添加学习率调度策略
"""

import os
import json
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..utils import clear_cuda_cache, print_gpu_memory_info


class EarlyStopping:
    """早停机制，当验证损失不再下降时停止训练"""

    def __init__(self, patience=10, min_delta=0.0001, mode='min'):
        """
        Args:
            patience: 容忍多少个epoch验证损失不改善
            min_delta: 改善的最小阈值
            mode: 'min'表示损失越小越好，'max'表示准确率越高越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            return False

        if self.mode == 'min':
            improved = val_loss < (self.best_loss - self.min_delta)
        else:
            improved = val_loss > (self.best_loss + self.min_delta)

        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

    def status(self):
        return f"早停计数: {self.counter}/{self.patience}, 最佳损失: {self.best_loss:.6f} (Epoch {self.best_epoch})"


class MultiStageTrainer:
    """
    多阶段训练器（改进版）

    支持五个训练阶段（渐进式解冻）：
    1. pretrain_dam: 预训练DAM的instance head
    2. warmup_sr: SR + Mapper 预热（DAM完全冻结）
    3. joint_partial: 联合训练（解冻Instance Head）
    4. joint_encoder: 联合训练（解冻DAM最后N层）
    5. finetune: 微调（全部解冻）
    
    特性：
    - 混合精度训练（AMP）节省显存
    - 自动显存清理
    - 阶段感知学习率调度
    """

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            save_dir,
            log_dir,
            num_epochs=10,
            save_freq=10,
            val_freq=5,
            grad_clip=1.0,
            early_stopping=None,
            stage_configs=None,  # 详细的阶段配置字典
            use_amp=False,
            clear_cache_freq=10,
            amp_mode='conservative',
            initial_lr=1e-4,  # 初始学习率，用于阶段感知调度
            use_cached_dam_encoder=False,  # 是否使用预缓存的DAM Encoder特征
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.num_epochs = num_epochs
        self.save_freq = save_freq
        self.val_freq = val_freq
        self.grad_clip = grad_clip
        self.early_stopping = early_stopping
        self.initial_lr = initial_lr
        self.use_cached_dam_encoder = use_cached_dam_encoder
        device_str = getattr(device, 'type', str(device).lower()) if hasattr(device, 'type') else str(device).lower()
        self.use_amp = use_amp and torch.cuda.is_available() and device_str != 'cpu'
        self.clear_cache_freq = clear_cache_freq

        # 阶段配置（默认5阶段渐进式解冻）
        if stage_configs is None:
            # 默认配置
            pretrain_epochs = max(3, int(num_epochs * 0.3))
            warmup_epochs = max(2, int(num_epochs * 0.4))
            joint_partial_epochs = num_epochs - pretrain_epochs - warmup_epochs
            
            self.stage_configs = {
                'pretrain_dam': {
                    'epochs': pretrain_epochs,
                    'train_dam_encoder': False,
                    'train_dam_decoder': False,
                    'train_instance_head': True,
                    'train_sr': False,
                    'train_mapper': False,
                    'lr_scale': 1.0,
                    'unfreeze_encoder_layers': 0,
                },
                'warmup_sr': {
                    'epochs': warmup_epochs,
                    'train_dam_encoder': False,
                    'train_dam_decoder': False,
                    'train_instance_head': False,
                    'train_sr': True,
                    'train_mapper': True,
                    'lr_scale': 1.0,
                    'unfreeze_encoder_layers': 0,
                },
                'joint_partial': {
                    'epochs': joint_partial_epochs,
                    'train_dam_encoder': False,
                    'train_dam_decoder': False,
                    'train_instance_head': True,
                    'train_sr': True,
                    'train_mapper': True,
                    'lr_scale': 0.5,
                    'unfreeze_encoder_layers': 0,
                }
            }
        else:
            self.stage_configs = stage_configs

        # 计算累积epoch用于阶段判断
        self._build_stage_epochs()

        # TensorBoard
        self.writer = SummaryWriter(log_dir)

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
        self.current_stage = None
        
        # AMP scaler
        self.amp_mode = amp_mode
        if self.use_amp and not torch.cuda.is_available():
            print("警告: 检测到CPU环境，自动禁用AMP")
            self.use_amp = False
        
        if self.use_amp:
            if amp_mode == 'conservative':
                self.scaler = GradScaler(
                    init_scale=512.0,
                    growth_factor=1.2,
                    backoff_factor=0.5,
                    growth_interval=2000,
                    enabled=True
                )
            else:
                self.scaler = GradScaler(
                    init_scale=2048.0,
                    growth_factor=1.5,
                    backoff_factor=0.5,
                    growth_interval=500,
                    enabled=True
                )
        else:
            self.scaler = None
        
        print(f"混合精度训练(AMP): {'启用' if self.use_amp else '禁用'}")
        if self.use_amp:
            print(f"AMP模式: {amp_mode}")
        
        # 打印阶段配置
        print(f"\n分阶段训练配置（渐进式解冻）:")
        for stage, config in self.stage_configs.items():
            print(f"  {stage}: {config['epochs']} epochs, lr_scale={config['lr_scale']}")

    def _build_stage_epochs(self):
        """构建阶段到epoch范围的映射"""
        self.stage_epochs = {}
        cumulative = 0
        for stage, config in self.stage_configs.items():
            start = cumulative
            end = cumulative + config['epochs']
            self.stage_epochs[stage] = (start, end)
            cumulative = end
        self.total_epochs = cumulative

    def get_current_stage(self, epoch):
        """根据当前epoch确定训练阶段"""
        for stage, (start, end) in self.stage_epochs.items():
            if start <= epoch < end:
                return stage
        # 如果超出范围，返回最后一个阶段
        return list(self.stage_configs.keys())[-1]

    def setup_stage(self, stage):
        """配置当前训练阶段（渐进式解冻）"""
        if stage == self.current_stage:
            return

        self.current_stage = stage
        config = self.stage_configs[stage]
        
        print(f"\n{'=' * 50}")
        print(f"进入训练阶段: {stage}")
        print(f"{'=' * 50}")

        # 设置损失权重
        self.criterion.set_stage(stage)

        # 阶段感知学习率调整
        new_lr = self.initial_lr * config['lr_scale']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"  - 学习率设置为: {new_lr:.6f} (scale={config['lr_scale']})")

        # 配置模型参数的可训练性
        # 1. DAM Encoder
        if not self.use_cached_dam_encoder:
            if config['train_dam_encoder'] == True:
                # 全部解冻
                for param in self.model.dam_model.pretrained.parameters():
                    param.requires_grad = True
                print("  - DAM encoder: 全部解冻")
            elif config['train_dam_encoder'] == 'partial':
                # 部分解冻（最后N层）
                total_layers = len(self.model.dam_model.pretrained.blocks)
                unfreeze_n = config.get('unfreeze_encoder_layers', 4)
                for i, block in enumerate(self.model.dam_model.pretrained.blocks):
                    for param in block.parameters():
                        param.requires_grad = (i >= total_layers - unfreeze_n)
                print(f"  - DAM encoder: 解冻最后{unfreeze_n}层")
            else:
                for param in self.model.dam_model.pretrained.parameters():
                    param.requires_grad = False
                print("  - DAM encoder: 冻结")

        # 2. DAM Decoder (depth_head)
        if config['train_dam_decoder']:
            for param in self.model.dam_model.depth_head.parameters():
                param.requires_grad = True
            print("  - DAM decoder: 解冻")
        else:
            for param in self.model.dam_model.depth_head.parameters():
                param.requires_grad = False
            print("  - DAM decoder: 冻结")

        # 3. Instance Head
        if config['train_instance_head']:
            for param in self.model.dam_model.instance_head.parameters():
                param.requires_grad = True
            self.model.dam_model.norm_min.requires_grad = True
            self.model.dam_model.norm_max.requires_grad = True
            print("  - Instance head: 训练")
        else:
            for param in self.model.dam_model.instance_head.parameters():
                param.requires_grad = False
            print("  - Instance head: 冻结")

        # 4. SR Network
        if config['train_sr']:
            for param in self.model.sr_network.parameters():
                param.requires_grad = True
            print("  - SR network: 训练")
        else:
            for param in self.model.sr_network.parameters():
                param.requires_grad = False
            print("  - SR network: 冻结")

        # 5. Mapper Network
        if config['train_mapper']:
            for param in self.model.mapper_network.parameters():
                param.requires_grad = True
            print("  - Mapper network: 训练")
        else:
            for param in self.model.mapper_network.parameters():
                param.requires_grad = False
            print("  - Mapper network: 冻结")

    def train_epoch(self):
        """训练一个epoch（支持AMP）"""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'hrdem': 0.0,
            'mapping': 0.0,
            'dam_enhanced': 0.0,
            'instance_reg': 0.0,
            'prototype_div': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch} [{self.current_stage}]")

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            copernicus = batch['copernicus'].to(self.device)
            google = batch['google'].to(self.device)
            usgs = batch['usgs'].to(self.device)

            dam_encoder_features = None
            if self.use_cached_dam_encoder and 'dam_encoder_features' in batch:
                dam_encoder_features = batch['dam_encoder_features']
                dam_encoder_features = [
                    (feat[0].to(self.device), feat[1].to(self.device))
                    for feat in dam_encoder_features
                ]

            # 使用AMP进行前向传播
            if self.use_amp:
                with autocast():

                    output = self.model(google, copernicus, dam_encoder_features=dam_encoder_features)
                    hrdem = output['hrdem']
                    mapped_lrdem = output['mapped_lrdem']
                    dam_enhanced = output['dam_output'].get('enhanced_depth', None)
                    instance_biases = output['dam_output'].get('prototype_biases', None)
                    prototypes = self.model.dam_model.instance_head.prototypes

                    # 获取主导权图（如果存在）
                    dominance_map = output.get('dominance_map', None)
                    
                    # 计算损失
                    loss, loss_dict = self.criterion(
                        hrdem=hrdem,
                        usgs_dem=usgs,
                        mapped_lrdem=mapped_lrdem,
                        copernicus_dem=F.adaptive_avg_pool2d(copernicus, mapped_lrdem.shape[-2:]),
                        dam_enhanced_depth=dam_enhanced,
                        instance_biases=instance_biases,
                        prototypes=prototypes,
                        dominance_map=dominance_map,
                    )

                # 检查损失是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n警告: Epoch {self.current_epoch}, Batch {batch_idx} 出现NaN/Inf损失，跳过")
                    # 跳过此batch（不调用scaler.update，因为未进行反向传播）
                    # GradScaler会自动检测inf/NaN并跳过更新
                    continue
                
                # 反向传播（使用scaler）
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # 梯度裁剪（需要在unscale之后）
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.grad_clip
                    )

                # 检查梯度是否包含NaN
                has_nan_grad = False
                for p in self.model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"\n警告: Epoch {self.current_epoch}, Batch {batch_idx} 出现NaN梯度，跳过")
                    # 跳过此batch（不调用scaler.update，因为梯度无效）
                    self.optimizer.zero_grad()
                    continue

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:

                output = self.model(google, copernicus, dam_encoder_features=dam_encoder_features)

                hrdem = output['hrdem']
                mapped_lrdem = output['mapped_lrdem']
                dam_enhanced = output['dam_output'].get('enhanced_depth', None)
                instance_biases = output['dam_output'].get('prototype_biases', None)
                prototypes = self.model.dam_model.instance_head.prototypes
                dominance_map = output.get('dominance_map', None)

                # 计算损失
                loss, loss_dict = self.criterion(
                    hrdem=hrdem,
                    usgs_dem=usgs,
                    mapped_lrdem=mapped_lrdem,
                    copernicus_dem=F.adaptive_avg_pool2d(copernicus, mapped_lrdem.shape[-2:]),
                    dam_enhanced_depth=dam_enhanced,
                    instance_biases=instance_biases,
                    prototypes=prototypes,
                    dominance_map=dominance_map,
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, self.model.parameters()),
                        self.grad_clip
                    )

                self.optimizer.step()

            # 累加损失
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'hrdem': f"{loss_dict.get('hrdem', 0):.4f}",
                'dam': f"{loss_dict.get('dam_enhanced', 0):.4f}",
                'map': f"{loss_dict.get('mapping', 0):.4f}"
            })
            
            # 定期清理显存
            if self.clear_cache_freq > 0 and batch_idx % self.clear_cache_freq == 0:
                clear_cuda_cache()

        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)

        return epoch_losses

    @torch.no_grad()
    def validate(self):
        """验证（支持AMP）"""
        self.model.eval()

        val_losses = {
            'total': 0.0,
            'hrdem': 0.0,
            'mapping': 0.0,
            'dam_enhanced': 0.0,
            'instance_reg': 0.0,
        }

        # 用于计算额外的评估指标
        rmse_list = []
        mae_list = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            copernicus = batch['copernicus'].to(self.device)
            google = batch['google'].to(self.device)
            usgs = batch['usgs'].to(self.device)

            dam_encoder_features = None
            if self.use_cached_dam_encoder and 'dam_encoder_features' in batch:
                dam_encoder_features = batch['dam_encoder_features']
                dam_encoder_features = [
                    (feat[0].to(self.device), feat[1].to(self.device))
                    for feat in dam_encoder_features
                ]

            # 前向传播（验证时也使用AMP节省显存）
            if self.use_amp:
                with autocast():
                    output = self.model(google, copernicus, dam_encoder_features=dam_encoder_features)
                    hrdem = output['hrdem']
                    mapped_lrdem = output['mapped_lrdem']
                    dam_enhanced = output['dam_output'].get('enhanced_depth', None)
                    instance_biases = output['dam_output'].get('prototype_biases', None)
                    prototypes = self.model.dam_model.instance_head.prototypes

                    # 获取主导权图
                    dominance_map = output.get('dominance_map', None)
                    
                    # 计算损失
                    loss, loss_dict = self.criterion(
                        hrdem=hrdem,
                        usgs_dem=usgs,
                        mapped_lrdem=mapped_lrdem,
                        copernicus_dem=F.adaptive_avg_pool2d(copernicus, mapped_lrdem.shape[-2:]),
                        dam_enhanced_depth=dam_enhanced,
                        instance_biases=instance_biases,
                        prototypes=prototypes,
                        dominance_map=dominance_map,
                    )
            else:
                output = self.model(google, copernicus, dam_encoder_features=dam_encoder_features)
                hrdem = output['hrdem']
                mapped_lrdem = output['mapped_lrdem']
                dam_enhanced = output['dam_output'].get('enhanced_depth', None)
                instance_biases = output['dam_output'].get('prototype_biases', None)
                prototypes = self.model.dam_model.instance_head.prototypes
                dominance_map = output.get('dominance_map', None)

                # 计算损失
                loss, loss_dict = self.criterion(
                    hrdem=hrdem,
                    usgs_dem=usgs,
                    mapped_lrdem=mapped_lrdem,
                    copernicus_dem=F.adaptive_avg_pool2d(copernicus, mapped_lrdem.shape[-2:]),
                    dam_enhanced_depth=dam_enhanced,
                    instance_biases=instance_biases,
                    prototypes=prototypes,
                    dominance_map=dominance_map,
                )

            # 累加损失
            for key in val_losses:
                if key in loss_dict:
                    val_losses[key] += loss_dict[key]

            # 计算评估指标（使用函数式接口避免创建临时对象）
            rmse = torch.sqrt(F.mse_loss(hrdem, usgs)).item()
            mae = F.l1_loss(hrdem, usgs).item()

            rmse_list.append(rmse)
            mae_list.append(mae)

        # 计算平均损失
        for key in val_losses:
            val_losses[key] /= len(self.val_loader)

        # 添加评估指标
        val_losses['rmse_mean'] = np.mean(rmse_list)
        val_losses['rmse_std'] = np.std(rmse_list)
        val_losses['mae_mean'] = np.mean(mae_list)
        val_losses['mae_std'] = np.std(mae_list)

        return val_losses

    def log_metrics(self, train_losses, val_losses=None):
        """记录指标到TensorBoard"""
        epoch = self.current_epoch

        # 记录训练损失
        for key, value in train_losses.items():
            self.writer.add_scalar(f'Train/{key}', value, epoch)

        # 记录验证损失
        if val_losses is not None:
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)

        # 记录当前阶段
        stage_map = {'pretrain_dam': 0, 'joint': 1, 'finetune': 2}
        self.writer.add_scalar('Train/stage', stage_map.get(self.current_stage, 0), epoch)

    def save_checkpoint(self, is_best=False, stage_name=None):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'current_stage': self.current_stage,
            'stage_epochs': self.stage_epochs
        }

        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # 保存阶段检查点
        if stage_name is not None:
            stage_path = os.path.join(self.save_dir, f'checkpoint_{stage_name}.pth')
            torch.save(checkpoint, stage_path)

        # 保存定期检查点
        if self.current_epoch % self.save_freq == 0:
            epoch_path = os.path.join(
                self.save_dir,
                f'checkpoint_epoch_{self.current_epoch}.pth'
            )
            torch.save(checkpoint, epoch_path)

        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型，验证损失: {self.best_val_loss:.4f}")

    def train(self):
        """主训练循环"""
        print(f"开始训练，共 {self.num_epochs} 个epoch")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
        print(f"阶段配置: {self.stage_epochs}")

        if self.early_stopping:
            print(f"早停机制: 耐心值={self.early_stopping.patience}, 最小改善={self.early_stopping.min_delta}")
        
        if self.use_amp and self.scaler is not None:
            print(f"AMP Scale初始值: {self.scaler.get_scale()}")

        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch

                # 确定并设置当前阶段
                stage = self.get_current_stage(epoch)
                self.setup_stage(stage)

                print(f"\n{'=' * 50}")
                print(f"Epoch {epoch}/{self.num_epochs} - Stage: {stage}")
                print(f"{'=' * 50}")

                # 训练
                start_time = time.time()
                train_losses = self.train_epoch()
                train_time = time.time() - start_time

                self.train_history.append(train_losses)

                print(f"\n训练损失:")
                for key, value in train_losses.items():
                    if value > 0:
                        print(f"  {key}: {value:.4f}")
                print(f"训练时间: {train_time:.2f}s")

                # 学习率调度
                if self.scheduler is not None:
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"当前学习率: {current_lr:.6f}")
                    self.writer.add_scalar('Train/lr', current_lr, epoch)

                # 验证
                val_losses = None
                if epoch % self.val_freq == 0:
                    val_losses = self.validate()
                    self.val_history.append(val_losses)

                    print(f"\n验证损失:")
                    for key, value in val_losses.items():
                        print(f"  {key}: {value:.4f}")

                    # 检查是否是最佳模型
                    if val_losses['total'] < self.best_val_loss:
                        self.best_val_loss = val_losses['total']
                        self.save_checkpoint(is_best=True)

                    if self.early_stopping:
                        should_stop = self.early_stopping(val_losses['total'], epoch)
                        print(f"  {self.early_stopping.status()}")
                        if should_stop:
                            print(f"\n{'=' * 50}")
                            print(f"早停触发！验证损失连续 {self.early_stopping.patience} 个epoch未改善")
                            print(
                                f"最佳验证损失: {self.early_stopping.best_loss:.4f} (Epoch {self.early_stopping.best_epoch})")
                            print(f"{'=' * 50}")
                            break

                # 记录指标
                self.log_metrics(train_losses, val_losses)

                # 保存检查点（在阶段切换时保存）
                for s, (start, end) in self.stage_epochs.items():
                    if epoch + 1 == end:
                        self.save_checkpoint(stage_name=f"stage_{s}_end")
                        break
                        break
                else:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            print("\n\n检测到训练中断信号(KeyboardInterrupt)")
            self.save_checkpoint(is_best=False)
            interrupt_path = os.path.join(self.save_dir, 'interrupt_checkpoint.pth')
            latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
            if os.path.exists(latest_path):
                import shutil
                shutil.copy(latest_path, interrupt_path)
            print(f"已保存中断时的检查点到: {interrupt_path}")
            raise

        # 训练结束
        print(f"\n{'=' * 50}")
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"{'=' * 50}")

        # 保存训练历史
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history,
                'stage_epochs': self.stage_epochs
            }, f, indent=2)

        self.writer.close()