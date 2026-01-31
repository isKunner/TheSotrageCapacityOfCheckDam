"""
DEM超分辨率模型训练代码

损失函数：
1. RMSE损失：HRDEM与USGS DEM之间的均方根误差
2. 映射损失：mapped LRDEM与Copernicus DEM之间的误差
3. 实例分割损失（可选）：用于约束实例分割的效果
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import time
import json
from typing import Dict, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from LocalPath import dam_root_path, dam_project_path
from dataset import create_dataloaders, DEMSuperResolutionDataset, collect_valid_samples
from dam_model import create_dam_model
from super_resolution_model import create_super_resolution_system




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

class RMSELoss(nn.Module):
    """RMSE损失"""
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target))


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    包含：
    1. HRDEM与USGS DEM的RMSE损失
    2. Mapped LRDEM与Copernicus DEM的RMSE损失
    3. 实例分割的辅助损失（可选）
    """
    
    def __init__(
        self,
        hrdem_weight=1.0,
        mapping_weight=0.5,
        instance_weight=0.1
    ):
        super(CombinedLoss, self).__init__()
        
        self.hrdem_weight = hrdem_weight
        self.mapping_weight = mapping_weight
        self.instance_weight = instance_weight
        
        self.rmse_loss = RMSELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        hrdem,
        usgs_dem,
        mapped_lrdem,
        copernicus_dem,
        instance_biases=None
    ):
        """
        计算组合损失
        
        Args:
            hrdem: 预测的高分辨率DEM (B, 1, H, W)
            usgs_dem: USGS DEM真值 (B, 1, H, W)
            mapped_lrdem: 映射后的低分辨率DEM (B, 1, H, W)
            copernicus_dem: Copernicus DEM (B, 1, H, W)
            instance_biases: 实例偏置值 (B, num_instances)，可选
        
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失的字典
        """
        # HRDEM损失（主要损失）
        hrdem_loss = self.rmse_loss(hrdem, usgs_dem)
        
        # 映射损失
        mapping_loss = self.rmse_loss(mapped_lrdem, copernicus_dem)
        
        # 总损失
        total_loss = (
            self.hrdem_weight * hrdem_loss +
            self.mapping_weight * mapping_loss
        )
        
        # 实例偏置正则化损失（鼓励偏置值接近0，避免过度修正）
        instance_reg_loss = torch.tensor(0.0, device=hrdem.device)
        if instance_biases is not None:
            instance_reg_loss = torch.mean(instance_biases ** 2)
            total_loss += self.instance_weight * instance_reg_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'hrdem': hrdem_loss.item(),
            'mapping': mapping_loss.item(),
            'instance_reg': instance_reg_loss.item()
        }
        
        return total_loss, loss_dict


class Trainer:
    """训练器"""
    
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
        num_epochs=100,
        save_freq=10,
        val_freq=5,
        grad_clip=1.0,
        early_stopping=None
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
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'hrdem': 0.0,
            'mapping': 0.0,
            'instance_reg': 0.0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            copernicus = batch['copernicus'].to(self.device)
            google = batch['google'].to(self.device)
            usgs = batch['usgs'].to(self.device)
            
            # 前向传播
            output = self.model(google, copernicus)
            
            hrdem = output['hrdem']
            mapped_lrdem = output['mapped_lrdem']
            instance_biases = output['dam_output'].get('instance_biases', None)
            
            # 计算损失
            loss, loss_dict = self.criterion(
                hrdem, usgs,
                mapped_lrdem, copernicus,
                instance_biases
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip
                )
            
            self.optimizer.step()
            
            # 累加损失
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'hrdem': f"{loss_dict['hrdem']:.4f}",
                'map': f"{loss_dict['mapping']:.4f}"
            })
        
        # 计算平均损失
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'hrdem': 0.0,
            'mapping': 0.0,
            'instance_reg': 0.0
        }
        
        # 用于计算额外的评估指标
        rmse_list = []
        mae_list = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            copernicus = batch['copernicus'].to(self.device)
            google = batch['google'].to(self.device)
            usgs = batch['usgs'].to(self.device)
            
            # 前向传播
            output = self.model(google, copernicus)
            
            hrdem = output['hrdem']
            mapped_lrdem = output['mapped_lrdem']
            instance_biases = output['dam_output'].get('instance_biases', None)
            
            # 计算损失
            loss, loss_dict = self.criterion(
                hrdem, usgs,
                mapped_lrdem, copernicus,
                instance_biases
            )
            
            # 累加损失
            for key in val_losses:
                val_losses[key] += loss_dict[key]
            
            # 计算评估指标
            rmse = torch.sqrt(nn.MSELoss()(hrdem, usgs)).item()
            mae = nn.L1Loss()(hrdem, usgs).item()
            
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
    
    def save_checkpoint(self, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
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

        if self.early_stopping:
            print(f"早停机制: 耐心值={self.early_stopping.patience}, 最小改善={self.early_stopping.min_delta}")

        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch

                print(f"\n{'='*50}")
                print(f"Epoch {epoch}/{self.num_epochs}")
                print(f"{'='*50}")

                # 训练
                start_time = time.time()
                train_losses = self.train_epoch()
                train_time = time.time() - start_time

                self.train_history.append(train_losses)

                print(f"\n训练损失:")
                for key, value in train_losses.items():
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

                # 保存检查点
                self.save_checkpoint()

        except KeyboardInterrupt:
            print("\n\n检测到训练中断信号(KeyboardInterrupt)")
            self.save_checkpoint(is_best=False)
            print(f"已保存中断时的检查点到: {self.save_dir}/interrupt_checkpoint.pth")
            raise
        
        # 训练结束
        print(f"\n{'='*50}")
        print("训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"{'='*50}")
        
        # 保存训练历史
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train': self.train_history,
                'val': self.val_history
            }, f, indent=2)
        
        self.writer.close()


def main(
    # 数据参数
    data_dir=r"D:\研究文件\ResearchData\USA",
    batch_size=2,
    num_workers=4,
    
    # 模型参数
    dam_encoder='vitl',
    dam_pretrained_path=None,  # DAM预训练权重路径
    num_instances=64,
    sr_channels=64,
    sr_residual_blocks=8,
    mapper_base_channels=32,
    use_instance_guidance=True,  # 是否使用实例引导
    
    # 训练参数
    num_epochs=100,
    lr=1e-4,
    weight_decay=1e-5,
    scheduler_step=30,
    scheduler_gamma=0.5,
    grad_clip=1.0,
    
    # 损失权重
    hrdem_weight=1.0,
    mapping_weight=0.5,
    instance_weight=0.1,
    
    # 其他参数
    save_dir='./checkpoints',
    log_dir='./logs',
    device='cuda',
    seed=42,
    
    # 项目路径（用于自动查找DAM权重）
    project_dir=r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything"
):
    """
    主函数
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        num_workers: 数据加载worker数
        dam_encoder: DAM编码器类型
        dam_pretrained_path: DAM预训练权重路径
        num_instances: 实例数量
        sr_channels: 超分辨率网络基础通道数
        sr_residual_blocks: 超分辨率网络残差块数
        mapper_base_channels: 映射网络基础通道数
        use_instance_guidance: 是否使用实例引导
        num_epochs: 训练epoch数
        lr: 学习率
        weight_decay: 权重衰减
        scheduler_step: 学习率调度步长
        scheduler_gamma: 学习率调度衰减因子
        grad_clip: 梯度裁剪阈值
        hrdem_weight: HRDEM损失权重
        mapping_weight: 映射损失权重
        instance_weight: 实例损失权重
        save_dir: 模型保存目录
        log_dir: 日志目录
        device: 设备
        seed: 随机种子
        project_dir: 项目目录（用于自动查找DAM权重）
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 创建设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        base_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )
    
    # 自动查找DAM预训练权重
    if dam_pretrained_path is None:
        # 构建可能的权重文件路径
        possible_weights = [
            os.path.join(project_dir, 'checkpoints', f'depth_anything_v2_{dam_encoder}.pth'),
            os.path.join(project_dir, 'checkpoints', f'depth_anything_v2_{dam_encoder}_encoder.pth'),
            os.path.join(project_dir, 'checkpoints', f'dam_v2_{dam_encoder}.pth'),
            os.path.join(project_dir, 'checkpoints', 'depth_anything_v2.pth'),
        ]
        
        for weight_path in possible_weights:
            if os.path.exists(weight_path):
                dam_pretrained_path = weight_path
                print(f"\n自动找到DAM预训练权重: {dam_pretrained_path}")
                break
        
        if dam_pretrained_path is None:
            print("\n警告: 未找到DAM预训练权重，将使用随机初始化的权重")
            print(f"请在以下路径放置权重文件: {os.path.join(project_dir, 'checkpoints')}")
    else:
        print(f"\n使用指定的DAM预训练权重: {dam_pretrained_path}")
    
    # 创建DAM模型
    print("\n创建DAM模型...")
    dam_model = create_dam_model(
        encoder=dam_encoder,
        pretrained_path=dam_pretrained_path,
        num_instances=num_instances,
        device=device
    )
    
    # 创建超分辨率系统
    print("\n创建超分辨率系统...")
    model = create_super_resolution_system(
        dam_model=dam_model,
        sr_channels=sr_channels,
        sr_residual_blocks=sr_residual_blocks,
        mapper_base_channels=mapper_base_channels,
        use_instance_guidance=use_instance_guidance,
        device=device
    )
    
    print(f"实例引导: {'启用' if use_instance_guidance else '禁用'}")
    
    # 统计可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {total_params - trainable_params:,}")
    
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
        instance_weight=instance_weight
    )
    
    # 创建训练器
    trainer = Trainer(
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
        grad_clip=grad_clip
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DEM超分辨率训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=dam_root_path)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    
    # 模型参数
    parser.add_argument('--dam_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--dam_pretrained_path', type=str, default=None)
    parser.add_argument('--num_instances', type=int, default=64)
    parser.add_argument('--sr_channels', type=int, default=64)
    parser.add_argument('--sr_residual_blocks', type=int, default=8)
    parser.add_argument('--mapper_base_channels', type=int, default=32)
    parser.add_argument('--use_instance_guidance', action='store_true', default=True)
    parser.add_argument('--no_instance_guidance', action='store_false', dest='use_instance_guidance',
                        help='禁用实例引导')
    parser.add_argument('--project_dir', type=str, 
                        default=dam_project_path)
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--scheduler_step', type=int, default=30)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # 损失权重
    parser.add_argument('--hrdem_weight', type=float, default=1.0)
    parser.add_argument('--mapping_weight', type=float, default=0.5)
    parser.add_argument('--instance_weight', type=float, default=0.1)
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 运行主函数
    main(**vars(args))
