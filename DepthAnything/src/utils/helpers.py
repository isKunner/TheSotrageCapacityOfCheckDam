"""
通用工具函数
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: str = 'cuda') -> torch.device:
    """
    获取设备
    
    Args:
        device_name: 设备名称
    
    Returns:
        device: torch设备
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        stats: 参数字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    save_path: str,
    **kwargs
):
    """
    保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前epoch
        save_path: 保存路径
        **kwargs: 其他需要保存的数据
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }
    checkpoint.update(kwargs)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Dict:
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备
    
    Returns:
        checkpoint: 检查点字典
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        if checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
    
    Returns:
        lr: 学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def clear_cuda_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info() -> Dict[str, float]:
    """
    获取GPU显存信息
    
    Returns:
        dict: 包含allocated, reserved, free的信息（单位：GB）
    """
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'free': 0, 'total': 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total - reserved
    
    return {
        'allocated': allocated,
        'reserved': reserved,
        'free': free,
        'total': total
    }


def print_gpu_memory_info(prefix=""):
    """打印GPU显存信息"""
    info = get_gpu_memory_info()
    print(f"{prefix}GPU Memory: Allocated={info['allocated']:.2f}GB, "
          f"Reserved={info['reserved']:.2f}GB, Free={info['free']:.2f}GB, "
          f"Total={info['total']:.2f}GB")
