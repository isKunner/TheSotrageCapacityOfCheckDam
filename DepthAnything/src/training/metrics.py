"""
评估指标模块

包含各种评估指标的计算
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class MetricTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.metrics = {}
        self.history = []
    
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def get_average(self, key: str) -> float:
        """获取平均值"""
        if key in self.metrics and len(self.metrics[key]) > 0:
            return np.mean(self.metrics[key])
        return 0.0
    
    def get_all_averages(self) -> Dict[str, float]:
        """获取所有指标的平均值"""
        return {key: self.get_average(key) for key in self.metrics.keys()}
    
    def reset(self):
        """重置"""
        self.metrics = {}
    
    def save_history(self, epoch: int):
        """保存历史"""
        self.history.append({
            'epoch': epoch,
            'metrics': self.get_all_averages()
        })


def calculate_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        pred: 预测值 (B, C, H, W) 或 (B, H, W)
        target: 目标值 (B, C, H, W) 或 (B, H, W)
    
    Returns:
        metrics: 包含各种指标的字典
    """
    # 确保是4维张量
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    
    # 计算各种指标
    mse = nn.MSELoss()(pred, target).item()
    rmse = np.sqrt(mse)
    mae = nn.L1Loss()(pred, target).item()
    
    # MAPE（避免除以0）
    mask = target != 0
    if mask.sum() > 0:
        mape = torch.mean(
            torch.abs((pred[mask] - target[mask]) / target[mask])
        ).item() * 100
    else:
        mape = 0.0
    
    # Bias
    bias = torch.mean(pred - target).item()
    
    # 相对误差
    relative_error = torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8)).item() * 100
    
    # 最大误差
    max_error = torch.max(torch.abs(pred - target)).item()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'bias': bias,
        'relative_error': relative_error,
        'max_error': max_error
    }


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算PSNR（峰值信噪比）
    
    Args:
        pred: 预测值
        target: 目标值
        max_val: 最大值
    
    Returns:
        psnr: PSNR值
    """
    mse = nn.MSELoss()(pred, target).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))
