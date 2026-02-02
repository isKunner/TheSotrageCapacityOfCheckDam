"""
学习率调度模块
"""

import torch
from torch.optim.lr_scheduler import LRScheduler
import math


class WarmupCosineScheduler(LRScheduler):
    """
    带warmup的余弦退火学习率调度器
    
    前warmup_epochs个epoch线性增加学习率
    之后使用余弦退火
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        eta_min=0,
        last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup阶段：线性增加
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [
                self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
