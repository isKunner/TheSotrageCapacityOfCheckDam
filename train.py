#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: test
# @Time    : 2026/1/17 15:19
# @Author  : Kevin
# @Describe:
import os
from multiprocessing import freeze_support
from DepthAnything.src import train_sr_model
from LocalPath import dam_root_path

if __name__ == '__main__':
    freeze_support()
    train_sr_model(data_dir=dam_root_path,
                   cache_dir=os.path.join(dam_root_path, "data_cache_zscore"),
                   batch_size=1,
                   num_workers=4,
                   device='cuda',
                   use_amp=True,
                   num_prototypes=4,
                   embedding_dim=4,
                   sr_channels=8,
                   sr_residual_blocks=2,
                   mapper_base_channels=8,
                   checkpoints_dir=r"./DepthAnything/checkpoints",
                   save_dir=r"./DepthAnything/checkpoints",
                   log_dir=r"./DepthAnything/logs",
                   val_freq=2,
                   save_freq=2)