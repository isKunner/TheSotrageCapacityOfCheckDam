#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: inference
# @Time    : 2026/2/5 17:09
# @Author  : Kevin
# @Describe:

import os.path as osp

from LocalPath import dam_project_path
from DepthAnything.src import run_dem_super_resolution

run_dem_super_resolution(
    test_dir=osp.join(dam_project_path, "Test"),
    copernicus_folder="Copernicus_1.0m_1024pixel",
    google_folder="WMG_1.0m_1024pixel",
    checkpoint=osp.join(dam_project_path, "checkpoints", "depth_anything_v2_vits.pth"),
    output_dir=osp.join(dam_project_path, "20260205_test"),
    encoder="vits",
    batch_size=1,
    device='cpu'
)