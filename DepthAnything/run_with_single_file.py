#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: run_with_single_file
# @Time    : 2026/1/28 19:32
# @Author  : Kevin
# @Describe:

import cv2
import numpy as np
import os

import rasterio
import torch
import matplotlib
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
sys.path.insert(0, parent_dir)

from DepthAnything.src.models.depth_anything_v2.dpt import DepthAnythingV2
from DEMAndRemoteSensingUtils import crop_source_to_reference, batch_modify_tifs_vectorized
from LocalPath import ROOT_DIR

sys.path.insert(0, str(ROOT_DIR))

# ==================== 用户配置区域 ====================
# 输入输出配置
INPUT_IMAGE_PATH = r'C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Test\WMG_1.0m_1024pixel\375644_1103401_Google.tif'
REFERENCE_DEM_PATH = r'D:\研究文件\ResearchData\USA\USGSDEM\GeoDAR_v11_dams_of_USA_group1\130.tif'
OUTPUT_DIR = r'./'

# 模型配置
ENCODER = 'vits'  # 可选: 'vits', 'vitb', 'vitl', 'vitg'
INPUT_SIZE = 1022
DEVICE = 'cpu'  # 可选: 'cuda', 'cpu'

# 可视化配置
PRED_ONLY = True  # True: 只保存深度图, False: 原图+深度图拼接
GRAYSCALE = False  # True: 灰度图, False: 彩色热力图(Spectral_r)


# ====================================================
def read_tif_as_bgr(tif_path):
    """
    使用 rasterio 读取 TIF 并转换为 OpenCV BGR 格式
    """
    with rasterio.open(tif_path) as src:
        # 读取所有波段 (bands, height, width)
        image = src.read()

        image = image[:, 1:-1, 1:-1]

        # 检查波段数
        if image.shape[0] >= 3:
            # 多波段图像（取前3个波段作为RGB）
            r = image[0]
            g = image[1]
            b = image[2]

            # 归一化到 0-255 (uint8)
            def normalize_band(band):
                band_min, band_max = band.min(), band.max()
                if band_max > band_min:
                    band = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
                else:
                    band = np.zeros_like(band, dtype=np.uint8)
                return band

            r = normalize_band(r)
            g = normalize_band(g)
            b = normalize_band(b)

            # 合并为RGB
            rgb_image = np.stack([r, g, b], axis=2)  # (H, W, 3)
            # 转为BGR (OpenCV格式)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        else:
            # 单波段灰度图
            band = image[0]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                gray = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            else:
                gray = np.zeros_like(band, dtype=np.uint8)

            # 转为3通道BGR
            bgr_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return bgr_image

def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 配置模型参数
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # 加载模型
    print(f"正在加载模型: Depth Anything V2 ({ENCODER})...")
    depth_anything = DepthAnythingV2(**model_configs[ENCODER])
    depth_anything.load_state_dict(torch.load(
        fr'C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\checkpoints\depth_anything_v2_{ENCODER}.pth',
        map_location=DEVICE
    ))
    depth_anything = depth_anything.to(DEVICE).eval()

    # 读取图像
    print(f"正在处理: {INPUT_IMAGE_PATH}")
    raw_image = read_tif_as_bgr(INPUT_IMAGE_PATH)
    if raw_image is None:
        raise ValueError(f"无法读取图像: {INPUT_IMAGE_PATH}")

    # 推理 - 单张（但使用list保持与工具函数兼容）
    # infer_batch支持单张，传入list of 1 image
    depth_list = depth_anything.infer_batch([raw_image], INPUT_SIZE)
    depth = depth_list[0]  # (H, W)

    # 归一化到 0-1
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)

    # 添加batch维度以适配batch_modify_tifs_vectorized (B, H, W) -> (1, H, W)
    depth_batch = depth_normalized[np.newaxis, ...]

    # 生成输出文件名（基于输入文件名）
    base_name = os.path.splitext(os.path.basename(INPUT_IMAGE_PATH))[0]
    output_tif_name = f"{base_name}_depth.tif"
    output_tif_path = os.path.join(OUTPUT_DIR, output_tif_name)

    # 保存为GeoTIFF（保持地理参考）
    # print(f"正在保存GeoTIFF: {output_tif_path}")
    # batch_modify_tifs_vectorized(
    #     tif_paths=[REFERENCE_DEM_PATH],  # 参考TIF路径（单元素列表）
    #     input_matrices=depth_batch,  # (1, H, W)
    #     output_paths=[output_tif_path]  # 输出路径（单元素列表）
    # )

    # 可视化保存
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    if GRAYSCALE:
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
    else:
        # 彩色热力图
        colored = cmap(depth_normalized)  # (H, W, 4) RGBA
        depth_vis = (colored[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)  # 转为BGR

    # 保存可视化图像
    if PRED_ONLY:
        output_img_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")
        cv2.imwrite(output_img_path, depth_vis)
    else:
        # 拼接原图和深度图
        split_region = np.full((raw_image.shape[0], 50, 3), 255, dtype=np.uint8)
        combined = cv2.hconcat([raw_image, split_region, depth_vis])
        output_img_path = os.path.join(OUTPUT_DIR, f"{base_name}_compare.png")
        cv2.imwrite(output_img_path, combined)

    print(f"可视化结果已保存: {output_img_path}")
    print("处理完成！")


if __name__ == '__main__':
    main()