#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: clean_nodata
# @Time    : 2025/8/22 11:46
# @Author  : Kevin
# @Describe: 去除DEM.tif中的全为Nodata的行或者列

import rasterio
import numpy as np
from rasterio.transform import Affine

def remove_nodata_rows_cols(input_path, output_path):
    """
    移除DEM数据中全为Nodata的行和列

    参数:
        input_path: 输入DEM文件路径
        output_path: 输出处理后的DEM文件路径
    """
    with rasterio.open(input_path) as src:
        # 读取数据和元数据
        dem_data = src.read(1)  # 读取第一个波段
        nodata = src.nodata
        meta = src.meta.copy()
        # 原始变换矩阵
        transform = src.transform

        # 如果没有定义nodata值，尝试从数据中识别（这里假设-9999是常见的nodata值）
        if nodata is None:
            nodata = -9999
            print(f"警告：未找到nodata定义，将使用默认值 {nodata}")

        # 找到非nodata的行和列索引
        # 对于行：如果一行中存在非nodata值，则保留该行
        valid_rows = np.any(dem_data != nodata, axis=1)
        # 对于列：如果一列中存在非nodata值，则保留该列
        valid_cols = np.any(dem_data != nodata, axis=0)

        # 检查是否有有效数据
        if not np.any(valid_rows) or not np.any(valid_cols):
            raise ValueError("输入数据中没有有效数据，全部为Nodata")

        # 裁剪数据，只保留有效行和列
        cleaned_data = dem_data[valid_rows, :][:, valid_cols]

        # 计算新的变换矩阵（调整地理坐标）

        # 计算新的左上角坐标
        # 第一行、列中含有有效数据的位置
        new_x = transform.c + np.where(valid_cols)[0][0] * transform.a
        new_y = transform.f + np.where(valid_rows)[0][0] * transform.e
        # 创建新的变换矩阵
        new_transform = Affine(transform.a, transform.b, new_x,
                               transform.d, transform.e, new_y)

        # 更新元数据
        meta.update({
            'height': cleaned_data.shape[0],
            'width': cleaned_data.shape[1],
            'transform': new_transform
        })

        # 写入处理后的文件
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(cleaned_data, 1)

        print(f"处理完成！原始尺寸: {dem_data.shape}, 处理后尺寸: {cleaned_data.shape}")
        print(f"移除了 {dem_data.shape[0] - cleaned_data.shape[0]} 行和 {dem_data.shape[1] - cleaned_data.shape[1]} 列")

# 使用示例
if __name__ == "__main__":
    input_dem = r"C:\Users\Kevin\Documents\ResearchData\WangMao\WMGdem.tif"  # 替换为你的输入DEM文件路径
    output_dem = r"C:\Users\Kevin\Documents\ResearchData\WangMao\cleaned_dem.tif"  # 输出文件路径
    remove_nodata_rows_cols(input_dem, output_dem)

