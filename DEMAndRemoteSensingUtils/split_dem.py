#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: clip_tif
# @Time    : 2025/7/24 17:28
# @Author  : Kevin
# @Describe: 对tif进行裁剪，裁剪为32*32大小的数据

import os

import numpy as np
import rasterio
from rasterio.transform import Affine

def split_tif(input_path, output_dir, tile_size, overlap):
    """
    按指定瓦片大小和重叠像素裁剪TIF文件
    :param input_path: 输入TIF文件路径
    :param output_dir: 瓦片输出目录
    :param tile_size: 瓦片尺寸（长宽相同，像素数）
    :param overlap: 重叠像素数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取TIF数据
    with rasterio.open(input_path) as src:
        data = src.read(1)  # 读取第一波段
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        height, width = data.shape

        step = tile_size - overlap

        if step <= 0:
            raise ValueError("步长必须为正数（瓦片尺寸需大于重叠像素）")

        # 计算瓦片数量
        h_count = (height - tile_size + step - 1) // step + 1  # 向上取整
        w_count = (width - tile_size + step - 1) // step + 1

        # 循环裁剪瓦片
        for h_idx in range(h_count):
            for w_idx in range(w_count):
                # 计算当前瓦片起始/结束索引
                start_h = h_idx * step
                end_h = start_h + tile_size
                start_w = w_idx * step
                end_w = start_w + tile_size

                # 处理边缘瓦片（避免超出范围）
                if end_h > height:
                    end_h = height
                    start_h = end_h - tile_size
                if end_w > width:
                    end_w = width
                    start_w = end_w - tile_size

                # 提取瓦片数据
                tile_data = data[start_h:end_h, start_w:end_w]

                if np.all((tile_data == nodata) | np.isnan(tile_data)):
                    continue

                # 计算瓦片地理变换
                tile_transform = Affine(
                    transform.a, transform.b, transform.xoff + start_w * transform.a,
                    transform.d, transform.e, transform.yoff + start_h * transform.e
                )

                # 保存瓦片
                tile_name = f"tile_{h_idx}_{w_idx}.tif"
                tile_path = os.path.join(output_dir, tile_name)
                with rasterio.open(
                        tile_path, 'w',
                        driver='GTiff',
                        height=tile_data.shape[0],
                        width=tile_data.shape[1],
                        count=1,
                        dtype=tile_data.dtype,
                        crs=crs,
                        transform=tile_transform,
                        nodata=nodata
                ) as dst:
                    dst.write(tile_data, 1)


if __name__ == '__main__':
    split_tif(input_path=r'C:\Users\Kevin\Desktop\result\test_30_copernicus.tif', output_dir=r'C:\Users\Kevin\Desktop\result\Copernicus_30', tile_size=50, overlap=1)