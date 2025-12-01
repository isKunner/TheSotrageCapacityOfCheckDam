#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: utils
# @Time    : 2025/8/16 09:03
# @Author  : Kevin
# @Describe:

import rasterio

def read_tif(path):
    """读取TIFF数据及地理信息（替代gdal实现）"""
    with rasterio.open(path) as src:
        # 读取数据数组
        data = src.read(1)  # 读取第一个波段
        # 获取地理变换信息（仿射矩阵）
        geotrans = src.transform
        # 获取投影信息
        proj = src.crs.to_wkt()  # 转换为WKT格式，与gdal输出格式一致
        # 获取无数据值
        nodata = src.nodata if src.nodata is not None else -9999.0
    return data, geotrans, proj, nodata

def write_tif(save_path, data, geotrans, proj, nodata_value=-9999.0):
    """保存TIFF文件（替代gdal实现）"""
    # 获取数据形状
    rows, cols = data.shape

    # 设置TIFF文件的元数据
    profile = {
        'driver': 'GTiff',
        'width': cols,
        'height': rows,
        'count': 1,  # 波段数
        'dtype': data.dtype,
        'crs': proj,
        'transform': geotrans,
        'nodata': nodata_value,
        'compress': 'lzw'  # 启用压缩，可根据需要修改
    }

    # 写入数据
    with rasterio.open(save_path, 'w', **profile) as dst:
        dst.write(data, 1)  # 写入第一个波段