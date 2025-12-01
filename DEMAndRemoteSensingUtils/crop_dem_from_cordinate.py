#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: crop_dem_from_cordinate
# @Time    : 2025/9/4 17:53
# @Author  : Kevin
# @Describe:

import rasterio
import rasterio.mask
import math
from shapely.geometry import box
import json


def add_buffer_to_bounds(lon_min, lat_min, lon_max, lat_max, buffer_distance_km):
    """
    给经纬度边界添加缓冲距离（单位：公里）

    参数:
        lon_min, lat_min, lon_max, lat_max: 原始边界
        buffer_distance_km: 缓冲距离，单位公里

    返回:
        带缓冲的新边界 (new_lon_min, new_lat_min, new_lon_max, new_lat_max)
    """
    # 地球半径（公里）
    earth_radius_km = 6371.0

    # 将缓冲距离转换为弧度
    buffer_rad = buffer_distance_km / earth_radius_km

    # 计算纬度方向的缓冲（近似值）
    lat_buffer_deg = math.degrees(buffer_rad)

    # 计算经度方向的缓冲（基于平均纬度）
    avg_lat_rad = math.radians((lat_min + lat_max) / 2)
    lon_buffer_deg = math.degrees(buffer_rad / math.cos(avg_lat_rad))

    # 应用缓冲
    new_lon_min = lon_min - lon_buffer_deg
    new_lat_min = lat_min - lat_buffer_deg
    new_lon_max = lon_max + lon_buffer_deg
    new_lat_max = lat_max + lat_buffer_deg

    return new_lon_min, new_lat_min, new_lon_max, new_lat_max


def crop_tif_by_bounds(input_tif_path, output_tif_path,
                       lon_min, lat_min, lon_max, lat_max,
                       buffer_distance_km=2):
    """
    根据经纬度边界裁剪TIF文件，可选添加缓冲

    参数:
        input_tif_path: 输入TIF文件路径
        output_tif_path: 输出裁剪后的TIF文件路径
        lon_min, lat_min, lon_max, lat_max: 裁剪边界
        buffer_distance_km: 缓冲距离，单位公里，默认为0
    """
    # 添加缓冲
    if buffer_distance_km > 0:
        lon_min, lat_min, lon_max, lat_max = add_buffer_to_bounds(
            lon_min, lat_min, lon_max, lat_max, buffer_distance_km)

    # 创建裁剪区域的几何形状
    bbox = box(lon_min, lat_min, lon_max, lat_max)
    geojson = [json.loads(json.dumps(bbox.__geo_interface__))]

    # 打开输入文件并裁剪
    with rasterio.open(input_tif_path) as src:
        # 检查数据的坐标参考系统是否为WGS84 (EPSG:4326)
        # 如果不是，rasterio会自动进行坐标转换
        out_image, out_transform = rasterio.mask.mask(src, geojson, crop=True)
        out_meta = src.meta.copy()

        # 更新元数据
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # 写入输出文件
        with rasterio.open(output_tif_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(f"裁剪完成，输出文件: {output_tif_path}")


# 使用示例
if __name__ == "__main__":
    # 输入文件路径
    input_tif = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    # 输出文件路径
    output_tif = r"C:\Users\Kevin\Desktop\test\output_cropped.tif"

    # 定义裁剪边界（经纬度）
    lon_min, lat_min = 110.347, 37.595  # 最小经度，最小纬度
    lon_max, lat_max = 110.348, 37.596  # 最大经度，最大纬度

    # 缓冲距离（公里）
    buffer_km = 2  # 100公里缓冲

    # 执行裁剪
    try:
        crop_tif_by_bounds(input_tif, output_tif,
                           lon_min, lat_min, lon_max, lat_max,
                           buffer_distance_km=buffer_km)
        print("操作成功完成")
    except Exception as e:
        print(f"操作失败: {str(e)}")
