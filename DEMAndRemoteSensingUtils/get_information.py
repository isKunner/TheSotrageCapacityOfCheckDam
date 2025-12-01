#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: get_information
# @Time    : 2025/7/26 22:27
# @Author  : Kevin
# @Describe: 获取Tif栅格数据的基本属性信息

import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
import pyproj
from Logger import LoggerManager

def get_pixel_size_accurate(raster_path, is_print=False):
    logger = LoggerManager.get_logger()

    """
    计算栅格的大小，以m为单位
    :param raster_path:
    :return:
    """

    with rasterio.open(raster_path) as src:
        # 1. 确认数据使用地理坐标系（经纬度）
        if not src.crs.is_geographic:
            pixel_size_m_x = abs(src.transform[0])  # X方向像元大小（米）
            pixel_size_m_y = abs(src.transform[4])
            return pixel_size_m_x, pixel_size_m_y

        # 2. 计算栅格中心经纬度（作为转换基准点）
        center_lon = (src.bounds.left + src.bounds.right) / 2
        center_lat = (src.bounds.top + src.bounds.bottom) / 2

        # 3. 创建从地理坐标系到UTM的转换器（使用中心经纬度确定UTM带）
        utm_zone = int((center_lon + 180) / 6) + 1
        # 判断南北半球
        if center_lat > 0:
            utm_crs = f"EPSG:326{utm_zone}"  # 北半球
        else:
            utm_crs = f"EPSG:327{utm_zone}"  # 南半球

        transformer = Transformer.from_crs(
            src.crs.to_string(),  # 源坐标系（地理坐标系）
            utm_crs,  # 目标坐标系（UTM投影）
            always_xy=True  # 确保经纬度顺序为x=经度，y=纬度
        )

        # 4. 计算像元大小（度）
        pixel_size_deg_x = src.transform[0]  # X方向像元大小（度）
        pixel_size_deg_y = abs(src.transform[4])  # Y方向像元大小（度，取绝对值）

        # 5. 将中心经纬度转换为UTM坐标
        x_center, y_center = transformer.transform(center_lon, center_lat)

        # 6. 计算经纬度方向各偏移1个像元后的UTM坐标
        lon_east = center_lon + pixel_size_deg_x  # 向东偏移1个像元
        lat_north = center_lat + pixel_size_deg_y  # 向北偏移1个像元
        x_east, y_east = transformer.transform(lon_east, center_lat)
        x_north, y_north = transformer.transform(center_lon, lat_north)

        # 7. 计算实际距离（米）
        pixel_size_m_x = abs(x_east - x_center)  # X方向像元大小（米）
        pixel_size_m_y = abs(y_north - y_center)  # Y方向像元大小（米）

        # 计算分辨率比例（检查是否为正方形像素）
        ratio = pixel_size_m_x / pixel_size_m_y

        if is_print:

            logger.info(f"数据中心经纬度：{center_lon:.6f}°, {center_lat:.6f}°")
            logger.info(f"对应UTM带：{'北半球' if center_lat > 0 else '南半球'} Zone {utm_zone}")
            logger.info(f"像元大小（度）：X={pixel_size_deg_x:.10f}°, Y={pixel_size_deg_y:.10f}°")
            logger.info(f"像元大小（米）：X={pixel_size_m_x:.2f}m, Y={pixel_size_m_y:.2f}m")
            logger.info(f"X/Y分辨率比例：{ratio:.4f}（理想值为1，表示正方形像素）")

        return pixel_size_m_x, pixel_size_m_y

def get_tif_latlon_bounds(tif_path):
    logger = LoggerManager.get_logger()
    """获取TIFF文件的经纬度坐标范围（无论原坐标系如何）"""
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # 原始坐标范围，格式为(left, bottom, right, top)
        src_crs = src.crs  # 原始坐标系

        # 判断是否已经是地理坐标系（经纬度），EPSG:4326是WGS84经纬度
        if src_crs.is_geographic or src_crs.to_epsg() == 4326:
            latlon_bounds = bounds
        else:
            # 创建转换器：原始坐标系 → WGS84经纬度
            transformer = Transformer.from_crs(
                src_crs, "EPSG:4326", always_xy=True
            )
            # 转换四个角点的坐标
            left, bottom = transformer.transform(bounds.left, bounds.bottom)
            right, top = transformer.transform(bounds.right, bounds.top)
            latlon_bounds = (left, bottom, right, top)

        # print(f"原始坐标系: {src_crs}")
        # print(f"原始坐标范围 (left, bottom, right, top): {bounds}")
        logger.info(f"经纬度范围 (lon_min, lat_min, lon_max, lat_max): {latlon_bounds}")
        return latlon_bounds

def get_crs_transformer(src_crs, dst_crs="EPSG:4326"):
    """创建坐标参考系转换器"""
    if src_crs == dst_crs:
        return None

    # 确保输入是正确的CRS格式
    if isinstance(src_crs, str):
        src_crs = pyproj.CRS(src_crs)
    if isinstance(dst_crs, str):
        dst_crs = pyproj.CRS(dst_crs)

    return Transformer.from_crs(src_crs, dst_crs, always_xy=True)

def geo_to_pixel(src, lon, lat, is_cv=False):
    logger = LoggerManager.get_logger()
    """
    将经纬度(WGS84)转换为大DEM中的像素坐标

    cv2是(cols, rows)，PIL.image和np.array是(rows, cols)
    """
    # 获取大DEM的坐标参考系
    large_crs = src.crs

    # 如果大DEM不是WGS84，将经纬度转换为大DEM的投影坐标
    if large_crs.to_string() != "EPSG:4326":
        transformer = get_crs_transformer("EPSG:4326", large_crs)
        if transformer:
            proj_x, proj_y = transformer.transform(lon, lat)
        else:
            proj_x, proj_y = lon, lat
    else:
        proj_x, proj_y = lon, lat
    row, col = rowcol(src.transform, proj_x, proj_y)

    logger.info(f"投影坐标: {proj_x}, {proj_y} → 像素坐标: {col}, {row}")

    if is_cv:
        return col, row

    return row, col  # 交换row和col的顺序

def pixel_to_geo(src, row, col):
    logger = LoggerManager.get_logger()
    """
    将像素坐标转换为经纬度(WGS84)
    输入：col（像素列坐标）, row（像素行坐标）
    返回：(lon, lat)（经度，纬度），严格遵循地理坐标顺序
    """
    # 获取原始坐标参考系
    src_crs = src.crs

    # 计算投影坐标（输入为像素列、行）
    proj_x, proj_y = src.transform * (col, row)

    # 转换为WGS84经纬度
    if src_crs.to_string() == "EPSG:4326":
        lon, lat = proj_x, proj_y  # 直接对应(lon, lat)
    else:
        transformer = get_crs_transformer(src_crs, "EPSG:4326")
        if transformer:
            lon, lat = transformer.transform(proj_x, proj_y)  # 转换后为(lon, lat)
        else:
            lon, lat = proj_x, proj_y  # 降级处理，仍保持返回格式

    logger.info(f"像素坐标: {col}, {row} → 投影坐标: {proj_x}, {proj_y} → 经纬度: {lon}, {lat}")

    return lon, lat  # 明确返回(经度, 纬度)