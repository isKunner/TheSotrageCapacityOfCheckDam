#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: run_with_geojson
# @Time    : 2026/1/25 11:09
# @Author  : Kevin
# @Describe: 根据现有的TIF数据来下载DEM，但前提是得确保下小于2°的TIF，否则可能失效

import os
import re
import json
import math
from typing import Set, Dict, List
import rasterio
import pyproj
from tqdm import tqdm


def get_copernicus_tiles_from_tif(tif_path: str) -> List[str]:
    """
    根据TIF四角坐标，计算需要哪些Copernicus瓦片（1°x1°）
    返回瓦片基础名列表，如 ["Copernicus_DSM_10_N34_00_W081_00_DEM"]
    """
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # (left, bottom, right, top)
        src_crs = src.crs

        # 转为WGS84（GEE导出的通常是4326，但保险起见）
        if src_crs and src_crs.to_epsg() != 4326:
            transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            left, bottom = transformer.transform(bounds.left, bounds.bottom)
            right, top = transformer.transform(bounds.right, bounds.top)
        else:
            left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

    # 计算覆盖的整数度网格（西南角坐标系统）
    lat_min = int(math.floor(min(bottom, top)))
    lat_max = int(math.floor(max(bottom, top)))
    lon_min = int(math.floor(min(left, right)))
    lon_max = int(math.floor(max(left, right)))

    tiles = []
    for lat in range(lat_min, lat_max + 1):
        ns = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
        for lon in range(lon_min, lon_max + 1):
            if lon < 0:
                ew = f"W{abs(lon):03d}"  # 西经取绝对值，如 -81 -> W081
            else:
                ew = f"E{lon:03d}"
            tile_name = f"Copernicus_DSM_10_{ns}_00_{ew}_00_DEM"
            tiles.append(tile_name)

    return tiles


def tile_to_polygon(tile_name: str) -> str:
    """
    生成缩小范围的多边形（避免边界误差导致多下载）
    使用瓦片中心 ±0.05° 的区域
    """
    match = re.search(r'([NS])(\d{2})_00_([WE])(\d{3})_00', tile_name)
    if not match:
        raise ValueError(f"无法解析瓦片名: {tile_name}")

    lat_dir, lat_val, lon_dir, lon_val = match.groups()
    lat = int(lat_val)
    lon = int(lon_val)

    # 换算瓦片边界
    if lat_dir == 'N':
        min_lat, max_lat = lat, lat + 1
    else:
        min_lat, max_lat = -(lat + 1), -lat

    if lon_dir == 'W':
        min_lon, max_lon = -lon, -(lon - 1)  # 例如 -81 到 -80
    else:
        min_lon, max_lon = lon, lon + 1

    # 计算中心点
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    # 缩小范围：中心±0.05度（完全远离1度边界）
    buffer = 0.05

    return (f"{center_lon - buffer:.6f} {center_lat - buffer:.6f}, "
            f"{center_lon + buffer:.6f} {center_lat - buffer:.6f}, "
            f"{center_lon + buffer:.6f} {center_lat + buffer:.6f}, "
            f"{center_lon - buffer:.6f} {center_lat + buffer:.6f}, "
            f"{center_lon - buffer:.6f} {center_lat - buffer:.6f}")


def download_copernicus_for_group(
        dam_google_remote_root_dir: str,
        dem_downloader,  # DemDownloader 实例（已初始化）
        access_token: str,  # CDSE访问令牌
        output_dir: str,
        downloaded_tracker: Set[str],  # 全局已下载集合（实时判重核心）
        group_name: str,
        down_dict_info: Dict  # 记录字典
):
    """
    完全对标 down_load_file 的接口
    遍历TIF -> 计算瓦片 -> 实时判重 -> 下载缺失
    """
    os.makedirs(output_dir, exist_ok=True)

    tif_files = [f for f in os.listdir(dam_google_remote_root_dir) if f.endswith('.tif')]

    for google_remote_file in tqdm(tif_files, desc=f"处理 {group_name}"):
        try:

            # print(google_remote_file)

            tif_path = os.path.join(dam_google_remote_root_dir, google_remote_file)

            # 1. 计算该TIF需要哪些Copernicus瓦片（通常1-4个）
            required_tiles = get_copernicus_tiles_from_tif(tif_path)

            # 初始化记录结构
            if group_name not in down_dict_info:
                down_dict_info[group_name] = {}
            down_dict_info[group_name][google_remote_file] = {}

            # 2. 逐个检查/下载（核心：实时判重）
            for tile_name in required_tiles:

                # **实时判重检查**：内存集合中是否存在？
                if tile_name in downloaded_tracker:
                    down_dict_info[group_name][google_remote_file][tile_name] = "exists_in_session"
                    continue

                # 本地文件检查（之前下载的）
                has_tif = os.path.exists(os.path.join(output_dir, f"{tile_name}.tif"))

                if has_tif:
                    downloaded_tracker.add(tile_name)  # 加入内存，后续TIF不再下载
                    down_dict_info[group_name][google_remote_file][tile_name] = "exists_locally"
                    continue

                # 3. 需要下载：调用原工具API
                try:
                    # 设置查询区域为该瓦片精确边界
                    dem_downloader.polygon = tile_to_polygon(tile_name)

                    # 查询产品ID（你的原工具方法）
                    search_url = dem_downloader.create_url()
                    dem_ids = dem_downloader.retrieve_dem_list(search_url)

                    if not dem_ids or dem_ids == 1:
                        print(f"[!] {tile_name} 未找到产品")
                        down_dict_info[group_name][google_remote_file][tile_name] = "product_not_found"
                        continue

                    # 执行下载（通常只有1个ID） 并且只下载一个
                    print(f"[下载] {google_remote_file} -> {tile_name}")
                    dem_downloader.downloading_dem(dem_ids[0], access_token)

                    # **关键**：立即加入集合，后续TIF看到这个瓦片会直接跳过
                    downloaded_tracker.add(tile_name)
                    down_dict_info[group_name][google_remote_file][tile_name] = "downloaded"

                    # 实时保存（防中断）
                    with open(os.path.join(output_dir, "Copernicus_DownloadInfo.json"), 'w', encoding='utf-8') as f:
                        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"[错误] {tile_name}: {e}")
                    down_dict_info[group_name][google_remote_file][tile_name] = f"error:{e}"

        except Exception as e:
            print(f'[错误] 处理 {google_remote_file}: {e}')

    # 最终保存
    info_path = os.path.join(output_dir, "Copernicus_DownloadInfo.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    print(f"[完成] {group_name}: 已累积 {len(downloaded_tracker)} 个唯一瓦片")


# ==================== 使用方式（和你的USGS_download.py完全一致）====================

if __name__ == '__main__':
    from cdse_copernicus_dem_downloader import DemDownloader, DemDownloaderException
    from credentials.credentials import Credentials
    from LocalPath import (
        dam_google_remote_root_path,
        dam_copernicus_dem_root_path
    )

    # 1. 初始化（一次性）
    dem_downloader = DemDownloader()
    dem_downloader.dem_resolution = "30"  # 或 "90"
    dem_downloader.dem_format = "DGED"  # 或 "DTED"

    # 2. 获取Token（一次性）
    auth = Credentials()
    access_token, _ = dem_downloader.get_access_token(auth.id, auth.password)

    # 3. 全局判重集合（关键点：跨组共享）
    downloaded_tiles = set()
    down_dict_info = {}

    # 4. 处理多组（和你的循环完全一样）
    group_names = ["GeoDAR_v11_dams_of_USA_group14"]
    # group_names = ["GeoDAR_v11_dams_of_USA_group10", "GeoDAR_v11_dams_of_USA_group11", "GeoDAR_v11_dams_of_USA_group12", "GeoDAR_v11_dams_of_USA_group13_1"]

    for group_name in group_names:
        current_tif_dir = os.path.join(dam_google_remote_root_path, group_name)
        output_dir = os.path.join(dam_copernicus_dem_root_path, group_name)

        dem_downloader.dem_directory = output_dir

        download_copernicus_for_group(
            dam_google_remote_root_dir=current_tif_dir,
            dem_downloader=dem_downloader,
            access_token=access_token,
            output_dir=output_dir,
            downloaded_tracker=downloaded_tiles,  # 同一个集合贯穿所有组！
            group_name=group_name,
            down_dict_info=down_dict_info
        )

    print(f"[全部完成] 共下载 {len(downloaded_tiles)} 个唯一Copernicus瓦片")
