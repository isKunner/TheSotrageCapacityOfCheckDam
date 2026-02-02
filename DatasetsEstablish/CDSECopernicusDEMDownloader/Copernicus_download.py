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

from .cdse_copernicus_dem_downloader import DemDownloader
from .credentials.credentials import Credentials


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


def download_copernicu_file(
        dam_google_remote_root_dir: str,
        dem_downloader,
        access_token: str,
        downloaded_tracker: Set[str],
        down_dict_info: Dict
):
    """
    History TIF -> Calculate tiles -> Real-time weight -> Download missing
    """

    group_name = os.path.basename(dam_google_remote_root_dir)

    tif_files = [f for f in os.listdir(dam_google_remote_root_dir) if f.endswith('.tif')]

    for google_remote_file in tqdm(tif_files, desc=f"处理 {group_name}"):
        try:

            # print(google_remote_file)

            tif_path = os.path.join(dam_google_remote_root_dir, google_remote_file)

            # 1. Calculate which Copernicus tiles are needed for this TIF (usually 1-4)
            required_tiles = get_copernicus_tiles_from_tif(tif_path)

            # Initialize the record structure
            if group_name not in down_dict_info:
                down_dict_info[group_name] = {}
            down_dict_info[group_name][google_remote_file] = {}

            # 2. Check/download one by one (core: real-time weighing)
            for tile_name in required_tiles:

                # **Real-time weight check**: Is it present in the memory collection?
                if tile_name in downloaded_tracker:
                    down_dict_info[group_name][google_remote_file][tile_name] = "exists_in_session"
                    continue

                # Local file check (previously downloaded)
                has_tif = os.path.exists(os.path.join(dem_downloader.dem_directory, f"{tile_name}.tif"))

                if has_tif:
                    downloaded_tracker.add(tile_name)  # 加入内存，后续TIF不再下载
                    down_dict_info[group_name][google_remote_file][tile_name] = "exists_locally"
                    continue

                # 3. Download required: Call the original tool API
                try:
                    # Set the query area to the exact boundary of the tile
                    dem_downloader.polygon = tile_to_polygon(tile_name)

                    # Query Product ID
                    search_url = dem_downloader.create_url()
                    dem_ids = dem_downloader.retrieve_dem_list(search_url)

                    if not dem_ids or dem_ids == 1:
                        print(f"[!] {tile_name} 未找到产品")
                        down_dict_info[group_name][google_remote_file][tile_name] = "product_not_found"
                        continue

                    # Perform a download (usually only 1 ID) and download only one
                    print(f"[下载] {google_remote_file} -> {tile_name}")
                    dem_downloader.downloading_dem(dem_ids[0], access_token)

                    # **Key**: Join the collection immediately, and the subsequent TIF will skip the tile directly
                    downloaded_tracker.add(tile_name)
                    down_dict_info[group_name][google_remote_file][tile_name] = "downloaded"

                    # Real-time saving (interruption-proof)
                    with open(os.path.join(dem_downloader.dem_directory, "Copernicus_DownloadInfo.json"), 'w', encoding='utf-8') as f:
                        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

                except Exception as e:
                    print(f"[错误] {tile_name}: {e}")
                    down_dict_info[group_name][google_remote_file][tile_name] = f"error:{e}"

        except Exception as e:
            print(f'[错误] 处理 {google_remote_file}: {e}')

    # 最终保存
    info_path = os.path.join(dem_downloader.dem_directory, "Copernicus_DownloadInfo.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    print(f"[完成] {group_name}: 已累积 {len(downloaded_tracker)} 个唯一瓦片")


# ==================== 使用方式（和你的USGS_download.py完全一致）====================
def download_copernicus(ref_inputs, output_dirs, resolution="30", dem_format="DGED"):
    dem_downloader = DemDownloader()
    dem_downloader.dem_resolution = resolution  # 或 "90"
    dem_downloader.dem_format = dem_format  # 或 "DTED"

    auth = Credentials()
    access_token, _ = dem_downloader.get_access_token(auth.id, auth.password)

    downloaded_tiles = set()
    down_dict_info = {}

    for res_input, output_dir in zip(ref_inputs, output_dirs):
        dem_downloader.dem_directory = output_dir
        os.makedirs(output_dir, exist_ok=True)
        download_copernicu_file(
            dam_google_remote_root_dir=res_input,
            dem_downloader=dem_downloader,
            access_token=access_token,
            downloaded_tracker=downloaded_tiles,
            down_dict_info=down_dict_info
        )

if __name__ == '__main__':

    from LocalPath import (
        dam_google_remote_root_path,
        dam_copernicus_dem_root_path
    )
