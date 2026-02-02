#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: generation_Copernicus_from_DAM
# @Time    : 2026/1/30 16:00
# @Author  : Kevin
# @Describe:


import os
import os.path as osp
import sys
import math
import numpy as np
import rasterio
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds, Window

current_file_path = osp.abspath(__file__)
sys.path.append(osp.join(osp.dirname(current_file_path), ".."))

from LocalPath import (
    dam_root_path, dam_google_remote_root_path,
    dam_copernicus_dem_root_path,
    dam_usgs_dem_down_link_file, dam_usgs_dem_down_file, dam_usgs_dem_delete_info
)


def merge_grouped_tifs(input_dir, output_prefix, n_groups=2, overlap_strategy='mean'):
    """
    将tif文件按数量分成n组，每组生成一个合并文件
    例如n_groups=2：前50%文件 -> _part1.tif，后50% -> _part2.tif
    """
    tif_files = sorted([
        osp.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith('.tif') and osp.isfile(osp.join(input_dir, f))
    ])

    if not tif_files:
        print(f"警告: {input_dir} 中没有tif文件")
        return []

    total = len(tif_files)
    files_per_group = total // n_groups

    print(f"\n[分组拼接] 共 {total} 个文件，分 {n_groups} 组")
    output_files = []

    for group_idx in range(n_groups):
        start_idx = group_idx * files_per_group
        end_idx = start_idx + files_per_group if group_idx < n_groups - 1 else total

        group_files = tif_files[start_idx:end_idx]
        output_file = f"{output_prefix}_part{group_idx + 1}.tif"
        output_files.append(output_file)

        if osp.exists(output_file):
            print(f"  组{group_idx + 1}({len(group_files)}个文件): {output_file} 已存在")
            continue

        print(f"  组{group_idx + 1}({len(group_files)}个文件): 生成 {output_file}")
        _merge_single_group(group_files, output_file, overlap_strategy)

    return output_files


def _merge_single_group(tif_files, output_path, overlap_strategy='mean'):
    """合并单组，使用窗口分块避免内存溢出"""

    with rasterio.open(tif_files[0]) as ref_ds:
        ref_crs = ref_ds.crs
        ref_bands = ref_ds.count
        ref_dtype = ref_ds.dtypes[0]
        ref_nodata = ref_ds.nodata if ref_ds.nodata is not None else (
            np.nan if np.issubdtype(ref_dtype, np.floating) else 0)
        pixel_size_x = abs(ref_ds.transform.a)
        pixel_size_y = abs(ref_ds.transform.e)

    # 计算边界
    src_datasets = []
    all_bounds = []
    for path in tif_files:
        ds = rasterio.open(path)
        src_datasets.append(ds)
        b = ds.bounds
        all_bounds.extend([b.left, b.right, b.bottom, b.top])

    global_left = min(all_bounds[::4])
    global_right = max(all_bounds[1::4])
    global_bottom = min(all_bounds[2::4])
    global_top = max(all_bounds[3::4])

    global_width = int(round((global_right - global_left) / pixel_size_x))
    global_height = int(round((global_top - global_bottom) / pixel_size_y))

    # 创建空文件
    global_transform = rasterio.Affine(pixel_size_x, 0, global_left, 0, -pixel_size_y, global_top)
    with rasterio.open(output_path, 'w', driver='GTiff',
                       height=global_height, width=global_width,
                       count=ref_bands, dtype=ref_dtype, crs=ref_crs,
                       transform=global_transform, nodata=ref_nodata,
                       compress='LZW', tiled=True, blockxsize=256, blockysize=256):
        pass

    # 分块处理（每次处理4096x4096像素）
    block_size = 4096
    n_row_blocks = math.ceil(global_height / block_size)
    n_col_blocks = math.ceil(global_width / block_size)

    for row_block in range(n_row_blocks):
        for col_block in range(n_col_blocks):
            row_start = row_block * block_size
            col_start = col_block * block_size
            row_end = min(row_start + block_size, global_height)
            col_end = min(col_start + block_size, global_width)
            win_h, win_w = row_end - row_start, col_end - col_start

            # 窗口地理范围
            win_left = global_left + col_start * pixel_size_x
            win_top = global_top - row_start * pixel_size_y

            # 累积数据
            if overlap_strategy == 'mean':
                win_data = np.zeros((ref_bands, win_h, win_w), dtype=ref_dtype)
                count_array = np.zeros((win_h, win_w), dtype=np.uint32)
            else:
                win_data = np.full((ref_bands, win_h, win_w), ref_nodata, dtype=ref_dtype)

            # 从每个源文件读取该窗口
            for ds in src_datasets:
                ds_bounds = ds.bounds
                # 快速空间检查
                if (ds_bounds.right <= win_left or ds_bounds.left >= win_left + win_w * pixel_size_x or
                        ds_bounds.top <= win_top - win_h * pixel_size_y or ds_bounds.bottom >= win_top):
                    continue

                # 计算源窗口
                src_win = from_bounds(
                    win_left, win_top - win_h * pixel_size_y,
                              win_left + win_w * pixel_size_x, win_top,
                    ds.transform
                )
                src_win = src_win.intersection(Window(0, 0, ds.width, ds.height))

                if src_win.width <= 0 or src_win.height <= 0:
                    continue

                data = ds.read(window=src_win)

                # 计算目标位置
                dst_row = int(round((win_top - ds_bounds.top) / pixel_size_y)) - row_start
                dst_col = int(round((ds_bounds.left - win_left) / pixel_size_x)) - col_start
                dst_row = max(0, dst_row)
                dst_col = max(0, dst_col)

                h, w = data.shape[1], data.shape[2]
                h = min(h, win_h - dst_row)
                w = min(w, win_w - dst_col)

                if h <= 0 or w <= 0:
                    continue

                # 合并
                for band in range(ref_bands):
                    band_data = data[band, :h, :w]
                    target = win_data[band, dst_row:dst_row + h, dst_col:dst_col + w]

                    if np.isnan(ref_nodata):
                        valid = ~np.isnan(band_data)
                    else:
                        valid = (band_data != ref_nodata)

                    if overlap_strategy == 'mean':
                        target[valid] += band_data[valid]
                        if band == 0:
                            count_array[dst_row:dst_row + h, dst_col:dst_col + w][valid] += 1
                    else:
                        target[valid] = band_data[valid]

            # 均值处理
            if overlap_strategy == 'mean':
                for band in range(ref_bands):
                    valid = count_array > 0
                    win_data[band][valid] = (win_data[band][valid] / count_array[valid]).astype(ref_dtype)
                    win_data[band][~valid] = ref_nodata

            # 写入
            with rasterio.open(output_path, 'r+') as dst:
                dst.write(win_data, window=Window(col_start, row_start, win_w, win_h))

    for ds in src_datasets:
        ds.close()
    print(f"    ✓ 完成")


def crop_source_to_reference_multi(source_paths, reference_dir, output_dir,
                                   output_suffix="", resampling_method=Resampling.bilinear,
                                   log_csv=None):
    """
    支持多个源文件（如part1, part2），自动合并重叠区域（取平均）
    """
    if not isinstance(source_paths, list):
        source_paths = [source_paths]

    os.makedirs(output_dir, exist_ok=True)
    ref_files = [f for f in os.listdir(reference_dir) if f.lower().endswith('.tif')]

    print(f"\n[多源裁剪] 源文件: {len(source_paths)}个, 参考: {len(ref_files)}个")

    # 打开所有源
    src_datasets = [rasterio.open(p) for p in source_paths if osp.exists(p)]
    if not src_datasets:
        print("错误: 无有效源文件")
        return

    for ref_name in ref_files:
        ref_path = osp.join(reference_dir, ref_name)
        out_path = osp.join(output_dir, osp.splitext(ref_name)[0] + output_suffix + ".tif")

        with rasterio.open(ref_path) as ref_ds:
            ref_crs, ref_bounds = ref_ds.crs, ref_ds.bounds
            ref_h, ref_w = ref_ds.shape
            ref_transform, ref_nodata = ref_ds.transform, ref_ds.nodata or np.nan
            ref_dtype = ref_ds.dtypes[0]

            # 找出有重叠的源
            valid_sources = []
            for src_ds in src_datasets:
                src_b = src_ds.bounds
                if ref_crs != src_ds.crs:
                    src_b = transform_bounds(src_ds.crs, ref_crs, *src_b)

                if (ref_bounds.left < src_b.right and ref_bounds.right > src_b.left and
                        ref_bounds.top > src_b.bottom and ref_bounds.bottom < src_b.top):
                    valid_sources.append(src_ds)

            if not valid_sources:
                continue

            print(f"  {ref_name} (匹配{len(valid_sources)}个源)")

            # 累积数组
            accum = np.zeros((ref_ds.count, ref_h, ref_w), dtype=np.float64)
            counts = np.zeros((ref_h, ref_w), dtype=np.uint8)

            for src_ds in valid_sources:
                src_crs = src_ds.crs
                src_nodata = src_ds.nodata or np.nan

                # 重投影该源到参考网格
                if ref_crs != src_crs:
                    src_bounds = transform_bounds(ref_crs, src_crs, *ref_bounds)
                else:
                    src_bounds = ref_bounds

                # 读取窗口
                px = abs(src_ds.transform.a)
                py = abs(src_ds.transform.e)
                win = from_bounds(
                    src_bounds.left - px, src_bounds.bottom - py,
                    src_bounds.right + px, src_bounds.top + py,
                    src_ds.transform
                )
                win = win.intersection(Window(0, 0, src_ds.width, src_ds.height))

                if win.width <= 0 or win.height <= 0:
                    continue

                src_data = src_ds.read(window=win)
                src_trans = rasterio.windows.transform(win, src_ds.transform)

                # 重投影到参考影像
                reprojected = np.full((src_ds.count, ref_h, ref_w), ref_nodata, dtype=ref_dtype)
                for i in range(src_ds.count):
                    reproject(source=src_data[i], destination=reprojected[i],
                              src_transform=src_trans, src_crs=src_crs,
                              dst_transform=ref_transform, dst_crs=ref_crs,
                              resampling=resampling_method,
                              src_nodata=src_nodata, dst_nodata=ref_nodata)

                # 有效掩码
                if np.isnan(ref_nodata):
                    valid = ~np.isnan(reprojected[0])
                else:
                    valid = (reprojected[0] != ref_nodata)

                # 累加
                for band in range(min(accum.shape[0], reprojected.shape[0])):
                    accum[band][valid] += reprojected[band][valid]
                counts[valid] += 1

            # 平均
            for band in range(accum.shape[0]):
                valid = counts > 0
                accum[band][valid] /= counts[valid]
                accum[band][~valid] = ref_nodata

            # 保存
            with rasterio.open(out_path, 'w', driver='GTiff',
                               height=ref_h, width=ref_w,
                               count=accum.shape[0], dtype=ref_dtype,
                               crs=ref_crs, transform=ref_transform,
                               nodata=ref_nodata, compress='LZW') as dst:
                dst.write(accum.astype(ref_dtype))

    for ds in src_datasets:
        ds.close()


if __name__ == '__main__':
    n_groups = 2  # 设置分组数，2=两半，3=三等分

    for file in os.listdir(dam_copernicus_dem_root_path):

        if file != "GeoDAR_v11_dams_of_USA_group14":
            continue

        current_dir = osp.join(dam_copernicus_dem_root_path, file)

        if osp.isdir(current_dir) and "paired" not in file:
            # 生成 part1.tif, part2.tif
            part_files = merge_grouped_tifs(current_dir, current_dir, n_groups=n_groups)

            # 处理这n个文件（自动合并重叠）
            if part_files:
                crop_source_to_reference_multi(
                    source_paths=part_files,
                    reference_dir=osp.join(dam_google_remote_root_path, file),
                    output_dir=osp.join(dam_copernicus_dem_root_path, file + "_paired"),
                    output_suffix="",
                    resampling_method=Resampling.bilinear,
                    log_csv=osp.join(osp.join(dam_copernicus_dem_root_path, file + "_paired"), "log.csv")
                )