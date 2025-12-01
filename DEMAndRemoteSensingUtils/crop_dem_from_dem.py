#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: crop_dem_from_dem
# @Time    : 2025/7/24 17:52
# @Author  : Kevin
# @Describe: 根据已有的裁剪好的tif进行重新的裁剪采样(方形)
import math
import os
import rasterio
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.windows import from_bounds, Window
import numpy as np

def extract_matching_files(input_dir, extracted_input_file, output_dir, input_file_name=None, output_path=None):
    """
    从目标影像中提取与源瓦片对应的区域并重采样
    Args:
        input_dir: 裁剪的区域的文件的目录
        extracted_input_file: 需要被裁剪的数据
        output_dir: 保存的目录的文件夹
        input_file_name: 如果为None，则对input_dir下的所有目录进行重采样，否则只对单独的那个进行重采样
        output_path: 保存的文件的路径，完整的（适用于单个的路径）
    """
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(extracted_input_file) as target_ds:

        b_crs = target_ds.crs
        b_bounds = target_ds.bounds  # 目标影像全局范围
        b_nodata = target_ds.nodata if target_ds.nodata is not None else None  # 不假设NoData值
        b_count = target_ds.count
        b_transform = target_ds.transform  # 目标影像全局transform
        b_full_mask = target_ds.dataset_mask()  # 0=无效，255=有效

        if input_file_name is None:
            tile_names = [name for name in os.listdir(input_dir) if name.endswith('.tif')]
        else:
            tile_names = [input_file_name]

        for tile_name in tile_names:
            tile_path = os.path.join(input_dir, tile_name)
            try:
                with rasterio.open(tile_path) as a_tile:
                    a_crs = a_tile.crs
                    a_bounds = a_tile.bounds  # 源瓦片范围
                    a_height, a_width = a_tile.shape
                    a_transform = a_tile.transform  # 源瓦片transform
                    a_nodata = a_tile.nodata if a_tile.nodata is not None else 0

                    # 1. 转换坐标，用于判断两个块是否有重叠（一般都是有的，以防万一）；同时转化为了在b坐标系下的位置，就知道了应该提取b的哪个位置
                    if a_crs != b_crs:
                        # 转换源瓦片边界到目标坐标系（rasterio原生方式）
                        a_bounds_in_b_crs = transform_bounds(
                            a_crs, b_crs,
                            a_bounds.left, a_bounds.bottom,
                            a_bounds.right, a_bounds.top
                        )
                        a_left, a_bottom, a_right, a_top = a_bounds_in_b_crs
                    else:
                        a_left, a_bottom, a_right, a_top = (
                            a_bounds.left, a_bounds.bottom,
                            a_bounds.right, a_bounds.top
                        )

                    # 2. 检查重叠
                    overlap = (
                        a_left < b_bounds.right and
                        a_right > b_bounds.left and
                        a_top > b_bounds.bottom and
                        a_bottom < b_bounds.top
                    )
                    if not overlap:
                        print(f"警告: 瓦片 {tile_name} 与目标影像无重叠，跳过")
                        continue

                    # 3. 扩大窗口范围（多扩1像素）
                    # 计算目标影像的像素尺寸（x和y方向）
                    pixel_size_x = abs(b_transform.a)
                    pixel_size_y = abs(b_transform.e)
                    # 向四周各扩大1个像素（确保覆盖）
                    a_left_expanded = a_left - pixel_size_x
                    a_right_expanded = a_right + pixel_size_x
                    a_bottom_expanded = a_bottom - pixel_size_y
                    a_top_expanded = a_top + pixel_size_y

                    # 4. 计算窗口（基于扩大后的范围）
                    # TIF的变换矩阵：Affine(30, 0, 450000, 0, -30, 3500000)
                    # 地理范围：(450100, 349900, 450200, 350000)  # 米

                    # 转换为像素范围：
                    # 列：(450100-450000)/30 = 3.33 → 左边界
                    # 列：(450200-450000)/30 = 6.67 → 右边界
                    # 行：(3500000-350000)/30 = 0 → 上边界
                    # 行：(3500000-349900)/30 = 3.33 → 下边界

                    # 结果：Window(col_off=3.33, row_off=0, width=3.34, height=3.33)

                    window = rasterio.windows.from_bounds(
                        a_left_expanded, a_bottom_expanded,
                        a_right_expanded, a_top_expanded,
                        transform=b_transform
                    )

                    # 与目标影像边界取交集
                    # 计算出的窗口：Window(col_off=-5, row_off=100, width=50, height=30)
                    # 影像实际范围：Window(0, 0, 1000, 800)

                    # intersection后：
                    # Window(col_off=0, row_off=100, width=45, height=30)
                    # 自动裁剪掉超出的部分
                    window = window.intersection(Window(0, 0, target_ds.width, target_ds.height))

                    # 5. 窗口坐标计算（用floor和ceil确保完整包含）
                    win_col_min = math.floor(window.col_off)
                    win_row_min = math.floor(window.row_off)
                    win_col_max = math.ceil(window.col_off + window.width)
                    win_row_max = math.ceil(window.row_off + window.height)
                    # 确保不超出范围
                    win_col_min = max(0, win_col_min)
                    win_row_min = max(0, win_row_min)
                    win_col_max = min(target_ds.width, win_col_max)
                    win_row_max = min(target_ds.height, win_row_max)
                    window = Window(
                        col_off=win_col_min,
                        row_off=win_row_min,
                        width=win_col_max - win_col_min,
                        height=win_row_max - win_row_min
                    )

                    # 6. 读取目标影像窗口数据
                    b_data = target_ds.read(window=window)
                    # 根据窗口在原影像中的位置，计算该窗口区域的新左上角坐标
                    # 用于后续的重投影操作
                    # 原变换：Affine(30, 0, 450000, 0, -30, 3500000)
                    # 窗口：Window(col_off=10, row_off=20, width=100, height=100)

                    # 新变换：
                    # 新左上角x = 450000 + 10×30 = 450300
                    # 新左上角y = 3500000 + 20×(-30) = 3499400
                    # 结果：Affine(30, 0, 450300, 0, -30, 3499400)
                    b_window_transform = rasterio.windows.transform(window, b_transform)

                    # 7. 调试信息
                    # print(f"\n瓦片 {tile_name} 处理信息:")
                    # print(f"源瓦片范围（目标坐标系）: 左={a_left:.2f}, 右={a_right:.2f}, 下={a_bottom:.2f}, 上={a_top:.2f}")
                    # print(f"扩大后的窗口范围: 列={win_col_min}-{win_col_max}, 行={win_row_min}-{win_row_max}, 形状={b_data.shape}")

                    # 8. 读取窗口掩码并检查中间有效性
                    window_mask = b_full_mask[win_row_min:win_row_max, win_col_min:win_col_max]
                    valid_pixels_in_mask = np.sum(window_mask == 255)
                    total_pixels_in_mask = window_mask.size
                    mask_valid_ratio = valid_pixels_in_mask / total_pixels_in_mask if total_pixels_in_mask > 0 else 0
                    # print(f"窗口掩码有效率: {mask_valid_ratio*100:.2f}% (中间区域应接近100%)")

                    # 9. 初始化输出数组
                    dst_array = np.full((b_count, a_height, a_width), a_nodata, dtype=b_data.dtype)

                    # 10. 重投影（用最近邻，确保中间像素映射）
                    for band_idx in range(b_count):
                        reproject(
                            source=b_data[band_idx],
                            destination=dst_array[band_idx],
                            src_transform=b_window_transform,
                            src_crs=b_crs,
                            dst_transform=a_transform,
                            dst_crs=a_crs,
                            resampling=Resampling.nearest,
                            src_nodata=b_nodata,
                            dst_nodata=a_nodata,
                            src_mask=window_mask if mask_valid_ratio > 0 else None
                        )

                    # 11. 验证中间区域
                    mid_row = a_height // 2
                    mid_col = a_width // 2
                    # 检查中间20x20区域（更大范围，更准确）
                    mid_region = dst_array[:,
                                          max(0, mid_row-10):min(a_height, mid_row+10),
                                          max(0, mid_col-10):min(a_width, mid_col+10)]
                    mid_valid = np.sum(mid_region != a_nodata)
                    mid_total = mid_region.size
                    if mid_valid/mid_total < 0.99:
                        print(f"中间区域有效率低于99%，请检查！")

                    # 12. 保存结果
                    if output_path is None:
                        output_path = os.path.join(output_dir, tile_name)
                        print(output_path)
                    with rasterio.open(
                        output_path, 'w',
                        driver='GTiff',
                        height=a_height,
                        width=a_width,
                        count=b_count,
                        dtype=dst_array.dtype,
                        crs=a_crs,
                        transform=a_transform,
                        nodata=a_nodata,
                        compress='LZW'
                    ) as dst:
                        dst.write(dst_array)
                        dst_mask = np.all(dst_array != a_nodata, axis=0)
                        dst.write_mask(dst_mask)
                    output_path = None

            except Exception as e:
                output_path = None
                print(f"处理瓦片 {tile_name} 时出错: {e}")

if __name__ == '__main__':

    extract_matching_files(input_dir=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Google",
                           extracted_input_file=r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif",
                           output_dir=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\DEM",
                           input_file_name=None,)