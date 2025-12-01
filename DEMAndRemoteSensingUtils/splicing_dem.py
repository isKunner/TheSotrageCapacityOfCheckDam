#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: splicing_dem
# @Time    : 2025/8/4 16:13
# @Author  : Kevin
# @Describe: 将DEM拼接为一个完整的大的DEM，支持有重叠区域

import os
import numpy as np
import rasterio

def merge_geo_referenced_tifs(input_dir, output_path, overlap_strategy='mean'):
    """
    自定义拼接统一坐标系TIF，支持重叠区域平均值
    """
    # 1. 收集所有TIF文件
    tif_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith('.tif') and os.path.isfile(os.path.join(input_dir, f))
    ]

    if not tif_files:
        raise ValueError(f"输入文件夹 {input_dir} 未找到TIF文件")
    print(f"找到 {len(tif_files)} 个TIF文件，开始处理...")

    # 2. 读取参考文件元数据
    with rasterio.open(tif_files[0]) as ref_ds:
        ref_crs = ref_ds.crs
        ref_bands = ref_ds.count
        ref_dtype = ref_ds.dtypes[0]
        ref_nodata = ref_ds.nodata
        # 关键：如果源文件没有定义nodata，手动指定（根据数据类型调整）
        if ref_nodata is None:
            ref_nodata = np.nan if np.issubdtype(ref_dtype, np.floating) else 0
        ref_transform = ref_ds.transform
        pixel_size_x = abs(ref_transform.a)
        pixel_size_y = abs(ref_transform.e)

    # 3. 验证所有文件并保持打开状态
    src_datasets = []
    for tif_path in tif_files:
        ds = rasterio.open(tif_path)
        if ds.crs != ref_crs:
            ds.close()
            raise ValueError(f"文件 {tif_path} 坐标系不一致")
        if ds.count != ref_bands:
            ds.close()
            raise ValueError(f"文件 {tif_path} 波段数不一致")
        # 检查当前文件的nodata是否与参考一致（避免掩码判断错误）
        if ds.nodata is not None and not np.isclose(ds.nodata, ref_nodata):
            print(f"警告：文件 {tif_path} 的nodata值与参考文件不一致，可能导致数据丢失")
        src_datasets.append(ds)

    # 4. 计算全局地理边界
    all_bounds = []
    for ds in src_datasets:
        bounds = ds.bounds
        all_bounds.extend([bounds.left, bounds.right, bounds.bottom, bounds.top])

    global_left = min(all_bounds[::4])
    global_right = max(all_bounds[1::4])
    global_bottom = min(all_bounds[2::4])
    global_top = max(all_bounds[3::4])

    # 5. 计算拼接后尺寸和变换矩阵
    global_width = int(round((global_right - global_left) / pixel_size_x))
    global_height = int(round((global_top - global_bottom) / pixel_size_y))
    global_transform = rasterio.Affine(
        pixel_size_x, 0, global_left,
        0, -pixel_size_y, global_top
    )

    # 6. 初始化结果数组
    if overlap_strategy == 'mean':
        # 均值计算需要累加，初始化为0（避免Nodata污染计算）
        merged_data = np.zeros((ref_bands, global_height, global_width), dtype=ref_dtype)
    else:
        # 其他策略仍初始化为Nodata
        merged_data = np.full(
            (ref_bands, global_height, global_width),
            ref_nodata,
            dtype=ref_dtype
        )
    count_array = np.zeros((global_height, global_width), dtype=np.uint32)

    # 7. 遍历每个TIF处理数据
    for idx, ds in enumerate(src_datasets):
        ds_name = os.path.basename(ds.name)

        # 获取当前TIF的地理边界和数据
        ds_bounds = ds.bounds
        ds_data = ds.read()  # 形状：(bands, height, width)

        # 检查源数据是否有有效像素（关键：避免源数据本身全是nodata）
        band_valid = []
        for band in range(ref_bands):
            valid = np.sum(ds_data[band] != ref_nodata)
            band_valid.append(valid)
        if all(v == 0 for v in band_valid):
            print(f"  警告：文件 {ds_name} 所有波段均为nodata，跳过处理")
            continue

        # 计算当前TIF在全局栅格中的位置
        start_x = max(0, int(round((ds_bounds.left - global_left) / pixel_size_x)))
        start_y = max(0, int(round((global_top - ds_bounds.top) / pixel_size_y)))
        end_x = min(global_width, int(round((ds_bounds.right - global_left) / pixel_size_x)))
        end_y = min(global_height, int(round((global_top - ds_bounds.bottom) / pixel_size_y)))

        # 检查目标区域是否有效（避免坐标计算错误导致区域为空）
        if start_x >= end_x or start_y >= end_y:
            print(f"  警告：文件 {ds_name} 目标区域为空，跳过处理")
            continue

        # 尺寸适配
        ds_height, ds_width = ds_data.shape[1], ds_data.shape[2]
        if (end_y - start_y != ds_height) or (end_x - start_x != ds_width):
            end_y = start_y + ds_height
            end_x = start_x + ds_width
            end_y = min(end_y, global_height)
            end_x = min(end_x, global_width)
            ds_height = end_y - start_y
            ds_width = end_x - start_x
            ds_data = ds_data[:, :ds_height, :ds_width]

        # 处理每个波段
        for band in range(ref_bands):
            band_data = ds_data[band]
            target_region = merged_data[band, start_y:end_y, start_x:end_x]

            # 计算有效掩码（关键：确保正确识别有效像素）
            if np.isnan(ref_nodata):
                # 处理浮点型nodata（nan不能直接用==判断）
                valid_mask = ~np.isnan(band_data)
            else:
                valid_mask = (band_data != ref_nodata)

            valid_count = np.sum(valid_mask)
            if valid_count == 0:
                continue  # 无有效像素，跳过

            # 根据策略写入数据
            if overlap_strategy == 'mean':
                target_region[valid_mask] += band_data[valid_mask]
                count_array[start_y:end_y, start_x:end_x][valid_mask] += 1
            elif overlap_strategy == 'first':
                empty_mask = (target_region == ref_nodata) & valid_mask
                target_region[empty_mask] = band_data[empty_mask]
            elif overlap_strategy == 'last':
                target_region[valid_mask] = band_data[valid_mask]
            elif overlap_strategy == 'max':
                target_region[valid_mask] = np.maximum(target_region[valid_mask], band_data[valid_mask])
            elif overlap_strategy == 'min':
                target_region[valid_mask] = np.minimum(target_region[valid_mask], band_data[valid_mask])

    # count_array = count_array // 3

    # 8. 处理mean策略（计算平均值）
    if overlap_strategy == 'mean':
        for band in range(ref_bands):
            non_zero_mask = (count_array > 0)
            # 避免除零和空值
            if np.sum(non_zero_mask) == 0:
                print(f"  波段 {band + 1} 无重叠区域，无需计算平均值")
                continue
            # 用浮点数除法保留精度，再转换为目标类型
            merged_data[band, non_zero_mask] = (
                    merged_data[band, non_zero_mask] / count_array[non_zero_mask]
            ).astype(ref_dtype)

            merged_data[band, ~non_zero_mask] = ref_nodata

    # 9. 保存结果
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=global_height,
            width=global_width,
            count=ref_bands,
            dtype=ref_dtype,
            crs=ref_crs,
            transform=global_transform,
            nodata=ref_nodata
    ) as dst:
        dst.write(merged_data)

    print(f"\n拼接完成！结果已保存至: {output_path}")

    # 关闭所有文件
    for ds in src_datasets:
        ds.close()


if __name__ == "__main__":
    input_folder = r"C:\Users\Kevin\Desktop\result\Copernicus_10"
    output_file = r"C:\Users\Kevin\Desktop\result\test_30_copernicus_temp.tif"

    merge_geo_referenced_tifs(
        input_dir=input_folder,
        output_path=output_file,
        overlap_strategy='mean'
    )
