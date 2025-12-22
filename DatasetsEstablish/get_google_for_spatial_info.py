#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os.path

import arcpy
import sys

# -------------------------- 1. 基础参数配置 --------------------------
aprx = arcpy.mp.ArcGISProject('CURRENT')
arcpy.env.addOutputsToMap = False

layout = aprx.listLayouts("Export_Map")[0]  # 或指定具体布局名称
map_frame = layout.listElements("MAPFRAME_ELEMENT")[0]


mapx = map_frame.map

# 图层配置
google_layer = mapx.listLayers("Google_Satellite_GIS思维")[0]
copernicus_map_group = mapx.listLayers("PrivateData")[0]
silted_land_group = mapx.listLayers("SiltedLand")[0]

# 获取目标要素层
target_dam_layer = None
for lyr in silted_land_group.listLayers():
    if lyr.name == "selected_silted_land":
        target_dam_layer = lyr
        break
if not target_dam_layer:
    print("错误：未在SiltedLand图层组中找到selected_silted_land！", file=sys.stderr)
    sys.exit(1)

# 获取DEM
dem_layer = None
for lyr in copernicus_map_group.listLayers():
    if lyr.name == "Loess_Plateau_Copernicus.tif":
        dem_layer = lyr
        break
if not dem_layer:
    print("错误：未在PrivateData图层组中找到Loess_Plateau_Copernicus.tif！", file=sys.stderr)
    sys.exit(1)

output_dir = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\test"
remote_sensing_output_dir = os.path.join(output_dir, "Google")
dem_output_dir = os.path.join(output_dir, "DEM")
if not os.path.exists(remote_sensing_output_dir):
    os.makedirs(remote_sensing_output_dir)
if not os.path.exists(dem_output_dir):
    os.makedirs(dem_output_dir)


# -------------------------- 2. DEM网格参数（关键对齐逻辑） --------------------------
dem_desc = arcpy.Describe(dem_layer)
dem_sr = dem_desc.spatialReference
dem_cell_lon = dem_desc.meanCellWidth  # 经度方向分辨率（度）
dem_cell_lat = dem_desc.meanCellHeight  # 纬度方向分辨率（度）
lon_degree_buffer = dem_cell_lon * 3
lat_degree_buffer = dem_cell_lat * 3
print(f"lon: {lon_degree_buffer}, lat: {lat_degree_buffer}")

# 验证DEM分辨率有效性
if dem_cell_lon <= 0 or dem_cell_lat <= 0:
    print("错误：DEM分辨率无效（非正数）！", file=sys.stderr)
    sys.exit(1)

# 强制使用x/y方向分辨率的精确值（避免平均导致的偏差）
dem_origin_lon = dem_desc.extent.XMin  # 网格原点经度（最小经度）
dem_origin_lat = dem_desc.extent.YMin  # 网格原点纬度（最小纬度）

# -------------------------- 3. 图层可见性 --------------------------
google_layer.visible = True
copernicus_map_group.visible = False
silted_land_group.visible = False

# -------------------------- 4. 处理要素并导出TIFF --------------------------
try:
    total_count = int(arcpy.GetCount_management(target_dam_layer)[0])
    if total_count == 0:
        print("错误：目标要素层中无要素！", file=sys.stderr)
        sys.exit(1)
except Exception as e:
    print(f"错误：获取要素总数失败 - {str(e)}", file=sys.stderr)
    sys.exit(1)

current_count = 0
error_ids = []  # 存储报错的ID

with arcpy.da.SearchCursor(target_dam_layer, ["OBJECTID", "SHAPE@"], spatial_reference=dem_sr) as cursor:
    for row in cursor:
        try:
            oid = row[0]
            dam_geometry = row[1]

            # 获取要素经纬度范围
            dam_extent = dam_geometry.extent
            dam_min_lon, dam_max_lon = dam_extent.XMin, dam_extent.XMax
            dam_min_lat, dam_max_lat = dam_extent.YMin, dam_extent.YMax

            dam_min_lon -= lon_degree_buffer
            dam_max_lon += lon_degree_buffer
            dam_min_lat -= lat_degree_buffer
            dam_max_lat += lat_degree_buffer

            # -------------------------- 关键：DEM网格精确对齐 --------------------------
            # 1. 计算要素范围在DEM网格中的索引（左/下边界索引）
            # 公式：索引 = floor((要素坐标 - 原点坐标) / 分辨率)
            min_lon_index = int((dam_min_lon - dem_origin_lon) // dem_cell_lon)
            min_lat_index = int((dam_min_lat - dem_origin_lat) // dem_cell_lat)
            # 右/上边界索引（+1确保包含要素）
            max_lon_index = int((dam_max_lon - dem_origin_lon) // dem_cell_lon) + 1
            max_lat_index = int((dam_max_lat - dem_origin_lat) // dem_cell_lat) + 1

            # 2. 计算包含要素的最小正方形网格数
            lon_grid_count = max_lon_index - min_lon_index  # 经度方向网格数
            lat_grid_count = max_lat_index - min_lat_index  # 纬度方向网格数
            square_grid_count = max(lon_grid_count, lat_grid_count)  # 正方形网格数

            # 3. 计算正方形中心索引（确保要素居中）
            center_lon_index = min_lon_index + (lon_grid_count // 2)
            center_lat_index = min_lat_index + (lat_grid_count // 2)

            # 4. 计算对齐后的正方形范围（严格基于DEM网格）
            # 左边界 = 原点 + 起始索引 × 分辨率
            new_min_lon = dem_origin_lon + (center_lon_index - square_grid_count // 2) * dem_cell_lon
            new_max_lon = new_min_lon + square_grid_count * dem_cell_lon  # 右边界 = 左边界 + 总宽度
            new_min_lat = dem_origin_lat + (center_lat_index - square_grid_count // 2) * dem_cell_lat
            new_max_lat = new_min_lat + square_grid_count * dem_cell_lat  # 上边界 = 下边界 + 总高度

            # 5. 强制范围为正方形（避免计算误差）
            assert abs((new_max_lon - new_min_lon) - (new_max_lat - new_min_lat)) < 1e-8, "范围不是正方形"

            # -------------------------- 导出TIFF --------------------------
            aligned_extent = arcpy.Extent(new_min_lon, new_min_lat, new_max_lon, new_max_lat)
            aligned_extent.spatialReference = dem_sr
            aprx.activeView.camera.setExtent(aligned_extent)

            output_path = f"{remote_sensing_output_dir}\\{oid}.tif"
            aprx.activeView.exportToTIFF(
                output_path,
                width=1024,
                height=1024,
                tiff_compression=1,
                color_mode="24-BIT_TRUE_COLOR",
                geoTIFF_tags=True
            )

            # -------------------------- 导出对应范围的DEM数据 --------------------------
            # 用Clip工具裁剪DEM到aligned_extent范围
            dem_output_path = f"{dem_output_dir}\\{oid}.tif"  # DEM命名格式：DEM_ID.tif
            arcpy.management.Clip(
                in_raster=dem_layer,  # 输入原始DEM
                rectangle=f"{new_min_lon} {new_min_lat} {new_max_lon} {new_max_lat}",  # 裁剪范围（左 下 右 上）
                out_raster=dem_output_path,  # 输出裁剪后的DEM
                nodata_value="-9999",  # DEM无数据值（根据原始DEM调整，常见-9999）
                clipping_geometry=False,
                maintain_clipping_extent="NO_MAINTAIN_EXTENT"
            )

        except Exception as e:
            # 仅记录报错的ID
            error_ids.append(f"ID {oid}：{str(e)}")
        finally:
            # 更新进度条
            current_count += 1
            progress = int((current_count / total_count) * 100)
            # 手动绘制进度条（\r覆盖当前行，不换行）
            bar = '#' * (progress // 2) + '-' * (50 - progress // 2)
            sys.stdout.write(f"\r[{bar}] {progress}% 完成 ({current_count}/{total_count})")
            sys.stdout.flush()

# 进度条结束后换行
print()

# 输出所有报错的ID
if error_ids:
    print("\n以下ID处理失败：")
    for err in error_ids:
        print(err, file=sys.stderr)
else:
    print("\n所有要素处理完成，无错误。")