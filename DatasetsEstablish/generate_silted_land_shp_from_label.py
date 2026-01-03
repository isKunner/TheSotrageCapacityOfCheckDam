#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: generate_silted_land_shp_from_label_with_grouping.py
# @Time : 2025/12/16 11:00
# @Author : Assistant (Based on your code)
# @Describe : Generates unified slope/road Shapefile from LabelMe JSONs, deduplicates, then groups components.
#             FIXED: Uses original geo coords for polygon storage, only projects a copy for calculations.
#             ADDED: Groups features based on angle similarity and spatial proximity.

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import json
import argparse
from typing import List

import cv2
import numpy as np
from tqdm import tqdm
from rtree import index

import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
from pyproj import CRS, Transformer

from DEMAndRemoteSensingUtils import get_geotransform_and_crs, pixel_to_geo_coords, calculate_meters_per_degree_precise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate and group slope/road Shapefile from LabelMe JSONs')

    # Path configurations
    parser.add_argument('--json-label-dir', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\GoogleLabel",
                        help='Directory containing LabelMe JSON files')
    parser.add_argument('--tif-dir', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\Google",
                        help='Directory containing TIF files for georeferencing')
    parser.add_argument('--output-shp-path', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG\check_dam_struct.shp",
                        help='Output Shapefile path')

    # Target labels
    parser.add_argument('--target-labels', nargs='+', default=["slope", "road"], help='Target labels to process')

    # Deduplication parameters
    parser.add_argument('--overlap-threshold', type=float, default=0.5,
                        help='Overlap threshold for deduplication')

    # Grouping parameters
    parser.add_argument('--buffer-distance-meters', type=float, default=1,  # Adjusted default
                        help='Buffer distance in meters for handling small gaps')
    parser.add_argument('--group-distance-meters', type=float, default=50,  # Adjusted default
                        help='Buffer distance in meters for handling small gaps')
    parser.add_argument('--angle-threshold-deg', type=float, default=10.0,
                        help='Angle threshold in degrees for grouping')

    return parser.parse_args()


def is_duplicate(new_geom: Polygon, label: str, existing_geoms: List[Polygon], spatial_index,
                 threshold: float) -> bool:
    """Use the R-tree spatial index to check if the new geometry is a duplicate."""
    new_area = new_geom.area
    if new_area == 0:
        return True  # Consider zero-area geom as duplicate

    candidate_ids = list(spatial_index.intersection(new_geom.bounds))
    for idx in candidate_ids:
        existing_geom = existing_geoms[idx]
        try:
            intersection = new_geom.intersection(existing_geom)
            if intersection.is_empty or intersection.area == 0:
                continue
            existing_area = existing_geom.area
            overlap_ratio_new = intersection.area / new_area if new_area > 0 else 0
            overlap_ratio_existing = intersection.area / existing_area if existing_area > 0 else 0

            if overlap_ratio_new > threshold or overlap_ratio_existing > threshold:
                return True
        except Exception as e:
            pass  # Shapely operations can sometimes fail silently during dup check
    return False


def process_labels_from_jsons_to_gdf(args):

    json_files = [f for f in os.listdir(args.json_label_dir) if f.lower().endswith('.json')]

    # Lists to store unique geometries and their attributes
    unique_geometries_geo_crs = []  # 存储最终在地理 CRS 下的 shapely Polygon (来自原始 geo_points)
    unique_labels = []
    opencv_angles_deg = []  # 存储符合 OpenCV 4.5+ 格式的角度 (度)
    center_coords = []
    geo_heights_deg = []
    geo_widths_deg = []
    geo_heights_meters = []  # 存储地理长度 (米)
    geo_widths_meters = []  # 存储地理宽度 (米)
    unique_filenames = []

    # Spatial indexes and structures categorized by labels
    spatial_indexes = {}
    geometries_by_label = {}  # 用于去重的几何体字典 (将在地理 CRS 中进行)

    total_extracted = 0
    total_duplicates = 0
    crs_wkt_for_gdf_geo = None  # To store the GEOGRAPHIC CRS from the first processed TIF
    output_crs_is_geographic = True  # Flag to track if final GDF CRS should be geographic

    pbar = tqdm(total=len(json_files), desc="Processing JSON files", unit="file")
    for json_filename in json_files:

        json_path = os.path.join(args.json_label_dir, json_filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        image_path_from_json = data.get("imagePath", "")
        tif_filename_stem = os.path.splitext(os.path.basename(image_path_from_json))[0]
        tif_filename_full = f"{tif_filename_stem}.tif"
        tif_path = os.path.join(args.tif_dir, tif_filename_full)

        geotransform, crs_wkt = get_geotransform_and_crs(tif_path)

        src_crs = CRS.from_wkt(crs_wkt) if crs_wkt else None

        if crs_wkt_for_gdf_geo is None and src_crs is not None:
            crs_wkt_for_gdf_geo = crs_wkt
            output_crs_is_geographic = src_crs.is_geographic

        shapes_data = data.get("shapes", [])
        features_found_in_file = 0
        duplicates_in_file = 0

        for shape_obj in shapes_data:
            label = shape_obj.get("label", "")
            points = shape_obj.get("points", [])

            if label not in spatial_indexes:
                spatial_indexes[label] = index.Index()
                geometries_by_label[label] = []

            if label in args.target_labels:
                geo_points = [pixel_to_geo_coords(px, py, geotransform) for px, py in points]

                polygon_geom_geo = Polygon(geo_points)

                total_extracted += 1

                is_dup = is_duplicate(polygon_geom_geo, label, geometries_by_label[label], spatial_indexes[label],
                                      args.overlap_threshold)

                center_geo, (width_deg, height_deg), angle_geo = cv2.minAreaRect(np.array(geo_points, dtype=np.float32))

                if not is_dup:
                    unique_geometries_geo_crs.append(polygon_geom_geo)
                    unique_labels.append(label)
                    opencv_angles_deg.append(angle_geo)
                    center_coords.append(center_geo)
                    geo_heights_deg.append(height_deg)
                    geo_widths_deg.append(width_deg)
                    unique_filenames.append(tif_filename_full)
                    features_found_in_file += 1

                    scale_lon, scale_lat = calculate_meters_per_degree_precise(center_geo[0], center_geo[1])

                    angle_rad = math.radians(angle_geo)

                    # 分解旋转矩形的宽度和高度在经纬度方向的分量（度）
                    # 宽度分量（经向和纬向）
                    width_lon_deg = width_deg * math.cos(angle_rad)
                    width_lat_deg = width_deg * math.sin(angle_rad)

                    # 高度分量（经向和纬向）
                    height_lon_deg = height_deg * math.sin(angle_rad)
                    height_lat_deg = height_deg * math.cos(angle_rad)

                    # 转换为实际米数（考虑经纬度缩放差异）
                    width_m = math.hypot(
                        width_lon_deg * scale_lon,  # 经度方向距离（米）
                        width_lat_deg * scale_lat  # 纬度方向距离（米）
                    )
                    height_m = math.hypot(
                        height_lon_deg * scale_lon,  # 经度方向距离（米）
                        height_lat_deg * scale_lat  # 纬度方向距离（米）
                    )

                    # 存储实际距离（米）
                    geo_widths_meters.append(width_m)
                    geo_heights_meters.append(height_m)

                else:
                    total_duplicates += 1
                    duplicates_in_file += 1

        pbar.update(1)

    pbar.close()

    gdf = gpd.GeoDataFrame({
        'label': unique_labels,
        'angle_deg': opencv_angles_deg,
        'center_lon': [coord[0] for coord in center_coords],
        'center_lat': [coord[1] for coord in center_coords],
        'width_deg': geo_widths_deg,
        'height_deg': geo_heights_deg,
        'width_m': geo_widths_meters,
        'height_m': geo_heights_meters,
        'source': unique_filenames
    }, geometry=unique_geometries_geo_crs, crs=crs_wkt_for_gdf_geo)  # Set CRS to the geographic CRS from TIF

    if not gdf.empty:
        # 1. 对角度列排序并保留原始索引（确保分组逻辑正确）
        sorted_angles = gdf['angle_deg'].sort_values().reset_index()
        sorted_angles.columns = ['original_index', 'sorted_angle']  # 重命名列

        # 2. 初始化分组ID
        group_id = 0
        sorted_angles['angle_group_id'] = 0  # 存储分组结果
        base_angle = sorted_angles['sorted_angle'].iloc[0]  # 第一个角度作为基准

        # 3. 遍历排序后的角度，按阈值分组
        for i in range(1, len(sorted_angles)):
            current_angle = sorted_angles['sorted_angle'].iloc[i]
            # 若当前角度与基准角度差距>5度，创建新组
            if current_angle - base_angle > 5:
                group_id += 1
                base_angle = current_angle  # 更新基准角度为当前角度
            sorted_angles.loc[i, 'angle_group_id'] = group_id

        # 4. 将分组结果映射回原始GeoDataFrame
        angle_group_map = dict(zip(sorted_angles['original_index'], sorted_angles['angle_group_id']))
        gdf['angle_id'] = gdf.index.map(angle_group_map)

        """
        
        可能会出现错误，应该是距离阈值越大越不容易出错，这相当于给矩形分块了，可能本来是同一个淤地坝但是正好被边缘分开
        
        gdf['dis_id'] = -1  # 初始化最终分组ID
        dis_id = 0

        for angle_id in gdf['angle_id'].unique():
            # 获取该 angle_id 组的所有组件
            angle_group = gdf[gdf['angle_id'] == angle_id]

            if len(angle_group) > 0:
                # 按经纬度排序
                sorted_indices = angle_group.sort_values(['center_lon', 'center_lat']).index.tolist()

                # 简化的经纬度阈值
                require_lon = args.group_distance_meters/111000.0
                require_lat = args.group_distance_meters/111000.0

                visited = set()

                # 遍历所有未访问的点
                for idx in sorted_indices:
                    if idx in visited:
                        continue

                    # 为新的一组分配ID
                    gdf.loc[idx, 'dis_id'] = dis_id

                    current_lon = gdf.loc[idx, 'center_lon']
                    current_lat = gdf.loc[idx, 'center_lat']

                    # 检查其他点是否在同一组内
                    for other_idx in sorted_indices:
                        if other_idx in visited:
                            continue

                        other_lon = gdf.loc[other_idx, 'center_lon']
                        other_lat = gdf.loc[other_idx, 'center_lat']

                        # 判断是否在同一组（经纬度差值都在阈值内）
                        if (abs(other_lon - current_lon) < require_lon and
                                abs(other_lat - current_lat) < require_lat):
                            gdf.loc[other_idx, 'dis_id'] = dis_id
                            visited.add(other_idx)

                    visited.add(idx)
                    dis_id += 1
    """

    gdf_buffered = gdf.copy()
    gdf_buffered['geometry'] = gdf.geometry.buffer(args.buffer_distance_meters/111000.0, join_style=2)

    result = find_connected_components_in_angle_group(gdf_buffered)
    group_id_counter = 0
    for group_info in result:
        indices = group_info['indices']
        type_num = 0
        for idx in indices:
            gdf.loc[idx, 'group_id'] = group_id_counter
            type_num += 1 if gdf.loc[idx, 'label']=='slope' else 10
        for idx in indices:
            gdf.loc[idx, 'group_type'] = type_num
        group_id_counter += 1

    gdf.to_file(args.output_shp_path, encoding='utf-8', driver='ESRI Shapefile')

def find_connected_components_in_angle_group(gdf_buffered, ID='angle_id'):
    """Find all connected components within each angle group."""
    # 获取所有唯一的angle_id
    unique_ids = gdf_buffered[ID].unique()

    all_groups_result = []

    # 对每个angle_id组分别处理
    for id in unique_ids:
        # 获取该angle_id组中所有未分配的组件
        unassigned_components = gdf_buffered[
            (gdf_buffered[ID] == id)
        ].index.tolist()

        if not unassigned_components:
            continue

        visited = set()

        # 遍历所有未分配的组件
        for start_idx in unassigned_components:
            if start_idx in visited:
                continue

            # 对每个未访问的组件进行BFS查找连通组件
            connected = set()
            queue = [start_idx]

            while queue:
                current_idx = queue.pop(0)
                if current_idx in connected or current_idx in visited:
                    continue
                connected.add(current_idx)
                visited.add(current_idx)

                current_buffered_geom = gdf_buffered.loc[current_idx, 'geometry']

                # 查找相交的未访问组件
                intersecting = gdf_buffered[
                    (gdf_buffered[ID] == id) &
                    (gdf_buffered.geometry.intersects(current_buffered_geom)) &
                    (~gdf_buffered.index.isin(visited))
                ].index
                queue.extend(intersecting)

            if connected:
                all_groups_result.append({
                    'indices': list(connected),
                    'angle_id': id
                })

    return all_groups_result



if __name__ == '__main__':

    process_labels_from_jsons_to_gdf(parse_args())