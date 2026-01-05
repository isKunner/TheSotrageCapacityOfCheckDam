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
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import math
import json
import argparse
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm
from rtree import index, Index

import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point

from DEMAndRemoteSensingUtils import get_geotransform_and_crs, pixel_to_geo_coords, calculate_meters_per_degree_precise


# ===================== Constant definition =====================
# Geographic conversion constant
METERS_PER_DEGREE_APPROX = 111000.0  # Approximation coefficient of latitude and longitude to latitude and longitude
BUFFER_JOIN_STYLE = 2  # Connection style for buffer geometry
# Group type constant
LABEL_TYPE_MAP = {
    "slope": 1,
    "road": 10
}
# Default threshold constant
DEFAULT_OVERLAP_THRESHOLD = 0.5
DEFAULT_ANGLE_THRESHOLD_DEG = 10.0
DEFAULT_BUFFER_DISTANCE_M = 1.0
DEFAULT_GROUP_DISTANCE_M = 50.0
# Deduplication logical constants
DUPLICATE_CHECK_LOGIC = "OR"  # OR: Any overlap rate meets the standard; AND: Both overlap rates are up to par
# ====================================================


def parse_args():
    """Parse command line arguments with explicit hard-coded params."""
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

    # Target labels (hard-coded support for slope/road only)
    parser.add_argument('--target-labels', nargs='+', default=["slope", "road"],
                        help='Target labels to process (HARD-CODED: only slope/road are supported)')

    # Deduplication parameters
    parser.add_argument('--overlap-threshold', type=float, default=DEFAULT_OVERLAP_THRESHOLD,
                        help='Overlap threshold for deduplication (0-1)')
    parser.add_argument('--duplicate-check-logic', type=str, default=DUPLICATE_CHECK_LOGIC,
                        choices=["OR", "AND"], help='Logic for duplicate check: OR/AND (overlap ratio new/existing)')

    # Grouping parameters
    parser.add_argument('--buffer-distance-meters', type=float, default=DEFAULT_BUFFER_DISTANCE_M,
                        help='Buffer distance (m) for small gap handling (HARD-CODED: converted via 1/111000 deg)')
    parser.add_argument('--group-distance-meters', type=float, default=DEFAULT_GROUP_DISTANCE_M,
                        help='Buffer distance (m) for silt dam group identification')
    parser.add_argument('--angle-threshold-deg', type=float, default=DEFAULT_ANGLE_THRESHOLD_DEG,
                        help='Angle threshold (deg) for angle-based grouping (0-90)')

    return parser.parse_args()


def is_duplicate(new_geom: Polygon, label: str, existing_geoms: List[Polygon], spatial_index,
                 threshold: float, check_logic: str = "OR") -> bool:
    """Use the R-tree spatial index to check if the new geometry is a duplicate."""
    if not new_geom.is_valid:
        new_geom = new_geom.buffer(0)
        if not new_geom.is_valid:
            return True

    new_area = new_geom.area
    if new_area <= 1e-9:
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

            if check_logic.upper() == "AND":
                is_dup = (overlap_ratio_new > threshold) and (overlap_ratio_existing > threshold)
            else:  # OR
                is_dup = (overlap_ratio_new > threshold) or (overlap_ratio_existing > threshold)

        except Exception as e:
            pass  # Shapely operations can sometimes fail silently during dup check
    return False


def get_json_files(json_dir: str) -> List[str]:
    """Get list of JSON files from directory (case-insensitive).

    Args:
        json_dir: Directory containing JSON files

    Returns:
        List of JSON filenames
    """
    return [f for f in os.listdir(json_dir) if f.lower().endswith('.json')]


def init_data_structures(target_labels: List[str]) -> tuple[
    dict[str, Index], dict[str, list[Any]], list[Any], list[Any], list[Any], list[Any], list[Any], list[Any], list[Any],
    list[Any], list[Any]]:
    """Initialize data structures for storing geometries and attributes.

    Args:
        target_labels: List of target labels to process

    Returns:
        Tuple of initialized data structures (spatial indexes, geometries by label, and attribute lists)
    """
    # Spatial indexes and geometries categorized by labels (for deduplication)
    spatial_indexes = {label: index.Index() for label in target_labels}
    geometries_by_label = {label: [] for label in target_labels}

    # Attribute lists for final GeoDataFrame
    unique_geometries_geo_crs = []  # Polygons in geographic CRS
    unique_labels = []
    opencv_angles_deg = []  # Angles in OpenCV format (degrees)
    center_coords = []  # Center coordinates (lon/lat)
    geo_heights_deg = []  # Height in degrees
    geo_widths_deg = []  # Width in degrees
    geo_heights_meters = []  # Height in meters
    geo_widths_meters = []  # Width in meters
    unique_filenames = []  # Source TIF filenames

    return (
        spatial_indexes, geometries_by_label,
        unique_geometries_geo_crs, unique_labels, opencv_angles_deg,
        center_coords, geo_heights_deg, geo_widths_deg,
        geo_heights_meters, geo_widths_meters, unique_filenames
    )


def calculate_geo_dimensions_meters(center_geo: Tuple[float, float], width_deg: float,
                                    height_deg: float, angle_geo: float) -> Tuple[float, float]:
    """Calculate width/height in meters from geographic degrees (considering lat/lon scaling).

    Args:
        center_geo: (lon, lat) of geometry center
        width_deg: Width in degrees (from minAreaRect)
        height_deg: Height in degrees (from minAreaRect)
        angle_geo: Angle in degrees (from minAreaRect)

    Returns:
        (width_m, height_m): Dimensions in meters
    """
    # Get meter-per-degree scale for current location
    scale_lon, scale_lat = calculate_meters_per_degree_precise(center_geo[0], center_geo[1])
    angle_rad = math.radians(angle_geo)

    # Decompose width/height into lon/lat components (degrees)
    width_lon_deg = width_deg * math.cos(angle_rad)
    width_lat_deg = width_deg * math.sin(angle_rad)

    height_lon_deg = height_deg * math.sin(angle_rad)
    height_lat_deg = height_deg * math.cos(angle_rad)

    # Convert to meters using hypotenuse (account for both lon/lat directions)
    width_m = math.hypot(width_lon_deg * scale_lon, width_lat_deg * scale_lat)
    height_m = math.hypot(height_lon_deg * scale_lon, height_lat_deg * scale_lat)

    return width_m, height_m


def process_single_shape(shape_obj: Dict, geotransform: Any, tif_filename_full: str,
                         spatial_indexes: Dict, geometries_by_label: Dict, args: Any,
                         attr_lists: Tuple[List, List, List, List, List, List, List, List, List]) -> Tuple[int, int]:
    """Process single shape object from LabelMe JSON.

    Args:
        shape_obj: Single shape dict from JSON "shapes" list
        geotransform: GDAL geotransform for TIF file
        tif_filename_full: Source TIF filename
        spatial_indexes: Spatial indexes by label
        geometries_by_label: Geometries list by label
        args: Command line arguments
        attr_lists: Tuple of attribute lists to populate

    Returns:
        (features_added, duplicates_found): Count of new features and duplicates
    """
    label = shape_obj.get("label", "")
    points = shape_obj.get("points", [])

    # Skip if label not in target labels
    if label not in args.target_labels:
        return 0, 0

    # Convert pixel points to geographic coordinates
    geo_points = [pixel_to_geo_coords(px, py, geotransform) for px, py in points]

    try:
        polygon_geom_geo = Polygon(geo_points)
        # 修复无效几何（如自相交）
        if not polygon_geom_geo.is_valid:
            polygon_geom_geo = polygon_geom_geo.buffer(0)  # 自动修复自相交
            if not polygon_geom_geo.is_valid:
                return 0, 1  # 无法修复则视为无效/重复
        # 过滤面积为0的几何
        if polygon_geom_geo.area <= 1e-9:
            return 0, 1
    except Exception as e:
        # 多边形创建失败（如点数<3）
        return 0, 1

    # Check for duplicates
    is_dup = is_duplicate(
        polygon_geom_geo, label,
        geometries_by_label[label],
        spatial_indexes[label],
        args.overlap_threshold,
        check_logic=args.duplicate_check_logic
    )

    if is_dup:
        return 0, 1

    # Calculate geometry properties using OpenCV minAreaRect
    center_geo, (width_deg, height_deg), angle_geo = cv2.minAreaRect(
        np.array(geo_points, dtype=np.float32)
    )

    # Calculate meters from degrees
    width_m, height_m = calculate_geo_dimensions_meters(
        center_geo, width_deg, height_deg, angle_geo
    )

    # Unpack attribute lists and populate
    (unique_geoms, unique_labels_list, angles, centers, heights_deg, widths_deg,
     heights_m, widths_m, filenames) = attr_lists

    unique_geoms.append(polygon_geom_geo)
    unique_labels_list.append(label)
    angles.append(angle_geo)
    centers.append(center_geo)
    heights_deg.append(height_deg)
    widths_deg.append(width_deg)
    heights_m.append(height_m)
    widths_m.append(width_m)
    filenames.append(tif_filename_full)

    # Add to spatial index/geometry list for future duplicate checks
    geometries_by_label[label].append(polygon_geom_geo)
    # spatial_indexes[label].insert(len(geometries_by_label[label]) - 1, polygon_geom_geo.bounds)

    return 1, 0


def process_single_json_file(json_filename: str, json_dir: str, tif_dir: str,
                             spatial_indexes: Dict, geometries_by_label: Dict, args: Any,
                             attr_lists: Tuple[List, ...]) -> Tuple[int, int, str]:
    """Process single LabelMe JSON file and extract geometries/attributes.

    Args:
        json_filename: Name of JSON file to process
        json_dir: Directory containing JSON file
        tif_dir: Directory containing TIF files
        spatial_indexes: Spatial indexes by label
        geometries_by_label: Geometries list by label
        args: Command line arguments
        attr_lists: Tuple of attribute lists to populate

    Returns:
        (features_found, duplicates_found, crs_wkt): Count of features/duplicates, and CRS WKT from TIF
    """
    json_path = os.path.join(json_dir, json_filename)

    # Read JSON content
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Get corresponding TIF file
    image_path_from_json = data.get("imagePath", "")
    tif_filename_stem = os.path.splitext(os.path.basename(image_path_from_json))[0]
    tif_filename_full = f"{tif_filename_stem}.tif"
    tif_path = os.path.join(tif_dir, tif_filename_full)

    # Get geotransform and CRS from TIF
    geotransform, crs_wkt = get_geotransform_and_crs(tif_path)

    # Process each shape in JSON
    shapes_data = data.get("shapes", [])
    features_found = 0
    duplicates_found = 0

    for shape_obj in shapes_data:
        feat_added, dup_found = process_single_shape(
            shape_obj, geotransform, tif_filename_full,
            spatial_indexes, geometries_by_label, args,
            attr_lists
        )
        features_found += feat_added
        duplicates_found += dup_found

    return features_found, duplicates_found, crs_wkt


def create_geo_dataframe(attr_lists: Tuple[List, ...], crs_wkt: str) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from collected attributes and geometries.

    Args:
        attr_lists: Tuple of attribute lists
        crs_wkt: CRS WKT string for geographic coordinate system

    Returns:
        GeoDataFrame with all features and attributes
    """
    (unique_geometries, unique_labels, opencv_angles, center_coords,
     geo_heights_deg, geo_widths_deg, geo_heights_m, geo_widths_m, filenames) = attr_lists

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'label': unique_labels,
        'angle_deg': opencv_angles,
        'center_lon': [coord[0] for coord in center_coords],
        'center_lat': [coord[1] for coord in center_coords],
        'width_deg': geo_widths_deg,
        'height_deg': geo_heights_deg,
        'width_m': geo_widths_m,
        'height_m': geo_heights_m,
        'source': filenames
    }, geometry=unique_geometries, crs=crs_wkt)

    return gdf


def assign_angle_groups(gdf: gpd.GeoDataFrame, angle_threshold: float = 5.0) -> gpd.GeoDataFrame:
    """Assign angle group IDs based on angle similarity threshold.

    Args:
        gdf: Input GeoDataFrame with 'angle_deg' column
        angle_threshold: Maximum angle difference for same group (degrees)

    Returns:
        GeoDataFrame with added 'angle_id' column
    """
    if gdf.empty:
        return gdf

    # Sort angles and preserve original index
    sorted_angles = gdf['angle_deg'].sort_values().reset_index()
    sorted_angles.columns = ['original_index', 'sorted_angle']

    # Initialize grouping
    group_id = 0
    sorted_angles['angle_group_id'] = 0
    base_angle = sorted_angles['sorted_angle'].iloc[0]

    # Assign angle groups
    for i in range(1, len(sorted_angles)):
        current_angle = sorted_angles['sorted_angle'].iloc[i]
        if current_angle - base_angle > angle_threshold:
            group_id += 1
            base_angle = current_angle
        sorted_angles.loc[i, 'angle_group_id'] = group_id

    # Map group IDs back to original GeoDataFrame
    angle_group_map = dict(zip(sorted_angles['original_index'], sorted_angles['angle_group_id']))
    gdf['angle_id'] = gdf.index.map(angle_group_map)

    return gdf


def build_spatial_index_batch(geoms: List[Polygon]) -> Index:
    """
    批量构建空间索引（优化版）
    核心优化：
    1. 过滤无效几何，避免索引异常
    2. 列表遍历替代iterrows，提升效率
    3. 0-based连续ID，索引更紧凑
    """
    idx = index.Index()  # 初始化空索引
    for idx_id, geom in enumerate(geoms):
        # 只插入有效几何，跳过无效/空几何（避免Shapely报错）
        if geom and geom.is_valid and not geom.is_empty:
            idx.insert(idx_id, geom.bounds)  # 用连续0-based ID作为索引ID
    return idx


def find_connected_components_in_angle_group(gdf_buffered: gpd.GeoDataFrame, id_col: str = 'angle_id') -> List[Dict]:
    """Find connected components within each angle group using spatial index and BFS.

    Args:
        gdf_buffered: GeoDataFrame with buffered geometries
        id_col: Column name for angle group ID

    Returns:
        List of group info dicts (indices and angle_id)
    """
    all_groups_result = []
    unique_ids = gdf_buffered[id_col].unique()

    for id_val in unique_ids:
        subset = gdf_buffered[gdf_buffered[id_col] == id_val]
        if subset.empty:
            continue

        # Build spatial index for current subset
        geom_list = subset['geometry'].tolist()
        spatial_idx = build_spatial_index_batch(geom_list)
        idx_mapping = {batch_id: original_idx for batch_id, original_idx in enumerate(subset.index)}

        # Find connected components using BFS
        visited = set()
        for start_idx in subset.index:
            if start_idx in visited:
                continue

            connected = set()
            queue = deque([start_idx])
            visited.add(start_idx)

            while queue:
                current_idx = queue.popleft()
                connected.add(current_idx)
                current_geom = subset.loc[current_idx, 'geometry']

                if not current_geom.is_valid:
                    continue

                # Find candidate geometries via spatial index
                batch_candidate_ids = list(spatial_idx.intersection(current_geom.bounds))
                candidate_ids = [idx_mapping[bid] for bid in batch_candidate_ids if bid in idx_mapping]

                for cand_idx in candidate_ids:
                    if cand_idx not in visited and subset.loc[cand_idx, 'geometry'].intersects(current_geom):
                        visited.add(cand_idx)
                        queue.append(cand_idx)

            if connected:
                all_groups_result.append({
                    'indices': list(connected),
                    'angle_id': id_val
                })

    return all_groups_result


def assign_connected_component_groups(gdf: gpd.GeoDataFrame, buffer_distance_m: float) -> gpd.GeoDataFrame:
    """Assign final group IDs based on connected components (angle + spatial proximity).

    Args:
        gdf: Input GeoDataFrame with 'angle_id' column
        buffer_distance_m: Buffer distance in meters (converted to degrees)

    Returns:
        GeoDataFrame with added 'group_id' and 'group_type' columns
    """
    if gdf.empty:
        return gdf

    # Create buffered copy (convert meters to degrees: ~1m = 1/111000 deg)
    gdf_buffered = gdf.copy()
    gdf_buffered['geometry'] = gdf.geometry.buffer(buffer_distance_m / METERS_PER_DEGREE_APPROX, join_style=BUFFER_JOIN_STYLE)

    # Find connected components
    connected_groups = find_connected_components_in_angle_group(gdf_buffered)

    # Assign group IDs and calculate group types
    group_id_counter = 0
    gdf['group_id'] = -1
    gdf['group_type'] = -1

    for group_info in connected_groups:
        indices = group_info['indices']
        # Calculate group type (slope=1, road=10, sum for mixed)
        type_num = sum(1 if gdf.loc[idx, 'label'] == 'slope' else 10 for idx in indices)

        # Assign group ID and type
        gdf.loc[indices, 'group_id'] = group_id_counter
        gdf.loc[indices, 'group_type'] = type_num
        group_id_counter += 1

    return gdf


def save_gdf_to_shp(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """Save GeoDataFrame to ESRI Shapefile with UTF-8 encoding.

    Args:
        gdf: GeoDataFrame to save
        output_path: Path to output Shapefile
    """
    if not gdf.empty:
        gdf.to_file(output_path, encoding='utf-8', driver='ESRI Shapefile')
        print(f"Successfully saved Shapefile to: {output_path}")
    else:
        print("No features to save - output Shapefile not created")


def process_labels_from_jsons_to_gdf(args):
    """Main pipeline to process LabelMe JSONs and generate grouped Shapefile.

    Args:
        args: Command line arguments from parse_args()
    """
    # Step 1: Get list of JSON files
    json_files = get_json_files(args.json_label_dir)
    if not json_files:
        print("No JSON files found in specified directory")
        return

    # Step 2: Initialize data structures
    (spatial_indexes, geometries_by_label,
     unique_geometries, unique_labels, opencv_angles,
     center_coords, geo_heights_deg, geo_widths_deg,
     geo_heights_m, geo_widths_m, filenames) = init_data_structures(args.target_labels)

    # Step 3: Process each JSON file
    crs_wkt_for_gdf = None
    total_extracted = 0
    total_duplicates = 0

    total_shapes_processed = 0
    total_valid_shapes = 0
    total_dups = 0

    pbar = tqdm(total=len(json_files), desc="Processing JSON files", unit="file")
    for json_filename in json_files:
        # Process single JSON file
        features_found, duplicates_found, crs_wkt = process_single_json_file(
            json_filename, args.json_label_dir, args.tif_dir,
            spatial_indexes, geometries_by_label, args,
            (unique_geometries, unique_labels, opencv_angles, center_coords,
             geo_heights_deg, geo_widths_deg, geo_heights_m, geo_widths_m, filenames)
        )

        with open(os.path.join(args.json_label_dir, json_filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            shapes_in_file = len(data.get("shapes", []))
            total_shapes_processed += shapes_in_file

        # Update counters and CRS
        total_extracted += features_found + duplicates_found
        total_duplicates += duplicates_found
        if crs_wkt_for_gdf is None and crs_wkt:
            crs_wkt_for_gdf = crs_wkt

        pbar.set_postfix({
            "shapes": total_shapes_processed,
            "valid": total_valid_shapes,
            "dups": total_dups
        })

        pbar.update(1)

    pbar.close()

    print("\nBatch building spatial indexes for all labels...")
    for label in args.target_labels:
        if geometries_by_label[label]:  # 只有该标签有几何时才构建
            spatial_indexes[label] = build_spatial_index_batch(geometries_by_label[label])

    # Print summary stats
    print(f"\nProcessing Summary:")
    print(f"Total features extracted: {total_extracted}")
    print(f"Total duplicates removed: {total_duplicates}")
    print(f"Unique features retained: {len(unique_geometries)}")

    # Step 4: Create GeoDataFrame
    if not unique_geometries or not crs_wkt_for_gdf:
        print("No valid geometries or CRS found - cannot create GeoDataFrame")
        return

    gdf = create_geo_dataframe(
        (unique_geometries, unique_labels, opencv_angles, center_coords,
         geo_heights_deg, geo_widths_deg, geo_heights_m, geo_widths_m, filenames),
        crs_wkt_for_gdf
    )

    # Step 5: Assign angle groups
    gdf = assign_angle_groups(gdf, angle_threshold=5.0)

    # Step 6: Assign connected component groups
    gdf = assign_connected_component_groups(gdf, args.buffer_distance_meters)

    # Step 7: Save to Shapefile
    save_gdf_to_shp(gdf, args.output_shp_path)


if __name__ == '__main__':

    process_labels_from_jsons_to_gdf(parse_args())