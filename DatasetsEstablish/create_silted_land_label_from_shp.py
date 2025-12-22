#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: create_silted_land_label_from_shp
# @Time    : 2025/12/1 20:54
# @Author  : Kevin
# @Describe: Create LabelMe JSON labels from silted_land.shp, appending to existing JSONs if they exist.

import os
import json
import csv
import logging  # For logging to file
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import rasterio
from tqdm import tqdm


def setup_logging(log_file_path):
    """Configures logging to write to a file."""
    logger = logging.getLogger('shp_to_labelme_logger')
    logger.setLevel(logging.INFO)

    # Avoid adding handlers if they already exist (prevents duplicates if script is imported/re-run in some environments)
    if not logger.handlers:
        # Create file handler
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') # 'w' to overwrite log each run, 'a' to append
        fh.setLevel(logging.INFO)

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(fh)
    return logger

def main():
    # --- Configuration ---
    shp_file_path = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_silted_land.shp"
    source_dir = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Google"
    label_dir = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Label"
    log_dir = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Log" # New: Directory for logs
    label_class_name = "silted_land_with_OBIA_datasets"
    expected_image_size = (1024, 1024)
    # 计算相对路径
    image_path_prefix = os.path.relpath(source_dir, label_dir)

    # Ensure directories exist
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True) # Create log directory

    # Setup logging
    timestamp = "20251201_2100" # You might want to generate this dynamically based on current time
    log_file_path = os.path.join(log_dir, f"processing_log_{timestamp}.log")
    error_csv_path = os.path.join(log_dir, f"errors_{timestamp}.csv")

    logger = setup_logging(log_file_path)
    logger.info("="*50)
    logger.info("Starting Shapefile to LabelMe JSON processing")
    logger.info("="*50)
    logger.info(f"Shapefile Path: {shp_file_path}")
    logger.info(f"Source TIF Directory: {source_dir}")
    logger.info(f"Output Label Directory: {label_dir}")
    logger.info(f"Log File: {log_file_path}")
    logger.info(f"Error CSV File: {error_csv_path}")

    # Initialize error CSV
    error_fieldnames = ['OBJECTID', 'Error_Type', 'Error_Message']
    with open(error_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
        writer.writeheader()


    # --- 1. Read the Shapefile ---
    logger.info(f"Reading Shapefile: {shp_file_path}")
    try:
        gdf = gpd.read_file(shp_file_path)
        logger.info(f"Successfully read Shapefile with {len(gdf)} features.")
    except Exception as e:
        error_msg = f"CRITICAL_ERROR_READING_SHAPEFILE: {e}"
        logger.error(error_msg)
        # Log to CSV as well for easy parsing
        with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
            writer.writerow({'OBJECTID': 'N/A', 'Error_Type': 'CRITICAL_ERROR_READING_SHAPEFILE', 'Error_Message': str(e)})
        return # Stop execution if shapefile cannot be read

    if 'OBJECTID' not in gdf.columns:
        error_msg = "'OBJECTID' column not found in the Shapefile."
        logger.error(error_msg)
        logger.error(f"Available columns: {list(gdf.columns)}")
        with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
            writer.writerow({'OBJECTID': 'N/A', 'Error_Type': 'MISSING_OBJECTID_COLUMN', 'Error_Message': error_msg})
        return

    shp_crs = gdf.crs
    logger.info(f"Shapefile CRS: {shp_crs}")

    # --- 2. Iterate through features ---
    processed_count = 0
    skipped_because_exists = 0

    # Wrap the loop with tqdm for progress bar
    pbar = tqdm(total=len(gdf), desc="Processing features", unit="feature")

    for index, row in gdf.iterrows():
        object_id = row['OBJECTID']
        original_geometry = row['geometry']
        # Use debug level for fine-grained info inside the loop
        # logger.debug(f"Processing feature OBJECTID: {object_id}")

        # --- 3. Construct corresponding TIF path ---
        tif_filename = f"{object_id}.tif"
        tif_path = os.path.join(source_dir, tif_filename)

        # --- 4. Check if TIF exists ---
        if not os.path.exists(tif_path):
            # logger.debug(f"Skipping (TIF not found): {tif_path}")
            # Log missing TIF as an error type
            error_msg = f"TIF file not found at {tif_path}"
            logger.warning(f"OBJECTID {object_id}: {error_msg}") # Use warning or info? Let's use warning for missing input files
            with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                writer.writerow({'OBJECTID': object_id, 'Error_Type': 'MISSING_TIF_FILE', 'Error_Message': error_msg})
            pbar.update(1)
            continue

        # --- 5. Construct corresponding JSON path ---
        json_filename = f"{object_id}.json"
        json_path = os.path.join(label_dir, json_filename)

        # --- 6. Check if JSON already exists and load it ---
        existing_shapes = []
        base_json_data = {}
        if os.path.exists(json_path):
            # logger.debug(f"Found existing JSON: {json_path}")
            try:
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
                existing_shapes = existing_data.get("shapes", [])
                base_json_data = {k: v for k, v in existing_data.items() if k != "shapes"}
                # logger.debug(f"Loaded {len(existing_shapes)} existing shapes.")
            except Exception as e:
                error_msg = f"Could not read existing JSON {json_path}: {e}"
                logger.warning(f"OBJECTID {object_id}: {error_msg}")
                with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                    writer.writerow({'OBJECTID': object_id, 'Error_Type': 'ERROR_READING_EXISTING_JSON', 'Error_Message': str(e)})
                # Continue processing, treat as if no existing file
                existing_shapes = []
                base_json_data = {}


        # --- 7. Read TIF metadata ---
        try:
            with rasterio.open(tif_path) as src:
                tif_transform = src.transform
                tif_width = src.width
                tif_height = src.height
                tif_crs = src.crs
                # tif_bounds = src.bounds # Not used currently

            # logger.debug(f"TIF Info - Size: {tif_width}x{tif_height}, CRS: {tif_crs}")
        except Exception as e:
            error_msg = f"Error reading TIF metadata {tif_path}: {e}"
            logger.error(f"OBJECTID {object_id}: {error_msg}")
            with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                writer.writerow({'OBJECTID': object_id, 'Error_Type': 'ERROR_READING_TIF_METADATA', 'Error_Message': str(e)})
            pbar.update(1)
            continue # Skip if TIF metadata can't be read


        # --- 8. Validate TIF size (optional check) ---
        # if (tif_width, tif_height) != expected_image_size:
        #     logger.warning(f"OBJECTID {object_id}: TIF size {tif_width}x{tif_height} does not match expected {expected_image_size}.")

        # --- 9. Reproject geometry to TIF's CRS if necessary ---
        if shp_crs != tif_crs:
            logger.warning(f"OBJECTID {object_id}: Shapefile CRS ({shp_crs}) differs from TIF CRS ({tif_crs}). Attempting reprojection...")
            try:
                # Reproject using GeoPandas
                temp_gdf = gpd.GeoDataFrame([{'geometry': original_geometry}], crs=shp_crs)
                reprojected_gdf = temp_gdf.to_crs(tif_crs)
                geometry = reprojected_gdf.iloc[0]['geometry']
                logger.info(f"OBJECTID {object_id}: Successfully reprojected geometry.")
            except Exception as e:
                error_msg = f"Error reprojecting geometry for OBJECTID {object_id}: {e}"
                logger.error(error_msg)
                with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                    writer.writerow({'OBJECTID': object_id, 'Error_Type': 'ERROR_REPROJECTING_GEOMETRY', 'Error_Message': str(e)})
                pbar.update(1)
                continue
        else:
            geometry = original_geometry
            # logger.debug(f"OBJECTID {object_id}: Geometry CRS matches TIF CRS.")

        # --- 10. Transform geometry coordinates to pixel coordinates ---
        # Features in a shapefile can be a single polygon or a multipolygon made up of multiple disconnected parts
        polygons_to_process = []
        if isinstance(geometry, Polygon):
            polygons_to_process.append(geometry)
        elif isinstance(geometry, MultiPolygon):
            polygons_to_process.extend(list(geometry.geoms))
        else:
            error_msg = f"Skipping feature {object_id} with unsupported geometry type: {type(geometry)}"
            logger.warning(error_msg)
            with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                writer.writerow({'OBJECTID': object_id, 'Error_Type': 'UNSUPPORTED_GEOMETRY_TYPE', 'Error_Message': error_msg})
            pbar.update(1)
            continue

        new_shapes_for_this_feature = []
        for i, poly in enumerate(polygons_to_process):
            if not poly.is_empty:
                geo_coords = list(poly.exterior.coords)
                try:
                    xs_geo, ys_geo = zip(*geo_coords)
                    # rowcol returns (row, col) which is (y_pixel, x_pixel)
                    rows_pixels, cols_pixels = rasterio.transform.rowcol(tif_transform, xs_geo, ys_geo)
                    # Convert to list of [x_pixel, y_pixel] pairs for LabelMe
                    pixel_coords = [[float(col), float(row)] for col, row in zip(cols_pixels, rows_pixels)]

                    # --- Check for duplicate/new shape ---
                    is_duplicate = False
                    for existing_shape in existing_shapes:
                         if (existing_shape.get("label") == label_class_name and
                             existing_shape.get("shape_type") == "polygon" and
                             len(existing_shape.get("points", [])) == len(pixel_coords)):
                              existing_points = existing_shape.get("points", [])
                              if sorted(existing_points) == sorted(pixel_coords):
                                   is_duplicate = True
                                   break

                    if is_duplicate:
                         # logger.debug(f"OBJECTID {object_id}: Skipping duplicate polygon part {i+1}.")
                         continue # Skip adding this shape

                    if len(pixel_coords) >= 3:
                         shape_data = {
                            "label": label_class_name,
                            "points": pixel_coords,
                            "group_id": None,
                            "description": "",
                            "difficult": False,
                            "shape_type": "polygon",
                            "flags": {}
                        }
                         new_shapes_for_this_feature.append(shape_data)
                         # logger.debug(f"OBJECTID {object_id}: Added new polygon part {i+1} with {len(pixel_coords)} points.")
                    else:
                         warning_msg = f"Converted polygon part {i+1} has insufficient points (<3)."
                         logger.warning(f"OBJECTID {object_id}: {warning_msg}")
                         with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                            writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                            writer.writerow({'OBJECTID': object_id, 'Error_Type': 'INSUFFICIENT_POINTS_IN_POLYGON', 'Error_Message': warning_msg})

                except Exception as e:
                     error_msg = f"Error transforming coordinates for polygon part {i+1}: {e}"
                     logger.error(f"OBJECTID {object_id}: {error_msg}")
                     with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                        writer.writerow({'OBJECTID': object_id, 'Error_Type': 'ERROR_TRANSFORMING_COORDINATES', 'Error_Message': str(e)})
            else:
                 warning_msg = f"Skipping empty polygon part {i+1}."
                 logger.warning(f"OBJECTID {object_id}: {warning_msg}")
                 with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                    writer.writerow({'OBJECTID': object_id, 'Error_Type': 'EMPTY_POLYGON_PART', 'Error_Message': warning_msg})

        # --- 11. Combine existing and new shapes ---
        combined_shapes = existing_shapes + new_shapes_for_this_feature

        if not combined_shapes:
             if existing_shapes:
                  # logger.debug(f"OBJECTID {object_id}: No new shapes added, keeping existing JSON unchanged.")
                  skipped_because_exists += 1
                  pbar.update(1)
                  continue
             else:
                  # logger.debug(f"OBJECTID {object_id}: No valid polygons found, skipping JSON creation.")
                  pbar.update(1)
                  continue

        # --- 12. Construct final LabelMe JSON data ---
        final_labelme_data = base_json_data.copy()
        final_labelme_data.update({
            "version": final_labelme_data.get("version", "5.5.0"),
            "flags": final_labelme_data.get("flags", {}),
            "shapes": combined_shapes,
            "imagePath": final_labelme_data.get("imagePath", os.path.join(image_path_prefix, tif_filename)),
            "imageData": final_labelme_data.get("imageData", None),
            "imageHeight": final_labelme_data.get("imageHeight", tif_height),
            "imageWidth": final_labelme_data.get("imageWidth", tif_width)
        })

        # --- 13. Save/Update JSON ---
        try:
            with open(json_path, 'w') as f:
                json.dump(final_labelme_data, f, indent=2)
            if new_shapes_for_this_feature:
                logger.info(f"OBJECTID {object_id}: Updated LabelMe JSON ({len(new_shapes_for_this_feature)} new shapes added).")
                processed_count += 1
            else:
                # logger.debug(f"OBJECTID {object_id}: Rewrote LabelMe JSON (no new shapes, kept {len(existing_shapes)} existing).")
                # Count as processed if it was checked and potentially updated?
                # For now, let's just increment processed_count if the file existed and was (re)written
                # Or keep it as is. Let's say only counts if new shapes were added.
                pass
        except Exception as e:
            error_msg = f"Error saving/updating JSON {json_path}: {e}"
            logger.error(f"OBJECTID {object_id}: {error_msg}")
            with open(error_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=error_fieldnames)
                writer.writerow({'OBJECTID': object_id, 'Error_Type': 'ERROR_SAVING_JSON', 'Error_Message': str(e)})

        pbar.update(1) # Update progress bar at the end of the loop iteration

    pbar.close() # Close the progress bar

    # --- Summary ---
    summary_msg = f"\n--- Finished processing ---"
    logger.info(summary_msg)
    summary_msg = f"Successfully added shapes to/created {processed_count} LabelMe JSON files in {label_dir}"
    logger.info(summary_msg)
    if skipped_because_exists > 0:
        summary_msg = f"Skipped updating {skipped_because_exists} files that already contained relevant shapes."
        logger.info(summary_msg)
    logger.info("="*50)
    logger.info("Processing finished")
    logger.info("="*50)


if __name__ == "__main__":
    main()