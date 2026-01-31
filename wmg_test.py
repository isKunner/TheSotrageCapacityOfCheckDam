import argparse
import gc
import os

from DatasetsEstablish import process_labels_from_jsons_to_gdf
from DEMAndRemoteSensingUtils import check_dam_info_extract, process_checkdam_capacity, filter_shp

wangmaogou_root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG"

def wang_mao_gou_with_2m_data():
    wangmaogou_root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG"
    os.makedirs(wangmaogou_root_path, exist_ok=True)

    json_dir = os.path.join(wangmaogou_root_path, "GoogleLabel")
    image_dir = os.path.join(wangmaogou_root_path, "Google")
    label_shp = os.path.join(wangmaogou_root_path, "CheckDamLabel.shp")
    labels = ["slope", "road"]
    overlap_threshold = 0.5
    buffer_distance_meters = 1
    group_distance_meters = 2
    angle_threshold_deg = 10

    args = argparse.Namespace(
        json_label_dir=json_dir,
        tif_dir=image_dir,
        output_shp_path=label_shp,
        target_labels=labels,
        overlap_threshold=overlap_threshold,
        buffer_distance_meters=buffer_distance_meters,
        group_distance_meters=group_distance_meters,
        angle_threshold_deg=angle_threshold_deg
    )

    process_labels_from_jsons_to_gdf(args)
    del args
    gc.collect()

    check_dam_info_shp = os.path.join(wangmaogou_root_path, "CheckDamInfo.shp")
    base_dem = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    clip_dem = os.path.join(wangmaogou_root_path, "WMG.tif")
    flow_accum = os.path.join(wangmaogou_root_path, "WMG_FlowAccum.tif")
    elev_dem = r"C:\Users\Kevin\Documents\ResearchData\WangMao\cleaned_dem.tif"
    modified_dem = os.path.join(wangmaogou_root_path, "WGM_MODIFY.tif")
    buffer = 2
    extend = 100
    min_width_flow = 100
    min_width_elev = 0.5
    mode = 1
    args = argparse.Namespace(
        shp_path=label_shp,
        output_shp_path=check_dam_info_shp,
        input_tif=base_dem,
        output_tif=clip_dem,
        output_tif_flow_accumulation=flow_accum,
        elev_tif=elev_dem,
        modified_tif=modified_dem,
        buffer=buffer,
        extend=extend,
        min_width_flow=min_width_flow,
        min_width_elev=min_width_elev,
        mode=mode
    )

    check_dam_info_extract(args)
    del args
    gc.collect()

    clip_dem = os.path.join(wangmaogou_root_path, "WGM_FILLED.tif")
    all_info_shp = os.path.join(wangmaogou_root_path, "CheckDamInfoAll.shp")

    process_checkdam_capacity(dem_path=modified_dem, input_shp_path=check_dam_info_shp, output_shp_path=all_info_shp, output_dem_path=clip_dem)
    gc.collect()

    silted_land_shp = os.path.join(wangmaogou_root_path, "SiltedLand.shp")
    filter_shp(input_shp_path=all_info_shp, output_shp_path=silted_land_shp, filter_type="control_area", filter_mode="include")

def wang_mao_gou_with_2m_to_30m_data():
    original_root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG"
    root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\2_to_30m_WMG"
    os.makedirs(root_path, exist_ok=True)

    label_shp = os.path.join(original_root_path, "CheckDamLabel.shp")

    check_dam_info_shp = os.path.join(root_path, "CheckDamInfo.shp")
    base_dem = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    clip_dem = os.path.join(root_path, "WMG.tif")
    flow_accum = os.path.join(root_path, "WMG_FlowAccum.tif")
    elev_dem = r"C:\Users\Kevin\Documents\ResearchData\WangMao\2_to_30_WMG.tif"
    modified_dem = os.path.join(root_path, "WGM_MODIFY.tif")
    buffer = 2
    extend = 100
    min_width_flow = 60
    min_width_elev = 30
    mode=3
    args = argparse.Namespace(
        shp_path=label_shp,
        output_shp_path=check_dam_info_shp,
        input_tif=base_dem,
        output_tif=clip_dem,
        output_tif_flow_accumulation=flow_accum,
        elev_tif=elev_dem,
        modified_tif=modified_dem,
        buffer=buffer,
        extend=extend,
        min_width_flow=min_width_flow,
        min_width_elev=min_width_elev,
        mode=mode
    )

    check_dam_info_extract(args)
    del args
    gc.collect()

    clip_dem = os.path.join(root_path, "WGM_FILLED.tif")
    all_info_shp = os.path.join(root_path, "CheckDamInfoAll.shp")

    process_checkdam_capacity(dem_path=modified_dem, input_shp_path=check_dam_info_shp, output_shp_path=all_info_shp, output_dem_path=clip_dem)
    gc.collect()

    silted_land_shp = os.path.join(root_path, "SiltedLand.shp")
    filter_shp(input_shp_path=all_info_shp, output_shp_path=silted_land_shp, filter_type="control_area", filter_mode="include")


def wang_mao_gou_with_30m_copernicus_data():
    original_root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG"
    root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\Copernicus_30_WMG"
    os.makedirs(root_path, exist_ok=True)

    label_shp = os.path.join(original_root_path, "CheckDamLabel.shp")

    check_dam_info_shp = os.path.join(root_path, "CheckDamInfo.shp")
    base_dem = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    clip_dem = os.path.join(root_path, "WMG.tif")
    flow_accum = os.path.join(root_path, "WMG_FlowAccum.tif")
    elev_dem = clip_dem
    modified_dem = os.path.join(root_path, "WGM_MODIFY.tif")
    buffer = 2
    extend = 100
    min_width_flow = 60
    min_width_elev = 30
    mode = 3
    args = argparse.Namespace(
        shp_path=label_shp,
        output_shp_path=check_dam_info_shp,
        input_tif=base_dem,
        output_tif=clip_dem,
        output_tif_flow_accumulation=flow_accum,
        elev_tif=elev_dem,
        modified_tif=modified_dem,
        buffer=buffer,
        extend=extend,
        min_width_flow=min_width_flow,
        min_width_elev=min_width_elev,
        mode=mode
    )

    check_dam_info_extract(args)
    del args
    gc.collect()

    clip_dem = os.path.join(root_path, "WGM_FILLED.tif")
    all_info_shp = os.path.join(root_path, "CheckDamInfoAll.shp")

    process_checkdam_capacity(dem_path=modified_dem, input_shp_path=check_dam_info_shp, output_shp_path=all_info_shp,
                              output_dem_path=clip_dem)
    gc.collect()

    silted_land_shp = os.path.join(root_path, "SiltedLand.shp")
    filter_shp(input_shp_path=all_info_shp, output_shp_path=silted_land_shp, filter_type="control_area",
               filter_mode="include")


def wang_mao_gou_with_30m_to_10m_copernicus_data():
    original_root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG"
    root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\Copernicus_30_to_10_WMG"
    os.makedirs(root_path, exist_ok=True)

    label_shp = os.path.join(original_root_path, "CheckDamLabel.shp")

    check_dam_info_shp = os.path.join(root_path, "CheckDamInfo.shp")
    base_dem = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    clip_dem = os.path.join(root_path, "WMG.tif")
    flow_accum = os.path.join(root_path, "WMG_FlowAccum.tif")
    elev_dem = r"C:\Users\Kevin\Documents\ResearchData\WangMao\30_to_10_Copernicus_WMG.tif"
    modified_dem = os.path.join(root_path, "WGM_MODIFY.tif")
    buffer = 2
    extend = 100
    min_width_flow = 60
    min_width_elev = 10
    mode = 3
    args = argparse.Namespace(
        shp_path=label_shp,
        output_shp_path=check_dam_info_shp,
        input_tif=base_dem,
        output_tif=clip_dem,
        output_tif_flow_accumulation=flow_accum,
        elev_tif=elev_dem,
        modified_tif=modified_dem,
        buffer=buffer,
        extend=extend,
        min_width_flow=min_width_flow,
        min_width_elev=min_width_elev,
        mode=mode
    )

    check_dam_info_extract(args)
    del args
    gc.collect()

    clip_dem = os.path.join(root_path, "WGM_FILLED.tif")
    all_info_shp = os.path.join(root_path, "CheckDamInfoAll.shp")

    process_checkdam_capacity(dem_path=modified_dem, input_shp_path=check_dam_info_shp, output_shp_path=all_info_shp,
                              output_dem_path=clip_dem)
    gc.collect()

    silted_land_shp = os.path.join(root_path, "SiltedLand.shp")
    filter_shp(input_shp_path=all_info_shp, output_shp_path=silted_land_shp, filter_type="control_area",
               filter_mode="include")


def wang_mao_gou_with_30m_to_10m_copernicus_tfasr_data():
    original_root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG"
    root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\Copernicus_tfasr"
    os.makedirs(root_path, exist_ok=True)

    label_shp = os.path.join(original_root_path, "CheckDamLabel.shp")

    check_dam_info_shp = os.path.join(root_path, "CheckDamInfo.shp")
    base_dem = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    clip_dem = os.path.join(root_path, "WMG.tif")
    flow_accum = os.path.join(root_path, "WMG_FlowAccum.tif")
    elev_dem = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\Copernicus_tfasr\test_30_copernicus_temp.tif"
    modified_dem = os.path.join(root_path, "WGM_MODIFY.tif")
    buffer = 2
    extend = 100
    min_width_flow = 60
    min_width_elev = 10
    mode = 3
    args = argparse.Namespace(
        shp_path=label_shp,
        output_shp_path=check_dam_info_shp,
        input_tif=base_dem,
        output_tif=clip_dem,
        output_tif_flow_accumulation=flow_accum,
        elev_tif=elev_dem,
        modified_tif=modified_dem,
        buffer=buffer,
        extend=extend,
        min_width_flow=min_width_flow,
        min_width_elev=min_width_elev,
        mode=mode
    )

    check_dam_info_extract(args)
    del args
    gc.collect()

    clip_dem = os.path.join(root_path, "WGM_FILLED.tif")
    all_info_shp = os.path.join(root_path, "CheckDamInfoAll.shp")

    process_checkdam_capacity(dem_path=modified_dem, input_shp_path=check_dam_info_shp, output_shp_path=all_info_shp,
                              output_dem_path=clip_dem)
    gc.collect()

    silted_land_shp = os.path.join(root_path, "SiltedLand.shp")
    filter_shp(input_shp_path=all_info_shp, output_shp_path=silted_land_shp, filter_type="control_area",
               filter_mode="include")

def wang_mao_gou_with_2m_to_30m_to_10m_data():
    original_root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\WMG"
    root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\2_to_30m_to_10m_WMG"
    os.makedirs(root_path, exist_ok=True)

    label_shp = os.path.join(original_root_path, "CheckDamLabel.shp")

    check_dam_info_shp = os.path.join(root_path, "CheckDamInfo.shp")
    base_dem = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    clip_dem = os.path.join(root_path, "WMG.tif")
    flow_accum = os.path.join(root_path, "WMG_FlowAccum.tif")
    elev_dem = r"C:\Users\Kevin\Documents\ResearchData\WangMao\2_to_30_to_10_WMG.tif"
    modified_dem = os.path.join(root_path, "WGM_MODIFY.tif")
    buffer = 2
    extend = 100
    min_width_flow = 60
    min_width_elev = 10
    mode=3
    args = argparse.Namespace(
        shp_path=label_shp,
        output_shp_path=check_dam_info_shp,
        input_tif=base_dem,
        output_tif=clip_dem,
        output_tif_flow_accumulation=flow_accum,
        elev_tif=elev_dem,
        modified_tif=modified_dem,
        buffer=buffer,
        extend=extend,
        min_width_flow=min_width_flow,
        min_width_elev=min_width_elev,
        mode=mode
    )

    check_dam_info_extract(args)
    del args
    gc.collect()

    clip_dem = os.path.join(root_path, "WGM_FILLED.tif")
    all_info_shp = os.path.join(root_path, "CheckDamInfoAll.shp")

    process_checkdam_capacity(dem_path=modified_dem, input_shp_path=check_dam_info_shp, output_shp_path=all_info_shp, output_dem_path=clip_dem)
    gc.collect()

    silted_land_shp = os.path.join(root_path, "SiltedLand.shp")
    filter_shp(input_shp_path=all_info_shp, output_shp_path=silted_land_shp, filter_type="control_area", filter_mode="include")


def zhou_tun_gou_with_2m_data():
    root_path = r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\Test\ZTG"
    os.makedirs(root_path, exist_ok=True)

    json_dir = os.path.join(root_path, "GoogleLabel")
    image_dir = os.path.join(root_path, "Google")
    label_shp = os.path.join(root_path, "CheckDamLabel.shp")
    labels = ["slope", "road"]
    overlap_threshold = 0.5
    buffer_distance_meters = 1
    group_distance_meters = 2
    angle_threshold_deg = 10
    duplicate_check_logic = 'and'

    args = argparse.Namespace(
        json_label_dir=json_dir,
        tif_dir=image_dir,
        output_shp_path=label_shp,
        target_labels=labels,
        overlap_threshold=overlap_threshold,
        buffer_distance_meters=buffer_distance_meters,
        group_distance_meters=group_distance_meters,
        angle_threshold_deg=angle_threshold_deg,
        duplicate_check_logic=duplicate_check_logic
    )

    process_labels_from_jsons_to_gdf(args)
    del args
    gc.collect()

    check_dam_info_shp = os.path.join(root_path, "CheckDamInfo.shp")
    base_dem = r"C:\Users\Kevin\Documents\ResearchData\Copernicus\Loess_Plateau_Copernicus.tif"
    clip_dem = os.path.join(root_path, "ZTG.tif")
    flow_accum = os.path.join(root_path, "ZTG_FlowAccum.tif")
    elev_dem = r"C:\Users\Kevin\Documents\ResearchData\ZhouTun\zhou_tun_gou_wgs84.tif"
    modified_dem = os.path.join(root_path, "ZTG_MODIFY.tif")
    buffer = 2
    extend = 100
    min_width_flow = 100
    min_width_elev = 0.5
    mode = 1
    args = argparse.Namespace(
        shp_path=label_shp,
        output_shp_path=check_dam_info_shp,
        input_tif=base_dem,
        output_tif=clip_dem,
        output_tif_flow_accumulation=flow_accum,
        elev_tif=elev_dem,
        modified_tif=modified_dem,
        buffer=buffer,
        extend=extend,
        min_width_flow=min_width_flow,
        min_width_elev=min_width_elev,
        mode=mode
    )

    check_dam_info_extract(args)
    del args
    gc.collect()

    clip_dem = os.path.join(root_path, "ZTG_FILLED.tif")
    all_info_shp = os.path.join(root_path, "CheckDamInfoAll.shp")

    process_checkdam_capacity(dem_path=modified_dem, input_shp_path=check_dam_info_shp, output_shp_path=all_info_shp, output_dem_path=clip_dem)
    gc.collect()

    silted_land_shp = os.path.join(root_path, "SiltedLand.shp")
    filter_shp(input_shp_path=all_info_shp, output_shp_path=silted_land_shp, filter_type="control_area", filter_mode="include")

if __name__ == '__main__':
    # wang_mao_gou_with_2m_data()
    # wang_mao_gou_with_2m_to_30m_data()
    # wang_mao_gou_with_2m_to_30m_to_10m_data()
    # wang_mao_gou_with_30m_copernicus_data()
    # wang_mao_gou_with_30m_to_10m_copernicus_data()
    # wang_mao_gou_with_30m_to_10m_copernicus_tfasr_data()

    zhou_tun_gou_with_2m_data()