#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: dam_download
# @Time    : 2026/2/2 11:40
# @Author  : Kevin
# @Describe:

import os
import time
from os import times

from DEMAndRemoteSensingUtils import merge_geo_referenced_tifs, crop_source_to_reference

from DatasetsEstablish.USGSDEMDownloader import usgs_down_index, usgs_load_file, check_load_file, \
    generate_html_for_usgs_down
from DatasetsEstablish.CDSECopernicusDEMDownloader import download_copernicus
from DatasetsEstablish.USGSDEMDownloader.USGS_down_process import generation_usgs
from LocalPath import dam_usgs_dem_index_dir, dam_google_remote_root_path, USA_States_shp, dam_usgs_dem_down_link_file, \
    dam_usgs_dem_root_path, dam_usgs_dem_down_html_dir, dam_copernicus_dem_root_path, dam_usgs_dem_delete_info

# usgs_down_index(dam_usgs_dem_index_dir)

# for file in os.listdir(dam_google_remote_root_path):
#     usgs_load_file(os.path.join(dam_google_remote_root_path, file), dam_usgs_dem_index_dir, USA_States_shp, dam_usgs_dem_down_link_file)
#
# check_load_file(dam_usgs_dem_root_path, dam_usgs_dem_down_link_file)
#
# generate_html_for_usgs_down(dam_usgs_dem_down_link_file, dam_usgs_dem_down_html_dir)

# while True:
#
#     generation_usgs(dam_google_remote_root_path,
#                     dam_usgs_dem_down_link_file,
#                     r"C:\Users\Kevin\Downloads\Edge",
#                     dam_usgs_dem_root_path,
#                     is_delete_file=True,
#                     usgs_dem_delete_info=dam_usgs_dem_delete_info
#                     )
#
#     time.sleep(10*60)

state_names = ["TX"]
download_copernicus([os.path.join(dam_google_remote_root_path, sub_file) for sub_file in state_names],
                        [os.path.join(dam_copernicus_dem_root_path, sub_file+"_original") for sub_file in state_names])


# state_names = ["FL"]
# for state_name in state_names:
#
#
#     merge_geo_referenced_tifs(os.path.join(dam_copernicus_dem_root_path, state_name+"_original"),
#                               output_path=os.path.join(dam_copernicus_dem_root_path, f"{state_name}_original", state_name + ".tif"))
#     crop_source_to_reference(source_raster_path=os.path.join(dam_copernicus_dem_root_path, state_name+"_original", f"{state_name}.tif"),
#                              reference_inputs=os.path.join(dam_google_remote_root_path, state_name),
#                              output_destination=os.path.join(dam_copernicus_dem_root_path, state_name)
#                              )
