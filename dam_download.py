#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: dam_download
# @Time    : 2026/2/2 11:40
# @Author  : Kevin
# @Describe:

import os
from DatasetsEstablish.USGSDEMDownloader import usgs_down_index, usgs_load_file, check_load_file, \
    generate_html_for_usgs_down
from DatasetsEstablish.CDSECopernicusDEMDownloader import download_copernicus
from LocalPath import dam_usgs_dem_index_dir, dam_google_remote_root_path, USA_States_shp, dam_usgs_dem_down_link_file, \
    dam_usgs_dem_root_path, dam_usgs_dem_down_html_dir, dam_copernicus_dem_root_path

# usgs_down_index(dam_usgs_dem_index_dir)

# for file in os.listdir(dam_google_remote_root_path):
#     usgs_load_file(os.path.join(dam_google_remote_root_path, file), dam_usgs_dem_index_dir, USA_States_shp, dam_usgs_dem_down_link_file)

# check_load_file(dam_usgs_dem_root_path, dam_usgs_dem_down_link_file)

# generate_html_for_usgs_down(dam_usgs_dem_down_link_file, dam_usgs_dem_down_html_dir)

download_copernicus([os.path.join(dam_google_remote_root_path, sub_file) for sub_file in ["TX"]], [os.path.join(dam_copernicus_dem_root_path, sub_file) for sub_file in ["TX"]])