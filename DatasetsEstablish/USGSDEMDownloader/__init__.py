#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: __init__
# @Time    : 2026/2/1 10:18
# @Author  : Kevin
# @Describe:

from .USGS_download_index import usgs_down_index
from .USGS_download import usgs_load_file, check_load_file
from .USGS_generation_html import generate_html_for_usgs_down

"""

from DatasetsEstablish.USGSDEMDownloader import usgs_down_index, usgs_load_file, generate_html_for_usgs_down
from LocalPath import dam_google_remote_root_path, USA_States_shp, dam_usgs_dem_index_dir

google = os.path.join(dam_google_remote_root_path, r"GeoDAR_v11_dams_of_USA_group1")

usgs_down_index(dam_usgs_dem_index_dir)
usgs_load_file(google, dam_usgs_dem_index_dir, USA_States_shp, "./test/json")
generate_html_for_usgs_down("./test/json", "./test/html")

"""