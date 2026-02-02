#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: __init__.py
# @Time    : 2026/1/17 15:51
# @Author  : Kevin
# @Describe:

import os.path as osp
from pathlib import Path

from .constant_variable import Loess_Plateau_Copernicus, USA_States_shp, White_Box_exe_dir

ROOT_DIR = Path(__file__).resolve().parent.parent

dam_project_path = osp.join(ROOT_DIR, "DepthAnything")


# Depth Anything for Super Resolution
# Datasets
dam_root_path = osp.join(ROOT_DIR, r"D:\研究文件\ResearchData\USA_ByState")
# Sub Datasets
dam_copernicus_dem_root_path = osp.join(dam_root_path, "CopernicusDEM")
# Sub Datasets
dam_google_remote_root_path = osp.join(dam_root_path, "GoogleRemoteSensing")
# Sub Datasets
dam_usgs_dem_root_path = osp.join(dam_root_path, "USGSDEM")
dam_usgs_dem_index_dir = osp.join(dam_usgs_dem_root_path, "INDEX")
dam_usgs_dem_down_html_dir = osp.join(dam_usgs_dem_root_path, "HTML")
dam_usgs_dem_down_link_file = osp.join(dam_usgs_dem_root_path, "DownloadLink.json")
dam_usgs_dem_delete_info = osp.join(dam_usgs_dem_root_path, "DeleteInfo.csv")
dam_usgs_dem_down_file = r"C:\Users\Kevin\Downloads\Edge"


# 导出变量
__all__ = [
    'ROOT_DIR',

    'dam_project_path',

    'Loess_Plateau_Copernicus',
    'White_Box_exe_dir',

    'USA_States_shp',
    'dam_root_path',
    'dam_copernicus_dem_root_path',
    'dam_google_remote_root_path',
    "dam_usgs_dem_delete_info",
    'dam_usgs_dem_root_path',
    'dam_usgs_dem_index_dir',
    'dam_usgs_dem_down_link_file',
    'dam_usgs_dem_down_file'
]
