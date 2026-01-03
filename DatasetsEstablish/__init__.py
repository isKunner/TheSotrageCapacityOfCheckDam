#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: __init__.py
# @Time    : 2025/12/1 20:52
# @Author  : Kevin
# @Describe:

from .generate_silted_land_shp_from_label import process_labels_from_jsons_to_gdf

__all__ = ['process_labels_from_jsons_to_gdf']