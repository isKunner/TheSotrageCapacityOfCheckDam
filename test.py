#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: test
# @Time    : 2026/1/17 15:19
# @Author  : Kevin
# @Describe:
import os
import re
import shutil
from zipfile import ZipFile

dem_files = os.listdir(r"D:\研究文件\ResearchData\USA\CopernicusDEM\GeoDAR_v11_dams_of_USA_group15")

for dem_file in dem_files:

    if not dem_file.endswith(".zip"):
        continue

    print(dem_file)

    with ZipFile(os.path.join(r"D:\研究文件\ResearchData\USA\CopernicusDEM\GeoDAR_v11_dams_of_USA_group15", dem_file)) as zip_object:
        for member in zip_object.namelist():
            if re.match(r".*DEM.tif$", member) or re.match(r".*DEM.dt[12]$", member):
                filename = os.path.basename(member)
                # skip directories
                if not filename:
                    continue
                # copy file (taken from zipfile's extract)
                source = zip_object.open(member)
                target = open(os.path.join(r"D:\研究文件\ResearchData\USA\CopernicusDEM\GeoDAR_v11_dams_of_USA_group15", filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

