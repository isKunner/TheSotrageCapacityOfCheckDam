#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: generation_data_for_DAM
# @Time    : 2026/1/26 10:08
# @Author  : Kevin
# @Describe: 生成DepthAnything的数据对
import json
import os
import os.path as osp
import sys
import time
from datetime import datetime

import pandas as pd

current_file_path = osp.abspath(__file__)
sys.path.append(osp.join(osp.dirname(current_file_path), ".."))

from LocalPath import dam_root_path, dam_google_remote_root_path, dam_usgs_dem_root_path, dam_copernicus_dem_root_path, \
    dam_usgs_dem_down_link, dam_usgs_dem_down_file, dam_usgs_dem_delete_info
from DEMAndRemoteSensingUtils import merge_geo_referenced_tifs, crop_source_to_reference, merge_sources_to_reference


def generation_usgs():
    with open(dam_usgs_dem_down_link, 'r', encoding='utf-8') as f:
        download_link = json.load(f)

    for key, value in download_link.items():
        current_dir = osp.join(dam_usgs_dem_root_path, key)
        google_dir = osp.join(dam_google_remote_root_path, key)
        os.makedirs(current_dir, exist_ok=True)
        for target_file, links in value.items():
            if not osp.exists(osp.join(current_dir, target_file)):
                could_merge = True
                for link in links:
                    down_file = link.split("/")[-1]
                    if not osp.exists(osp.join(dam_usgs_dem_down_file, down_file)):
                        could_merge = False
                        break
                if could_merge:
                    reference_path = osp.join(google_dir, target_file)
                    source_paths = [osp.join(dam_usgs_dem_down_file, link.split("/")[-1]) for link in links]
                    output_file = osp.join(current_dir, target_file)
                    merge_sources_to_reference(reference_path, source_paths, output_file)
                else:
                    print(f"{key} {target_file} 缺少文件")

def delete_usgs_file():
    with open(dam_usgs_dem_down_link, 'r', encoding='utf-8') as f:
        download_link = json.load(f)

    link_counts = {}
    for key, value in download_link.items():
        for target_file, links in value.items():
            for link in links:
                link_counts[link] = link_counts.get(link, 0) + 1

    single_occurrence_links = [link for link, count in link_counts.items() if count == 1]

    print(f"总链接数: {sum(link_counts.values())}")
    print(f"不同链接数: {len(link_counts)}")
    print(f"只出现一次的链接数: {len(single_occurrence_links)}")

    deletion_records = []

    for key, value in download_link.items():
        current_dir = osp.join(dam_usgs_dem_root_path, key)
        google_dir = osp.join(dam_google_remote_root_path, key)
        os.makedirs(current_dir, exist_ok=True)
        try:
            for target_file, links in value.items():
                if osp.exists(osp.join(current_dir, target_file)):
                    for link in links:
                        down_file = link.split("/")[-1]
                        if osp.exists(osp.join(dam_usgs_dem_down_file, down_file)):
                            link_counts[link]-=1
                            if link_counts[link] == 0:
                                file_path = osp.join(dam_usgs_dem_down_file, down_file)

                                # 添加删除记录到列表
                                deletion_records.append({
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'deleted_file': down_file,
                                    'link_url': link
                                })

                                os.remove(file_path)
                                print(f"删除文件: {file_path}")
        except Exception as e:
            print(f"处理文件时出错: {e}")
            if deletion_records:
                # 将新记录转换为DataFrame
                new_records_df = pd.DataFrame(deletion_records)

                # 检查CSV文件是否存在，如果存在则读取现有数据
                if os.path.isfile(dam_usgs_dem_delete_info):
                    existing_df = pd.read_csv(dam_usgs_dem_delete_info)
                    # 合并现有数据和新数据
                    combined_df = pd.concat([existing_df, new_records_df], ignore_index=True)
                else:
                    # 如果文件不存在，只使用新数据
                    combined_df = new_records_df
                combined_df.to_csv(dam_usgs_dem_delete_info, index=False)

    if deletion_records:
        # 将新记录转换为DataFrame
        new_records_df = pd.DataFrame(deletion_records)

        # 检查CSV文件是否存在，如果存在则读取现有数据
        if os.path.isfile(dam_usgs_dem_delete_info):
            existing_df = pd.read_csv(dam_usgs_dem_delete_info)
            # 合并现有数据和新数据
            combined_df = pd.concat([existing_df, new_records_df], ignore_index=True)
        else:
            # 如果文件不存在，只使用新数据
            combined_df = new_records_df

        combined_df.to_csv(dam_usgs_dem_delete_info, index=False)


if __name__ == '__main__':

    task = 1

    if task == 0:
        for i in range(312):
            if not osp.exists(osp.join(r"D:\研究文件\ResearchData\USA\USGSDEM\GeoDAR_v11_dams_of_USA_group1", f"{i}.tif")):
                print(f"{i}.tif is not existed!!!")


    elif task == 1:

        while True:

            generation_usgs()
            delete_usgs_file()
            time.sleep(10 * 60)

    elif task == 2:

        for file in os.listdir(dam_copernicus_dem_root_path):

            if file!="GeoDAR_v11_dams_of_USA_group14":
                continue

            current_dir = osp.join(dam_copernicus_dem_root_path, file)
            if osp.isdir(current_dir) and "paired" not in file:
                print(current_dir + ".tif")
                if not osp.exists(current_dir + ".tif"):
                    merge_geo_referenced_tifs(current_dir, current_dir + ".tif")

            current_dir = osp.join(dam_google_remote_root_path, file)
            copernicus_source_file = osp.join(dam_copernicus_dem_root_path, file+".tif")
            output_dir = osp.join(dam_copernicus_dem_root_path, file+"_paired")
            if osp.exists(copernicus_source_file):
                crop_source_to_reference(copernicus_source_file, current_dir, output_dir, log_csv=osp.join(output_dir, "log.csv"))