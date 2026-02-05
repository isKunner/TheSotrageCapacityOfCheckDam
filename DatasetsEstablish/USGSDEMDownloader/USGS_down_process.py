#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: USGS_down_process
# @Time    : 2026/2/1 11:47
# @Author  : Kevin
# @Describe:

import json
import os
import os.path as osp
from datetime import datetime

import pandas as pd

from DEMAndRemoteSensingUtils import merge_geo_referenced_tifs, crop_source_to_reference, merge_sources_to_reference

def generation_usgs(google_remote_root_dir, usgs_dem_index_file, dam_usgs_dem_down_file, usgs_dem_root_path, is_delete_file=False, usgs_dem_delete_info=None):

    """

    :param google_remote_root_dir: Get the range you want to crop
    :param usgs_dem_index_file: Each file to be cropped needs a DEM to be downloaded
    :param dam_usgs_dem_down_file: Path to download DEM
    :param usgs_dem_root_path: Save the path of the cropped DEM
    :return:
    """

    with open(usgs_dem_index_file, 'r', encoding='utf-8') as f:
        download_link = json.load(f)

    for key, value in download_link.items():
        current_dir = osp.join(usgs_dem_root_path, key)
        google_dir = osp.join(google_remote_root_dir, key)
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
                    print(f"{key} {target_file} Missing documents")

    if is_delete_file:
        link_counts = {}
        for key, value in download_link.items():
            for target_file, links in value.items():
                for link in links:
                    link_counts[link] = link_counts.get(link, 0) + 1

        deletion_records = []

        for key, value in download_link.items():
            current_dir = osp.join(usgs_dem_root_path, key)
            try:
                for target_file, links in value.items():
                    if osp.exists(osp.join(current_dir, target_file)):
                        for link in links:
                            down_file = link.split("/")[-1]
                            if osp.exists(osp.join(dam_usgs_dem_down_file, down_file)):
                                link_counts[link] -= 1
                                if link_counts[link] == 0:
                                    file_path = osp.join(dam_usgs_dem_down_file, down_file)

                                    # 添加删除记录到列表
                                    deletion_records.append({
                                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        'deleted_file': down_file,
                                        'link_url': link
                                    })

                                    os.remove(file_path)
                                    print(f"Delete files: {file_path}")
            except Exception as e:
                print(f"Error processing files: {e}")
                if deletion_records:
                    # 将新记录转换为DataFrame
                    new_records_df = pd.DataFrame(deletion_records)

                    # 检查CSV文件是否存在，如果存在则读取现有数据
                    if os.path.isfile(usgs_dem_delete_info):
                        existing_df = pd.read_csv(usgs_dem_delete_info)
                        # 合并现有数据和新数据
                        combined_df = pd.concat([existing_df, new_records_df], ignore_index=True)
                    else:
                        # 如果文件不存在，只使用新数据
                        combined_df = new_records_df
                    combined_df.to_csv(usgs_dem_delete_info, index=False)

        if deletion_records:
            # 将新记录转换为DataFrame
            new_records_df = pd.DataFrame(deletion_records)

            # 检查CSV文件是否存在，如果存在则读取现有数据
            if os.path.isfile(usgs_dem_delete_info):
                existing_df = pd.read_csv(usgs_dem_delete_info)
                # 合并现有数据和新数据
                combined_df = pd.concat([existing_df, new_records_df], ignore_index=True)
            else:
                # 如果文件不存在，只使用新数据
                combined_df = new_records_df

            combined_df.to_csv(usgs_dem_delete_info, index=False)