#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: USGS_download
# @Time    : 2026/1/26 21:42
# @Author  : Kevin
# @Describe:
import json
import os.path
import re
from math import ceil
from urllib.parse import urlparse

import rasterio
import requests
import utm
import geopandas as gpd
from shapely.geometry import Point

import concurrent.futures
from tqdm import tqdm

from LocalPath import dam_google_remote_root_path, dam_usgs_dem_index_root_path, USA_States, dam_usgs_dem_root_path

state_to_abbr = {
    "alabama": "al",
    "alaska": "ak",
    "arizona": "az",
    "arkansas": "ar",
    "california": "ca",
    "colorado": "co",
    "connecticut": "ct",
    "delaware": "de",
    "florida": "fl",
    "georgia": "ga",
    "hawaii": "hi",
    "idaho": "id",
    "illinois": "il",
    "indiana": "in",
    "iowa": "ia",
    "kansas": "ks",
    "kentucky": "ky",
    "louisiana": "la",
    "maine": "me",
    "maryland": "md",
    "massachusetts": "ma",
    "michigan": "mi",
    "minnesota": "mn",
    "mississippi": "ms",
    "missouri": "mo",
    "montana": "mt",
    "nebraska": "ne",
    "nevada": "nv",
    "new hampshire": "nh",
    "new jersey": "nj",
    "new mexico": "nm",
    "new york": "ny",
    "north carolina": "nc",
    "north dakota": "nd",
    "ohio": "oh",
    "oklahoma": "ok",
    "oregon": "or",
    "pennsylvania": "pa",
    "rhode island": "ri",
    "south carolina": "sc",
    "south dakota": "sd",
    "tennessee": "tn",
    "texas": "tx",
    "utah": "ut",
    "vermont": "vt",
    "virginia": "va",
    "washington": "wa",
    "west virginia": "wv",
    "wisconsin": "wi",
    "wyoming": "wy"
}



def download_single_file(url, download_dir):
    """
    下载单个文件
    """
    try:
        filename = os.path.basename(urlparse(url).path)
        filepath = os.path.join(download_dir, filename)

        if os.path.exists(filepath):
            return True

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False


def get_file_name_part(lon, lat):
    easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)
    return f"x{int(easting//10000)}y{ceil(northing/10000)}"

def get_tif_bounds(google_file_path):
    """
    读取TIF文件并获取四个角坐标点
    返回格式: (左下, 右下, 右上, 左上)
    """
    with rasterio.open(google_file_path) as dataset:
        # 获取TIF文件的边界信息
        bounds = dataset.bounds  # (left, bottom, right, top)
        crs = dataset.crs

        left, bottom, right, top = bounds

        # 四个角点坐标
        lower_left = left, bottom  # 左下角
        lower_right = right, bottom  # 右下角
        upper_right = right, top  # 右上角
        upper_left = left, top  # 左上角

        return [lower_left, lower_right, upper_right, upper_left]


def down_load_file(dam_google_remote_root_dir, index_dict, usa_states_gdf, down_dict_info, group_name):
    """
    改进版：记录所有文件，包括失败的，并打印详细调试信息
    """
    import re
    import math

    # 获取所有tif文件并排序
    all_files = [f for f in os.listdir(dam_google_remote_root_dir) if f.endswith(".tif")]
    all_files.sort(key=lambda x: int(re.match(r'(\d+)\.tif', x).group(1)) if re.match(r'(\d+)\.tif', x) else 0)

    for google_remote_file in tqdm(all_files, desc=f"处理 {group_name}"):
        # 强制初始化记录
        if group_name not in down_dict_info:
            down_dict_info[group_name] = {}
        down_dict_info[group_name][google_remote_file] = {}

        try:
            # 1. 获取四角坐标
            tif_path = os.path.join(dam_google_remote_root_dir, google_remote_file)
            coordinates = get_tif_bounds(tif_path)
            coordinates_set = set()

            # print(f"\n【处理】{google_remote_file}")
            # print(f"  四角坐标: {coordinates}")

            # 2. 判断州和计算file_name_part
            for i, (lon, lat) in enumerate(coordinates):
                point = Point(lon, lat)
                matched = False

                for idx, state_row in usa_states_gdf.iterrows():
                    state_name = str(state_row['NAME']).lower().strip()

                    if state_row['geometry'].contains(point):
                        state_abbr = state_to_abbr.get(state_name)
                        if state_abbr:
                            file_part = get_file_name_part(lon, lat)
                            coordinates_set.add((state_abbr.upper(), file_part))
                            matched = True
                            # print(
                            #     f"  角点{i + 1} ({lon:.6f}, {lat:.6f}): 州={state_abbr.upper()}, file_part={file_part}")
                            break
                        else:
                            print(f"  ⚠️ 警告: 州 '{state_row['NAME']}' 无缩写映射")

                if not matched:
                    print(f"  ❌ 角点{i + 1} ({lon:.6f}, {lat:.6f}): 未匹配到任何州")

            # 3. 检查是否找到州
            if not coordinates_set:
                down_dict_info[group_name][google_remote_file]["__error__"] = "no_state_found"
                down_dict_info[group_name][google_remote_file]["__detail__"] = f"四角坐标: {coordinates}"
                print(f"  【失败】未匹配到任何州")
                continue

            # 4. 在索引中查找链接（关键改进：打印详细匹配信息）
            link_found = False
            for state, file_name_part in coordinates_set:
                # print(f"\n  【查找索引】州={state}, 寻找 file_part='{file_name_part}'")

                if state not in index_dict:
                    down_dict_info[group_name][google_remote_file]["__error__"] = f"state_not_in_index:{state}"
                    print(f"    ❌ 索引字典中无州 '{state}'")
                    print(f"    可用州列表: {list(index_dict.keys())[:10]}...")  # 打印前10个
                    continue

                # 打印索引样本（前3个和后3个，看命名格式）
                idx_list = index_dict[state]
                # print(f"    索引文件数: {len(idx_list)}")
                # print(f"    索引样本(前3): {idx_list[:3]}")
                # if len(idx_list) > 6:
                #     print(f"    索引样本(后3): {idx_list[-3:]}")

                # 尝试匹配
                matched_links = []
                for link in idx_list:
                    if file_name_part in link:
                        matched_links.append(link)
                        down_dict_info[group_name][google_remote_file][link] = True
                        link_found = True
                        # print(f"    ✓ 匹配: {link}")

                if not matched_links:
                    print(f"    ❌ 未找到包含 '{file_name_part}' 的链接")
                    # 尝试模糊匹配：找相似的部分
                    similar = [l for l in idx_list if file_name_part[:5] in l or file_name_part[-5:] in l][:3]
                    if similar:
                        print(f"    相似链接(供参考): {similar}")

            # 5. 最终失败记录
            if not link_found:
                down_dict_info[group_name][google_remote_file]["__error__"] = "no_link_found"
                down_dict_info[group_name][google_remote_file]["__detail__"] = {
                    "states_searched": [c[0] for c in coordinates_set],
                    "file_parts": [c[1] for c in coordinates_set],
                    "coordinates": coordinates
                }
                print(f"\n  【最终失败】未找到任何匹配链接")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            down_dict_info[group_name][google_remote_file]["__error__"] = "exception"
            down_dict_info[group_name][google_remote_file]["__detail__"] = error_msg
            print(f'\n  【异常】{google_remote_file}: {e}')

        # 每10个保存一次
        if len(down_dict_info[group_name]) % 10 == 0:
            with open(os.path.join(dam_usgs_dem_root_path, "DownloadInfo.json"), 'w', encoding='utf-8') as f:
                json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(os.path.join(dam_usgs_dem_root_path, "DownloadInfo.json"), 'w', encoding='utf-8') as f:
        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    # 统计
    total = len(down_dict_info.get(group_name, {}))
    success = sum(1 for f, links in down_dict_info.get(group_name, {}).items()
                  if not any(k.startswith("__") for k in links.keys()))
    failed = total - success
    print(f"\n{'=' * 60}")
    print(f"[{group_name}] 完成: 总计{total}, 成功{success}, 失败{failed}")
    print(f"{'=' * 60}")


def down_load_file_v2(dam_google_remote_root_dir, index_dict, usa_states_gdf, output_dir, down_dict_info, group_name):
    """
    改进版：先查本地州，本地找不到再扩大范围查临近州
    """
    all_files = [f for f in os.listdir(dam_google_remote_root_dir) if f.endswith(".tif")]
    all_files.sort(key=lambda x: int(re.match(r'(\d+)\.tif', x).group(1)) if re.match(r'(\d+)\.tif', x) else 0)

    for google_remote_file in tqdm(all_files, desc=f"处理 {group_name}"):
        if group_name not in down_dict_info:
            down_dict_info[group_name] = {}
        down_dict_info[group_name][google_remote_file] = {}

        try:
            tif_path = os.path.join(dam_google_remote_root_dir, google_remote_file)
            coordinates = get_tif_bounds(tif_path)

            print(f"\n【处理】{google_remote_file}")
            print(f"  四角: {[(round(lon, 4), round(lat, 4)) for lon, lat in coordinates]}")

            # 第一步：分离本地州和临近州
            local_candidates = []  # 严格包含的州（distance=0）
            nearby_candidates = []  # 临近州（distance>0但在buffer内）

            for lon, lat in coordinates:
                point = Point(lon, lat)
                file_part = get_file_name_part(lon, lat)

                for idx, row in usa_states_gdf.iterrows():
                    state_name = str(row['NAME']).strip().lower()
                    geom = row['geometry']
                    abbr = state_to_abbr.get(state_name)
                    if not abbr:
                        continue

                    distance = geom.distance(point)

                    if geom.contains(point):
                        # 严格包含：本地州
                        local_candidates.append((abbr.upper(), file_part, 0))
                        print(f"  🏠 本地州: {abbr.upper()}, file_part={file_part}")
                    elif distance < 1:  # 1度≈100km范围内
                        # 临近州
                        nearby_candidates.append((abbr.upper(), file_part, distance))
                        print(f"  📍 临近州: {abbr.upper()} (距离{distance:.4f}°), file_part={file_part}")

            # 去重
            seen_local = set()
            unique_local = []
            for c in local_candidates:
                if c[:2] not in seen_local:
                    seen_local.add(c[:2])
                    unique_local.append(c)

            seen_nearby = set()
            unique_nearby = []
            for c in sorted(nearby_candidates, key=lambda x: x[2]):  # 按距离排序
                if c[:2] not in seen_local and c[:2] not in seen_nearby:  # 避免和本地重复
                    seen_nearby.add(c[:2])
                    unique_nearby.append(c)

            link_found = False
            searched_states = []

            # 第二步：优先查本地州（严格包含的）
            if unique_local:
                print(f"  🔍 阶段1：搜索本地州 {[c[0] for c in unique_local]}...")
                for state, file_part, _ in unique_local:
                    if state not in index_dict:
                        print(f"    ⚠️ 本地州{state}无索引文件")
                        continue

                    searched_states.append(f"{state}(本地)")

                    # 精确匹配
                    for link in index_dict[state]:
                        if file_part in link:
                            down_dict_info[group_name][google_remote_file][link] = True
                            down_dict_info[group_name][google_remote_file]["__source__"] = f"{state}(本地)"
                            link_found = True
                            print(f"    ✅ 本地州{state}找到: {link[:60]}...")
                            break

                    if link_found:
                        break

                    # # 模糊匹配（邻居格子）
                    # if not link_found:
                    #     match = re.match(r'x(\d+)y(\d+)', file_part)
                    #     if match:
                    #         x, y = int(match.group(1)), int(match.group(2))
                    #         neighbors = [f"x{x - 1}y{y}", f"x{x + 1}y{y}", f"x{x}y{y - 1}", f"x{x}y{y + 1}"]
                    #         for neighbor in neighbors:
                    #             for link in index_dict[state]:
                    #                 if neighbor in link:
                    #                     print(f"    ⚠️ 本地州{state}近似匹配({neighbor}): {link[:60]}...")
                    #                     down_dict_info[group_name][google_remote_file][link] = True
                    #                     down_dict_info[group_name][google_remote_file][
                    #                         "__source__"] = f"{state}(本地近似)"
                    #                     down_dict_info[group_name][google_remote_file][
                    #                         "__note__"] = f"{file_part}->{neighbor}"
                    #                     link_found = True
                    #                     break
                    #             if link_found:
                    #                 break
                    #
                    # if link_found:
                    #     break

            # 第三步：本地州没找到，扩大范围查临近州
            if not link_found and unique_nearby:
                print(f"  🔍 阶段2：本地州未找到，扩大搜索临近州 {[c[0] for c in unique_nearby]}...")

                for state, file_part, dist in unique_nearby:
                    if state not in index_dict:
                        continue

                    searched_states.append(f"{state}(临近,距{dist:.2f}°)")
                    print(f"    查临近州 {state} (距离{dist:.2f}°)...")

                    # 同样先精确匹配
                    for link in index_dict[state]:
                        if file_part in link:
                            down_dict_info[group_name][google_remote_file][link] = True
                            down_dict_info[group_name][google_remote_file]["__source__"] = f"{state}(临近州)"
                            down_dict_info[group_name][google_remote_file][
                                "__note__"] = f"本地州未找到，在{state}找到(距{dist:.2f}°)"
                            link_found = True
                            print(f"    ✅ 临近州{state}找到: {link[:60]}...")
                            break

                    if link_found:
                        break

            # 第四步：记录失败
            if not link_found:
                down_dict_info[group_name][google_remote_file]["__error__"] = "no_link_found"
                down_dict_info[group_name][google_remote_file]["__detail__"] = {
                    "本地州": [(c[0], c[1]) for c in unique_local],
                    # "临近州": [(c[0], c[1], round(c[2], 4)) for c in unique_nearby],
                    "已搜索": searched_states,
                    "tip": "本地州及100km内临近州均未找到该网格数据"
                }
                print(f"    ❌ 失败: 本地州{[c[0] for c in unique_local]}及临近州均未找到")

        except Exception as e:
            import traceback
            down_dict_info[group_name][google_remote_file]["__error__"] = "exception"
            down_dict_info[group_name][google_remote_file]["__detail__"] = str(e)

        # 每10个保存
        if len(down_dict_info[group_name]) % 10 == 0:
            with open(os.path.join(dam_usgs_dem_root_path, "DownloadInfo.json"), 'w') as f:
                json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(os.path.join(dam_usgs_dem_root_path, "DownloadInfo.json"), 'w') as f:
        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':

    index_dict = {}
    for file_name in os.listdir(dam_usgs_dem_index_root_path):
        key = file_name.split("_")[0]
        with open(os.path.join(dam_usgs_dem_index_root_path, file_name), "r", encoding="UTF-8") as f:
            for line in f:
                index_dict.setdefault(key, []).append(line.strip())

    usa_states_gdf = gpd.read_file(USA_States)

    # group_names = ["GeoDAR_v11_dams_of_USA_group1"]
    group_names = ["GeoDAR_v11_dams_of_USA_group1", "GeoDAR_v11_dams_of_USA_group10", "GeoDAR_v11_dams_of_USA_group11", "GeoDAR_v11_dams_of_USA_group12", "GeoDAR_v11_dams_of_USA_group13_1", "GeoDAR_v11_dams_of_USA_group13_2", "GeoDAR_v11_dams_of_USA_group14"]

    down_dict_info = {}

    for group_name in group_names:
        current_dam_google_remote_root_dir = os.path.join(dam_google_remote_root_path, group_name)
        output_dir = os.path.join(dam_usgs_dem_root_path, group_name)
        os.makedirs(output_dir, exist_ok=True)
        down_load_file(current_dam_google_remote_root_dir, index_dict, usa_states_gdf, group_name=group_name, down_dict_info=down_dict_info)

    with open(os.path.join(dam_usgs_dem_root_path, "DownloadInfo.json"), 'w', encoding='utf-8') as f:
        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)
