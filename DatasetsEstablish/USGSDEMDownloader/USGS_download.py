#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: USGS_download
# @Time    : 2026/1/26 21:42
# @Author  : Kevin
# @Describe:

import re
import json
import os.path
from math import ceil

import requests
from urllib.parse import urlparse
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import utm
import geopandas as gpd
import rasterio
from shapely.geometry import Point

from tqdm import tqdm

state_to_abbr = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    "american samoa": "AS", "district of columbia": "DC", "guam": "GU",
    "puerto rico": "PR", "commonwealth of the northern mariana islands": "MP",
    "united states virgin islands": "VI"
}

def download_single_file(url, download_dir):
    """
    Download individual files
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
    Read the TIF file and get the four angular coordinate points
    Return format: (Bottom left, Bottom right, Top right, Top left)
    """
    with rasterio.open(google_file_path) as dataset:
        bounds = dataset.bounds  # (left, bottom, right, top)

        left, bottom, right, top = bounds

        lower_left = left, bottom
        lower_right = right, bottom
        upper_right = right, top
        upper_left = left, top

        return [lower_left, lower_right, upper_right, upper_left]


def usgs_load_file(google_remote_root_dir, usgs_dem_index_dir, usa_states_shp_path, usgs_dem_down_link):
    """
    Record all files, including failed ones, and print out detailed debugging information

    google_remote_root_dir: 用来计算存放目标位置的范围，注意，这里暂定的范围不超过10000米，如果有需要改进还需要改进
    usgs_dem_index_dir: 存放USGS_download_index.py生成的目录的路径
    usa_states_shp_path: usa的shp文件路径，用来判断所处地理位置的州

    """

    usa_states_gdf = gpd.read_file(usa_states_shp_path)

    index_dict = {}
    for file_name in os.listdir(usgs_dem_index_dir):
        key = file_name.split("_")[0]
        with open(os.path.join(usgs_dem_index_dir, file_name), "r", encoding="UTF-8") as f:
            for line in f:
                index_dict.setdefault(key, []).append(line.strip())

    # Get all TIF files and sort them
    all_files = [f for f in os.listdir(google_remote_root_dir) if f.endswith(".tif")]
    all_files.sort(key=lambda x: int(re.match(r'(\d+)\.tif', x).group(1)) if re.match(r'(\d+)\.tif', x) else 0)

    if os.path.exists(usgs_dem_down_link):
        with open(usgs_dem_down_link, "r", encoding="UTF-8") as f:
            down_dict_info = json.load(f)
    else:
        down_dict_info = {}

    group_name = os.path.basename(google_remote_root_dir)

    for google_remote_file in tqdm(all_files, desc=f"processing{group_name}"):
        if group_name not in down_dict_info:
            down_dict_info[group_name] = {}
        down_dict_info[group_name][google_remote_file] = {}

        try:
            # 1. Get the coordinates of the squares
            tif_path = os.path.join(google_remote_root_dir, google_remote_file)
            coordinates = get_tif_bounds(tif_path)
            coordinates_set = set()

            # 2. Judge the state and calculate file_name_part
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
                            break
                        else:
                            print(f"  ⚠️ Warning: State '{state_row['NAME']}' does not have an abbreviation map")

                if not matched:
                    print(f"  ❌ Corner {i + 1} ({lon:.6f}, {lat:.6f}): Not matched to any state")

            # 3. Check if the state is found
            if not coordinates_set:
                down_dict_info[group_name][google_remote_file]["__error__"] = "no_state_found"
                down_dict_info[group_name][google_remote_file]["__detail__"] = f"Four-corner coordinates: {coordinates}"
                print(f"  Failed: No matching to any state")
                continue

            # 4. Find links in the index (key improvement: print detailed match information)
            link_found = False
            for state, file_name_part in coordinates_set:

                if state not in index_dict:
                    down_dict_info[group_name][google_remote_file]["__error__"] = f"state_not_in_index:{state}"
                    print(f"    ❌There are no states for {google_remote_file} in the index dictionary '{state}'")
                    continue

                idx_list = index_dict[state]

                matched_links = []
                for link in idx_list:
                    if file_name_part in link:
                        matched_links.append(link)
                        down_dict_info[group_name][google_remote_file][link] = False
                        link_found = True

                if not matched_links:
                    # print(f"    ❌ No link with '{file_name_part}' was found for {google_remote_file}")
                    similar = [l for l in idx_list if file_name_part[:5] in l or file_name_part[-5:] in l][:3]
                    # if similar:
                    #     print(f"   Similar links (for reference): {similar}")

            # 5. Eventual failure record
            if not link_found:
                down_dict_info[group_name][google_remote_file]["__error__"] = "no_link_found"
                down_dict_info[group_name][google_remote_file]["__detail__"] = {
                    "states_searched": [c[0] for c in coordinates_set],
                    "file_parts": [c[1] for c in coordinates_set],
                    "coordinates": coordinates
                }
                # print(f"\n  Final Failure: No matching links found")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            down_dict_info[group_name][google_remote_file]["__error__"] = "exception"
            down_dict_info[group_name][google_remote_file]["__detail__"] = error_msg
            print(f'\n  Exception: {google_remote_file}: {e}')

        # 每10个保存一次
        if len(down_dict_info[group_name]) % 10 == 0:
            with open(usgs_dem_down_link, 'w', encoding='utf-8') as f:
                json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(usgs_dem_down_link, 'w', encoding='utf-8') as f:
        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    # 统计
    total = len(down_dict_info.get(group_name, {}))
    success = sum(1 for f, links in down_dict_info.get(group_name, {}).items()
                  if not any(k.startswith("__") for k in links.keys()))
    failed = total - success
    print(f"\n{'=' * 60}")
    print(f"{group_name} Complete： Total {total}， Successful {success}， Failed {failed}")
    print(f"{'=' * 60}")


def analyze_download_statistics(usgs_dem_down_link):
    """
    分析下载统计信息，包括总体和各group_name分布
    """
    if not os.path.exists(usgs_dem_down_link):
        print("统计文件不存在")
        return

    with open(usgs_dem_down_link, "r", encoding="UTF-8") as f:
        down_dict_info = json.load(f)

    # 总体统计
    total_files = 0
    downloaded_files = 0
    failed_files = 0
    pending_files = 0  # 有链接但未下载

    # 各组统计
    group_stats = {}

    for group_name, group_info in down_dict_info.items():
        group_total = len(group_info)
        group_downloaded = 0
        group_failed = 0
        group_pending = 0

        for file_name, links in group_info.items():
            total_files += 1

            # 检查是否为错误状态
            if "__error__" in links:
                failed_files += 1
                group_failed += 1
                continue

            # 检查下载状态
            has_links = False
            is_downloaded = False

            for link, status in links.items():
                if not link.startswith("__"):  # 排除错误标识
                    has_links = True
                    if status:  # 已下载
                        is_downloaded = True
                        break

            if is_downloaded:
                downloaded_files += 1
                group_downloaded += 1
            elif not has_links:  # 没有可用链接
                failed_files += 1
                group_failed += 1
            else:  # 有链接但未下载
                pending_files += 1
                group_pending += 1

        # 记录组统计
        group_stats[group_name] = {
            'total': group_total,
            'downloaded': group_downloaded,
            'failed': group_failed,
            'pending': group_pending
        }

    # 打印总体统计
    print("="*60)
    print("总体下载统计:")
    print(f"总文件数: {total_files}")
    print(f"已下载: {downloaded_files}")
    print(f"失败/无链接: {failed_files}")
    print(f"待下载: {pending_files}")
    print(f"下载成功率: {(downloaded_files/total_files)*100:.2f}%")
    print("="*60)

    # 打印各组统计
    print("\n各Group分布统计:")
    print("-"*80)
    print(f"{'Group Name':<20} {'Total':<8} {'Downloaded':<12} {'Failed':<8} {'Pending':<8} {'Success Rate':<12}")
    print("-"*80)

    for group_name, stats in group_stats.items():
        success_rate = (stats['downloaded']/stats['total'])*100 if stats['total'] > 0 else 0
        print(f"{group_name:<20} {stats['total']:<8} {stats['downloaded']:<12} "
              f"{stats['failed']:<8} {stats['pending']:<8} {success_rate:<12.2f}%")

    print("-"*80)

    # 显示失败详情（可选）
    show_failure_details = input("\n是否显示失败详情? (y/n): ").lower() == 'y'
    if show_failure_details:
        print("\n失败详情:")
        for group_name, group_info in down_dict_info.items():
            failed_count = 0
            for file_name, links in group_info.items():
                if "__error__" in links:
                    failed_count += 1
                    print(f"  {group_name}/{file_name}: {links['__error__']}")
            if failed_count > 0:
                print(f"  Group {group_name}: 共 {failed_count} 个失败文件")


def get_detailed_breakdown(usgs_dem_down_link):
    """
    获取详细的下载状态分解
    """
    if not os.path.exists(usgs_dem_down_link):
        return None

    with open(usgs_dem_down_link, "r", encoding="UTF-8") as f:
        down_dict_info = json.load(f)

    breakdown = {
        'total_files': 0,
        'downloaded': [],
        'failed_no_state': [],      # 无法匹配到州
        'failed_no_link': [],       # 无法找到对应链接
        'failed_exception': [],     # 处理异常
        'failed_state_not_in_index': [],  # 州不在索引中
        'pending_with_links': []    # 有待下载链接的文件
    }

    for group_name, group_info in down_dict_info.items():
        for file_name, links in group_info.items():
            breakdown['total_files'] += 1

            if '__error__' in links:
                error_type = links['__error__']
                if error_type == 'no_state_found':
                    breakdown['failed_no_state'].append(f"{group_name}/{file_name}")
                elif error_type == 'no_link_found':
                    breakdown['failed_no_link'].append(f"{group_name}/{file_name}")
                elif error_type == 'exception':
                    breakdown['failed_exception'].append(f"{group_name}/{file_name}")
                elif error_type.startswith('state_not_in_index'):
                    breakdown['failed_state_not_in_index'].append(f"{group_name}/{file_name}")
            else:
                # 检查是否有已下载的链接
                is_downloaded = any(status for link, status in links.items() if not link.startswith('__'))
                if is_downloaded:
                    breakdown['downloaded'].append(f"{group_name}/{file_name}")
                else:
                    # 有链接但未下载
                    has_any_links = any(not link.startswith('__') for link in links.keys())
                    if has_any_links:
                        breakdown['pending_with_links'].append(f"{group_name}/{file_name}")

    return breakdown


def check_load_file(usa_dem_root_dir, usgs_dem_down_link):

    if os.path.exists(usgs_dem_down_link):
        with open(usgs_dem_down_link, "r", encoding="UTF-8") as f:
            down_dict_info = json.load(f)
    else:
        return None

    for group_name, group_info in down_dict_info.items():
        for file_name, links in group_info.items():
            file_path = os.path.join(usa_dem_root_dir, group_name, file_name)
            if os.path.exists(file_path):
                try:
                    for link, downloaded in links.items():
                        links[link] = True
                except Exception as e:
                    print(f"{file_path} is exists, but the content is error!")
                    print(f"Error processing {file_name}: {e}")

    with open(usgs_dem_down_link, "w", encoding="UTF-8") as f:
        json.dump(down_dict_info, f, ensure_ascii=False, indent=2)

    # 运行统计分析
    analyze_download_statistics(usgs_dem_down_link)

    # 或者获取详细分解信息
    detailed_breakdown = get_detailed_breakdown(usgs_dem_down_link)
    if detailed_breakdown:
        print(f"已下载文件: {len(detailed_breakdown['downloaded'])}")
        print(f"失败-无法匹配州: {len(detailed_breakdown['failed_no_state'])}")
        print(f"失败-无法找到链接: {len(detailed_breakdown['failed_no_link'])}")
        print(f"待下载: {len(detailed_breakdown['pending_with_links'])}")
