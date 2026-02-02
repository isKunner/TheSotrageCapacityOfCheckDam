#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: __init__
# @Time    : 2026/1/26 10:18
# @Author  : Kevin
# @Describe:

import re
import os
import time

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

USGS_PROJECTS_URL = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/1m/Projects/"

def get_all_projects():
    """获取所有USGS 1米DEM项目列表"""
    print("=" * 70)
    print("Step 1: Get a list of USGS projects")
    print("=" * 70)
    print(f"Connections: {USGS_PROJECTS_URL}")

    try:
        response = requests.get(
            USGS_PROJECTS_URL,
            timeout=60,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        projects = re.findall(r'href="([^"]+)/"', response.text)
        projects = [p for p in projects if re.match(r'^[A-Z]{2}_', p) and p not in ['..']]

        print(f"✓ Successfully obtain {len(projects)} valid projects")
        return projects

    except Exception as e:
        print(f"✗ Acquisition failure: {e}")
        return []


def download_link_file(project_name, output_dir):

    """Download the linked files for individual projects"""

    link_file_url = f"{USGS_PROJECTS_URL}{project_name}/0_file_download_links.txt"

    output_path = os.path.join(output_dir, f"{project_name}.txt")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        return "skipped"

    try:
        response = requests.get(
            link_file_url,
            timeout=60,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code == 404:
            return "not_found"
        elif response.status_code != 200:
            return f"http_error_{response.status_code}"

        with open(output_path, 'wb') as f:
            f.write(response.content)

        file_size = len(response.content) / 1024
        return f"success_{file_size:.1f}KB"

    except Exception as e:
        return f"error_{e}"


def usgs_down_index(output_dir, delay=1):
    """主函数：下载所有项目的链接文件（修改为平铺结构）"""
    print("\n" + "=" * 70)
    print("Step 1: Start downloading linked files in batches")
    print("=" * 70)
    print(f"Output directory: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    projects = get_all_projects()
    if not projects:
        print("✗ The list of items is not available")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nPrepare to download the linked files for {len(projects)} projects...")

    stats = {
        'total': len(projects),
        'success': 0,
        'skipped': 0,
        'not_found': 0,
        'failed': 0
    }

    bar_width = 50

    for i, project in enumerate(projects, 1):
        # 显示进度条
        progress = i / len(projects)
        filled = int(bar_width * progress)
        bar = '█' * filled + '-' * (bar_width - filled)

        result = download_link_file(project, output_dir)

        if result.startswith("success"):
            stats['success'] += 1
            status = "✓"
        elif result == "skipped":
            stats['skipped'] += 1
            status = "⏭"
        elif result == "not_found":
            stats['not_found'] += 1
            status = "⚠"
        else:
            stats['failed'] += 1
            status = "✗"

        print(f'\r[{bar}] {i}/{len(projects)} {status} {project:<50}', end='', flush=True)

        # 礼貌延迟
        time.sleep(delay)

    # 打印最终统计
    print("\n\n" + "=" * 70)
    print("Download the completion statistics")
    print("=" * 70)
    print(f"Total number of projects: {stats['total']}")
    print(f"Successfully: {stats['success']}")
    print(f"Skip (already exists): {stats['skipped']}")
    print(f"Not Found(404): {stats['not_found']}")
    print(f"Failed: {stats['failed']}")

    # 显示最终目录内容
    print(f"\n{'=' * 70}")
    print(f"Output directory contents: {output_dir}")
    print("=" * 70)
    files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    print(f"{len(files)} linked files have been downloaded")

    # 列出前10个文件
    print("\nTop 10 files:")
    for i, f in enumerate(files[:10], 1):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {i:3d}. {f:<50} ({size / 1024:.1f} KB)")

    if len(files) > 10:
        print(f"  ... There are also {len(files) - 10} files")

    print("=" * 70)


# ====================================================================
# 主程序入口
# ====================================================================

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         USGS 3DEP 1米DEM链接文件批量下载工具                         ║")
    print("║         保存结构：所有文件平铺在单一目录下                             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    usgs_down_index("./test", delay=1)

