#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: USGS_index_metadata_download
# @Time    : 2026/1/26 16:04
# @Author  : Kevin
# @Describe:

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName: download_metadata_and_build_index.py
# @Time: 2026/1/17 15:19
# @Author: Kevin
# @Description: 下载USGS元数据XML并建立数据库索引

import requests
import warnings
from urllib3.exceptions import InsecureRequestWarning
import re
import os
import sqlite3
import time
from pathlib import Path

# 禁用SSL警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# ====================================================================
# 配置参数
# ====================================================================
INPUT_INDEX_DIR = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Data\USA\USGSDEM\Index"
OUTPUT_METADATA_DIR = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Data\USA\USGSDEM\Metadata"
DB_PATH = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Data\USA\USGSDEM\usgs_dem_metadata.db"

USGS_BASE_URL = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/1m/Projects/"

# ====================================================================
# 核心函数：从文本提取边界（fallback）
# ====================================================================

def extract_bounding_from_text(text, project_name):
    """
    从元数据文本中提取边界坐标（作为XML解析失败时的fallback）

    参数:
        text: 元数据文本内容
        project_name: 项目名称

    返回:
        dict: {minx, maxx, miny, maxy} 或 None
    """
    # 查找"Bounding Coordinates"后面的4个数字
    pattern = r'Bounding\s+Coordinates?[-:]?\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)'
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return {
            'minx': float(match.group(1)),
            'maxx': float(match.group(2)),
            'miny': float(match.group(3)),
            'maxy': float(match.group(4))
        }

    # 如果还找不到，尝试查找任意4个连续的经纬度数字（最后4个通常是边界）
    numbers = re.findall(r'-?\d+\.\d+', text)
    if len(numbers) >= 4:
        return {
            'minx': float(numbers[-4]),
            'maxx': float(numbers[-3]),
            'miny': float(numbers[-2]),
            'maxy': float(numbers[-1])
        }

    print(f"  ✗ 无法提取边界: {project_name}")
    return None


# ====================================================================
# 核心函数：下载元数据XML
# ====================================================================

def download_metadata_for_project(project_name, tile_url, max_retries=3):
    """
    下载项目的元数据XML文件

    参数:
        project_name: 项目名称，如 "MI_31Co_Gladwin_2016"
        tile_url: 项目第一个瓦片的完整URL
        max_retries: 最大重试次数

    返回:
        bytes: XML内容，失败返回None
    """
    # 从瓦片文件名推断元数据文件名
    # 格式1: USGS_one_meter_x69y486_MI_31Co_Gladwin_2016.xml
    # 格式2: USGS_1M_19_x38y508_ME_CrownofMaine_2018_A18.xml

    tile_basename = os.path.basename(tile_url).replace('.tif', '')

    # 尝试两种可能的元数据文件名格式
    possible_names = [
        f"{tile_basename}.xml",  # 格式1
        f"{tile_basename}_meta.xml", # 格式2
        tile_basename.replace('_one_meter_', '_1M_') + ".xml",  # 格式3,
        tile_basename.replace('_one_meter_', '_1M_') + "_meta.xml"  # 格式4
    ]

    for meta_name in possible_names:

        metadata_url = f"{USGS_BASE_URL}{project_name}/metadata/{meta_name}"

        for attempt in range(max_retries):
            try:
                print(f"      尝试: {meta_name} (第{attempt+1}次)")
                response = requests.get(metadata_url, timeout=60, verify=False)

                if response.status_code == 200:
                    print(f"      ✓ 成功下载: {meta_name}")
                    return response.content, meta_name
                elif response.status_code == 404:
                    print(f"      ⚠ 文件不存在: {meta_name}")
                    break  # 404无需重试，尝试下一个文件名
                else:
                    print(f"      ✗ HTTP {response.status_code}")

            except Exception as e:
                print(f"      ✗ 错误: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        time.sleep(1)  # 尝试下一个文件名前延迟

    print(f"      ✗ 所有尝试失败: {project_name}")

    return None, None


# ====================================================================
# 核心函数：从XML或文本提取边界
# ====================================================================

def extract_bounding_from_content(content, project_name):
    """
    从元数据内容（XML或文本）中提取边界坐标

    参数:
        content: 字节或字符串内容
        project_name: 项目名称（用于日志）

    返回:
        dict: {minx, maxx, miny, maxy, delta_lon, delta_lat} 或 None
    """
    if not content:
        return None

    # 转换为字符串（如果是bytes）
    text = content.decode('utf-8') if isinstance(content, bytes) else content

    # 优先尝试从文本中提取（更快更可靠）
    bounds = extract_bounding_from_text(text, project_name)

    if bounds:
        # 计算步长
        bounds['delta_lon'] = bounds['maxx'] - bounds['minx']
        bounds['delta_lat'] = bounds['maxy'] - bounds['miny']
        return bounds

    return None


# ====================================================================
# 核心函数：从链接文件提取瓦片URL
# ====================================================================

def get_first_tile_url(link_file_path):
    """
    从链接文件中提取第一个有效的瓦片URL

    参数:
        link_file_path: 链接文件路径

    返回:
        str: 第一个瓦片URL，失败返回None
    """
    try:
        with open(link_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and url.endswith('.tif'):
                    return url
    except Exception as e:
        print(f"✗ 读取链接文件失败: {e}")

    return None


# ====================================================================
# 核心函数：提取项目名和瓦片索引
# ====================================================================

def extract_project_and_index(tile_url):
    """
    从瓦片URL中提取项目名和索引

    参数:
        tile_url: 完整瓦片URL

    返回:
        tuple: (project_name, x, y) 或 None
    """
    filename = os.path.basename(tile_url)

    # 提取索引 x??y??
    match = re.search(r'x(\d+)y(\d+)', filename)
    if not match:
        return None

    x = int(match.group(1))
    y = int(match.group(2))

    # ===== 修复：更灵活的项目名提取 =====
    # 先移除文件扩展名
    name_without_ext = filename.replace('.tif', '')

    # 模式1: USGS_one_meter_x69y486_ 或 USGS_1M_16_x65y376_ 或 USGS_1m_x60y362_
    # 使质量等级部分(\d+)?可选，并支持大小写不敏感
    pattern = r'USGS_(one_meter|1M|1m)(?:_\d+)?_x\d+y\d+_'
    project = re.sub(pattern, '', name_without_ext, flags=re.IGNORECASE)

    # 清理可能残留的'meta'字样
    project = project.replace('_meta', '')

    return project, x, y


# ====================================================================
# 核心函数：创建数据库并插入数据
# ====================================================================

def create_metadata_database(db_path, index_dir):
    """
    主函数：创建SQLite数据库并插入元数据信息

    参数:
        db_path: 数据库文件输出路径
        index_dir: 链接文件输入目录路径

    返回:
        dict: 统计信息
    """
    print("="*70)
    print("步骤3：创建元数据索引数据库")
    print("="*70)
    print(f"数据库: {db_path}")
    print(f"输入目录: {index_dir}")

    # 确保输出目录存在
    metadata_dir = Path(OUTPUT_METADATA_DIR)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # 连接到SQLite数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建表结构
    cursor.execute('DROP TABLE IF EXISTS project_metadata')

    cursor.execute('''
        CREATE TABLE project_metadata (
            project_name TEXT PRIMARY KEY,
            first_tile_x INTEGER,
            first_tile_y INTEGER,
            minx REAL,
            maxx REAL,
            miny REAL,
            maxy REAL,
            delta_lon REAL,
            delta_lat REAL,
            center_lon REAL,
            center_lat REAL,
            metadata_url TEXT,
            metadata_file_path TEXT,
            download_status TEXT
        )
    ''')

    # 创建索引
    cursor.execute('CREATE INDEX idx_project_name ON project_metadata(project_name)')

    # 统计信息
    stats = {
        'total_projects': 0,
        'successful': 0,
        'failed': 0,
        'metadata_downloaded': 0,
        'boundary_extracted': 0
    }

    # 扫描所有链接文件
    index_path = Path(index_dir)
    link_files = list(index_path.glob("*.txt"))

    print(f"\n扫描到 {len(link_files)} 个链接文件")

    for i, link_file in enumerate(link_files, 1):
        project_name = link_file.stem

        print(f"\n[{i}/{len(link_files)}] 处理项目: {project_name}")

        # 读取第一个瓦片URL
        first_url = get_first_tile_url(link_file)
        if not first_url:
            print(f"  ✗ 无法获取瓦片URL")
            stats['failed'] += 1
            continue

        print(f"  首个瓦片: {os.path.basename(first_url)}")

        # 提取索引和项目名
        url_info = extract_project_and_index(first_url)
        if not url_info:
            print(f"  ✗ 无法解析URL")
            stats['failed'] += 1
            continue

        extracted_project, x, y = url_info

        # 验证项目名匹配
        if project_name not in extracted_project:
            print(f"  ⚠ 项目名不匹配: {extracted_project} != {project_name}")

        # 尝试下载元数据
        print(f"  下载元数据...")

        xml_content, metadata_filename = download_metadata_for_project(project_name, first_url)

        bounds = None
        metadata_url = None
        metadata_file_path = None

        if xml_content:
            stats['metadata_downloaded'] += 1

            # 提取边界
            bounds = extract_bounding_from_content(xml_content, project_name)

            if bounds:
                stats['boundary_extracted'] += 1
                print(f"  ✓ 边界提取成功")

                # 保存元数据到文件（按州/项目组织）
                state_code = project_name[:2]
                project_metadata_dir = metadata_dir / state_code / project_name
                project_metadata_dir.mkdir(parents=True, exist_ok=True)

                metadata_file_path = project_metadata_dir / metadata_filename
                metadata_url = f"{USGS_BASE_URL}{project_name}/metadata/{metadata_filename}"

                # 保存XML文件
                with open(metadata_file_path, 'wb') as f:
                    f.write(xml_content)

                print(f"    保存到: {metadata_file_path}")
            else:
                print(f"  ⚠ 边界提取失败")
        else:
            print(f"  ✗ 元数据下载失败")

        # 准备数据库记录
        if bounds:
            record = {
                'project_name': project_name,
                'first_tile_x': x,
                'first_tile_y': y,
                'minx': bounds['minx'],
                'maxx': bounds['maxx'],
                'miny': bounds['miny'],
                'maxy': bounds['maxy'],
                'delta_lon': bounds.get('delta_lon', 0),
                'delta_lat': bounds.get('delta_lat', 0),
                'center_lon': (bounds['minx'] + bounds['maxx']) / 2,
                'center_lat': (bounds['miny'] + bounds['maxy']) / 2,
                'metadata_url': metadata_url,
                'metadata_file_path': str(metadata_file_path) if metadata_file_path else None,
                'download_status': 'success' if xml_content else 'failed'
            }
        else:
            # 边界提取失败，只记录基本信息
            record = {
                'project_name': project_name,
                'first_tile_x': x,
                'first_tile_y': y,
                'minx': None, 'maxx': None, 'miny': None, 'maxy': None,
                'delta_lon': None, 'delta_lat': None,
                'center_lon': None, 'center_lat': None,
                'metadata_url': None,
                'metadata_file_path': None,
                'download_status': 'failed'
            }

        # 插入数据库
        try:
            cursor.execute('''
                INSERT INTO project_metadata (
                    project_name, first_tile_x, first_tile_y,
                    minx, maxx, miny, maxy,
                    delta_lon, delta_lat,
                    center_lon, center_lat,
                    metadata_url, metadata_file_path, download_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record['project_name'], record['first_tile_x'], record['first_tile_y'],
                record['minx'], record['maxx'], record['miny'], record['maxy'],
                record['delta_lon'], record['delta_lat'],
                record['center_lon'], record['center_lat'],
                record['metadata_url'], record['metadata_file_path'], record['download_status']
            ))

            stats['successful'] += 1
            print(f"  ✓ 数据库记录已插入")

        except sqlite3.IntegrityError:
            print(f"  ⚠ 项目已存在，跳过")

        # 每10个项目提交一次
        if i % 10 == 0:
            conn.commit()
            print(f"  已提交 {i} 个项目")

    # 最终提交
    conn.commit()

    # 输出统计信息
    cursor.execute('''
        SELECT COUNT(DISTINCT SUBSTR(project_name, 1, 2)) FROM project_metadata
    ''')
    state_count = cursor.fetchone()[0]

    print("\n" + "="*70)
    print("数据库创建完成")
    print("="*70)
    print(f"总项目数: {stats['total_projects']}")
    print(f"成功: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"元数据下载成功: {stats['metadata_downloaded']}")
    print(f"边界提取成功: {stats['boundary_extracted']}")
    print(f"覆盖州数: {state_count}")
    print("="*70)

    # 关闭数据库连接
    conn.close()

    return stats


# ====================================================================
# 主程序入口
# ====================================================================

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         USGS 3DEP元数据XML下载与索引数据库创建工具                    ║")
    print("║         步骤3：为每个项目下载元数据并建立SQLite索引                    ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # 确保输出目录存在
    Path(OUTPUT_METADATA_DIR).mkdir(parents=True, exist_ok=True)

    confirm = input(f"\n开始操作?\n  输入目录: {INPUT_INDEX_DIR}\n  元数据输出: {OUTPUT_METADATA_DIR}\n  数据库: {DB_PATH}\n\n确认(y/n): ")

    if confirm.lower() != 'y':
        print("操作取消")
        exit()

    # 执行操作
    stats = create_metadata_database(DB_PATH, INPUT_INDEX_DIR)

    print("\n✓ 所有操作完成！")
    print(f"数据库文件: {DB_PATH}")
    print(f"文件大小: {Path(DB_PATH).stat().st_size / (1024*1024):.1f} MB")

    # 显示元数据目录统计
    metadata_count = sum(1 for _ in Path(OUTPUT_METADATA_DIR).rglob("*.xml"))
    print(f"下载的XML文件: {metadata_count} 个")