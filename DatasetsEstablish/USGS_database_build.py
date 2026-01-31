#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName: rebuild_database_from_existing_xml.py

import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import os
from typing import Tuple, Optional, Dict

# ====================================================================
# 配置参数
# ====================================================================
INDEX_DIR = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Data\USA\USGSDEM\Index"
XML_METADATA_DIR = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Data\USA\USGSDEM\Metadata"
DB_PATH = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Data\USA\USGSDEM\usgs_dem_metadata.db"


# ====================================================================
# 核心函数：从XML文件精确提取边界（修复版）
# ====================================================================

def extract_bounds_from_xml_file(xml_file_path: Path) -> Optional[Dict]:
    """
    从已下载的XML文件中精确提取WGS84边界坐标

    返回: {'minx': west, 'maxx': east, 'miny': south, 'maxy': north} 或 None
    """
    try:
        # 解析XML文件
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 精确查找bounding元素
        bounding = root.find('.//bounding')
        if bounding is None:
            print(f"  ⚠ 未找到<bounding>元素: {xml_file_path.name}")
            return None

        # 提取四个边界值（注意miny对应south，maxy对应north）
        elems = {
            'minx': bounding.find('westbc'),
            'maxx': bounding.find('eastbc'),
            'miny': bounding.find('southbc'),  # 纬度最小值（南）
            'maxy': bounding.find('northbc')  # 纬度最大值（北）
        }

        # 检查所有元素都存在且有文本内容
        for key, elem in elems.items():
            if elem is None or elem.text is None or elem.text.strip() == '':
                print(f"  ⚠ 缺少{key}元素或内容为空")
                return None

        # 转换为浮点数
        try:
            bounds = {k: float(v.text) for k, v in elems.items()}
        except (ValueError, TypeError) as e:
            print(f"  ⚠ 坐标转换失败: {e}")
            return None

        # 验证坐标合理性（WGS84范围）
        if not (-180 <= bounds['minx'] <= 180 and -180 <= bounds['maxx'] <= 180):
            print(f"  ⚠ 经度超出WGS84范围: {bounds['minx']}, {bounds['maxx']}")
            return None
        if not (-90 <= bounds['miny'] <= 90 and -90 <= bounds['maxy'] <= 90):
            print(f"  ⚠ 纬度超出WGS84范围: {bounds['miny']}, {bounds['maxy']}")
            return None

        # 验证边界顺序（min < max）
        if bounds['minx'] >= bounds['maxx'] or bounds['miny'] >= bounds['maxy']:
            print(f"  ⚠ 边界顺序错误: minx={bounds['minx']}, maxx={bounds['maxx']}")
            return None

        # 计算范围
        bounds['delta_lon'] = bounds['maxx'] - bounds['minx']
        bounds['delta_lat'] = bounds['maxy'] - bounds['miny']

        print(f"  ✓ 提取边界: {bounds['minx']:.4f} to {bounds['maxx']:.4f}, "
              f"{bounds['miny']:.4f} to {bounds['maxy']:.4f}")

        return bounds

    except ET.ParseError as e:
        print(f"  ✗ XML解析失败: {xml_file_path.name} - {e}")
        return None
    except Exception as e:
        print(f"  ✗ 未知错误: {xml_file_path.name} - {e}")
        return None


# ====================================================================
# 核心函数：从索引文件提取瓦片索引
# ====================================================================

def extract_tile_index_from_url(tile_url: str) -> Optional[Tuple[int, int]]:
    """从TIF URL中提取x,y网格索引"""
    if not tile_url:
        return None

    filename = os.path.basename(tile_url)
    match = re.search(r'x(\d+)y(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


# ====================================================================
# 核心函数：重建数据库
# ====================================================================

def rebuild_database():
    """从已下载的XML文件重建数据库"""
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         重建数据库（从已下载的XML文件）                             ║")
    print("║         修复：强制使用XML解析提取WGS84边界坐标                      ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # 删除旧数据库
    db_file = Path(DB_PATH)
    if db_file.exists():
        db_file.unlink()
        print(f"\n✓ 删除旧数据库: {DB_PATH}")

    # 创建新数据库
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

    # 统计
    stats = {
        'total_projects': 0,
        'successful': 0,
        'failed': 0,
        'xml_found': 0,
        'boundary_extracted': 0
    }

    # 扫描索引目录
    index_path = Path(INDEX_DIR)
    link_files = list(index_path.glob("*.txt"))
    stats['total_projects'] = len(link_files)

    print(f"\n扫描到 {len(link_files)} 个项目")

    for i, link_file in enumerate(link_files, 1):
        project_name = link_file.stem
        print(f"\n[{i}/{len(link_files)}] 处理项目: {project_name}")

        # 步骤1: 从索引文件获取第一个瓦片URL
        tile_url = None
        try:
            with open(link_file, 'r', encoding='utf-8') as f:
                for line in f:
                    url = line.strip()
                    if url and url.endswith('.tif'):
                        tile_url = url
                        break
        except Exception as e:
            print(f"  ✗ 读取索引文件失败: {e}")
            stats['failed'] += 1
            continue

        if not tile_url:
            print(f"  ✗ 找不到瓦片URL")
            stats['failed'] += 1
            continue

        # 提取瓦片索引
        tile_index = extract_tile_index_from_url(tile_url)
        if not tile_index:
            print(f"  ✗ 无法提取瓦片索引")
            stats['failed'] += 1
            continue

        x, y = tile_index

        # 步骤2: 查找对应的XML文件
        # XML文件路径: Metadata/{州代码}/{项目名称}/*.xml
        state_code = project_name[:2]
        xml_search_dir = Path(XML_METADATA_DIR) / state_code / project_name

        if not xml_search_dir.exists():
            print(f"  ⚠ XML目录不存在: {xml_search_dir}")
            stats['failed'] += 1
            continue

        # 查找XML文件（任意一个即可）
        xml_files = list(xml_search_dir.glob("*.xml"))
        if not xml_files:
            print(f"  ⚠ 未找到XML文件: {xml_search_dir}")
            stats['failed'] += 1
            continue

        xml_file = xml_files[0]  # 使用第一个找到的XML文件
        stats['xml_found'] += 1

        # 步骤3: 从XML提取边界
        print(f"  读取XML: {xml_file.name}")
        bounds = extract_bounds_from_xml_file(xml_file)

        if not bounds:
            stats['failed'] += 1
            continue

        stats['boundary_extracted'] += 1

        # 步骤4: 插入数据库
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
                project_name,
                x, y,
                bounds['minx'], bounds['maxx'], bounds['miny'], bounds['maxy'],
                bounds['delta_lon'], bounds['delta_lat'],
                (bounds['minx'] + bounds['maxx']) / 2,
                (bounds['miny'] + bounds['maxy']) / 2,
                tile_url.replace('.tif', '.xml'),
                str(xml_file),
                'success'
            ))

            stats['successful'] += 1
            print(f"  ✓ 数据库记录已插入")

        except sqlite3.Error as e:
            print(f"  ✗ 数据库插入失败: {e}")
            stats['failed'] += 1

        # 每10个提交一次
        if i % 10 == 0:
            conn.commit()

    # 最终统计
    conn.commit()

    cursor.execute('SELECT COUNT(DISTINCT SUBSTR(project_name, 1, 2)) FROM project_metadata')
    state_count = cursor.fetchone()[0]

    print("\n" + "=" * 70)
    print("数据库重建完成")
    print("=" * 70)
    print(f"总项目数: {stats['total_projects']}")
    print(f"成功: {stats['successful']}")
    print(f"失败: {stats['failed']}")
    print(f"找到XML: {stats['xml_found']}")
    print(f"提取边界: {stats['boundary_extracted']}")
    print(f"覆盖州数: {state_count}")
    print("=" * 70)

    # 关闭连接
    conn.close()

    return stats


# ====================================================================
# 主程序入口
# ====================================================================

if __name__ == "__main__":
    # 确认路径
    print("路径确认:")
    print(f"  索引目录: {INDEX_DIR}")
    print(f"  XML目录:  {XML_METADATA_DIR}")
    print(f"  数据库:   {DB_PATH}")

    confirm = input("\n开始重建数据库? (y/n): ")
    if confirm.lower() != 'y':
        print("操作取消")
        exit()

    # 执行重建
    stats = rebuild_database()

    print(f"\n✓ 重建完成！")
    print(f"数据库文件: {DB_PATH} ({Path(DB_PATH).stat().st_size / (1024 * 1024):.1f} MB)")