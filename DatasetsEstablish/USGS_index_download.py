import requests
import warnings
from urllib3.exceptions import InsecureRequestWarning
import re
import os
import time
from pathlib import Path

# 禁用SSL警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# ====================================================================
# 配置参数
# ====================================================================

# 输出目录（修改为您的路径）
OUTPUT_BASE_DIR = r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\Data\USA\USGSDEM\Index"

# USGS项目根目录
USGS_PROJECTS_URL = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/1m/Projects/"


# ====================================================================
# 核心下载函数（已修改）
# ====================================================================

def get_all_projects():
    """获取所有USGS 1米DEM项目列表"""
    print("=" * 70)
    print("步骤1：获取USGS项目列表")
    print("=" * 70)
    print(f"连接: {USGS_PROJECTS_URL}")

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

        print(f"✓ 成功获取 {len(projects)} 个有效项目")
        return projects

    except Exception as e:
        print(f"✗ 获取失败: {e}")
        return []


def download_link_file(project_name, output_dir):
    """下载单个项目的链接文件（修改为平铺保存）"""
    link_file_url = f"{USGS_PROJECTS_URL}{project_name}/0_file_download_links.txt"

    # 直接保存为 {project_name}.txt（不创建子目录）
    output_path = os.path.join(output_dir, f"{project_name}.txt")

    # 检查是否已存在且非空
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

        # 保存文件（直接保存到输出目录）
        with open(output_path, 'wb') as f:
            f.write(response.content)

        file_size = len(response.content) / 1024
        return f"success_{file_size:.1f}KB"

    except Exception as e:
        return f"error_{e}"


def download_all_projects(output_dir, delay=1):
    """主函数：下载所有项目的链接文件（修改为平铺结构）"""
    print("\n" + "=" * 70)
    print("步骤2：开始批量下载链接文件（平铺保存）")
    print("=" * 70)
    print(f"输出目录: {output_dir}")

    projects = get_all_projects()
    if not projects:
        print("✗ 无法获取项目列表")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n准备下载 {len(projects)} 个项目的链接文件...")

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

        # 下载文件
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

        # 更新进度显示
        print(f'\r[{bar}] {i}/{len(projects)} {status} {project:<50}', end='', flush=True)

        # 礼貌延迟
        time.sleep(delay)

    # 打印最终统计
    print("\n\n" + "=" * 70)
    print("下载完成统计")
    print("=" * 70)
    print(f"总项目数: {stats['total']}")
    print(f"成功: {stats['success']}")
    print(f"跳过(已存在): {stats['skipped']}")
    print(f"未找到(404): {stats['not_found']}")
    print(f"失败: {stats['failed']}")

    # 显示最终目录内容
    print(f"\n{'=' * 70}")
    print(f"输出目录内容: {OUTPUT_BASE_DIR}")
    print("=" * 70)
    files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    print(f"已下载 {len(files)} 个链接文件")

    # 列出前10个文件
    print("\n前10个文件:")
    for i, f in enumerate(files[:10], 1):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {i:3d}. {f:<50} ({size / 1024:.1f} KB)")

    if len(files) > 10:
        print(f"  ... 还有 {len(files) - 10} 个文件")

    print("=" * 70)


# ====================================================================
# 主程序入口
# ====================================================================

if __name__ == "__main__":
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║         USGS 3DEP 1米DEM链接文件批量下载工具                         ║")
    print("║         保存结构：所有文件平铺在单一目录下                             ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # 确保输出目录存在
    output_dir = Path(OUTPUT_BASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 开始下载
    download_all_projects(OUTPUT_BASE_DIR, delay=1)

    print("\n✓ 所有操作完成！")
    print(f"文件已保存到: {OUTPUT_BASE_DIR}")

    # 显示最终统计
    files = [f for f in os.listdir(OUTPUT_BASE_DIR) if f.endswith('.txt')]
    print(f"总计 {len(files)} 个链接文件")