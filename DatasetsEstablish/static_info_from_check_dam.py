#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: utils
# @Time    : 2025/4/28 09:05
# @Author  : Kevin
# @Describe: 进行基本的数据处理，包括统计淤地坝的数据、统计每个市有哪些淤地坝，并建立需要采样的数据
import json
import random

import geopandas as gpd
import numpy as np
import pandas as pd

def get_shp_file_info(shp_file_path):
    """
    获取shp文件的基本信息

    参数:
        shp_file_path (str): shp文件的路径

    返回:
        dict: 包含文件基本信息的字典
    """
    import geopandas as gpd

    try:
        # 读取shp文件
        gdf = gpd.read_file(shp_file_path)

        # 获取基本信息
        info = {
            "文件路径": shp_file_path,
            "记录总数": len(gdf),
            "几何类型": list(gdf.geom_type.unique()),
            "坐标系": str(gdf.crs),
            "字段列表": list(gdf.columns),
            "字段数量": len(gdf.columns)-1,  # 减去geometry字段
            "边界范围": {
                "最小X": gdf.total_bounds[0],
                "最小Y": gdf.total_bounds[1],
                "最大X": gdf.total_bounds[2],
                "最大Y": gdf.total_bounds[3]
            }
        }

        # 显示前几行数据结构（不显示实际数据）
        print(f"文件路径: {info['文件路径']}")
        print(f"记录总数: {info['记录总数']}")
        print(f"几何类型: {info['几何类型']}")
        print(f"坐标系: {info['坐标系']}")
        print(f"字段数量: {info['字段数量']}")
        print(f"字段列表: {info['字段列表']}")
        print(f"边界范围: {info['边界范围']}")

        return info

    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None


def convert_geojson_2_shp():

    """
    从天地图下载的省市县地图转换成shp文件 https://cloudcenter.tianditu.gov.cn/administrativeDivision
    需要下载对应的三分地图
    :return:
    """

    for i in ['省', '市', '县']:

        gdf = gpd.read_file(rf"C:\Users\Kevin\Desktop\中国_{i}.geojson")

        print(gdf.geom_type.unique())

        # 获取所有唯一的几何类型
        unique_geom_types = gdf.geom_type.unique()

        # 按几何类型分别保存
        for geom_type in unique_geom_types:
            subset = gdf[gdf.geom_type == geom_type]
            output_path = rf"C:\Users\Kevin\Desktop\中国_{i}_{geom_type}.shp"
            subset.to_file(output_path)

def check_dam_correct(check_dam_file):

    """
    原始的shp文件中有很多objectid为0的淤地坝，对其进行了赋值，后面用不到了
    :return:
    """

    dams = gpd.read_file(check_dam_file)
    # 检查是否有重复的OBJECTID
    duplicate_mask = dams.duplicated(subset=['OBJECTID'], keep=False)
    if duplicate_mask.any():
        print(f"发现 {duplicate_mask.sum()} 条重复的OBJECTID记录")

        # 获取当前最大的OBJECTID
        max_id = dams['OBJECTID'].max()

        # 生成唯一的新ID（从max_id+1开始递增）
        new_ids = range(max_id + 1, max_id + 1 + duplicate_mask.sum())

        # 替换重复的OBJECTID
        dams.loc[duplicate_mask, 'OBJECTID'] = new_ids

        # 验证是否还有重复
        assert not dams['OBJECTID'].duplicated().any(), "仍有重复OBJECTID！"
        print("已将所有OBJECTID调整为唯一值")
    else:
        print("未发现重复的OBJECTID")

    # 保存修改后的文件（可选）
    dams.to_file(check_dam_file)

def statistical_check_dams(province_file, city_file, county_file, check_dam_file, csv_file):

    """
    根据三级的shp地图，生成对应的每个区域的淤地坝的数量地图，用以进行分析，其中有600多个处在县的交界线上，随机分配给了其中一个
    :return:
    """

    # 读取省、市、县的Shapefile文件
    province = gpd.read_file(province_file)
    city = gpd.read_file(city_file)
    county = gpd.read_file(county_file)

    # 读取淤地坝的Shapefile文件
    dams = gpd.read_file(check_dam_file)
    # 转换坐标系
    dams = dams.to_crs(county.crs)

    # 统计每个县的淤地坝数量
    # 空间连接（Spatial Join），进行空间匹配
    county_with_dams = gpd.sjoin(county, dams, how='left', predicate='intersects')

    # 有的淤地坝处于分界线上，所以要进行去重处理
    county_with_dams = county_with_dams.drop_duplicates(subset=['OBJECTID'], keep='first')

    # 用gb和name两个是为了保险
    county_stats = county_with_dams.groupby(['gb', 'name'])["OBJECTID"].count().reset_index().rename(columns={"OBJECTID": "dam_count"})

    # 最前面三位不懂了，不用管，都相同
    # 准备市级数据（提取GB代码前4位作为市级代码）
    city['gb_7'] = city['gb'].astype(str).str[:7]  # 市级代码是GB前4位
    county_stats['gb_7'] = county_stats['gb'].astype(str).str[:7]  # 县级GB前4位匹配市级

    # 合并市级信息
    result = county_stats.merge(
        city[['gb_7', 'name']],
        left_on='gb_7',
        right_on='gb_7',
        how='left',
        suffixes=('_county', '_city')
    )

    # 准备省级数据（提取GB代码前2位作为省级代码）
    province['gb_5'] = province['gb'].astype(str).str[:5]  # 省级代码是GB前2位
    result['gb_5'] = result['gb'].astype(str).str[:5]  # 县级GB前2位匹配省级

    # 合并省级信息
    result = result.merge(
        province[['gb_5', 'name']],
        left_on='gb_5',
        right_on='gb_5',
        how='left',
        suffixes=('', '_province')
    )

    # 整理最终结果列
    final_result = result[[
        'name',  # 省级名称
        'name_city',  # 市级名称
        'name_county',  # 县级名称
        'dam_count'  # 淤地坝数量
    ]].copy()  # 添加.copy()创建独立副本

    # 重命名列
    final_result.columns = ['province', 'city', 'county', 'dam_count']

    # 删除缺失值
    final_result['dam_count'] = final_result['dam_count'].astype(int)  # 确保数据类型为整数
    final_result = final_result[final_result['dam_count'] > 0]  # 删除dam_count为NaN的行

    # 保存结果
    final_result.to_csv(csv_file, index=False, encoding='utf-8-sig')

def get_dams_by_admin_level_with_hierarchy(admin_file, check_dam_file, csv_file, shp_file,
                                        province_file=None, city_file=None, admin_level="county", sample_ratio=0.2):
    """
    统计每个行政区划包含的所有淤地坝OBJECTID，按指定行政级别获取0.2比例的数据作为统计量，
    输出CSV和shp结果，且向下取整，但最小值为1（前提是总数最小为1）
    同时包含上级行政区划信息

    参数:
        admin_file (str): 行政区划文件路径（省/市/县）
        check_dam_file (str): 淤地坝数据文件路径
        csv_file (str): 输出CSV文件路径
        shp_file (str): 输出shp文件路径
        province_file (str): 省级行政区划文件路径（用于关联）
        city_file (str): 市级行政区划文件路径（用于关联）
        admin_level (str): 行政级别 ("province"/"city"/"county")
    """
    # 1. 加载数据（确保坐标系一致）
    admin = gpd.read_file(admin_file)
    dams = gpd.read_file(check_dam_file).to_crs(admin.crs)

    # 如果是按县分区，加载省市数据用于关联
    if admin_level.lower() == "county" and province_file and city_file:
        province = gpd.read_file(province_file).to_crs(admin.crs)
        city = gpd.read_file(city_file).to_crs(admin.crs)

    # 2. 空间连接（保留所有匹配项）
    admin_with_dams = gpd.sjoin(admin, dams, how='left', predicate='intersects')

    # 去重
    admin_with_dams = admin_with_dams.drop_duplicates(subset=['OBJECTID'], keep='first')

    # 3. 按行政区划分组统计OBJECTID列表
    result = (
        admin_with_dams.groupby(['gb', 'name'])
        .agg({
            'OBJECTID': lambda x: list(x.dropna().astype(int)),
            'geometry': 'first'  # 保留行政区划几何
        })
        .rename(columns={'OBJECTID': 'dam_ids'})
        .reset_index()
    )

    # 4. 添加统计数量
    result['dam_count'] = result['dam_ids'].apply(len)
    result = result[result['dam_count'] > 0]  # 删除dam_count=0的记录

    # 5. 如果是按县分区，添加省市信息
    if admin_level.lower() == "county" and province_file and city_file:
        # 提取GB代码用于关联
        result['gb'] = result['gb'].astype(str)
        result['gb_7'] = result['gb'].str[:7]  # 市级代码
        result['gb_5'] = result['gb'].str[:5]  # 省级代码

        # 添加市级名称
        city_gb_name_map = dict(zip(city['gb'].astype(str).str[:7], city['name']))
        result['city_name'] = result['gb_7'].map(city_gb_name_map)

        # 添加省级名称
        province_gb_name_map = dict(zip(province['gb'].astype(str).str[:5], province['name']))
        result['province_name'] = result['gb_5'].map(province_gb_name_map)

    # 6. 进行采样
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    sampling_results = []
    all_sampled_ids = []  # 收集所有采样ID
    all_remaining_ids = []  # 收集所有剩余ID
    total_sampled = 0

    # 7. 构建数据
    for _, row in result.iterrows():
        admin_name = row['name']
        dam_ids = row['dam_ids']
        dam_count = row['dam_count']

        # 计算采样数量（向下取整，但最小值为1）
        sample_size = max(1, int(dam_count * sample_ratio))  # 使用int()实现向下取整

        # 随机采样不放回
        sampled_ids = random.sample(dam_ids, sample_size)
        remaining_ids = [id_ for id_ in dam_ids if id_ not in sampled_ids]

        all_sampled_ids.extend(sampled_ids)
        all_remaining_ids.extend(remaining_ids)

        # 构建结果数据，包含省市信息
        result_item = {
            "admin_name": admin_name,
            "total_dams": dam_count,
            "sampled_count": sample_size,
            "sampled_ids": sorted(sampled_ids),
            "remaining_ids": sorted(remaining_ids)
        }

        # 如果是县级别且有省市信息，添加省市名称
        if admin_level.lower() == "county" and 'city_name' in result.columns and 'province_name' in result.columns:
            result_item["city_name"] = row.get('city_name', '')
            result_item["province_name"] = row.get('province_name', '')

        sampling_results.append(result_item)
        total_sampled += sample_size

    # 8. 存储数据和元数据到JSON
    json_output = {
        "metadata": {
            "total_sampled": total_sampled,
            "sampling_ratio": sample_ratio,
            "random_seed": seed,
            "timestamp": pd.Timestamp.now().isoformat(),
            "all_sampled_ids": sorted(all_sampled_ids),
            "all_remaining_ids": sorted(all_remaining_ids),
            "total_remaining": len(all_remaining_ids),
            "total_original": len(all_sampled_ids) + len(all_remaining_ids),
            "admin_level": admin_level
        },
        "data": sampling_results
    }

    # 保存JSON结果
    json_file_path = csv_file.replace('.csv', '.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    # 9. 准备CSV输出数据
    csv_data = []
    for item in sampling_results:
        # 调整列顺序，将省市信息放在前面
        csv_item = {
            f"{admin_level}_name": item["admin_name"],
        }

        # 如果是县级别且有省市信息，添加省市名称到CSV（放在前面）
        if admin_level.lower() == "county" and "city_name" in item and "province_name" in item:
            csv_item["province_name"] = item["province_name"]
            csv_item["city_name"] = item["city_name"]

        # 添加其他数据
        csv_item["total_dams"] = item["total_dams"]
        csv_item["sampled_count"] = item["sampled_count"]

        csv_data.append(csv_item)

    # 保存CSV结果
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_file, index=False, encoding='utf-8-sig')

    # 10. 准备SHP输出数据
    # 从原始dams数据中筛选出被采样的淤地坝
    sampled_dams = dams[dams['OBJECTID'].isin(all_sampled_ids)].copy()

    # 添加采样信息
    sampled_dams['sampled'] = True

    # 保存SHP结果
    sampled_dams.to_file(shp_file, encoding='utf-8')

    print(f"按{admin_level}采样完成:")
    print(f"- 总采样数量: {total_sampled}")
    print(f"- 采样比例: {sample_ratio}")
    print(f"- CSV结果保存至: {csv_file}")
    print(f"- SHP结果保存至: {shp_file}")
    print(f"- JSON详细信息保存至: {json_file_path}")

    return sampling_results


if __name__ == '__main__':

    province_file = r"C:\Users\Kevin\Documents\ResearchData\AdministrativeDivision\ProvincialBoundary.shp"
    city_file = r"C:\Users\Kevin\Documents\ResearchData\AdministrativeDivision\CityBoundary.shp"
    county_file = r"C:\Users\Kevin\Documents\ResearchData\AdministrativeDivision\CountryBoundary.shp"
    check_dam_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\check_dam_dataset.shp"
    csv_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\sampled_dams.csv"
    selected_shp_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dam_dataset.shp"
    selected_csv_file = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dams.csv"

    # 获取shp文件的直观信息
    get_shp_file_info(check_dam_file)

    # 统计shp文件的地理分布
    # statistical_check_dams(province_file=province_file, city_file=city_file, county_file=county_file, check_dam_file=check_dam_file, csv_file=csv_file)

    # 数据集裁剪：8：2 获得了10006张
    # get_dams_by_admin_level_with_hierarchy(admin_file=county_file, check_dam_file=check_dam_file, csv_file=selected_csv_file, shp_file=selected_shp_file, province_file=province_file, city_file=city_file, admin_level='County')

    # 数据集采样：太多了，因此在上面的基础上再减少一半，最终获得4991张
    selected_shp_file_5000 = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dam_dataset_5000.shp"
    selected_csv_file_5000 = r"C:\Users\Kevin\Documents\ResearchData\SedimentRetentionOfCheckDam\selected_check_dams_5000.csv"
    # get_dams_by_admin_level_with_hierarchy(admin_file=county_file, check_dam_file=selected_shp_file,
    #                                        csv_file=selected_csv_file_5000, shp_file=selected_shp_file_5000,
    #                                        province_file=province_file, city_file=city_file, admin_level='County', sample_ratio=0.5)
