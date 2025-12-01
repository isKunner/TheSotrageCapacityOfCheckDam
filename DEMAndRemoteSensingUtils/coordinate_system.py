#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: utils
# @Time    : 2025/8/10 13:47
# @Author  : Kevin
# @Describe: 地理坐标系的处理工具

from osgeo import osr
import numpy as np


def create_coordinate_transformer(src_srs, target_srs=None):
    """
    创建可靠的坐标转换对象，支持源坐标系为地理/投影类型，确保转换结果准确

    参数:
        src_srs: 源坐标系 (osr.SpatialReference对象，必须有效)
        target_srs: 目标坐标系，默认WGS84(EPSG:4326)，支持EPSG代码/ WKT / osr对象

    返回:
        target_srs_obj: 目标坐标系空间参考对象
        transform_func: 坐标转换函数，输入(x,y[,z])返回转换后坐标
    """
    # --------------------------
    # 1. 源坐标系有效性校验
    # --------------------------
    if not isinstance(src_srs, osr.SpatialReference):
        raise TypeError("src_srs必须是osr.SpatialReference对象")
    try:
        # 尝试获取坐标系权威信息来验证有效性
        if not src_srs.GetAttrValue('AUTHORITY', 0):
            raise ValueError("源坐标系缺少权威信息，可能无效")
    except:
        raise ValueError("源坐标系无效，请检查src_srs")

    # 获取源坐标系关键信息
    src_is_geo = src_srs.IsGeographic()
    src_is_proj = src_srs.IsProjected()
    src_epsg = src_srs.GetAttrValue('AUTHORITY', 1)
    src_datum = src_srs.GetAttrValue("DATUM") or "未知"

    # --------------------------
    # 2. 目标坐标系处理与校验
    # --------------------------
    if target_srs is None:
        # 默认目标：WGS84地理坐标系
        target_srs_obj = osr.SpatialReference()
        target_srs_obj.ImportFromEPSG(4326)
    elif isinstance(target_srs, int):
        target_srs_obj = osr.SpatialReference()
        if target_srs_obj.ImportFromEPSG(target_srs) != 0:
            raise ValueError(f"EPSG代码无效: {target_srs}")
    elif isinstance(target_srs, str):
        target_srs_obj = osr.SpatialReference()
        if target_srs_obj.ImportFromWkt(target_srs) != 0:
            raise ValueError(f"WKT字符串无效: {target_srs[:50]}...")
    elif isinstance(target_srs, osr.SpatialReference):
        target_srs_obj = target_srs
        if not target_srs_obj.IsValid():
            raise ValueError("目标坐标系对象无效")
    else:
        raise TypeError(f"不支持的目标坐标系类型: {type(target_srs)}")

    # 获取目标坐标系关键信息
    target_is_geo = target_srs_obj.IsGeographic()
    target_is_proj = target_srs_obj.IsProjected()
    target_epsg = target_srs_obj.GetAttrValue('AUTHORITY', 1)
    target_datum = target_srs_obj.GetAttrValue("DATUM") or "未知"

    # 打印调试信息（关键）
    print(f"\n[坐标系信息]")
    print(f"源坐标系 - 类型: {'地理' if src_is_geo else '投影'}, EPSG: {src_epsg}, 基准面: {src_datum}")
    print(f"目标坐标系 - 类型: {'地理' if target_is_geo else '投影'}, EPSG: {target_epsg}, 基准面: {target_datum}")

    # --------------------------
    # 3. 坐标系组合合理性校验
    # --------------------------
    # 地理→地理：基准面不一致警告
    if src_is_geo and target_is_geo and src_datum != target_datum:
        print(f"⚠️ 警告：地理坐标系基准面不同（{src_datum} → {target_datum}），转换可能有误差")

    # 投影→投影：建议通过地理坐标系中转（如果基准面不同）
    if src_is_proj and target_is_proj:
        src_geo = src_srs.CloneGeogCS()
        target_geo = target_srs_obj.CloneGeogCS()
        if src_geo.GetAttrValue("DATUM") != target_geo.GetAttrValue("DATUM"):
            print(f"⚠️ 警告：投影基准面不同，将自动通过WGS84中转")
            # 强制使用中间转换
            return _create_composite_transform(src_srs, target_srs_obj)

    # --------------------------
    # 4. 创建并测试转换对象
    # --------------------------
    try:
        # 尝试直接转换
        direct_transform = osr.CoordinateTransformation(src_srs, target_srs_obj)

        # 生成合理的测试点（避免用(0,0)这种可能在无效区域的点）
        test_x, test_y = _get_valid_test_point(src_is_geo, src_epsg)

        # 测试转换
        test_result = direct_transform.TransformPoint(test_x, test_y)
        if not _is_valid_coordinate(test_result[0], test_result[1], target_is_geo):
            raise ValueError("直接转换结果超出合理范围")

        # 封装转换函数（统一接口）
        def transform_func(x, y, z=0):
            res = direct_transform.TransformPoint(x, y, z)
            return (res[0], res[1]) if len(res) >= 2 else (None, None)

        print("✅ 直接转换验证通过")
        return target_srs_obj, transform_func

    except Exception as e:
        print(f"❌ 直接转换失败: {str(e)}, 尝试中间转换...")
        # 尝试通过WGS84中转
        return _create_composite_transform(src_srs, target_srs_obj)


def _create_composite_transform(src_srs, target_srs_obj):
    """创建通过WGS84中转的复合转换"""
    try:
        wgs84 = osr.SpatialReference()
        wgs84.ImportFromEPSG(4326)

        # 源→WGS84转换
        transform1 = osr.CoordinateTransformation(src_srs, wgs84)
        # WGS84→目标转换
        transform2 = osr.CoordinateTransformation(wgs84, target_srs_obj)

        # 测试中转转换
        test_x, test_y = _get_valid_test_point(src_srs.IsGeographic(), src_srs.GetAttrValue('AUTHORITY', 1))
        step1 = transform1.TransformPoint(test_x, test_y)
        step2 = transform2.TransformPoint(step1[0], step1[1])
        if not _is_valid_coordinate(step2[0], step2[1], target_srs_obj.IsGeographic()):
            raise ValueError("中间转换结果超出合理范围")

        # 封装复合转换函数
        def composite_func(x, y, z=0):
            step1 = transform1.TransformPoint(x, y, z)
            step2 = transform2.TransformPoint(step1[0], step1[1], step1[2])
            return (step2[0], step2[1]) if len(step2) >= 2 else (None, None)

        print("✅ 中间转换验证通过")
        return target_srs_obj, composite_func

    except Exception as e2:
        raise ValueError(f"❌ 所有转换方案失败: {str(e2)}")


def _get_valid_test_point(is_geographic, epsg):
    """生成适合当前坐标系的测试点（避免无效区域）"""
    if is_geographic:
        # 地理坐标系：使用中纬度地区有效经纬度（避免极点、国际日期变更线附近）
        return 105.0, 35.0  # 中国中部附近经纬度
    else:
        # 投影坐标系：使用UTM等投影的典型有效范围（假设米制）
        if epsg and epsg.startswith('326'):  # UTM北半球
            return 500000, 4000000  # UTM典型坐标
        else:
            return 100000, 100000  # 通用投影坐标


def _is_valid_coordinate(x, y, is_target_geographic):
    """验证转换后的坐标是否在合理范围内"""
    if is_target_geographic:
        # 地理坐标：经度[-180,180]，纬度[-90,90]
        return (-180 <= x <= 180) and (-90 <= y <= 90)
    else:
        # 投影坐标：通常在[-1e7, 1e7]米范围内（根据常见投影调整）
        return (-1e7 <= x <= 1e7) and (-1e7 <= y <= 1e7)


def transform_coordinates(transform_func, x, y, z=0):
    """
    执行坐标转换，封装错误处理

    参数:
        transform_func: create_coordinate_transformer返回的转换函数
        x, y: 源坐标
        z: 高程（可选，默认0）

    返回:
        (tx, ty): 转换后的坐标
    """
    if not callable(transform_func):
        raise TypeError("transform_func必须是可调用的转换函数")

    try:
        tx, ty = transform_func(x, y, z)
        if tx is None or ty is None:
            raise ValueError("转换返回空值")
        return tx, ty
    except Exception as e:
        raise RuntimeError(f"坐标转换执行失败 (x={x}, y={y}): {str(e)}")
