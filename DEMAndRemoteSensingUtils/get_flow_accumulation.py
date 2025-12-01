#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: get_flow_direction
# @Time    : 2025/8/27 10:11
# @Author  : Kevin
# @Describe: 获取Tif文件的流向累计
import os
import shutil
import whitebox

# --------------------------
# 参数设置（根据实际数据修改）
# --------------------------
def calculate_flow_accumulation(dem_path, flow_accum_path, stream_path=None, threshold=1000):
    """
    计算DEM的汇流累积量，仅保存最终结果，自动清理中间文件

    参数:
        dem_path: 输入DEM的文件路径（.tif格式）
        flow_accum_path: 汇流累积量输出文件路径（.tif格式）
    """
    # 初始化Whitebox工具
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False  # 关闭冗余日志

    user_home = os.path.expanduser(r"~\Desktop")
    temp_dir = os.path.join(user_home, "checkdam_temp")
    os.makedirs(temp_dir, exist_ok=True)

    filled_dem_path = os.path.join(temp_dir, "filled_dem.tif")
    flow_dir_path = os.path.join(temp_dir, "flow_direction.tif")
    intermediate_fill_path = os.path.join(temp_dir, "intermediate_filled.tif")

    # 1. DEM预处理：填洼
    print("Start DEM filling the pothole...")
    fill_result1 = wbt.fill_depressions(
        dem=dem_path,
        output=intermediate_fill_path,
        fix_flats=True,
        flat_increment=0.01,
        max_depth=None  # 不限制填充深度，处理所有洼地
    )
    if fill_result1 != 0 or not os.path.exists(intermediate_fill_path):
        error_msg = f"The first stage of depression filling failed with code: {fill_result1}"
        if fill_result1 == 1:
            error_msg += " (Missing required input file)"
        elif fill_result1 == 2:
            error_msg += " (Invalid input value/combination)"
        elif fill_result1 == 3:
            error_msg += " (Error in input file)"
        elif fill_result1 == 4:
            error_msg += " (I/O error)"
        elif fill_result1 == 5:
            error_msg += " (Unsupported data type)"
        elif fill_result1 == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    print("Begin the second stage of deep depression filling...")
    # 第二阶段：使用更激进的参数处理残留洼地
    fill_result2 = wbt.fill_depressions(
        dem=intermediate_fill_path,
        output=filled_dem_path,
        fix_flats=True,
        flat_increment=0.005,  # 更小的平坦区域增量，确保水流连续性
        max_depth=None
    )
    if fill_result2 != 0 or not os.path.exists(filled_dem_path):
        error_msg = f"The second stage of depression filling failed with code: {fill_result2}"
        if fill_result2 == 1:
            error_msg += " (Missing required input file)"
        elif fill_result2 == 2:
            error_msg += " (Invalid input value/combination)"
        elif fill_result2 == 3:
            error_msg += " (Error in input file)"
        elif fill_result2 == 4:
            error_msg += " (I/O error)"
        elif fill_result2 == 5:
            error_msg += " (Unsupported data type)"
        elif fill_result2 == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    # 2. DEM预处理：削峰
    print("Start DEM peak shaving...")
    breach_result1 = wbt.breach_depressions(
        dem=filled_dem_path,
        output=filled_dem_path,
        max_depth=10.0,  # 限制浅洼地的最大处理深度（米）
        max_length=50,  # 限制 breach 通道长度（网格单元数）
        flat_increment=0.0001,  # 保持平坦区域连续性
        fill_pits=True  # 填充单像素坑洼
    )
    if breach_result1 != 0 or not os.path.exists(filled_dem_path):
        error_msg = f"The first DEM peak shaving process failed with code: {breach_result1}"
        if breach_result1 == 1:
            error_msg += " (Missing required input file)"
        elif breach_result1 == 2:
            error_msg += " (Invalid input value/combination)"
        elif breach_result1 == 3:
            error_msg += " (Error in input file)"
        elif breach_result1 == 4:
            error_msg += " (I/O error)"
        elif breach_result1 == 5:
            error_msg += " (Unsupported data type)"
        elif breach_result1 == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    breach_result2 = wbt.breach_depressions(
        dem=filled_dem_path,
        output=filled_dem_path,
        max_depth=None,  # 不限制深度，处理深洼地
        max_length=None,  # 不限制长度，处理狭长洼地
        flat_increment=0.0001,
        fill_pits=True
    )
    if breach_result2 != 0 or not os.path.exists(filled_dem_path):
        error_msg = f"The second DEM peak shaving process failed with code: {breach_result2}"
        if breach_result2 == 1:
            error_msg += " (Missing required input file)"
        elif breach_result2 == 2:
            error_msg += " (Invalid input value/combination)"
        elif breach_result2 == 3:
            error_msg += " (Error in input file)"
        elif breach_result2 == 4:
            error_msg += " (I/O error)"
        elif breach_result2 == 5:
            error_msg += " (Unsupported data type)"
        elif breach_result2 == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    # 3. 计算流向
    print("Start calculating the flow direction...")
    flow_dir_result = wbt.d8_pointer(
        dem=filled_dem_path,
        output=flow_dir_path,
        esri_pntr=False
    )
    if flow_dir_result != 0 or not os.path.exists(flow_dir_path):
        error_msg = f"Flow direction calculation failed with code: {flow_dir_result}"
        if flow_dir_result == 1:
            error_msg += " (Missing required input file)"
        elif flow_dir_result == 2:
            error_msg += " (Invalid input value/combination)"
        elif flow_dir_result == 3:
            error_msg += " (Error in input file)"
        elif flow_dir_result == 4:
            error_msg += " (I/O error)"
        elif flow_dir_result == 5:
            error_msg += " (Unsupported data type)"
        elif flow_dir_result == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    # 4. 计算汇流累积量（最终需要保存的结果）
    print("Start calculating the accumulated amount of confluence...")
    # 确保输出目录存在
    output_dir = os.path.dirname(flow_accum_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    accum_result = wbt.d8_flow_accumulation(
        i=flow_dir_path,
        output=flow_accum_path,
        pntr=True,
        esri_pntr=False
    )
    if accum_result != 0 or not os.path.exists(flow_accum_path):
        error_msg = f"The calculation of the accumulated amount of the confluence failed with code: {accum_result}"
        if accum_result == 1:
            error_msg += " (Missing required input file)"
        elif accum_result == 2:
            error_msg += " (Invalid input value/combination)"
        elif accum_result == 3:
            error_msg += " (Error in input file)"
        elif accum_result == 4:
            error_msg += " (I/O error)"
        elif accum_result == 5:
            error_msg += " (Unsupported data type)"
        elif accum_result == 32:
            error_msg += " (Timeout error)"
        raise RuntimeError(error_msg)

    print(f"The accumulated amount of the confluence is calculated, and the result is saved to: {flow_accum_path}")

    shutil.rmtree(temp_dir)

# 示例用法
if __name__ == "__main__":
    # 输入DEM路径
    input_dem = r"C:\Users\Kevin\Desktop\results\375858_1103666\375858_1103666_DEM_10.tif"
    # 输出汇流累积量路径
    output_accum = r"C:\Users\Kevin\Desktop\test.tif"
    # 输出流向路径
    output_stream = r"C:\Users\Kevin\Desktop\test_stream.tif"

    # 调用函数
    calculate_flow_accumulation(input_dem, output_accum, output_stream)