#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: dem_analysis_full.py
# @Time    : 2025/8/22 15:00
# @Author  : Data Analyst
# @Describe: 双 DEM 对比（通过迭代方式移除有效区域边界像素）

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import os
import sys
from scipy.ndimage import generic_filter
from scipy import stats

def analyze_two_dems(tif1_path, tif2_path, output_dir=None, band=1, window_size=3, edge_iterations=2):
    """
    核心改进：
    1. 通过迭代方式移除有效区域边界像素（每次迭代移除四周边界点）
    2. 边界点定义：四个方向有至少一个nodata或位于图像边缘的像素
    3. 共执行edge_iterations次迭代，逐步剥离边缘
    """
    plt.rcParams["font.family"] = ["Times New Roman", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    try:
        with rasterio.open(tif1_path) as src1, rasterio.open(tif2_path) as src2:
            # 检查尺寸
            if src1.shape != src2.shape:
                raise ValueError("两个 TIF 尺寸不一致！")
            if not np.allclose(src1.transform, src2.transform):
                print("警告：投影/坐标可能不一致，结果需验证")

            # 读取数据并转换为 float32（统一类型避免误差）
            data1 = src1.read(band).astype(np.float32)
            data2 = src2.read(band).astype(np.float32)
            nodata1 = src1.nodata
            nodata2 = src2.nodata
            rows, cols = data1.shape  # 获取图像尺寸

            # 生成 Nodata 掩码（双数据均有效才保留）
            def get_nodata_mask(data, nodata):
                if nodata is not None:
                    if np.issubdtype(data.dtype, np.floating) and np.isnan(nodata):
                        return np.isnan(data)
                    else:
                        return (data == nodata)
                else:
                    return np.zeros_like(data, dtype=bool)

            mask1 = get_nodata_mask(data1, nodata1)
            mask2 = get_nodata_mask(data2, nodata2)
            valid_mask = ~mask1 & ~mask2  # 初始有效区域掩码
            initial_valid_count = np.sum(valid_mask)
            print(f"初始有效像素数量: {initial_valid_count}")

            # 如果没有有效像素，直接返回
            if initial_valid_count == 0:
                print("无重叠有效数据！")
                return

            # 迭代移除边界像素
            current_mask = valid_mask.copy()
            for i in range(edge_iterations):
                # 找到当前有效区域的边界像素
                # 边界像素定义：四个方向有至少一个无效值，或位于图像边缘
                boundary_mask = np.zeros_like(current_mask, dtype=bool)

                # 遍历每个像素检查是否为边界点
                for r in range(rows):
                    for c in range(cols):
                        if current_mask[r, c]:  # 只检查当前有效的像素
                            # 图像边缘像素视为边界点
                            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                                boundary_mask[r, c] = True
                            else:
                                # 检查四个方向是否有无效值
                                if (not current_mask[r - 1, c] or  # 上
                                        not current_mask[r + 1, c] or  # 下
                                        not current_mask[r, c - 1] or  # 左
                                        not current_mask[r, c + 1]):  # 右
                                    boundary_mask[r, c] = True

                # 移除当前迭代找到的边界像素
                current_mask[boundary_mask] = False
                current_valid_count = np.sum(current_mask)

                # 如果已经没有有效像素，提前退出
                if current_valid_count == 0:
                    print(f"迭代 {i + 1} 次后已无有效像素，提前终止")
                    break

                print(f"迭代 {i + 1} 次后有效像素数量: {current_valid_count}")

            # 检查迭代后是否还有有效像素
            if np.sum(current_mask) == 0:
                print("移除边界像素后无有效数据！")
                return

            # 找到最终有效区域的边界框（用于裁剪显示）
            non_zero_rows = np.where(np.any(current_mask, axis=1))[0]
            non_zero_cols = np.where(np.any(current_mask, axis=0))[0]
            min_row, max_row = non_zero_rows[0], non_zero_rows[-1]
            min_col, max_col = non_zero_cols[0], non_zero_cols[-1]

            # 裁剪数据到有效区域
            data1_crop = data1[min_row:max_row + 1, min_col:max_col + 1]
            data2_crop = data2[min_row:max_row + 1, min_col:max_col + 1]
            current_mask_crop = current_mask[min_row:max_row + 1, min_col:max_col + 1]

            # 替换无效值为nan
            data1_crop = np.where(current_mask_crop, data1_crop, np.nan)
            data2_crop = np.where(current_mask_crop, data2_crop, np.nan)

            # 提取一维有效数据用于统计
            valid_data1 = data1[current_mask]
            valid_data2 = data2[current_mask]

            # ---------------------- 统计与检验 ----------------------
            stats1 = {
                '均值': np.nanmean(valid_data1),
                '中位数': np.nanmedian(valid_data1),
                '最小值': np.nanmin(valid_data1),
                '最大值': np.nanmax(valid_data1),
                '标准差': np.nanstd(valid_data1),
                '变异系数': np.nanstd(valid_data1) / np.nanmean(valid_data1) if np.nanmean(valid_data1) != 0 else 0
            }
            stats2 = {
                '均值': np.nanmean(valid_data2),
                '中位数': np.nanmedian(valid_data2),
                '最小值': np.nanmin(valid_data2),
                '最大值': np.nanmax(valid_data2),
                '标准差': np.nanstd(valid_data2),
                '变异系数': np.nanstd(valid_data2) / np.nanmean(valid_data2) if np.nanmean(valid_data2) != 0 else 0
            }

            diff = valid_data1 - valid_data2
            diff_stats = {
                '均值差': np.nanmean(diff),
                '中位数差': np.nanmedian(diff),
                '最大差值': np.nanmax(diff),
                '最小差值': np.nanmin(diff),
                '标准差': np.nanstd(diff)
            }

            # 正态性检验
            norm_test1 = stats.normaltest(valid_data1)
            norm_test2 = stats.normaltest(valid_data2)
            is_normal1 = norm_test1.pvalue > 0.05
            is_normal2 = norm_test2.pvalue > 0.05

            # 均值差异检验
            if is_normal1 and is_normal2:
                levene_test = stats.levene(valid_data1, valid_data2)
                equal_var = levene_test.pvalue > 0.05
                mean_test = stats.ttest_ind(valid_data1, valid_data2, equal_var=equal_var)
                mean_test_method = "独立样本 t 检验" if equal_var else "Welch's t 检验"
            else:
                mean_test = stats.mannwhitneyu(valid_data1, valid_data2)
                mean_test_method = "曼-惠特尼 U 检验"

            # 相关性检验
            if is_normal1 and is_normal2:
                corr, corr_p = stats.pearsonr(valid_data1, valid_data2)
                corr_method = "皮尔逊相关系数"
            else:
                corr, corr_p = stats.spearmanr(valid_data1, valid_data2)
                corr_method = "斯皮尔曼相关系数"

            # ---------------------- 局部差异分析 ----------------------
            def window_std(values):
                window_valid = values[~np.isnan(values)]
                return np.std(window_valid) if len(window_valid) >= 2 else np.nan

            # 基于最终有效区域计算局部标准差
            data1_pad = np.where(current_mask, data1, np.nan)
            data2_pad = np.where(current_mask, data2, np.nan)

            local_std1 = generic_filter(data1_pad, window_std, size=window_size, mode='constant', cval=np.nan)
            local_std2 = generic_filter(data2_pad, window_std, size=window_size, mode='constant', cval=np.nan)
            local_std_diff = local_std1 - local_std2
            local_std_diff_crop = local_std_diff[min_row:max_row + 1, min_col:max_col + 1]

            # ---------------------- 可视化 ----------------------
            dem_cmap = LinearSegmentedColormap.from_list("dem_cmap", ["#0000FF", "#00FF00", "#FFFF00", "#FF0000"])
            diff_cmap = plt.cm.RdBu

            fig = plt.figure(figsize=(16, 20))
            fig.suptitle(
                f"DEM 对比（迭代{edge_iterations}次移除边界像素）\n{os.path.basename(tif1_path)} vs {os.path.basename(tif2_path)}",
                fontsize=18, y=0.98)

            gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.8], hspace=0.25, wspace=0.2)

            # 1. DEM可视化
            ax1 = fig.add_subplot(gs[0, 0])
            im1 = ax1.imshow(data1_crop, cmap=dem_cmap)
            ax1.set_title(f"DEM1（已迭代移除边界像素）")
            ax1.set_aspect('equal')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            ax1.tick_params(axis='both', which='both', labelsize=8)

            ax2 = fig.add_subplot(gs[0, 1])
            im2 = ax2.imshow(data2_crop, cmap=dem_cmap)
            ax2.set_title(f"DEM2（已迭代移除边界像素）")
            ax2.set_aspect('equal')
            plt.colorbar(im2, ax=ax2, shrink=0.8)
            ax2.tick_params(axis='both', which='both', labelsize=8)

            # 2. 差值图和散点图
            ax3 = fig.add_subplot(gs[1, 0])
            diff_data_crop = data1_crop - data2_crop
            vmax = max(abs(np.nanmax(diff_data_crop)), abs(np.nanmin(diff_data_crop)))
            im_diff = ax3.imshow(diff_data_crop, cmap=diff_cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
            ax3.set_title("DEM1 - DEM2 差值")
            ax3.set_aspect('equal')
            plt.colorbar(im_diff, ax=ax3, shrink=0.8)
            ax3.tick_params(axis='both', which='both', labelsize=8)

            ax7 = fig.add_subplot(gs[3, 0])
            ax7.scatter(valid_data1, valid_data2, alpha=0.5, s=1)
            min_val = min(np.nanmin(valid_data1), np.nanmin(valid_data2))
            max_val = max(np.nanmax(valid_data1), np.nanmax(valid_data2))
            ax7.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
            ax7.set_title(f"{corr_method}: {corr:.3f} (P={corr_p:.3f})")
            ax7.set_xlabel("DEM1 数值")
            ax7.set_ylabel("DEM2 数值")
            ax7.grid(alpha=0.3)
            ax7.tick_params(axis='both', which='both', labelsize=8)

            # 3. 重叠直方图和累积分布曲线
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.hist(valid_data1, bins=50, alpha=0.5, label="DEM1", color='blue')
            ax5.hist(valid_data2, bins=50, alpha=0.5, label="DEM2", color='orange')
            ax5.axvline(stats1['均值'], color='blue', linestyle='--', label=f"DEM1 均值: {stats1['均值']:.2f}")
            ax5.axvline(stats2['均值'], color='orange', linestyle='--', label=f"DEM2 均值: {stats2['均值']:.2f}")
            ax5.set_title("数据分布直方图")
            ax5.set_xlabel("高程值")
            ax5.set_ylabel("频数")
            ax5.legend(fontsize=8)
            ax5.grid(alpha=0.3)
            ax5.tick_params(axis='both', which='both', labelsize=8)

            ax6 = fig.add_subplot(gs[2, 1])
            counts1, bins1 = np.histogram(valid_data1, bins=50, density=True)
            cdf1 = np.cumsum(counts1) / np.sum(counts1)
            counts2, bins2 = np.histogram(valid_data2, bins=50, density=True)
            cdf2 = np.cumsum(counts2) / np.sum(counts2)
            ax6.plot(bins1[1:], cdf1, 'b-', label="DEM1")
            ax6.plot(bins2[1:], cdf2, 'orange', label="DEM2")
            ax6.set_title("累积分布曲线")
            ax6.set_xlabel("高程值")
            ax6.set_ylabel("累积概率")
            ax6.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax6.legend(fontsize=8)
            ax6.grid(alpha=0.3)
            ax6.tick_params(axis='both', which='both', labelsize=8)

            # 4. 局部标准差差异和统计结果汇总
            ax4 = fig.add_subplot(gs[1, 1])
            im_std_diff = ax4.imshow(local_std_diff_crop, cmap=diff_cmap,
                                     norm=TwoSlopeNorm(vcenter=0,
                                                       vmin=-np.nanmax(abs(local_std_diff_crop)),
                                                       vmax=np.nanmax(abs(local_std_diff_crop))))
            ax4.set_title(f"局部标准差差异（{window_size}x{window_size} 窗口）")
            ax4.set_aspect('equal')
            plt.colorbar(im_std_diff, ax=ax4, shrink=0.8)
            ax4.tick_params(axis='both', which='both', labelsize=8)

            ax8 = fig.add_subplot(gs[3, 1])
            stats_text = [
                "=== 统计检验===",
                f"正态性检验（normaltest）: DEM1 P={norm_test1.pvalue:.3e}, DEM2 P={norm_test2.pvalue:.3e}",
                f"均值差异检验（{mean_test_method}）: P={mean_test.pvalue:.3e} (P<0.05 为显著差异)",
                f"{corr_method}: {corr:.3f} (P={corr_p:.3f})",
                "\n=== 差异统计 ===",
                f"均值差: {diff_stats['均值差']:.2f}",
                f"中位数差: {diff_stats['中位数差']:.2f}",
                f"最大差值: {diff_stats['最大差值']:.2f}",
                f"最小差值: {diff_stats['最小差值']:.2f}",
                "\n=== 数据处理说明 ===",
                f"通过 {edge_iterations} 次迭代移除边界像素",
                f"初始有效像素: {initial_valid_count}",
                f"最终有效像素: {len(valid_data1)}",
                "\n=== 边界定义 ===",
                "边界像素指：图像边缘像素或四个方向中至少有一个无效值的像素"
            ]

            # 合并文本并设置自动换行
            text_content = "\n".join(stats_text)
            text_obj = ax8.text(0.05, 0.95, text_content,
                                fontsize=10,
                                bbox=dict(facecolor='white', alpha=0.8, pad=8),
                                transform=ax8.transAxes,
                                verticalalignment='top',
                                wrap=True)

            # 调整文本宽度以实现自动换行
            def set_text_width():
                return 0.9  # 文本宽度为轴宽度的90%

            text_obj._get_wrap_line_width = set_text_width

            ax8.axis('off')

            # 保存结果
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(os.path.join(output_dir, "dem_comparison_crop.png"), dpi=300, bbox_inches='tight')
                print(f"结果已保存至: {output_dir}")

            plt.show()

    except Exception as e:
        print(f"分析失败: {e}")
        return


if __name__ == "__main__":
    # 固定文件路径
    tif1 = r"C:\Users\Kevin\Desktop\result\test_30_copernicus.tif"
    tif2 = r"C:\Users\Kevin\Desktop\result\test_30.tif"

    analyze_two_dems(
        tif1_path=tif1,
        tif2_path=tif2,
        output_dir=r"C:\Users\Kevin\Desktop\result\comparison",
        band=1,
        window_size=3,
        edge_iterations=5  # 迭代2次移除边界像素
    )
