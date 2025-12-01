# modify_dem.py
# 1. analysis.py
## 1.1 analyze_two_dems
对比两个DEM之间的差异，通过P检验等方式；这里主要是对比超分后的DEM和原始DEM之间的差异
# 2. clean_nodata.py
## 2.1 remove_nodata_rows_cols
当前使用场景：使用地理坐标系、非投影坐标系的DEM，且要去除的DEM是周围有多余的NoData的数据

| None | None | None | None | 
|------|------|------|------|
| 1    | 1    | 1    | None |
| 1    | 1    | 1    | None |

但是下面的场景会出错：

| None | None | None | None | 
|------|------|------|------|
| 1    | 1    | 1    | None |
| None | None | None | None |
| 1    | 1    | 1    | None |

## 2.2 相关知识
Affine：仿射变换
以王茂沟DEM数据（投影坐标系）为例：

Affine(a=2.0, b=0.0, c=440820.09892957023, d=0.0, e=-2, f=4165146.2840191615, g=0.0, h=0.0, i=1.0)

| 参数 | 	含义     | 	说明                       |
|----|---------|---------------------------|
| a  | 	x方向缩放	 | 经度/投影X 的列方向变化率            |
| b  | 	x-y旋转  | 	经度/投影X 的行方向变化率           |
| c  | 	x方向平移  | 	左上角x坐标                   |
| d  | 	y-x旋转  | 	纬度/投影Y 的列方向变化率           |
| e  | 	y方向缩放  | 	纬度/投影Y 的行方向变化率（负值表示图像向下） |
| f  | 	y方向平移  | 	左上角y坐标                   |
| g  | 	透视变换x  | 	固定为0                     |
| h  | 	透视变换y  | 	固定为0                     |
| i  | 	缩放因子   | 	固定为1                     |

坐标系的运算：\
x_geo = a × col + b × row + c \
y_geo = d × col + e × row + f

坐标系的矩阵运算公式：
$$
\begin{pmatrix}
a & b & c \\
d & e & f \\
g & h & i
\end{pmatrix}
\times
\begin{pmatrix}
x_{\text{pixel}} \\
y_{\text{pixel}} \\
1
\end{pmatrix}
=
\begin{pmatrix}
x_{\text{geo}} \\
y_{\text{geo}} \\
1
\end{pmatrix}
$$

# 3. coordinate_system
## 3.1 create_coordinate_transformer
创建一个可靠的坐标转换对象，用于获取一个在A坐标系下（A数据）的点的数据，在B坐标系（数据）下的点的位置\
核心思想：已有的位置->获取经纬度坐标->得到在另一个坐标系下的位置

| 源坐标系  | 	目标坐标系 | 	支持情况   | 	处理方式          |
|-------|--------|---------|----------------|
| 地理坐标系 | 	WGS84 | 	✅ 直接支持 | 	直接转换          |
| 投影坐标系 | 	WGS84 | 	✅ 直接支持 | 	直接转换          |
| 地理坐标系 | 	投影坐标系 | 	✅ 间接支持 | 	通过WGS84中转     |
| 投影坐标系 | 	地理坐标系 | 	✅ 间接支持 | 	通过WGS84中转     |
| 地理坐标系 | 	地理坐标系 | 	✅ 直接支持 | 	直接转换          |
| 投影坐标系 | 	投影坐标系 | 	✅ 条件支持 | 	同基准面直连，异基准面中转 |

# 4. crop_dem_from_cordinate.py
## 4.1 add_buffer_to_bounds
这个应用的场景：比如超分，超分后由于边缘的像素缺失效果不好，所以额外增加一个缓冲，超分后直接裁掉边缘

⚠️ 但其实这个裁剪的话，有问题，如果对多个块进行缓冲区的添加，他们的实际范围会不一样，因为是按照固定的赤道大小进行的经纬度计算，但可能实际位置会有变化，那么经纬度对应的实际距离有不同
- 如果数据本身是地理坐标系，如果裁剪的块的经纬度跨度都相同，那么扩展后还是一样的（在经纬度数值上），但实际的地理距离不同
- 如果数据本身是投影坐标系，由于是输入的经纬度，并且拿经纬度计算的，所以裁剪的区域不一样，仍需完善

## 4.2 crop_tif_by_bounds
| 根据给定的矩形框的范围进行数据的裁剪                                                                                |
|---------------------------------------------------------------------------------------------------|
| 输入参数:<br/>lon_min=110.347, lat_min=37.595<br/>lon_max=110.348, lat_max=37.596                     |
| 创建WGS84几何形状:<br/>Polygon([(110.347,37.595), (110.348,37.595), ...])                               |
| 读取TIF文件:<br/>坐标系: EPSG:4527 (投影坐标系)<br/>Affine: a=2.0, e=-2.0, c=440000, f=4166000                |
| 自动坐标转换:<br/>(110.347,37.595) → (440820.1, 4165146.3)<br/>(110.348,37.596) → (440822.1, 4165148.3) |
| 像素坐标计算:<br/>列: 410~411<br/>行: 425~426                                                             |
| 提取像素数据:<br/>out_image.shape = (1, 1, 1)  提取1×1个像素                                                 |
| 生成新变换矩阵:<br/>新左上角: (440820.0, 4165148.0)                                                          |
| 保存文件:<br/>输出裁剪后的小TIF文件                                                                            |

# 5. crop_dem_from_dem.py
## 5.1 extract_matching_files
根据已有的裁剪好的tif进行重新的裁剪采样(方形)
匹配到要裁剪的目标坐标系，作为区域划分的数据的坐标系不用管，会自动转换

# 6. get_flow_accumulation.py
## 6.1 calculate_flow_accumulation

参考的AI的代码，对具体的功能效果，并不是很确定
- 原始DEM
- 第一阶段填洼 (fill_depressions)
- 第二阶段填洼 (fill_depressions)  
- 第一阶段削峰 (breach_depressions)
- 第二阶段削峰 (breach_depressions)
- 流向计算 (d8_pointer)
- 汇流累积计算 (d8_flow_accumulation)
- 输出结果

[Whitebox Geospatial Analysis Tools (Whitebox GAT) 是一个强大的开源地理空间分析平台](https://www.whiteboxgeo.com/manual/wbt_book/intro.html)
- 地形分析常用工具：
    - wbt.slope()              # 坡度计算
  - wbt.aspect()             # 坡向计算
  - wbt.curvature()          # 曲率分析
  - wbt.ruggedness()         # 地形粗糙度
  - wbt.tpi()                # 地形位置指数
- 水文分析常用工具：
  - wbt.fill_depressions()   # 填洼处理
  - wbt.breach_depressions() # 削峰处理
  - wbt.d8_pointer()         # D8流向计算
  - wbt.d8_flow_accumulation() # D8汇流累积
  - wbt.watershed()          # 流域划分
- 图像处理常用工具
  - wbt.resample()           # 重采样
  - wbt.clip_raster_to_polygon() # 裁剪
  - wbt.mosaic()             # 影像镶嵌
  - wbt.change_vector_analysis() # 变化向量分析

# 7. get_information.py
## 7.1 get_pixel_size_accurate
计算栅格像元的实际大小（米）
- 情况1：投影坐标系 → 直接读取
- 情况2：地理坐标系 → 转换到UTM计算实际距离
## 7.2 get_tif_latlon_bounds
获取TIF文件的经纬度坐标范围
- 情况1：地理坐标系 → 直接读取
- 情况2：投影坐标系 → 转换到WGS计算经纬度
## 7.3 get_crs_transformer
创建两个坐标系之间的转换器
transformer = get_crs_transformer("EPSG:4326", "EPSG:32650")
lon, lat = 110.0, 35.0
x, y = transformer.transform(lon, lat)
## 7.4 geo_to_pixel
将经纬度转换为图像像素坐标
## 7.5 pixel_to_geo
将像素坐标转换为经纬度

# 8. resize_dem.py
## 8.1 unify_dem
将输入DEM统一到目标DEM的分辨率和坐标系（支持多波段），同时还有缓冲区
可能因为：
浮点数精度误差：坐标计算中的微小误差，可能导致列数出现变化

- 重采样算法：双线性插值可能在边界产生轻微偏差
- 像素对齐问题：新旧坐标系的像素边界不完全对齐
- 存在请问的不整齐，不过认为不影响大体的计算等

## 8.2 resample_to_target_resolution
简单重采样函数（适用于投影坐标系），输入想要的分辨率大小：m

## 8.3 resample_geography_to_target_resolution
重采样函数，可以从地理坐标系转换为投影坐标系后再进行重采样\
⚠️ 目前匹配不齐，可能是数据源的问题

# 9. splicing_dem.py
## 9.1 merge_georeferenced_tifs
- 将同一坐标系下，且Nodata等配置都要相同的多个 GeoTIFF 文件拼接成一个大的 GeoTIFF 文件，并处理重叠区域
- 整体思路是找出需要拼接的文件的范围，创建一个大的数组，然后把每个文件填充进去，且每次填充计数，以方便后面平滑
- 考虑了 NoData: 尝试处理源数据的 NoData 值，避免将其计入计算（尤其是在 mean 策略下）
- 计算全局边界和变换: 正确地计算了拼接后图像的地理范围、尺寸和仿射变换矩阵
- ⚠️ global_width 和 global_height 的计算: 使用 int(round(...)) 来计算最终图像的尺寸。这通常是合理的，但如果源文件的边界和分辨率之间存在微小的不匹配（浮点精度问题），可能会导致最终尺寸略有偏差。虽然 rasterio 通常能很好地处理这种情况，但在极端情况下仍需留意
- ⚠️ 前代码使用 ds.read() 一次性读取了整个源文件的数据。对于大型文件，这会消耗大量内存

# 10. split_dem.py
## 10.1 split_tif
- 实现了基于 step = tile_size - overlap 的滑动窗口逻辑来创建重叠
- NoData 处理（初步）: 尝试跳过所有像素都是 NoData 或 NaN 的瓦片，这是一个很好的优化，避免生成无意义的文件
- 目前只能处理单波段

# 11. utils.py
## 11.1 read_tif
成功读取了常用的栅格数据属性：像元值、仿射变换、坐标系、NoData 值

## 11.2 write_tif
输入参数涵盖了写入 GeoTIFF 所需的核心信息

## get_counters_by_value:
首先获取矩形中的所有类别，并分类别获取矩形，然后获取识别结果的拓展矩形，包括：{key=矩形类型, value=(矩形四点坐标，宽和长，中心坐标，矩形面积)}
调用函数：get_outline
## get_outline
获取矩形的外接矩形
调用函数：anti_aliasing
## anti_aliasing
消除分类别中的矩形块，防止出现问题