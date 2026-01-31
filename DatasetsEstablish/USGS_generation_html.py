import json
import os

# ===================== 配置项（请确认路径正确） =====================
# JSON文件路径
JSON_FILE_PATH = r"D:\研究文件\ResearchData\USA\USGSDEM\DownloadInfo.json"
# HTML文件输出目录（和JSON同目录）
HTML_OUTPUT_DIR = r"D:\研究文件\ResearchData\USA\USGSDEM"

# ===================== HTML模板（每个分组的页面模板） =====================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{group_name} - USGS DEM 下载链接</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Arial', sans-serif;
        }}
        body {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        h1 {{
            color: #2c3e50;
            margin-bottom: 2rem;
            border-bottom: 3px solid #3498db;
            padding-bottom: 0.5rem;
        }}
        .file-item {{
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .file-name {{
            color: #2980b9;
            font-size: 1.1rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
        }}
        .link-item {{
            margin-left: 1rem;
            padding: 0.5rem;
            background-color: #f8f9fa;
            border-left: 3px solid #3498db;
            border-radius: 4px;
            margin-bottom: 0.5rem;
        }}
        a {{
            color: #3498db;
            text-decoration: none;
            word-break: break-all;
        }}
        a:hover {{
            text-decoration: underline;
            color: #2980b9;
        }}
        .empty-message {{
            color: #7f8c8d;
            font-style: italic;
            text-align: center;
            padding: 2rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <h1>{group_name} - USGS DEM 下载链接</h1>
    {content}
</body>
</html>
"""

def is_error_entry(links):
    """
    检查是否为错误条目
    错误条目的键通常以 '__' 开头（如 '__error__', '__detail__' 等）
    """
    return any(key.startswith('__') for key in links.keys())



# ===================== 核心逻辑 =====================
def generate_html_per_group(json_path, output_dir):
    """为JSON中的每个分组单独生成HTML文件"""
    # 1. 检查输出目录是否存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")

    # 2. 读取JSON文件
    if not os.path.exists(json_path):
        print(f"错误：找不到JSON文件 {json_path}")
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            download_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误：JSON文件格式错误 - {e}")
        return False
    except Exception as e:
        print(f"错误：读取JSON文件失败 - {e}")
        return False

    # 3. 遍历每个分组，生成独立HTML
    if not download_data or len(download_data) == 0:
        print("提示：JSON文件中无任何分组数据")
        return False

    generated_count = 0
    for group_name, files in download_data.items():
        # 生成当前分组的HTML文件名（避免特殊字符，替换非法字符）
        safe_group_name = group_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace(
            '?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        html_file_name = f"{safe_group_name}_DEM_Links.html"
        html_file_path = os.path.join(output_dir, html_file_name)

        # 构建当前分组的HTML内容
        if not files or len(files) == 0:
            content = '<div class="empty-message">该分组暂无下载链接数据</div>'
        else:
            content = ""
            # 遍历当前分组下的所有文件
            for file_name, links in files.items():

                if is_error_entry(links):
                    continue

                content += f'<div class="file-item">'
                content += f'<div class="file-name">文件: {file_name}</div>'

                # 遍历当前文件的所有链接
                for link in links.keys():  # links的值是true，取key即可
                    content += f'<div class="link-item">'
                    content += f'<a href="{link}" target="_blank">{link}</a>'
                    content += f'</div>'

                content += f'</div>'  # 关闭file-item

        # 填充模板并写入文件
        final_html = HTML_TEMPLATE.format(group_name=group_name, content=content)
        try:
            with open(html_file_path, 'w', encoding='utf-8') as f:
                f.write(final_html)
            print(f"✅ 成功生成：{html_file_path}")
            generated_count += 1
        except Exception as e:
            print(f"❌ 生成 {group_name} 的HTML失败 - {e}")

    print(f"\n生成完成！共生成 {generated_count} 个HTML文件，保存路径：{output_dir}")
    return True


# ===================== 执行生成 =====================
if __name__ == '__main__':
    generate_html_per_group(JSON_FILE_PATH, HTML_OUTPUT_DIR)