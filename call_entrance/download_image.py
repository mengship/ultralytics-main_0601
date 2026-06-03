import pandas as pd
import requests
import os
import re
from urllib.parse import urlparse


def clean_url(value):
    """清洗和规范化URL"""
    if pd.isna(value):
        return None

    url = str(value).strip().strip('"').strip("'").strip()
    if not url or url.lower() in {'nan', 'none', 'null', 'na'}:
        return None

    if url.startswith('//'):
        url = 'https:' + url

    if not re.match(r'^https?://', url, flags=re.IGNORECASE):
        url = 'https://' + url

    return url


def infer_ext(url, content_type):
    """推断图片格式"""
    path_ext = os.path.splitext(urlparse(url).path)[1].lower()
    if path_ext in {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}:
        return path_ext

    ctype = (content_type or '').lower()
    if 'png' in ctype:
        return '.png'
    if 'gif' in ctype:
        return '.gif'
    if 'webp' in ctype:
        return '.webp'
    if 'bmp' in ctype:
        return '.bmp'
    return '.jpg'


def safe_text(value, fallback):
    """安全地处理文本值"""
    if pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    return re.sub(r'[\\/:*?"<>|]', '_', text)


def download_images(excel_file, output_dir, sheet_name=0, url_column='url',
                   filter_column=None, filter_value=None):
    """下载图片

    Args:
        excel_file: Excel文件路径
        output_dir: 输出目录
        sheet_name: Sheet页名称或索引（默认为0，即第一个sheet）
                   可以是sheet名称字符串 或 sheet索引整数
        url_column: URL所在的列名（默认为'url'）
        filter_column: 过滤列名（可选，例如'人工检查结果'）
        filter_value: 过滤值（可选，例如'AI没有识别结果'）
    """

    print("\n" + "="*70)
    print("📥 开始下载图片")
    print("="*70 + "\n")
    print(f"📄 Excel文件: {excel_file}")
    print(f"📊 Sheet页: {sheet_name}")
    print(f"📍 URL列名: {url_column}")
    if filter_column and filter_value:
        print(f"🔍 过滤条件: {filter_column} = '{filter_value}'")
    print(f"📁 输出目录: {output_dir}\n")

    # 检查Excel文件是否存在
    if not os.path.exists(excel_file):
        print(f"❌ 错误：Excel文件不存在")
        print(f"   路径: {os.path.abspath(excel_file)}")
        return

    # 读取Excel
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        print(f"✅ 已读取Excel文件，共 {len(df)} 行\n")
    except Exception as e:
        print(f"❌ 错误：无法读取Excel文件")
        print(f"   错误信息: {e}")
        print(f"   请检查:")
        print(f"   1. 文件是否存在: {os.path.abspath(excel_file)}")
        print(f"   2. Sheet页是否正确: {sheet_name}")
        return

    # 检查URL列是否存在
    if url_column not in df.columns:
        print(f"❌ 错误：列 '{url_column}' 不存在")
        print(f"   可用的列: {list(df.columns)}")
        return

    # 检查过滤列是否存在（如果指定了过滤条件）
    if filter_column:
        if filter_column not in df.columns:
            print(f"❌ 错误：过滤列 '{filter_column}' 不存在")
            print(f"   可用的列: {list(df.columns)}")
            return

        # 应用过滤条件
        original_count = len(df)
        df = df[df[filter_column] == filter_value].copy()
        filtered_count = len(df)
        print(f"🔍 过滤结果: {original_count} 行 → {filtered_count} 行（匹配 '{filter_value}'）\n")

        if filtered_count == 0:
            print(f"⚠️  警告：没有符合过滤条件的数据")
            return

        # 如果有过滤条件，创建子目录
        output_dir = os.path.join(output_dir, filter_value)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 输出目录已准备: {os.path.abspath(output_dir)}\n")

    ok, skip, fail = 0, 0, 0

    for idx, row in df.iterrows():
        raw_url = row[url_column]
        url = clean_url(raw_url)

        if not url:
            skip += 1
            print(f"[SKIP] row={idx + 2}, invalid url: {raw_url}")
            continue

        # stat_date = safe_text(row['stat_date'], 'unknown_date')
        # plate_number = safe_text(row['plate_number'], 'unknown_plate')
        # sys_name = safe_text(row['sys_name'], 'unknown_sysname')
        primarykey = safe_text(row['primarykey'], 'unknown_primarykey')

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                ext = infer_ext(url, resp.headers.get('Content-Type'))
                filename = os.path.join(output_dir, f'{primarykey}{ext}')
                with open(filename, 'wb') as f:
                    f.write(resp.content)
                ok += 1
                print(f"[OK] {filename}")
            else:
                fail += 1
                print(f"[FAIL] row={idx + 2}, status={resp.status_code}, url={url}")
        except Exception as e:
            fail += 1
            print(f"[FAIL] row={idx + 2}, url={url}, error={e}")

    print(f"\n{'='*70}")
    print(f"✅ 下载完成！")
    print(f"{'='*70}")
    print(f"📊 统计结果：")
    print(f"   - 成功: {ok} 张")
    print(f"   - 跳过: {skip} 张")
    print(f"   - 失败: {fail} 张")
    print(f"   - 总计: {ok + skip + fail} 张\n")


if __name__ == '__main__':
    # =================== 修改这里设置Excel文件、输出目录、Sheet页和URL列名 ===================
    # dt='0521'

    excelname ='20260602油量人工识别'
    EXCEL_FILE = '/Users/flash/Documents/Data_Work/99_临时中转站/0602训练油表图片/'+ excelname +'.xlsx'  # Excel文件路径
    OUTPUT_DIR = '/Users/flash/Documents/Data_Work/99_临时中转站/0602训练油表图片/'+ excelname          # 输出目录
    SHEET_NAME = 'Sheet1'                                        # Sheet页（0=第一个，或输入sheet名称）
    URL_COLUMN = '盘点照片'                                    # URL所在的列名（默认'url'，可改为其他列名）

    # 过滤条件（可选）
    FILTER_COLUMN = '人工检查结果'                              # 过滤列名，设为None则不过滤
    FILTER_VALUE = 'AI识别结果不准确'                            # 过滤值，仅当FILTER_COLUMN不为None时生效
    # AI没有识别结果
    # 白班示例
    # excelname = '0514白夜班下班打卡'
    # SHEET_NAME = '白班打卡'                         # Sheet页名称 白班
    # SHEET_NAME = '夜班打卡'                         # Sheet页名称 夜班
    # EXCEL_FILE = 'E:/predict/'+ dt +'/'+ excelname +'.xlsx'  # Excel文件路径
    # OUTPUT_DIR = 'E:/predict/'+ dt +'/'+ dt + SHEET_NAME          # 输出目录
    # URL_COLUMN = '下班里程图片'                                     # URL列名

    # 夜班示例
    # EXCEL_FILE = "E:\predict\\0408\\0408白夜班下班打卡.xlsx"  # Excel文件路径
    # OUTPUT_DIR =  "E:\predict\\0408\\0408重点网点夜班打卡"    # 输出目录
    # SHEET_NAME = '重点网点夜班打卡'                           # Sheet页名称
    # URL_COLUMN = 'url'                                       # URL列名

    # 用法示例：
    #   SHEET_NAME = 0        # 读取第一个sheet
    #   SHEET_NAME = 1        # 读取第二个sheet
    #   SHEET_NAME = 'Sheet1' # 读取名称为'Sheet1'的sheet
    #   URL_COLUMN = 'url'             # URL列名
    #   URL_COLUMN = '下班里程图片'    # 或其他列名
    #   FILTER_COLUMN = '人工检查结果'  # 过滤列名
    #   FILTER_VALUE = 'AI没有识别结果' # 过滤值
    #   FILTER_COLUMN = None           # 设为None则不过滤
    # =========================================================================

    print("\n" + "="*70)
    print("📌 图片下载工具")
    print("="*70)
    print(f"\n⚙️  配置：")
    print(f"   ├─ Excel文件: {EXCEL_FILE}")
    print(f"   ├─ Sheet页: {SHEET_NAME}")
    print(f"   ├─ URL列名: {URL_COLUMN}")
    if FILTER_COLUMN and FILTER_VALUE:
        print(f"   ├─ 过滤列名: {FILTER_COLUMN}")
        print(f"   ├─ 过滤值: {FILTER_VALUE}")
    print(f"   └─ 输出目录: {OUTPUT_DIR}\n")

    download_images(EXCEL_FILE, OUTPUT_DIR, SHEET_NAME, URL_COLUMN,
                   FILTER_COLUMN, FILTER_VALUE)
