#!/usr/bin/env python3
"""
目录文件名导出脚本
用法: python3 export_filenames.py <目标目录> [输出文件]
"""

import os
import sys
from pathlib import Path

def get_all_files(directory, ignore_hidden=True):
    """递归获取目录下所有文件路径"""
    files = []
    for root, dirs, filenames in os.walk(directory):
        # 忽略隐藏目录（以.开头）
        if ignore_hidden:
            dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in filenames:
            if ignore_hidden and filename.startswith('.'):
                continue
            # 获取相对于输入目录的路径
            rel_path = os.path.relpath(os.path.join(root, filename), directory)
            files.append(rel_path)
    return sorted(files)

def main():
    # 解析命令行参数
    if len(sys.argv) < 2:
        print("用法: python3 export_filenames.py <目标目录> [输出文件]")
        print("示例: python3 export_filenames.py ./my_project filelist.txt")
        sys.exit(1)

    target_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "filelist.txt"

    # 检查目录是否存在
    if not os.path.isdir(target_dir):
        print(f"错误: 目录 '{target_dir}' 不存在")
        sys.exit(1)

    # 获取所有文件
    files = get_all_files(target_dir)

    # 写入到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in files:
            f.write(file + '\n')

    print(f"✅ 完成！共找到 {len(files)} 个文件，已保存到 {output_file}")
    print(f"📁 目录: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    main()
