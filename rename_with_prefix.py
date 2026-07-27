#!/usr/bin/env python3
"""
给指定目录下的子目录中的文件添加目录前缀
"""
import os
from pathlib import Path


def rename_files_with_prefix(base_dir):
    """
    给base_dir下所有子目录中的文件添加目录前缀

    Args:
        base_dir: 基础目录路径
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"错误：目录不存在 {base_dir}")
        return

    # 遍历所有子目录
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]

    print(f"找到 {len(subdirs)} 个子目录")

    for subdir in subdirs:
        dir_name = subdir.name
        # 将目录名中的空格替换为下划线作为前缀
        prefix = dir_name.replace(' ', '_')

        print(f"\n处理目录: {dir_name}")
        print(f"使用前缀: {prefix}")

        # 获取该目录下的所有文件
        files = [f for f in subdir.iterdir() if f.is_file() and not f.name.startswith('.')]

        # 统计需要重命名的文件
        rename_count = 0
        skip_count = 0

        for file in files:
            old_name = file.name

            # 如果文件名已经以前缀开头，则跳过
            if old_name.startswith(f"{prefix}_"):
                skip_count += 1
                continue

            # 生成新文件名
            new_name = f"{prefix}_{old_name}"
            new_path = file.parent / new_name

            # 检查新文件名是否已存在
            if new_path.exists():
                print(f"  警告：目标文件已存在，跳过 {old_name}")
                skip_count += 1
                continue

            # 重命名文件
            try:
                file.rename(new_path)
                print(f"  ✓ {old_name} -> {new_name}")
                rename_count += 1
            except Exception as e:
                print(f"  ✗ 重命名失败 {old_name}: {e}")

        print(f"完成：重命名 {rename_count} 个文件，跳过 {skip_count} 个文件")


if __name__ == "__main__":
    base_directory = "/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/数据标记/王波补充标注里程框"

    print("=" * 60)
    print("文件重命名工具 - 添加目录前缀")
    print("=" * 60)
    print(f"目标目录: {base_directory}")
    print()

    rename_files_with_prefix(base_directory)
    print("\n操作完成！")
