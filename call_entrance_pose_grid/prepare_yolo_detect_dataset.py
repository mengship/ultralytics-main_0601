#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 YOLO 检测数据集：识别指针油表和格子油表

从 LabelMe JSON 标注中提取油表框，按油表类型生成 YOLO detection 数据集。

类别映射：
    0: pointer (指针油表)
    1: grid (格子油表)

处理规则：
    - 指针样本：提取 'oil' 框
    - 格子样本：提取 'oil1' 框
    - 支持 rectangle 和 rotation 两种标注形式
    - 按油表类型分层做 8:2 切分
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 YOLO 检测数据集（指针 + 格子油表）"
    )
    parser.add_argument(
        "--source-dir",
        default="/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/数据标记/test",
        help="源数据目录",
    )
    parser.add_argument(
        "--output-dir",
        default="call_entrance_pose_grid/dataset_detect",
        help="输出数据集目录",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="验证集比例",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    return parser.parse_args()


def rotation_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """将 rotation 标注转换为外接矩形 bbox

    Args:
        points: 4 个角点坐标

    Returns:
        (x1, y1, x2, y2) 外接矩形
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    return x1, y1, x2, y2


def rectangle_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """将 rectangle 标注转换为 bbox

    Args:
        points: 2 个对角点坐标 [[x1, y1], [x2, y2]] 或 4 个角点坐标

    Returns:
        (x1, y1, x2, y2)
    """
    if len(points) == 2:
        # 2 个对角点
        x1 = min(points[0][0], points[1][0])
        y1 = min(points[0][1], points[1][1])
        x2 = max(points[0][0], points[1][0])
        y2 = max(points[0][1], points[1][1])
    else:
        # 4 个角点，取外接框
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

    return x1, y1, x2, y2


def extract_oil_box(
    json_path: Path,
    fuel_type: str,
) -> Optional[Tuple[float, float, float, float, int, int]]:
    """从 JSON 中提取油表框

    Args:
        json_path: JSON 文件路径
        fuel_type: 油表类型 ('指针' 或 '格子')

    Returns:
        (x1, y1, x2, y2, img_width, img_height) 或 None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return None

    # 获取图像尺寸
    img_width = data.get('imageWidth')
    img_height = data.get('imageHeight')

    if img_width is None or img_height is None:
        return None

    # 根据油表类型选择要提取的标签
    target_label = 'oil' if fuel_type == '指针' else 'oil1'

    # 查找目标框
    shapes = data.get('shapes', [])
    for shape in shapes:
        if shape.get('label') == target_label:
            shape_type = shape.get('shape_type')
            points = shape.get('points', [])

            if not points:
                continue

            # 根据标注类型转换为 bbox
            if shape_type == 'rectangle':
                if len(points) >= 2:
                    x1, y1, x2, y2 = rectangle_to_bbox(points)
                    return x1, y1, x2, y2, img_width, img_height
            elif shape_type == 'rotation':
                if len(points) >= 4:
                    x1, y1, x2, y2 = rotation_to_bbox(points)
                    return x1, y1, x2, y2, img_width, img_height

    return None


def bbox_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """将 bbox 转换为 YOLO 格式（归一化的 cx cy w h）

    Args:
        x1, y1, x2, y2: bbox 坐标
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        (cx, cy, w, h) 归一化坐标
    """
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    # 裁剪到 [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h


def find_image_fuzzy(source_dir: Path, image_name: str) -> Optional[Path]:
    """模糊查找图片文件（容错机制）

    1. 精确匹配：按图片名称精确查找
    2. 模糊匹配：提取车牌号进行模糊匹配（如 CCP1767）

    Args:
        source_dir: 源数据目录
        image_name: Excel 中的图片名称

    Returns:
        找到的图片路径，或 None
    """
    # 方法 1：精确匹配
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']:
        candidate = source_dir / f"{Path(image_name).stem}{ext}"
        if candidate.exists():
            return candidate

    # 方法 2：模糊匹配（提取车牌号部分）
    # 从 "260630_CCP1767" 中提取 "CCP1767"
    parts = Path(image_name).stem.split('_')
    if len(parts) >= 2:
        plate_number = parts[-1]  # 提取最后一部分（车牌号）

        # 在目录中查找包含该车牌号的文件
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']:
            for file in source_dir.glob(f"*{plate_number}{ext}"):
                return file

    return None


def find_json_fuzzy(source_dir: Path, image_name: str) -> Optional[Path]:
    """模糊查找 JSON 文件（容错机制）

    Args:
        source_dir: 源数据目录
        image_name: Excel 中的图片名称

    Returns:
        找到的 JSON 路径，或 None
    """
    # 方法 1：精确匹配
    candidate = source_dir / f"{Path(image_name).stem}.json"
    if candidate.exists():
        return candidate

    # 方法 2：模糊匹配（提取车牌号部分）
    parts = Path(image_name).stem.split('_')
    if len(parts) >= 2:
        plate_number = parts[-1]

        # 在目录中查找包含该车牌号的 JSON
        for file in source_dir.glob(f"*{plate_number}.json"):
            return file

    return None


def process_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> None:
    """处理数据集主流程

    Args:
        source_dir: 源数据目录
        output_dir: 输出目录
        val_ratio: 验证集比例
        seed: 随机种子
    """
    # 读取 readme.xlsx
    readme_path = source_dir / "readme.xlsx"
    if not readme_path.exists():
        raise FileNotFoundError(f"未找到 readme.xlsx: {readme_path}")

    df = pd.read_excel(readme_path, sheet_name="Sheet1")

    # 将"图片名称"列转换为字符串类型（避免纯数字被读取为 int）
    df['图片名称'] = df['图片名称'].astype(str)

    print(f"读取到 {len(df)} 条记录")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 处理结果统计
    processed = {'pointer': [], 'grid': []}
    missing_report = {
        'missing_image': [],
        'missing_json': [],
        'missing_size': [],
        'missing_box': [],
        'fuel_type_unknown': [],
    }

    # 类别映射
    class_map = {'指针': 0, '格子': 1}

    # 处理每条记录
    for idx, row in df.iterrows():
        image_name = row['图片名称']
        fuel_type = row['油表类型']
        fuel_position = row['油表位置']

        # 检查油表类型
        if fuel_type not in class_map:
            missing_report['fuel_type_unknown'].append({
                'image': image_name,
                'fuel_type': fuel_type,
            })
            continue

        class_id = class_map[fuel_type]

        # 查找图片文件（支持模糊匹配）
        image_path = find_image_fuzzy(source_dir, image_name)

        if image_path is None:
            missing_report['missing_image'].append(image_name)
            continue

        # 查找 JSON 文件（支持模糊匹配）
        json_path = find_json_fuzzy(source_dir, image_name)
        if json_path is None:
            missing_report['missing_json'].append(image_name)
            continue

        # 提取油表框
        result = extract_oil_box(json_path, fuel_type)
        if result is None:
            missing_report['missing_box'].append({
                'image': image_name,
                'fuel_type': fuel_type,
            })
            continue

        x1, y1, x2, y2, img_width, img_height = result

        # 转换为 YOLO 格式
        cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2, img_width, img_height)

        # 记录成功样本
        fuel_type_en = 'pointer' if fuel_type == '指针' else 'grid'
        processed[fuel_type_en].append({
            'image_name': image_name,
            'image_path': image_path,
            'class_id': class_id,
            'label': f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}",
            'fuel_position': fuel_position,
        })

    print(f"\n处理结果统计:")
    print(f"  指针样本: {len(processed['pointer'])}")
    print(f"  格子样本: {len(processed['grid'])}")
    print(f"  缺失图片: {len(missing_report['missing_image'])}")
    print(f"  缺失 JSON: {len(missing_report['missing_json'])}")
    print(f"  缺失油表框: {len(missing_report['missing_box'])}")
    print(f"  油表类型未知: {len(missing_report['fuel_type_unknown'])}")

    # 按油表类型分层切分 train/val
    import random
    random.seed(seed)

    train_samples = []
    val_samples = []

    for fuel_type_en in ['pointer', 'grid']:
        samples = processed[fuel_type_en]
        random.shuffle(samples)

        split_idx = int(len(samples) * (1 - val_ratio))
        train_samples.extend(samples[:split_idx])
        val_samples.extend(samples[split_idx:])

    print(f"\n数据集切分:")
    print(f"  训练集: {len(train_samples)}")
    print(f"  验证集: {len(val_samples)}")

    # 复制图片和写标签
    for split, samples in [('train', train_samples), ('val', val_samples)]:
        for sample in samples:
            # 复制图片
            src_image = sample['image_path']
            dst_image = output_dir / split / 'images' / src_image.name
            shutil.copy2(src_image, dst_image)

            # 写标签
            label_path = output_dir / split / 'labels' / f"{src_image.stem}.txt"
            with open(label_path, 'w', encoding='utf-8') as f:
                f.write(sample['label'] + '\n')

    # 生成 data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'pointer',
            1: 'grid',
        },
    }

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"\nnames:\n")
        for class_id, class_name in data_yaml['names'].items():
            f.write(f"  {class_id}: {class_name}\n")

    print(f"\ndata.yaml 已生成: {yaml_path}")

    # 生成 metadata.json
    metadata = {
        'total_samples': len(train_samples) + len(val_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'pointer_samples': len(processed['pointer']),
        'grid_samples': len(processed['grid']),
        'class_map': {'pointer': 0, 'grid': 1},
        'val_ratio': val_ratio,
        'seed': seed,
    }

    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"metadata.json 已生成: {metadata_path}")

    # 生成 missing_report.json
    report_path = output_dir / 'missing_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(missing_report, f, indent=2, ensure_ascii=False)

    print(f"missing_report.json 已生成: {report_path}")

    print(f"\n✅ YOLO 检测数据集生成完成！")
    print(f"输出目录: {output_dir.absolute()}")


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"源数据目录不存在: {source_dir}")

    print("=" * 70)
    print("生成 YOLO 检测数据集：指针 + 格子油表")
    print("=" * 70)
    print(f"源数据目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"验证集比例: {args.val_ratio}")
    print(f"随机种子: {args.seed}")
    print("=" * 70)
    print()

    process_dataset(source_dir, output_dir, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
