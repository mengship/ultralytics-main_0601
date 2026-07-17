#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成指针油表 YOLO Pose 数据集

从 LabelMe JSON 标注中提取指针油表的关键点，生成 YOLO Pose 数据集。

关键点顺序：
    0 center
    1 tip
    2 empty
    3 full

YOLO Pose 标签格式：
    class cx cy w h center_x center_y tip_x tip_y empty_x empty_y full_x full_y
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成指针油表 YOLO Pose 数据集"
    )
    parser.add_argument(
        "--source-dir",
        default="/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/数据标记/test",
        help="源数据目录",
    )
    parser.add_argument(
        "--output-dir",
        default="call_entrance_pose_grid/dataset_pointer_pose",
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
    """将 rotation 标注转换为外接矩形 bbox"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def rectangle_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """将 rectangle 标注转换为 bbox

    LabelMe rectangle 可能给出 2 个点（对角点）或 4 个点（四个角）
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


def extract_pointer_pose_data(
    json_path: Path,
) -> Optional[Dict]:
    """从 JSON 中提取指针油表的框和关键点

    Args:
        json_path: JSON 文件路径

    Returns:
        包含 bbox 和关键点的字典，或 None
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

    # 查找 oil 框和关键点
    shapes = data.get('shapes', [])
    oil_box = None
    keypoints = {'center': None, 'tip': None, 'empty': None, 'full': None}

    for shape in shapes:
        label = shape.get('label')
        shape_type = shape.get('shape_type')
        points = shape.get('points', [])

        if not points:
            continue

        # 提取 oil 框
        if label == 'oil':
            if shape_type == 'rectangle' and len(points) >= 2:
                oil_box = rectangle_to_bbox(points)
            elif shape_type == 'rotation' and len(points) >= 4:
                oil_box = rotation_to_bbox(points)

        # 提取关键点
        elif label in keypoints and shape_type == 'point':
            if len(points) >= 1:
                keypoints[label] = (points[0][0], points[0][1])

    # 检查是否所有必需元素都存在
    if oil_box is None:
        return None

    if any(v is None for v in keypoints.values()):
        return None

    return {
        'bbox': oil_box,
        'keypoints': keypoints,
        'img_width': img_width,
        'img_height': img_height,
    }


def normalize_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """将 bbox 归一化为 YOLO 格式 (cx, cy, w, h)"""
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


def normalize_keypoint(
    x: float,
    y: float,
    img_width: int,
    img_height: int,
) -> Tuple[float, float]:
    """归一化关键点坐标"""
    nx = x / img_width
    ny = y / img_height

    # 裁剪到 [0, 1]
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))

    return nx, ny


def process_dataset(
    source_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> None:
    """处理数据集主流程"""
    # 读取 readme.xlsx
    readme_path = source_dir / "readme.xlsx"
    if not readme_path.exists():
        raise FileNotFoundError(f"未找到 readme.xlsx: {readme_path}")

    df = pd.read_excel(readme_path, sheet_name="Sheet1")
    print(f"读取到 {len(df)} 条记录")

    # 只保留指针油表样本
    df_pointer = df[df['油表类型'] == '指针']
    print(f"指针油表样本: {len(df_pointer)}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 处理结果统计
    processed = []
    missing_report = {
        'missing_image': [],
        'missing_json': [],
        'missing_size': [],
        'missing_oil': [],
        'missing_keypoints': [],
        'fuel_type_mismatch': [],
        'position_unknown': [],
    }

    # 处理每条记录
    for idx, row in df_pointer.iterrows():
        image_name = row['图片名称']
        fuel_type = row['油表类型']
        fuel_position = row['油表位置']

        # 查找图片文件
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG']:
            candidate = source_dir / f"{Path(image_name).stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            missing_report['missing_image'].append(image_name)
            continue

        # 查找 JSON 文件
        json_path = source_dir / f"{Path(image_name).stem}.json"
        if not json_path.exists():
            missing_report['missing_json'].append(image_name)
            continue

        # 提取指针油表数据
        result = extract_pointer_pose_data(json_path)
        if result is None:
            # 进一步判断缺失原因
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    shapes = data.get('shapes', [])
                    has_oil = any(s.get('label') == 'oil' for s in shapes)

                    if not has_oil:
                        missing_report['missing_oil'].append(image_name)
                    else:
                        missing_report['missing_keypoints'].append(image_name)
            except:
                missing_report['missing_json'].append(image_name)
            continue

        # 归一化 bbox
        x1, y1, x2, y2 = result['bbox']
        img_width = result['img_width']
        img_height = result['img_height']

        cx, cy, w, h = normalize_bbox(x1, y1, x2, y2, img_width, img_height)

        # 归一化关键点 (顺序: center, tip, empty, full)
        kpts = result['keypoints']
        center_x, center_y = normalize_keypoint(kpts['center'][0], kpts['center'][1], img_width, img_height)
        tip_x, tip_y = normalize_keypoint(kpts['tip'][0], kpts['tip'][1], img_width, img_height)
        empty_x, empty_y = normalize_keypoint(kpts['empty'][0], kpts['empty'][1], img_width, img_height)
        full_x, full_y = normalize_keypoint(kpts['full'][0], kpts['full'][1], img_width, img_height)

        # 生成 YOLO Pose 标签 (class=0, 固定)
        label = (
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} "
            f"{center_x:.6f} {center_y:.6f} {tip_x:.6f} {tip_y:.6f} "
            f"{empty_x:.6f} {empty_y:.6f} {full_x:.6f} {full_y:.6f}"
        )

        processed.append({
            'image_name': image_name,
            'image_path': image_path,
            'label': label,
            'fuel_position': fuel_position,
        })

    print(f"\n处理结果统计:")
    print(f"  成功处理: {len(processed)}")
    print(f"  缺失图片: {len(missing_report['missing_image'])}")
    print(f"  缺失 JSON: {len(missing_report['missing_json'])}")
    print(f"  缺失 oil 框: {len(missing_report['missing_oil'])}")
    print(f"  缺失关键点: {len(missing_report['missing_keypoints'])}")

    # 按油表位置分层切分 train/val
    import random
    random.seed(seed)

    # 按 fuel_position 分组
    position_groups = {}
    for sample in processed:
        pos = sample['fuel_position']
        if pos not in position_groups:
            position_groups[pos] = []
        position_groups[pos].append(sample)

    print(f"\n油表位置分布:")
    for pos, samples in position_groups.items():
        print(f"  {pos}: {len(samples)}")

    # 每组分别切分
    train_samples = []
    val_samples = []

    for pos, samples in position_groups.items():
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
    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {output_dir.absolute()}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"kpt_shape: [4, 2]\n")
        f.write(f"flip_idx: [0, 1, 2, 3]\n")
        f.write(f"\nnames:\n")
        f.write(f"  0: pointer_pose\n")

    print(f"\ndata.yaml 已生成: {yaml_path}")

    # 生成 metadata.json
    metadata = {
        'total_samples': len(processed),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'keypoint_order': ['center', 'tip', 'empty', 'full'],
        'kpt_shape': [4, 2],
        'position_distribution': {pos: len(samples) for pos, samples in position_groups.items()},
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

    print(f"\n✅ 指针油表 YOLO Pose 数据集生成完成！")
    print(f"输出目录: {output_dir.absolute()}")


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"源数据目录不存在: {source_dir}")

    print("=" * 70)
    print("生成指针油表 YOLO Pose 数据集")
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
