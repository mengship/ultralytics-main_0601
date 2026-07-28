#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成包含离散旋转增强的数据集

针对实际场景中图片旋转很随意的情况，预生成 0°, 90°, 180°, 270° 四个方向的数据。
每张原始图像生成 4 个版本，扩大数据集 4 倍。

优点：
- 覆盖所有可能的手机拍摄方向
- 训练时禁用旋转增强，避免样本过滤
- 所有方向的样本质量一致
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成包含离散旋转增强的 YOLO Pose 数据集"
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="源数据目录",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="输出数据集目录",
    )
    parser.add_argument(
        "--fuel-type",
        choices=["pointer", "grid"],
        required=True,
        help="油表类型（pointer 或 grid）",
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
    parser.add_argument(
        "--rotation-angles",
        nargs="+",
        type=int,
        default=[0, 90, 180, 270],
        help="离散旋转角度列表",
    )
    return parser.parse_args()


def rotate_image_90n(image: np.ndarray, angle: int) -> np.ndarray:
    """旋转图像 90 度的倍数（高效实现）

    Args:
        image: 输入图像
        angle: 旋转角度，必须是 90 的倍数

    Returns:
        旋转后的图像
    """
    angle = angle % 360

    if angle == 0:
        return image.copy()
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"角度必须是 90 的倍数，得到 {angle}")


def rotate_keypoint_90n(x: float, y: float, angle: int, width: int, height: int) -> Tuple[float, float]:
    """旋转关键点坐标（90 度倍数）

    Args:
        x, y: 原始坐标
        angle: 旋转角度
        width: 原始图像宽度
        height: 原始图像高度

    Returns:
        旋转后的坐标 (new_x, new_y)
    """
    angle = angle % 360

    if angle == 0:
        return x, y
    elif angle == 90:
        # 逆时针 90°
        return y, height - x
    elif angle == 180:
        return width - x, height - y
    elif angle == 270:
        # 顺时针 90° (逆时针 270°)
        return width - y, x
    else:
        raise ValueError(f"角度必须是 90 的倍数，得到 {angle}")


def extract_keypoints_from_json(json_path: Path, fuel_type: str) -> Optional[Dict]:
    """从 JSON 提取关键点和图像尺寸

    Args:
        json_path: JSON 文件路径
        fuel_type: 'pointer' 或 'grid'

    Returns:
        包含关键点和图像尺寸的字典，或 None
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    img_width = data.get('imageWidth')
    img_height = data.get('imageHeight')

    if img_width is None or img_height is None:
        return None

    # 定义需要的关键点
    if fuel_type == 'pointer':
        required_labels = ['center', 'tip', 'empty', 'full']
        oil_label = 'oil'
    else:  # grid
        required_labels = ['empty', 'full', 'tip']
        oil_label = 'oil1'

    shapes = data.get('shapes', [])
    keypoints = {}
    oil_box = None

    for shape in shapes:
        label = shape.get('label')
        shape_type = shape.get('shape_type')
        points = shape.get('points', [])

        if not points:
            continue

        # 提取关键点
        if label in required_labels and shape_type == 'point':
            keypoints[label] = (float(points[0][0]), float(points[0][1]))

        # 提取油表框（用于验证，但不保存到 YOLO 标签）
        if label == oil_label:
            if shape_type == 'rectangle':
                if len(points) == 2:
                    xs = [points[0][0], points[1][0]]
                    ys = [points[0][1], points[1][1]]
                elif len(points) >= 4:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                else:
                    continue
                oil_box = (min(xs), min(ys), max(xs), max(ys))
            elif shape_type == 'rotation' and len(points) >= 4:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                oil_box = (min(xs), min(ys), max(xs), max(ys))

    # 检查是否所有必需关键点都存在
    if len(keypoints) != len(required_labels):
        return None

    if oil_box is None:
        return None

    return {
        'keypoints': keypoints,
        'oil_box': oil_box,
        'img_width': img_width,
        'img_height': img_height,
    }


def keypoints_to_yolo_label(
    keypoints: Dict[str, Tuple[float, float]],
    oil_box: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
    fuel_type: str,
) -> str:
    """将关键点转换为 YOLO Pose 标签格式

    Args:
        keypoints: 关键点字典
        oil_box: 油表框 (x1, y1, x2, y2)
        img_width: 图像宽度
        img_height: 图像高度
        fuel_type: 'pointer' 或 'grid'

    Returns:
        YOLO 标签字符串
    """
    x1, y1, x2, y2 = oil_box

    # 归一化 bbox
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    # 裁剪到 [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    # 关键点顺序
    if fuel_type == 'pointer':
        kpt_order = ['center', 'tip', 'empty', 'full']
    else:
        kpt_order = ['empty', 'full', 'tip']

    # 归一化关键点
    kpt_strs = []
    for name in kpt_order:
        x, y = keypoints[name]
        nx = max(0.0, min(1.0, x / img_width))
        ny = max(0.0, min(1.0, y / img_height))
        kpt_strs.append(f"{nx:.6f} {ny:.6f} 2")  # visibility=2 (visible)

    # 拼接标签
    label = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(kpt_strs)
    return label


def process_one_image_with_rotations(
    image_path: Path,
    json_path: Path,
    fuel_type: str,
    rotation_angles: List[int],
    output_split_dir: Path,
    image_basename: str,
) -> int:
    """处理单张图像，生成多个旋转版本

    Returns:
        成功生成的版本数量
    """
    # 提取关键点
    data = extract_keypoints_from_json(json_path, fuel_type)
    if data is None:
        return 0

    keypoints = data['keypoints']
    oil_box = data['oil_box']
    img_width = data['img_width']
    img_height = data['img_height']

    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        return 0

    count = 0

    for angle in rotation_angles:
        # 旋转图像
        rotated_image = rotate_image_90n(image, angle)

        # 旋转后的尺寸
        if angle % 180 == 0:
            new_width, new_height = img_width, img_height
        else:
            new_width, new_height = img_height, img_width

        # 旋转关键点
        rotated_keypoints = {}
        for name, (x, y) in keypoints.items():
            new_x, new_y = rotate_keypoint_90n(x, y, angle, img_width, img_height)
            rotated_keypoints[name] = (new_x, new_y)

        # 旋转 bbox
        x1, y1, x2, y2 = oil_box
        corners = [
            rotate_keypoint_90n(x1, y1, angle, img_width, img_height),
            rotate_keypoint_90n(x2, y1, angle, img_width, img_height),
            rotate_keypoint_90n(x1, y2, angle, img_width, img_height),
            rotate_keypoint_90n(x2, y2, angle, img_width, img_height),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        rotated_box = (min(xs), min(ys), max(xs), max(ys))

        # 生成 YOLO 标签
        label = keypoints_to_yolo_label(
            rotated_keypoints,
            rotated_box,
            new_width,
            new_height,
            fuel_type,
        )

        # 保存图像和标签
        output_name = f"{image_basename}_rot{angle}"
        image_out_path = output_split_dir / 'images' / f"{output_name}.jpg"
        label_out_path = output_split_dir / 'labels' / f"{output_name}.txt"

        cv2.imwrite(str(image_out_path), rotated_image)

        with open(label_out_path, 'w', encoding='utf-8') as f:
            f.write(label + '\n')

        count += 1

    return count


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"源数据目录不存在: {source_dir}")

    print("=" * 70)
    print(f"生成包含旋转增强的 {args.fuel_type} 数据集")
    print("=" * 70)
    print(f"源数据目录: {source_dir}")
    print(f"输出目录: {output_dir}")
    print(f"油表类型: {args.fuel_type}")
    print(f"旋转角度: {args.rotation_angles}")
    print(f"验证集比例: {args.val_ratio}")
    print("=" * 70)
    print()

    # 读取 readme.xlsx
    readme_path = source_dir / "readme.xlsx"
    if not readme_path.exists():
        raise FileNotFoundError(f"未找到 readme.xlsx: {readme_path}")

    df = pd.read_excel(readme_path, sheet_name="Sheet1")
    df['图片名称'] = df['图片名称'].astype(str)

    # 过滤油表类型
    fuel_type_map = {'pointer': '指针', 'grid': '格子'}
    df_filtered = df[df['油表类型'] == fuel_type_map[args.fuel_type]]

    print(f"读取到 {len(df)} 条记录")
    print(f"筛选 {args.fuel_type} 类型: {len(df_filtered)} 条")
    print()

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 随机切分 train/val
    import random
    random.seed(args.seed)

    indices = list(df_filtered.index)
    random.shuffle(indices)

    split_idx = int(len(indices) * (1 - args.val_ratio))
    train_indices = set(indices[:split_idx])
    val_indices = set(indices[split_idx:])

    print(f"训练集: {len(train_indices)} 张原始图像")
    print(f"验证集: {len(val_indices)} 张原始图像")
    print()

    # 处理每张图像
    train_count = 0
    val_count = 0

    for idx, row in df_filtered.iterrows():
        image_name = row['图片名称']

        # 查找图像文件
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            candidate = source_dir / f"{Path(image_name).stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        # 查找 JSON 文件
        json_path = source_dir / f"{Path(image_name).stem}.json"
        if not json_path.exists():
            continue

        # 判断属于哪个集合
        if idx in train_indices:
            split = 'train'
        else:
            split = 'val'

        split_dir = output_dir / split

        # 处理图像（生成多个旋转版本）
        basename = image_path.stem
        count = process_one_image_with_rotations(
            image_path,
            json_path,
            args.fuel_type,
            args.rotation_angles,
            split_dir,
            basename,
        )

        if split == 'train':
            train_count += count
        else:
            val_count += count

    print()
    print("=" * 70)
    print("数据集生成完成")
    print("=" * 70)
    print(f"训练集: {train_count} 张图像（含旋转增强）")
    print(f"验证集: {val_count} 张图像（含旋转增强）")
    print(f"平均每张原始图像生成: {len(args.rotation_angles)} 个版本")
    print("=" * 70)

    # 生成 data.yaml
    yaml_content = f"""path: {output_dir.absolute()}
train: train/images
val: val/images

kpt_shape: [{len(['center', 'tip', 'empty', 'full'] if args.fuel_type == 'pointer' else ['empty', 'full', 'tip'])}, 3]

names:
  0: {args.fuel_type}
"""

    yaml_path = output_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"data.yaml 已生成: {yaml_path}")


if __name__ == "__main__":
    main()
