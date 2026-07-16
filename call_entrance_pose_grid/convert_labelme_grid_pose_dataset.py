#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LabelMe grid gauge annotations to YOLO Pose dataset.

This script converts grid-style fuel gauge annotations from LabelMe/X-AnyLabeling
format to YOLO Pose format.

Annotation rules:
    - oil1: grid gauge bounding box (rectangle or rotation)
    - empty: empty fuel position point
    - full: full fuel position point
    - tip: current fuel level point
    - odometer: mileage box (ignored)

Keypoint order (fixed):
    0 empty
    1 full
    2 tip

YOLO Pose label format:
    class cx cy w h empty_x empty_y full_x full_y tip_x tip_y

All coordinates are normalized to [0, 1].

Usage:
    # Convert with default settings (input: call_entrance_pose_grid/datasets)
    python call_entrance_pose_grid/convert_labelme_grid_pose_dataset.py

    # Specify custom input and output directories (one line)
    python call_entrance_pose_grid/convert_labelme_grid_pose_dataset.py --json-dir /path/to/labelme_jsons --output-dir /path/to/output

    # Or use backslash for multi-line (in shell script, not terminal copy-paste)
    python call_entrance_pose_grid/convert_labelme_grid_pose_dataset.py \
        --json-dir call_entrance_pose_grid/datasets \
        --output-dir call_entrance_pose_grid/dataset_convert_v2

    # Custom validation ratio
    python call_entrance_pose_grid/convert_labelme_grid_pose_dataset.py --val-ratio 0.3

Note on rotation bbox:
    - rotation type: 4 corner points of a rotated rectangle
    - Conversion: takes axis-aligned bounding box (AABB) of the 4 points
    - Limitation: may include background pixels outside the rotated box
    - Impact: minimal if rotation angle is small (< 15°)
    - Alternative: preprocess with rotation correction (not implemented)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml


KEYPOINT_ORDER = ("empty", "full", "tip")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class GridPoseObject:
    """格子油表 YOLO Pose 目标，包含 bbox 和 3 个关键点。"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (pixel coordinates)
    keypoints: Dict[str, Tuple[float, float]]  # {name: (x, y)} pixel coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe grid gauge annotations to YOLO Pose dataset."
    )
    parser.add_argument(
        "--json-dir",
        default="call_entrance_pose_grid/datasets",
        help="Input directory containing LabelMe JSON files and images.",
    )
    parser.add_argument(
        "--output-dir",
        default="call_entrance_pose_grid/dataset_convert",
        help="Output directory for YOLO Pose dataset.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio (default 0.2 = 20%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split.",
    )
    return parser.parse_args()


def find_image_for_json(json_path: Path) -> Optional[Path]:
    """根据 JSON 文件查找对应的图片文件。

    优先使用 JSON 中的 imagePath，如果找不到则尝试同名不同后缀。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 尝试从 JSON 中获取 imagePath
    image_path_str = data.get("imagePath", "")
    if image_path_str:
        # 尝试绝对路径
        image_path = Path(image_path_str)
        if image_path.is_absolute() and image_path.exists():
            return image_path

        # 尝试相对于 JSON 文件的路径
        image_path = json_path.parent / image_path_str
        if image_path.exists():
            return image_path

    # 尝试同名不同后缀
    stem = json_path.stem
    for ext in IMAGE_EXTS:
        candidate = json_path.parent / f"{stem}{ext}"
        if candidate.exists():
            return candidate

    return None


def parse_rotation_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """将 rotation 类型的 4 个点转换成外接矩形 bbox。

    Args:
        points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        (x1, y1, x2, y2): 外接矩形的左上角和右下角坐标
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    return x1, y1, x2, y2


def parse_rectangle_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
    """将 rectangle 类型的点转换成 bbox。

    LabelMe 的 rectangle 可能是：
    - 2 个点：对角线的两个点 [[x1, y1], [x2, y2]]
    - 4 个点：四个角点（顺时针或逆时针）

    Args:
        points: [[x1, y1], [x2, y2], ...] (2 或 4 个点)

    Returns:
        (x1, y1, x2, y2): 矩形的左上角和右下角坐标
    """
    if len(points) == 2:
        # 传统 rectangle：2 个对角点
        x1 = min(points[0][0], points[1][0])
        y1 = min(points[0][1], points[1][1])
        x2 = max(points[0][0], points[1][0])
        y2 = max(points[0][1], points[1][1])
    else:
        # LabelMe 2.x 的 rectangle：4 个角点，取外接矩形
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs)
        y2 = max(ys)

    return x1, y1, x2, y2


def parse_point(points: List[List[float]]) -> Tuple[float, float]:
    """解析 point 类型的坐标。

    Args:
        points: [[x, y]]

    Returns:
        (x, y): 点的坐标
    """
    return points[0][0], points[0][1]


def parse_labelme_json(json_path: Path) -> Optional[GridPoseObject]:
    """解析 LabelMe JSON 文件，提取 oil1 bbox 和 empty/full/tip 关键点。

    Args:
        json_path: LabelMe JSON 文件路径

    Returns:
        GridPoseObject 或 None（如果缺少必需的标注）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])

    # 提取 oil1 bbox
    oil1_bbox = None
    for shape in shapes:
        label = shape.get("label", "")
        if label == "oil1":
            shape_type = shape.get("shape_type", "")
            points = shape.get("points", [])

            if shape_type == "rotation":
                # rotation 类型：4 个点，取外接矩形
                if len(points) == 4:
                    oil1_bbox = parse_rotation_bbox(points)
            elif shape_type == "rectangle":
                # rectangle 类型：2 个点或 4 个点
                if len(points) in (2, 4):
                    oil1_bbox = parse_rectangle_bbox(points)
            break

    if oil1_bbox is None:
        return None

    # 提取关键点
    keypoints = {}
    for shape in shapes:
        label = shape.get("label", "")
        if label in KEYPOINT_ORDER:
            shape_type = shape.get("shape_type", "")
            if shape_type == "point":
                points = shape.get("points", [])
                if len(points) == 1:
                    keypoints[label] = parse_point(points)

    # 检查是否所有关键点都存在
    for kpt_name in KEYPOINT_ORDER:
        if kpt_name not in keypoints:
            return None

    return GridPoseObject(bbox=oil1_bbox, keypoints=keypoints)


def bbox_to_yolo(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """将像素坐标的 bbox (x1, y1, x2, y2) 转换成 YOLO 格式 (cx, cy, w, h)，并归一化到 [0, 1]。

    Args:
        bbox: (x1, y1, x2, y2) 像素坐标
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        (cx, cy, w, h): YOLO 格式的归一化坐标
    """
    x1, y1, x2, y2 = bbox

    # 计算中心点和宽高（像素坐标）
    cx_px = (x1 + x2) / 2.0
    cy_px = (y1 + y2) / 2.0
    w_px = x2 - x1
    h_px = y2 - y1

    # 归一化到 [0, 1]
    cx = cx_px / img_width
    cy = cy_px / img_height
    w = w_px / img_width
    h = h_px / img_height

    return cx, cy, w, h


def normalize_keypoint(
    keypoint: Tuple[float, float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float]:
    """将像素坐标的关键点归一化到 [0, 1]。

    Args:
        keypoint: (x, y) 像素坐标
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        (x, y): 归一化后的坐标
    """
    x, y = keypoint
    return x / img_width, y / img_height


def format_yolo_pose_label(
    bbox_yolo: Tuple[float, float, float, float],
    keypoints_norm: Dict[str, Tuple[float, float]],
) -> str:
    """格式化 YOLO Pose 标签行。

    格式：class cx cy w h empty_x empty_y full_x full_y tip_x tip_y

    Args:
        bbox_yolo: (cx, cy, w, h) 归一化后的 bbox
        keypoints_norm: {name: (x, y)} 归一化后的关键点

    Returns:
        YOLO Pose 标签字符串
    """
    values = [0]  # class 固定为 0
    values.extend(bbox_yolo)

    # 按固定顺序添加关键点
    for kpt_name in KEYPOINT_ORDER:
        values.extend(keypoints_norm[kpt_name])

    # 格式化：class 为整数，其他为浮点数（6 位小数）
    return " ".join([str(int(values[0]))] + [f"{v:.6f}" for v in values[1:]])


def process_json_file(
    json_path: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
) -> Optional[Dict]:
    """处理单个 JSON 文件，生成 YOLO Pose 标签并复制图片。

    Args:
        json_path: LabelMe JSON 文件路径
        output_images_dir: 输出图片目录
        output_labels_dir: 输出标签目录

    Returns:
        处理成功返回 metadata 字典，失败返回 None
    """
    # 查找对应的图片
    image_path = find_image_for_json(json_path)
    if image_path is None:
        return None

    # 读取图片尺寸
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    img_height, img_width = image.shape[:2]

    # 解析 JSON
    pose_obj = parse_labelme_json(json_path)
    if pose_obj is None:
        return None

    # 转换 bbox 到 YOLO 格式并归一化
    bbox_yolo = bbox_to_yolo(pose_obj.bbox, img_width, img_height)

    # 归一化关键点
    keypoints_norm = {
        name: normalize_keypoint(kpt, img_width, img_height)
        for name, kpt in pose_obj.keypoints.items()
    }

    # 格式化标签行
    label_line = format_yolo_pose_label(bbox_yolo, keypoints_norm)

    # 复制图片
    output_image_path = output_images_dir / image_path.name
    shutil.copy2(image_path, output_image_path)

    # 写标签文件
    label_filename = json_path.stem + ".txt"
    output_label_path = output_labels_dir / label_filename
    output_label_path.write_text(label_line + "\n", encoding="utf-8")

    # 返回 metadata
    return {
        "json_file": json_path.name,
        "image_file": image_path.name,
        "label_file": label_filename,
        "image_size": [img_width, img_height],
    }


def split_train_val(
    json_files: List[Path],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    """将 JSON 文件列表按比例划分为 train 和 val。

    Args:
        json_files: JSON 文件路径列表
        val_ratio: 验证集比例（0.0 ~ 1.0）
        seed: 随机种子

    Returns:
        (train_files, val_files): 训练集和验证集文件列表
    """
    random.seed(seed)
    shuffled = json_files.copy()
    random.shuffle(shuffled)

    val_count = int(len(shuffled) * val_ratio)
    val_files = shuffled[:val_count]
    train_files = shuffled[val_count:]

    return train_files, val_files


def write_data_yaml(
    output_dir: Path,
    train_count: int,
    val_count: int,
) -> None:
    """生成 YOLO Pose 数据集的 data.yaml 配置文件。

    Args:
        output_dir: 输出目录
        train_count: 训练集样本数
        val_count: 验证集样本数
    """
    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": [3, 2],  # 3 个关键点，每个 2 维 (x, y)
        "flip_idx": [0, 1, 2],  # 不做左右翻转映射
        "names": {0: "grid_pose"},
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"Generated data.yaml:")
    print(f"  Train: {train_count} samples")
    print(f"  Val: {val_count} samples")


def main() -> None:
    args = parse_args()

    json_dir = Path(args.json_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not json_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {json_dir}")

    print("=" * 70)
    print("Converting LabelMe Grid Gauge Annotations to YOLO Pose Dataset")
    print("=" * 70)
    print(f"Input directory: {json_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    print()

    # 收集所有 JSON 文件
    json_files = sorted(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files.")

    if len(json_files) == 0:
        print("No JSON files found. Exiting.")
        return

    # 划分 train / val
    train_files, val_files = split_train_val(json_files, args.val_ratio, args.seed)
    print(f"Split: {len(train_files)} train, {len(val_files)} val")
    print()

    # 创建输出目录
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 处理文件
    metadata = {"train": [], "val": []}
    missing_report = {"train": [], "val": []}

    for split, files in [("train", train_files), ("val", val_files)]:
        print(f"Processing {split} set...")
        output_images_dir = output_dir / split / "images"
        output_labels_dir = output_dir / split / "labels"

        for json_path in files:
            result = process_json_file(json_path, output_images_dir, output_labels_dir)
            if result is not None:
                result["split"] = split
                metadata[split].append(result)
            else:
                missing_report[split].append(json_path.name)

        print(f"  {split}: {len(metadata[split])} samples processed, "
              f"{len(missing_report[split])} skipped")

    print()

    # 写 data.yaml
    write_data_yaml(
        output_dir,
        len(metadata["train"]),
        len(metadata["val"]),
    )

    # 写 metadata
    metadata_path = output_dir / "grid_pose_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to: {metadata_path}")

    # 写 missing report
    missing_report_path = output_dir / "missing_grid_pose_report.json"
    with open(missing_report_path, "w", encoding="utf-8") as f:
        json.dump(missing_report, f, ensure_ascii=False, indent=2)
    print(f"Saved missing report to: {missing_report_path}")

    print()
    print("=" * 70)
    print("Conversion completed successfully!")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"  train/images: {len(metadata['train'])} images")
    print(f"  train/labels: {len(metadata['train'])} labels")
    print(f"  val/images: {len(metadata['val'])} images")
    print(f"  val/labels: {len(metadata['val'])} labels")
    print("=" * 70)


if __name__ == "__main__":
    main()
