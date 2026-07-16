#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crop YOLO Pose grid gauge dataset inside the annotated oil1 bounding boxes.

This script crops the full-image YOLO Pose dataset to focus on the grid gauge regions,
remapping both bounding boxes and keypoints to the cropped coordinate system.

Input:
    Full-image YOLO Pose dataset with grid gauge annotations
    Label format: class cx cy w h empty_x empty_y full_x full_y tip_x tip_y

Output:
    Cropped YOLO Pose dataset with remapped coordinates
    Same label format but coordinates are relative to cropped images

Usage:
    # Default settings
    python call_entrance_pose_grid/crop_yolo_grid_pose_dataset.py

    # Custom input and output
    python call_entrance_pose_grid/crop_yolo_grid_pose_dataset.py \
        --data call_entrance_pose_grid/dataset_convert/data.yaml \
        --output-dir call_entrance_pose_grid/dataset_convert_crop \
        --crop-padding 0.05

Keypoint order (fixed):
    0 empty
    1 full
    2 tip
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import yaml


KEYPOINT_ORDER = ("empty", "full", "tip")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop YOLO Pose grid gauge dataset inside annotated bounding boxes."
    )
    parser.add_argument(
        "--data",
        default="call_entrance_pose_grid/dataset_convert/data.yaml",
        help="Source YOLO Pose data.yaml path.",
    )
    parser.add_argument(
        "--output-dir",
        default="call_entrance_pose_grid/dataset_convert_crop",
        help="Output directory for the cropped YOLO Pose dataset.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.05,
        help="Extra padding ratio around bbox before cropping (default 0.05 = 5%%).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    """读取 YOLO 数据集的 data.yaml。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content: {path}")
    return data


def resolve_path(base: Path, value: object) -> Path:
    """解析相对路径或绝对路径。"""
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (base / path).resolve()


def resolve_split_dirs(data_yaml_path: Path) -> Dict[str, Path]:
    """把 data.yaml 里的相对路径解析成绝对路径。"""
    data = load_yaml(data_yaml_path)
    base = resolve_path(data_yaml_path.parent, data.get("path", data_yaml_path.parent))

    split_dirs: Dict[str, Path] = {}
    for split in ("train", "val"):
        if split not in data:
            raise KeyError(f"Missing '{split}' in data.yaml: {data_yaml_path}")
        split_dirs[split] = resolve_path(base, data[split])
    return split_dirs


def labels_dir_from_images(images_dir: Path) -> Path:
    """兼容常见 YOLO 目录结构：images/ 对应 labels/。"""
    candidates = []
    if images_dir.name == "images":
        candidates.append(images_dir.parent / "labels")
    candidates.append(images_dir.with_name("labels"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_image_file(images_dir: Path, stem: str) -> Optional[Path]:
    """根据 label 文件名反查图片文件。"""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(
        p for p in images_dir.glob(f"{stem}.*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return matches[0] if matches else None


def parse_label_line(line: str) -> Optional[Tuple[int, Tuple[float, ...], Tuple[float, ...]]]:
    """解析单行 YOLO Pose 标签。

    格式：class cx cy w h empty_x empty_y full_x full_y tip_x tip_y

    Returns:
        (class_id, bbox, keypoints) 或 None
        bbox: (cx, cy, w, h) 归一化坐标
        keypoints: (empty_x, empty_y, full_x, full_y, tip_x, tip_y) 归一化坐标
    """
    values = line.strip().split()
    if len(values) < 11:  # class + 4 bbox + 6 keypoints
        return None

    try:
        nums = [float(v) for v in values[:11]]
        cls = int(round(nums[0]))
        bbox = tuple(nums[1:5])  # cx, cy, w, h
        keypoints = tuple(nums[5:11])  # empty_x, empty_y, full_x, full_y, tip_x, tip_y
        return cls, bbox, keypoints
    except (ValueError, IndexError):
        return None


def denormalize_bbox(
    bbox: Tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """将归一化的 bbox (cx, cy, w, h) 还原成像素坐标 (x1, y1, x2, y2)。

    Args:
        bbox: (cx, cy, w, h) 归一化到 [0, 1]
        img_width: 图像宽度（像素）
        img_height: 图像高度（像素）

    Returns:
        (x1, y1, x2, y2): 像素坐标
    """
    cx, cy, w, h = bbox
    cx_px = cx * img_width
    cy_px = cy * img_height
    w_px = w * img_width
    h_px = h * img_height

    x1 = cx_px - w_px / 2.0
    y1 = cy_px - h_px / 2.0
    x2 = cx_px + w_px / 2.0
    y2 = cy_px + h_px / 2.0

    return x1, y1, x2, y2


def expand_and_clip_box(
    box_xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    """在 bbox 基础上增加 padding，再裁剪到图像范围内。

    这样可以防止裁剪时丢失边缘格子。

    Args:
        box_xyxy: (x1, y1, x2, y2) 像素坐标
        width: 图像宽度
        height: 图像高度
        padding: padding 比例（例如 0.05 表示 5%）

    Returns:
        (x1, y1, x2, y2): 扩展后的整数像素坐标，裁剪到图像边界内
    """
    x1, y1, x2, y2 = box_xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * padding
    pad_y = bh * padding

    crop_x1 = max(0.0, x1 - pad_x)
    crop_y1 = max(0.0, y1 - pad_y)
    crop_x2 = min(float(width), x2 + pad_x)
    crop_y2 = min(float(height), y2 + pad_y)

    ix1 = max(0, int(math.floor(crop_x1)))
    iy1 = max(0, int(math.floor(crop_y1)))
    ix2 = min(width, int(math.ceil(crop_x2)))
    iy2 = min(height, int(math.ceil(crop_y2)))

    if ix2 <= ix1:
        ix2 = min(width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(height, iy1 + 1)

    return ix1, iy1, ix2, iy2


def remap_bbox_to_crop(
    bbox_xyxy: Tuple[float, float, float, float],
    crop_xyxy: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    """将原图像素坐标的 bbox 重映射到 crop 坐标系，并转换为 YOLO 格式 (cx, cy, w, h)。

    步骤：
    1. 将原图 xyxy 转换到 crop 坐标系
    2. 归一化到 [0, 1]（相对于 crop 尺寸）
    3. 转换为 cx, cy, w, h 格式

    Args:
        bbox_xyxy: (x1, y1, x2, y2) 原图像素坐标
        crop_xyxy: (cx1, cy1, cx2, cy2) crop 区域像素坐标

    Returns:
        (cx, cy, w, h): 归一化到 crop 尺寸的 YOLO 格式
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx1, cy1, cx2, cy2 = crop_xyxy
    crop_w = max(1.0, float(cx2 - cx1))
    crop_h = max(1.0, float(cy2 - cy1))

    # 转换到 crop 坐标系并归一化
    nx1 = (x1 - cx1) / crop_w
    ny1 = (y1 - cy1) / crop_h
    nx2 = (x2 - cx1) / crop_w
    ny2 = (y2 - cy1) / crop_h

    # 截断到 [0, 1]
    nx1 = max(0.0, min(1.0, nx1))
    ny1 = max(0.0, min(1.0, ny1))
    nx2 = max(0.0, min(1.0, nx2))
    ny2 = max(0.0, min(1.0, ny2))

    # 转换为 YOLO 格式
    cx = (nx1 + nx2) / 2.0
    cy = (ny1 + ny2) / 2.0
    w = nx2 - nx1
    h = ny2 - ny1

    return cx, cy, w, h


def remap_keypoints_to_crop(
    keypoints: Tuple[float, ...],
    orig_width: int,
    orig_height: int,
    crop_xyxy: Tuple[int, int, int, int],
) -> Tuple[float, ...]:
    """将原图归一化的关键点重映射到 crop 坐标系。

    步骤：
    1. 将归一化关键点还原到原图像素坐标
    2. 转换到 crop 坐标系
    3. 重新归一化到 [0, 1]（相对于 crop 尺寸）

    Args:
        keypoints: (empty_x, empty_y, full_x, full_y, tip_x, tip_y) 原图归一化坐标
        orig_width: 原图宽度
        orig_height: 原图高度
        crop_xyxy: (cx1, cy1, cx2, cy2) crop 区域像素坐标

    Returns:
        (empty_x, empty_y, full_x, full_y, tip_x, tip_y): crop 归一化坐标
    """
    cx1, cy1, cx2, cy2 = crop_xyxy
    crop_w = max(1.0, float(cx2 - cx1))
    crop_h = max(1.0, float(cy2 - cy1))

    remapped = []
    for i in range(0, len(keypoints), 2):
        # 原图归一化坐标
        norm_x = keypoints[i]
        norm_y = keypoints[i + 1]

        # 还原到原图像素坐标
        pixel_x = norm_x * orig_width
        pixel_y = norm_y * orig_height

        # 转到 crop 坐标系并重新归一化
        new_x = (pixel_x - cx1) / crop_w
        new_y = (pixel_y - cy1) / crop_h

        # 截断到 [0, 1]
        new_x = max(0.0, min(1.0, new_x))
        new_y = max(0.0, min(1.0, new_y))

        remapped.extend([new_x, new_y])

    return tuple(remapped)


def format_label_row(
    cls: int,
    bbox: Tuple[float, float, float, float],
    keypoints: Tuple[float, ...],
) -> str:
    """格式化 YOLO Pose 标签行。"""
    values = [float(cls)]
    values.extend(bbox)
    values.extend(keypoints)
    return " ".join([str(int(values[0]))] + [f"{v:.6f}" for v in values[1:]])


def ensure_empty_dir(path: Path) -> None:
    """清空输出目录。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def crop_split(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    crop_padding: float,
) -> Dict[str, int]:
    """裁剪指定 split 的数据。"""
    out_images_dir = output_dir / split / "images"
    out_labels_dir = output_dir / split / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_labels": 0,
        "cropped": 0,
        "skipped_missing_image": 0,
        "skipped_missing_label": 0,
        "skipped_not_single_line": 0,
        "skipped_invalid_label": 0,
        "skipped_invalid_crop": 0,
    }

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        stats["total_labels"] += 1

        # 查找对应图片
        stem = label_path.stem
        image_path = find_image_file(images_dir, stem)
        if image_path is None:
            stats["skipped_missing_image"] += 1
            continue

        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            stats["skipped_missing_image"] += 1
            continue

        img_height, img_width = image.shape[:2]

        # 读取标签
        label_text = label_path.read_text(encoding="utf-8").strip()
        lines = [line for line in label_text.splitlines() if line.strip()]

        # 每个标签文件应该只有 1 行
        if len(lines) != 1:
            stats["skipped_not_single_line"] += 1
            continue

        # 解析标签
        parsed = parse_label_line(lines[0])
        if parsed is None:
            stats["skipped_invalid_label"] += 1
            continue

        cls, bbox_norm, keypoints_norm = parsed

        # 步骤 1: 将归一化的 bbox 还原到原图像素坐标
        bbox_xyxy = denormalize_bbox(bbox_norm, img_width, img_height)

        # 步骤 2: 扩展 padding 并裁剪到图像边界
        crop_xyxy = expand_and_clip_box(bbox_xyxy, img_width, img_height, crop_padding)
        cx1, cy1, cx2, cy2 = crop_xyxy

        # 步骤 3: 裁剪图像
        crop_image = image[cy1:cy2, cx1:cx2].copy()
        if crop_image.size == 0:
            stats["skipped_invalid_crop"] += 1
            continue

        # 步骤 4: 重映射 bbox 到 crop 坐标系
        crop_bbox = remap_bbox_to_crop(bbox_xyxy, crop_xyxy)

        # 步骤 5: 重映射关键点到 crop 坐标系
        crop_keypoints = remap_keypoints_to_crop(
            keypoints_norm, img_width, img_height, crop_xyxy
        )

        # 保存裁剪图像和标签
        crop_image_name = f"{stem}{image_path.suffix.lower()}"
        crop_label_name = f"{stem}.txt"

        cv2.imwrite(str(out_images_dir / crop_image_name), crop_image)
        (out_labels_dir / crop_label_name).write_text(
            format_label_row(cls, crop_bbox, crop_keypoints) + "\n",
            encoding="utf-8",
        )

        stats["cropped"] += 1

    return stats


def write_yaml(output_dir: Path, source_yaml: dict) -> None:
    """生成新的 data.yaml。"""
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": source_yaml.get("kpt_shape", [3, 2]),
        "flip_idx": source_yaml.get("flip_idx", [0, 1, 2]),
        "names": source_yaml.get("names", {0: "grid_pose"}),
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()

    data_yaml_path = Path(args.data).expanduser().resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()

    print("=" * 70)
    print("Cropping YOLO Pose Grid Gauge Dataset")
    print("=" * 70)
    print(f"Source data.yaml: {data_yaml_path}")
    print(f"Output directory: {output_dir}")
    print(f"Crop padding: {args.crop_padding} ({args.crop_padding * 100:.1f}%)")
    print("=" * 70)
    print()

    # 加载源数据集信息
    source_yaml = load_yaml(data_yaml_path)
    split_dirs = resolve_split_dirs(data_yaml_path)

    # 清空输出目录
    ensure_empty_dir(output_dir)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # 裁剪每个 split
    summary = {}
    for split in ("train", "val"):
        print(f"Processing {split} set...")
        images_dir = split_dirs[split]
        labels_dir = labels_dir_from_images(images_dir)

        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images dir for '{split}': {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labels dir for '{split}': {labels_dir}")

        summary[split] = crop_split(
            split=split,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            crop_padding=args.crop_padding,
        )

    # 写 data.yaml
    write_yaml(output_dir, source_yaml)

    # 写 crop_summary.json
    with open(output_dir / "crop_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("Cropping completed")
    print("=" * 70)
    for split in ("train", "val"):
        stats = summary.get(split, {})
        print(f"\n{split.upper()}:")
        print(f"  Total labels: {stats.get('total_labels', 0)}")
        print(f"  Cropped successfully: {stats.get('cropped', 0)}")
        print(f"  Skipped (missing image): {stats.get('skipped_missing_image', 0)}")
        print(f"  Skipped (missing label): {stats.get('skipped_missing_label', 0)}")
        print(f"  Skipped (not single line): {stats.get('skipped_not_single_line', 0)}")
        print(f"  Skipped (invalid label): {stats.get('skipped_invalid_label', 0)}")
        print(f"  Skipped (invalid crop): {stats.get('skipped_invalid_crop', 0)}")

    print(f"\nOutput: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
