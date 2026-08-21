#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Crop YOLO Pose dataset inside the annotated gauge boxes.

This script builds a cropped version of an existing YOLO Pose dataset:
    original full image -> crop inside the annotated box -> new YOLO Pose dataset

It is intended for a two-stage pipeline:
    1) YOLO detector finds the fuel gauge box
    2) YOLO Pose model runs on the cropped box image

Keypoint order:
    0 center
    1 tip
    2 empty
    3 full

Label format (YOLO Pose):
    class cx cy w h center_x center_y tip_x tip_y empty_x empty_y full_x full_y
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml


KEYPOINT_ORDER = ("center", "tip", "empty", "full")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class PoseObject:
    """单个 YOLO Pose 目标，包含类别、bbox 和关键点。"""
    cls: int
    box: Tuple[float, float, float, float]  # cx, cy, w, h (normalized)
    keypoints: Dict[str, Tuple[float, float]]  # {name: (x, y)} normalized
    visibilities: Optional[Dict[str, int]] = None  # {name: visibility} 可选的可见性标记


def default_data_path() -> Path:
    """默认输入数据集路径。"""
    return Path(__file__).resolve().parent / "dataset_convert" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop YOLO Pose dataset inside annotated boxes.")
    parser.add_argument(
        "--data",
        default=str(default_data_path()),
        help="Source YOLO Pose data.yaml path.",
    )
    parser.add_argument(
        "--output-dir",
        default="call_entrance_pose/dataset_convert_crop",
        help="Output directory for the cropped YOLO Pose dataset.",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.05,
        help="Extra padding ratio around each box before cropping. 0.05 means 5%% padding.",
    )
    return parser.parse_args()


def clamp01(value: float) -> float:
    """将浮点数截断到 [0, 1] 范围内。"""
    return max(0.0, min(1.0, value))


def load_yaml(path: Path) -> dict:
    """读取 YOLO 数据集的 data.yaml，里面包含 train / val / path 等信息。"""
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
    """把 data.yaml 里的相对路径解析成绝对路径，便于后面直接读图和读标签。"""
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
    """根据 label 文件名反查图片文件，支持多种常见图片后缀。"""
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(
        p for p in images_dir.glob(f"{stem}.*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return matches[0] if matches else None


def parse_label_line(line: str) -> Optional[PoseObject]:
    """解析单行 YOLO Pose 标签：
    支持两种格式：
    1. 不带可见性: class cx cy w h center_x center_y tip_x tip_y empty_x empty_y full_x full_y
    2. 带可见性:   class cx cy w h center_x center_y v1 tip_x tip_y v2 empty_x empty_y v3 full_x full_y v4
    """
    values = line.strip().split()

    # 判断格式：13个值（不带v）或 17个值（带v）
    if len(values) == 17:
        # 带可见性标记的格式
        try:
            cls = int(float(values[0]))
            box = tuple(float(v) for v in values[1:5])  # type: ignore[assignment]
            # 跳过可见性标记，只提取坐标
            keypoints = {
                "center": (float(values[5]), float(values[6])),   # v在values[7]
                "tip":    (float(values[8]), float(values[9])),   # v在values[10]
                "empty":  (float(values[11]), float(values[12])), # v在values[13]
                "full":   (float(values[14]), float(values[15])), # v在values[16]
            }
            visibilities = {
                "center": int(float(values[7])),
                "tip":    int(float(values[10])),
                "empty":  int(float(values[13])),
                "full":   int(float(values[16])),
            }
            return PoseObject(cls=cls, box=box, keypoints=keypoints, visibilities=visibilities)
        except (ValueError, IndexError):
            return None
    elif len(values) >= 13:
        # 不带可见性标记的格式（原有逻辑）
        try:
            nums = [float(v) for v in values[:13]]
            cls = int(round(nums[0]))
            box = tuple(nums[1:5])  # type: ignore[assignment]
            key_values = nums[5:13]
            keypoints = {
                name: (key_values[i * 2], key_values[i * 2 + 1])
                for i, name in enumerate(KEYPOINT_ORDER)
            }
            return PoseObject(cls=cls, box=box, keypoints=keypoints, visibilities=None)
        except (ValueError, IndexError):
            return None
    else:
        return None


def load_label_file(label_path: Path) -> List[PoseObject]:
    """从标签文件读取所有 YOLO Pose 目标（每个油表一行）。"""
    objects: List[PoseObject] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parsed = parse_label_line(raw_line)
        if parsed is not None:
            objects.append(parsed)
    return objects


def denormalize_box(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """把归一化的 cx/cy/w/h 还原成像素坐标的 xyxy，方便做裁剪。"""
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return x1, y1, x2, y2


def expand_and_clip_box(
    box_xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
    padding: float,
) -> Tuple[int, int, int, int]:
    """在原始标注框基础上增加一点 padding，再裁剪到图像范围内。

    这样可以防止把边缘重要信息裁掉。
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


def remap_box_to_crop(
    box_xyxy: Tuple[float, float, float, float],
    crop_xyxy: Tuple[int, int, int, int],
) -> Optional[Tuple[float, float, float, float]]:
    """把原图上的 bbox 重新映射到 crop 坐标系里，并转回 YOLO 所需的 cx/cy/w/h。

    步骤：
    1. 把 xyxy 从原图坐标转换到 crop 坐标
    2. 归一化到 [0, 1]
    3. 转换成 cx/cy/w/h 格式

    Returns:
        (cx, cy, w, h) 归一化坐标，如果bbox无效则返回None
    """
    x1, y1, x2, y2 = box_xyxy
    cx1, cy1, cx2, cy2 = crop_xyxy
    crop_w = max(1.0, float(cx2 - cx1))
    crop_h = max(1.0, float(cy2 - cy1))

    # 从原图坐标转到 crop 坐标，并归一化
    nx1 = (x1 - cx1) / crop_w
    ny1 = (y1 - cy1) / crop_h
    nx2 = (x2 - cx1) / crop_w
    ny2 = (y2 - cy1) / crop_h

    left = min(nx1, nx2)
    top = min(ny1, ny2)
    right = max(nx1, nx2)
    bottom = max(ny1, ny2)

    box_cx = clamp01((left + right) / 2.0)
    box_cy = clamp01((top + bottom) / 2.0)
    box_w = clamp01(right - left)
    box_h = clamp01(bottom - top)

    # 检查bbox是否有效：宽度和高度必须大于最小阈值
    if box_w < 0.01 or box_h < 0.01:
        return None

    return box_cx, box_cy, box_w, box_h


def remap_keypoints_to_crop(
    keypoints: Dict[str, Tuple[float, float]],
    orig_width: int,
    orig_height: int,
    crop_xyxy: Tuple[int, int, int, int],
) -> Dict[str, Tuple[float, float]]:
    """关键点也要跟着 crop 一起平移并重新归一化，这样训练时和裁剪图对齐。

    步骤：
    1. 把归一化关键点还原到原图像素坐标
    2. 转换到 crop 坐标系
    3. 重新归一化到 [0, 1]
    """
    cx1, cy1, cx2, cy2 = crop_xyxy
    crop_w = max(1.0, float(cx2 - cx1))
    crop_h = max(1.0, float(cy2 - cy1))

    remapped: Dict[str, Tuple[float, float]] = {}
    for name in KEYPOINT_ORDER:
        # 原图归一化坐标
        norm_x, norm_y = keypoints[name]
        # 还原到原图像素坐标
        pixel_x = norm_x * orig_width
        pixel_y = norm_y * orig_height
        # 转到 crop 坐标系并重新归一化
        new_x = clamp01((pixel_x - cx1) / crop_w)
        new_y = clamp01((pixel_y - cy1) / crop_h)
        remapped[name] = (new_x, new_y)
    return remapped


def format_label_row(obj: PoseObject) -> str:
    """写回 YOLO Pose 标签行，顺序必须和训练时保持一致。

    支持两种格式：
    1. 不带可见性: class cx cy w h x1 y1 x2 y2 x3 y3 x4 y4
    2. 带可见性:   class cx cy w h x1 y1 v1 x2 y2 v2 x3 y3 v3 x4 y4 v4
    """
    parts = [str(obj.cls)]

    # bbox
    parts.extend([f"{v:.6f}" for v in obj.box])

    # keypoints
    if obj.visibilities is not None:
        # 带可见性标记的格式: x y v
        for name in KEYPOINT_ORDER:
            x, y = obj.keypoints[name]
            v = obj.visibilities[name]
            parts.extend([f"{x:.6f}", f"{y:.6f}", str(v)])
    else:
        # 不带可见性标记的格式: x y
        for name in KEYPOINT_ORDER:
            x, y = obj.keypoints[name]
            parts.extend([f"{x:.6f}", f"{y:.6f}"])

    return " ".join(parts)


def ensure_empty_dir(path: Path) -> None:
    """清空输出目录，避免旧的 crop 数据混进去。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unique_stem(base_stem: str, used: set[str]) -> str:
    """同一张原图可能会裁出多个样本，这里保证输出文件名不重复。"""
    candidate = base_stem
    suffix = 1
    while candidate in used:
        candidate = f"{base_stem}_{suffix:02d}"
        suffix += 1
    used.add(candidate)
    return candidate


def write_yaml(output_dir: Path, source_yaml: dict) -> None:
    """输出一个新的 data.yaml，供裁剪后的数据集直接用于训练。"""
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": source_yaml.get("kpt_shape", [4, 2]),
        "flip_idx": source_yaml.get("flip_idx", [0, 1, 2, 3]),
        "names": source_yaml.get("names", {0: "oil_pose"}),
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)


def crop_split(
    split: str,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    crop_padding: float,
    used_names: set[str],
) -> Dict[str, int]:
    """逐个 split 做裁剪：train 一遍、val 一遍，保持原始划分不变。

    规则：
    - 每张图只有一个油表框（标签文件只有 1 行）
    - 读取 bbox cx cy w h
    - 还原成像素坐标 xyxy
    - 裁剪时稍微裁大一点（crop_padding）
    - 将 bbox 和关键点全部重映射到 crop 坐标系
    - 再重新归一化到 [0, 1]
    """
    out_images_dir = output_dir / split / "images"
    out_labels_dir = output_dir / split / "labels"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_labels": 0,
        "cropped": 0,
        "skipped_missing_image": 0,
        "skipped_not_single_line": 0,
        "skipped_invalid_crop": 0,
    }

    label_files = sorted(labels_dir.glob("*.txt"))
    for label_path in label_files:
        stats["total_labels"] += 1

        # 每张图只有一个油表框，标签文件应该只有 1 行
        stem = label_path.stem
        image_path = find_image_file(images_dir, stem)
        if image_path is None:
            stats["skipped_missing_image"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            stats["skipped_missing_image"] += 1
            continue

        objects = load_label_file(label_path)
        if len(objects) != 1:
            # 不是单行标签，跳过
            stats["skipped_not_single_line"] += 1
            continue

        obj = objects[0]
        height, width = image.shape[:2]

        # 步骤 1: 把归一化的 bbox 还原成像素坐标
        box_xyxy = denormalize_box(obj.box, width, height)

        # 步骤 2: 扩展 padding 并裁剪到图像边界
        crop_xyxy = expand_and_clip_box(box_xyxy, width, height, crop_padding)
        cx1, cy1, cx2, cy2 = crop_xyxy

        # 步骤 3: 用 OpenCV 裁剪图像（注意顺序是 [y1:y2, x1:x2]）
        crop = image[cy1:cy2, cx1:cx2].copy()
        if crop.size == 0:
            stats["skipped_invalid_crop"] += 1
            continue

        # 步骤 4: 把 bbox 重映射到 crop 坐标系
        crop_box = remap_box_to_crop(box_xyxy, crop_xyxy)
        if crop_box is None:
            # bbox 过小或无效，跳过此样本
            stats["skipped_invalid_crop"] += 1
            continue

        # 步骤 5: 把关键点重映射到 crop 坐标系
        crop_keypoints = remap_keypoints_to_crop(obj.keypoints, width, height, crop_xyxy)

        crop_obj = PoseObject(
            cls=obj.cls,
            box=crop_box,
            keypoints=crop_keypoints,
            visibilities=obj.visibilities,  # 保留可见性标记
        )

        # 生成唯一文件名
        crop_stem = unique_stem(stem, used_names)
        crop_image_name = f"{crop_stem}{image_path.suffix.lower()}"
        crop_label_name = f"{crop_stem}.txt"

        # 保存裁剪图和对应标签
        cv2.imwrite(str(out_images_dir / crop_image_name), crop)
        (out_labels_dir / crop_label_name).write_text(
            format_label_row(crop_obj) + "\n", encoding="utf-8"
        )

        stats["cropped"] += 1

    return stats


def build_crop_dataset(
    data_yaml_path: Path,
    output_dir: Path,
    crop_padding: float,
) -> Dict[str, Dict[str, int]]:
    """读取原始数据集信息，确认 train / val 目录，并生成新的 crop 数据集。"""
    source_yaml = load_yaml(data_yaml_path)
    split_dirs = resolve_split_dirs(data_yaml_path)

    source_root = resolve_path(
        data_yaml_path.parent, source_yaml.get("path", data_yaml_path.parent)
    )
    if output_dir.resolve() == source_root.resolve():
        raise ValueError(
            f"output-dir must not be the same as source dataset root: {source_root}"
        )

    # 每次都重建输出目录，保证 crop 数据是干净的
    ensure_empty_dir(output_dir)
    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    summary: Dict[str, Dict[str, int]] = {}

    for split in ("train", "val"):
        # 保持训练集和验证集各自独立裁剪，避免数据泄漏
        images_dir = split_dirs[split]
        labels_dir = labels_dir_from_images(images_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images dir for split '{split}': {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing labels dir for split '{split}': {labels_dir}")

        summary[split] = crop_split(
            split=split,
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            crop_padding=crop_padding,
            used_names=used_names,
        )

    write_yaml(output_dir, source_yaml)
    with open(output_dir / "crop_summary.json", "w", encoding="utf-8") as f:
        # 额外写一个摘要文件，便于后面核对裁剪样本数量
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    args = parse_args()
    data_yaml_path = Path(args.data).expanduser().resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()

    print("=" * 60)
    print("Cropping YOLO Pose Dataset")
    print("=" * 60)
    print(f"Source data.yaml: {data_yaml_path}")
    print(f"Output directory: {output_dir}")
    print(f"Crop padding: {args.crop_padding} ({args.crop_padding * 100:.1f}%)")
    print()

    summary = build_crop_dataset(data_yaml_path, output_dir, args.crop_padding)

    print("=" * 60)
    print("Cropping completed")
    print("=" * 60)
    for split in ("train", "val"):
        stats = summary.get(split, {})
        print(f"\n{split.upper()}:")
        print(f"  Total labels: {stats.get('total_labels', 0)}")
        print(f"  Cropped successfully: {stats.get('cropped', 0)}")
        print(f"  Skipped (missing image): {stats.get('skipped_missing_image', 0)}")
        print(f"  Skipped (not single line): {stats.get('skipped_not_single_line', 0)}")
        print(f"  Skipped (invalid crop): {stats.get('skipped_invalid_crop', 0)}")

    print(f"\nOutput dataset structure:")
    print(f"  {output_dir}/")
    print(f"    ├── data.yaml")
    print(f"    ├── crop_summary.json")
    print(f"    ├── train/images/")
    print(f"    ├── train/labels/")
    print(f"    ├── val/images/")
    print(f"    └── val/labels/")
    print()


if __name__ == "__main__":
    main()
