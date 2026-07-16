#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化裁剪后的 YOLO Pose 格子油表标签。

python call_entrance_pose_grid/visualize_crop_labels.py \
    --image call_entrance_pose_grid/dataset_convert_crop/train/images/xxx.jpg \
    --label call_entrance_pose_grid/dataset_convert_crop/train/labels/xxx.txt

python call_entrance_pose_grid/visualize_crop_labels.py \
    --image call_entrance_pose_grid/dataset_convert_crop/train/images/260622_NGC7836.jpg \
    --label call_entrance_pose_grid/dataset_convert_crop/train/labels/260622_NGC7836.txt

用于检查裁剪后的 bbox 和关键点是否正确重映射到裁剪图坐标系。
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


KEYPOINT_ORDER = ("empty", "full", "tip")
KEYPOINT_COLORS = {
    "empty": (0, 255, 0),    # 绿色 - 空油
    "full": (0, 0, 255),     # 红色 - 满油
    "tip": (255, 0, 0),      # 蓝色 - 当前油量
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化裁剪后的 YOLO Pose 格子油表标签"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="裁剪后的图片路径",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="裁剪后的标签文件路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出图片路径（默认在原图同目录下生成 _vis.jpg）",
    )
    return parser.parse_args()


def parse_label_line(line: str) -> Optional[Dict]:
    """解析 YOLO Pose 标签行。

    格式：class cx cy w h empty_x empty_y full_x full_y tip_x tip_y

    Returns:
        字典包含 class, bbox, keypoints，或 None
    """
    values = line.strip().split()
    if len(values) < 11:
        return None

    try:
        nums = [float(v) for v in values[:11]]
        return {
            "class": int(round(nums[0])),
            "bbox": {
                "cx": nums[1],
                "cy": nums[2],
                "w": nums[3],
                "h": nums[4],
            },
            "keypoints": {
                "empty": (nums[5], nums[6]),
                "full": (nums[7], nums[8]),
                "tip": (nums[9], nums[10]),
            },
        }
    except (ValueError, IndexError):
        return None


def denorm_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """将归一化坐标转换为像素坐标。"""
    return int(x * width), int(y * height)


def draw_bbox(img: np.ndarray, bbox: Dict, color=(128, 128, 128), thickness=2):
    """绘制 bbox。"""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def draw_keypoints_with_distance(
    img: np.ndarray,
    keypoints: Dict[str, Tuple[float, float]],
):
    """绘制关键点和距离线（不绘制文字标签）。"""
    h, w = img.shape[:2]

    # 转换为像素坐标
    points_px = {}
    for name, (x, y) in keypoints.items():
        px, py = denorm_point(x, y, w, h)
        points_px[name] = (px, py)

    # 绘制距离线
    # empty -> full (红色虚线)
    cv2.line(
        img,
        points_px["empty"],
        points_px["full"],
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    # empty -> tip (黄色实线)
    cv2.line(
        img,
        points_px["empty"],
        points_px["tip"],
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # 绘制关键点（不绘制文字标签）
    for name in KEYPOINT_ORDER:
        px, py = points_px[name]
        color = KEYPOINT_COLORS[name]

        # 绘制点
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.circle(img, (px, py), 10, (255, 255, 255), 2)

    # 计算距离比例
    empty_pt = points_px["empty"]
    full_pt = points_px["full"]
    tip_pt = points_px["tip"]

    total_distance = np.sqrt(
        (full_pt[0] - empty_pt[0]) ** 2 + (full_pt[1] - empty_pt[1]) ** 2
    )
    current_distance = np.sqrt(
        (tip_pt[0] - empty_pt[0]) ** 2 + (tip_pt[1] - empty_pt[1]) ** 2
    )

    if total_distance > 1e-6:
        fuel_ratio = current_distance / total_distance
        fuel_ratio = max(0.0, min(1.0, fuel_ratio))  # 截断到 [0, 1]
    else:
        fuel_ratio = 0.0

    return fuel_ratio, total_distance, current_distance


def put_text_with_bg(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.7,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
):
    """绘制带背景的文字。"""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 获取文字尺寸
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景
    x, y = pos
    cv2.rectangle(
        img,
        (x - 5, y - text_h - 5),
        (x + text_w + 5, y + baseline + 5),
        (0, 0, 0),
        -1,
    )

    # 绘制文字
    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)


def visualize_label(image_path: Path, label_path: Path, output_path: Path):
    """可视化单个标签。"""
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]

    # 读取标签
    label_text = label_path.read_text(encoding="utf-8").strip()
    if not label_text:
        raise ValueError(f"Empty label file: {label_path}")

    parsed = parse_label_line(label_text)
    if parsed is None:
        raise ValueError(f"Invalid label format: {label_text}")

    # 创建副本用于绘制
    vis_img = img.copy()

    # 1. 绘制 bbox
    draw_bbox(vis_img, parsed["bbox"], color=(128, 128, 128), thickness=2)

    # 2. 绘制关键点和距离线
    fuel_ratio, total_dist, current_dist = draw_keypoints_with_distance(
        vis_img, parsed["keypoints"]
    )

    # 保存结果（不添加文字信息）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)
    print(f"✓ Saved visualization to: {output_path}")

    # 打印详细信息
    print(f"\nLabel analysis:")
    print(f"  Fuel ratio: {fuel_ratio * 100:.1f}%")
    print(f"  Total distance (empty->full): {total_dist:.1f}px")
    print(f"  Current distance (empty->tip): {current_dist:.1f}px")
    print(f"  Bbox: cx={parsed['bbox']['cx']:.3f}, cy={parsed['bbox']['cy']:.3f}, "
          f"w={parsed['bbox']['w']:.3f}, h={parsed['bbox']['h']:.3f}")
    print(f"  Keypoints:")
    for name in KEYPOINT_ORDER:
        x, y = parsed["keypoints"][name]
        print(f"    {name}: ({x:.3f}, {y:.3f})")


def main():
    args = parse_args()

    image_path = Path(args.image).resolve()
    label_path = Path(args.label).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = image_path.parent / f"{image_path.stem}_vis.jpg"

    print("=" * 70)
    print("可视化裁剪后的 YOLO Pose 格子油表标签")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    visualize_label(image_path, label_path, output_path)
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
