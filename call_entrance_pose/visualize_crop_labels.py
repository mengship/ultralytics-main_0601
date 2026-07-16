#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化裁剪后的 YOLO Pose 标签，用于检查标签是否正确重映射。

python call_entrance_pose/visualize_crop_labels.py \
  --image "/Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/ultralytics-main_0601/call_entrance_pose/dataset_convert_crop/train/images/260626_CCC7152.jpg" \
  --label "/Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/ultralytics-main_0601/call_entrance_pose/dataset_convert_crop/train/labels/260626_CCC7152.txt"

增加角度计算和可视化，参考 predict_pose_fuel.py 的逻辑。
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


KEYPOINT_ORDER = ("center", "tip", "empty", "full")
KEYPOINT_COLORS = {
    "center": (255, 255, 255),  # 白色
    "tip": (0, 0, 255),         # 红色
    "empty": (255, 0, 0),       # 蓝色
    "full": (0, 200, 0),        # 绿色
}
TAU = math.tau


def parse_label_line(line: str) -> dict:
    """解析 YOLO Pose 标签行。"""
    values = line.strip().split()
    if len(values) < 13:
        raise ValueError(f"Invalid label format: {line}")

    nums = [float(v) for v in values[:13]]
    cls = int(round(nums[0]))
    box = {
        "cx": nums[1],
        "cy": nums[2],
        "w": nums[3],
        "h": nums[4],
    }

    keypoints = {}
    key_values = nums[5:13]
    for i, name in enumerate(KEYPOINT_ORDER):
        keypoints[name] = (key_values[i * 2], key_values[i * 2 + 1])

    return {"cls": cls, "box": box, "keypoints": keypoints}


def denorm_point(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    """将归一化坐标转换为像素坐标。"""
    return int(x * width), int(y * height)


def angle_of(point: Tuple[float, float], center: Tuple[float, float]) -> float:
    """计算点相对于中心的角度。"""
    return math.atan2(point[1] - center[1], point[0] - center[0])


def wrap_positive(angle: float) -> float:
    """Normalize angle to [0, 2π)."""
    value = angle % TAU
    return value + TAU if value < 0 else value


def clockwise_angle(start: float, end: float) -> float:
    """Calculate angle from start to end in clockwise direction."""
    return wrap_positive(end - start)


def counterclockwise_angle(start: float, end: float) -> float:
    """Calculate angle from start to end in counterclockwise direction."""
    return wrap_positive(start - end)


def angle_in_direction(start: float, target: float, direction: str) -> float:
    """Calculate angle from start to target along the specified direction."""
    if direction == "clockwise":
        return clockwise_angle(start, target)
    elif direction == "counterclockwise":
        return counterclockwise_angle(start, target)
    else:
        raise ValueError(f"Unsupported direction: {direction}")


def choose_effective_direction(
    empty_angle: float, full_angle: float, tip_angle: float
) -> Tuple[str, float]:
    """Choose the direction where tip lies between empty and full."""
    cw_full = clockwise_angle(empty_angle, full_angle)
    ccw_full = counterclockwise_angle(empty_angle, full_angle)
    cw_tip = clockwise_angle(empty_angle, tip_angle)
    ccw_tip = counterclockwise_angle(empty_angle, tip_angle)

    cw_valid = (cw_full > 1e-6) and (0 <= cw_tip <= cw_full)
    ccw_valid = (ccw_full > 1e-6) and (0 <= ccw_tip <= ccw_full)

    if cw_valid and ccw_valid:
        return ("clockwise", cw_full) if cw_full >= ccw_full else ("counterclockwise", ccw_full)
    elif cw_valid:
        return "clockwise", cw_full
    elif ccw_valid:
        return "counterclockwise", ccw_full
    else:
        return ("clockwise", cw_full) if cw_full >= ccw_full else ("counterclockwise", ccw_full)


def compute_fuel_ratio(points: Dict[str, Tuple[float, float]]) -> dict:
    """计算油量比例和角度信息。"""
    center = points["center"]
    tip_angle = angle_of(points["tip"], center)
    empty_angle = angle_of(points["empty"], center)
    full_angle = angle_of(points["full"], center)

    used_direction, max_angle = choose_effective_direction(empty_angle, full_angle, tip_angle)

    if max_angle < 1e-6:
        fuel_ratio = 0.0
        raw_fuel_ratio = 0.0
        tip_angle_progress = 0.0
        clamped = False
    else:
        tip_angle_progress = angle_in_direction(empty_angle, tip_angle, used_direction)
        raw_fuel_ratio = tip_angle_progress / max_angle
        fuel_ratio = max(0.0, min(1.0, raw_fuel_ratio))
        clamped = fuel_ratio != raw_fuel_ratio

    return {
        "fuel_ratio": fuel_ratio,
        "raw_fuel_ratio": raw_fuel_ratio,
        "clamped": clamped,
        "direction": used_direction,
        "span_deg": math.degrees(max_angle),
        "offset_deg": math.degrees(tip_angle_progress),
        "tip_angle": tip_angle,
        "empty_angle": empty_angle,
        "full_angle": full_angle,
    }


def put_label_with_outline(
    image: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    """Draw readable text on noisy dashboard images."""
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_clockwise_arc(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    start_deg: float,
    sweep_deg: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw a clockwise-positive arc in image coordinates."""
    remaining = max(0.0, min(360.0, sweep_deg))
    current = start_deg % 360.0

    while remaining > 1e-3:
        step = min(remaining, 360.0 - current)
        if step <= 1e-3:
            current = 0.0
            continue

        cv2.ellipse(
            image,
            center,
            (radius, radius),
            0,
            current,
            current + step,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        remaining -= step
        current = 0.0


def draw_angle_annotation(
    image: np.ndarray,
    center: Tuple[int, int],
    start_angle: float,
    target_angle: float,
    direction: str,
    radius: int,
    color: Tuple[int, int, int],
    label: str,
    label_offset: Tuple[int, int] = (0, 0),
) -> None:
    """Draw and label one angle arc from empty to tip/full."""
    sweep = angle_in_direction(start_angle, target_angle, direction)
    if sweep <= 1e-6:
        return

    if direction == "clockwise":
        cv_start_deg = math.degrees(wrap_positive(start_angle))
        mid_angle = start_angle + sweep / 2.0
    else:
        cv_start_deg = math.degrees(wrap_positive(target_angle))
        mid_angle = start_angle - sweep / 2.0

    draw_clockwise_arc(image, center, radius, cv_start_deg, math.degrees(sweep), color, thickness=2)

    text_radius = radius + 14
    text_x = int(center[0] + math.cos(mid_angle) * text_radius + label_offset[0])
    text_y = int(center[1] + math.sin(mid_angle) * text_radius + label_offset[1])
    put_label_with_outline(image, label, (text_x, text_y), color, scale=0.48)


def draw_bbox(img: np.ndarray, box: dict, color=(128, 128, 128), thickness=1):
    """绘制 bbox（灰色细线）。"""
    h, w = img.shape[:2]
    cx, cy, bw, bh = box["cx"], box["cy"], box["w"], box["h"]

    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def visualize_label(image_path: Path, label_path: Path, output_path: Path):
    """可视化单个标签，包括角度信息。"""
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

    # 归一化关键点转换为像素坐标
    points_normalized = parsed["keypoints"]
    points_pixel = {
        name: denorm_point(x, y, w, h) for name, (x, y) in points_normalized.items()
    }

    # 计算油量比例和角度
    metrics = compute_fuel_ratio(points_normalized)

    # 创建副本用于绘制
    vis_img = img.copy()

    # 1. 绘制 bbox（灰色细线）
    draw_bbox(vis_img, parsed["box"], color=(128, 128, 128), thickness=1)

    # 2. 绘制从 center 到各个关键点的连线
    center_pt = points_pixel["center"]
    for name in ["tip", "empty", "full"]:
        cv2.line(vis_img, center_pt, points_pixel[name], KEYPOINT_COLORS[name], 2)

    # 3. 绘制关键点
    for name, (px, py) in points_pixel.items():
        color = KEYPOINT_COLORS[name]
        cv2.circle(vis_img, (px, py), 5, color, -1)
        cv2.putText(
            vis_img, name, (px + 6, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    # 4. 绘制角度弧线
    center_float = points_normalized["center"]
    empty_angle = metrics["empty_angle"]
    tip_angle = metrics["tip_angle"]
    full_angle = metrics["full_angle"]
    direction = metrics["direction"]

    # 计算弧线半径
    distances = [
        math.hypot(
            points_normalized[name][0] - center_float[0],
            points_normalized[name][1] - center_float[1],
        ) * w  # 转换到像素空间
        for name in ("tip", "empty", "full")
    ]
    max_distance = max([d for d in distances if d > 1.0] or [40.0])
    tip_arc_radius = max(18, int(max_distance * 0.38))
    full_arc_radius = max(tip_arc_radius + 12, int(max_distance * 0.58))

    # 绘制 empty->tip 弧线（青色）
    draw_angle_annotation(
        vis_img,
        center_pt,
        empty_angle,
        tip_angle,
        direction,
        tip_arc_radius,
        (255, 255, 0),  # 青色
        f"empty->tip {metrics['offset_deg']:.1f}°",
        label_offset=(-8, 14),
    )

    # 绘制 empty->full 弧线（黄色）
    draw_angle_annotation(
        vis_img,
        center_pt,
        empty_angle,
        full_angle,
        direction,
        full_arc_radius,
        (0, 255, 255),  # 黄色
        f"empty->full {metrics['span_deg']:.1f}°",
        label_offset=(8, -10),
    )

    # 5. 添加油量比例信息
    ratio = metrics["fuel_ratio"]
    raw_ratio = metrics["raw_fuel_ratio"]
    label = f"fuel={ratio * 100:.1f}% raw={raw_ratio * 100:.1f}% {direction}"
    put_label_with_outline(vis_img, label, (10, 30), (0, 255, 255), scale=0.6)

    # 6. 添加图片信息
    info_text = f"{image_path.name} | {w}x{h}"
    put_label_with_outline(vis_img, info_text, (10, h - 15), (255, 255, 255), scale=0.45)

    # 保存结果
    cv2.imwrite(str(output_path), vis_img)
    print(f"Saved visualization to: {output_path}")

    # 打印详细信息
    print(f"\nMetrics:")
    print(f"  Fuel ratio: {ratio * 100:.1f}% (raw: {raw_ratio * 100:.1f}%)")
    print(f"  Direction: {direction}")
    print(f"  Empty->Full span: {metrics['span_deg']:.1f}°")
    print(f"  Empty->Tip offset: {metrics['offset_deg']:.1f}°")
    print(f"  Clamped: {metrics['clamped']}")

    return vis_img


def main():
    parser = argparse.ArgumentParser(description="可视化裁剪后的 YOLO Pose 标签（含角度）")
    parser.add_argument(
        "--image",
        required=True,
        help="图片路径",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="标签文件路径",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出图片路径（默认在原图同目录下生成 _vis.jpg）",
    )

    args = parser.parse_args()

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

    print("=" * 60)
    print("Visualizing Cropped YOLO Pose Label")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print(f"Output: {output_path}")
    print()

    # 读取并显示标签内容
    label_content = label_path.read_text(encoding="utf-8").strip()
    print(f"Label content:")
    print(f"  {label_content}")
    print()

    # 可视化
    visualize_label(image_path, label_path, output_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
