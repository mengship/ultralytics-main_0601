#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一二阶段油表预测：同时识别指针油表和格子油表

Stage 1: YOLO 检测模型识别原图中的油表框和类别
Stage 2: 根据类别选择对应的 Pose 模型预测关键点

类别路由：
    class_id == 0: 指针油表 -> pointer pose 模型 -> 角度计算
    class_id == 1: 格子油表 -> grid pose 模型 -> 距离计算
    其他: unsupported_cls

Pointer Pose 关键点顺序：
    0 center
    1 tip
    2 empty
    3 full

Grid Pose 关键点顺序：
    0 empty
    1 full
    2 tip

输出坐标均为裁剪小图空间，不映射回原图。
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# 导入 pointer 相关函数
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "call_entrance_pose"))
from predict_pose_fuel import (
    compute_fuel_ratio,
    best_detection_index,
    extract_points,
    angle_of,
    angle_in_direction,
    wrap_positive,
    put_label_with_outline,
    draw_angle_annotation,
)


POINTER_KEYPOINT_ORDER = ("center", "tip", "empty", "full")
GRID_KEYPOINT_ORDER = ("empty", "full", "tip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一二阶段油表预测：指针 + 格子"
    )
    parser.add_argument(
        "--det-model",
        required=True,
        help="第一阶段 YOLO 检测模型（识别油表框和类别）",
    )
    parser.add_argument(
        "--pointer-pose-model",
        required=True,
        help="指针油表 Pose 模型",
    )
    parser.add_argument(
        "--grid-pose-model",
        required=True,
        help="格子油表 Pose 模型",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="图片文件或目录",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="第一阶段检测置信度阈值",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=0.25,
        help="第二阶段 Pose 置信度阈值",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理图像尺寸",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.08,
        help="检测框裁剪前的扩大比例（如 0.08 = 8%%），防止裁掉关键区域",
    )
    parser.add_argument(
        "--direction",
        choices=("max_full_span", "tip_side", "auto", "clockwise", "counterclockwise"),
        default="max_full_span",
        help="指针油表角度方向规则（仅用于 pointer）",
    )
    parser.add_argument(
        "--output-csv",
        default="call_entrance_pose_grid/fuel_two_stage_predictions.csv",
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="保存可视化结果（在裁剪小图上）",
    )
    parser.add_argument(
        "--vis-dir",
        default="call_entrance_pose_grid/fuel_vis_two_stage",
        help="可视化结果保存目录",
    )
    parser.add_argument(
        "--crop-dir",
        default=None,
        help="可选：保存裁剪小图的目录",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备（如 '0', 'cpu', 'mps'）",
    )
    return parser.parse_args()


def expand_box(
    box_xyxy: Tuple[float, float, float, float],
    padding: float,
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """按 padding 比例扩大检测框，并裁剪到图像边界

    Args:
        box_xyxy: (x1, y1, x2, y2) 像素坐标
        padding: 扩大比例（如 0.08 = 8%）
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        (x1, y1, x2, y2) 整数坐标，裁剪到图像范围
    """
    x1, y1, x2, y2 = box_xyxy
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    pad_x = w * padding
    pad_y = h * padding

    x1_exp = max(0, x1 - pad_x)
    y1_exp = max(0, y1 - pad_y)
    x2_exp = min(img_width, x2 + pad_x)
    y2_exp = min(img_height, y2 + pad_y)

    ix1 = max(0, int(math.floor(x1_exp)))
    iy1 = max(0, int(math.floor(y1_exp)))
    ix2 = min(img_width, int(math.ceil(x2_exp)))
    iy2 = min(img_height, int(math.ceil(y2_exp)))

    # 确保裁剪区域有效
    if ix2 <= ix1:
        ix2 = min(img_width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(img_height, iy1 + 1)

    return ix1, iy1, ix2, iy2


# ========== Grid 相关函数 ==========

def extract_grid_points(result, index: int) -> Dict[str, Tuple[float, float]]:
    """从 Grid Pose 结果中提取关键点

    Grid 关键点顺序: 0.empty, 1.full, 2.tip
    """
    if result.keypoints is None or result.keypoints.xy is None:
        raise ValueError("No keypoints in result")

    keypoints = result.keypoints.xy[index].detach().cpu().numpy()
    if keypoints.shape[0] < len(GRID_KEYPOINT_ORDER):
        raise ValueError(f"Expected {len(GRID_KEYPOINT_ORDER)} keypoints, got {keypoints.shape[0]}")

    return {
        name: (float(keypoints[i][0]), float(keypoints[i][1]))
        for i, name in enumerate(GRID_KEYPOINT_ORDER)
    }


def compute_grid_fuel_ratio(points: Dict[str, Tuple[float, float]]) -> Dict[str, float | str | bool]:
    """计算格子油表燃料比例

    使用距离比例计算: distance(empty, tip) / distance(empty, full)

    Returns:
        包含 fuel_ratio、raw_fuel_ratio、clamped、total_distance、current_distance 等字段
    """
    empty = points["empty"]
    full = points["full"]
    tip = points["tip"]

    # 计算距离
    total_distance = math.hypot(full[0] - empty[0], full[1] - empty[1])
    current_distance = math.hypot(tip[0] - empty[0], tip[1] - empty[1])

    # 处理退化情况
    if total_distance < 1e-6:
        return {
            "fuel_ratio": 0.0,
            "raw_fuel_ratio": 0.0,
            "clamped": False,
            "total_distance": total_distance,
            "current_distance": current_distance,
        }

    # 计算比例
    raw_fuel_ratio = current_distance / total_distance
    fuel_ratio = max(0.0, min(1.0, raw_fuel_ratio))
    clamped = (fuel_ratio != raw_fuel_ratio)

    return {
        "fuel_ratio": fuel_ratio,
        "raw_fuel_ratio": raw_fuel_ratio,
        "clamped": clamped,
        "total_distance": total_distance,
        "current_distance": current_distance,
    }


def draw_grid_prediction_on_crop(
    crop_image: np.ndarray,
    points: Dict[str, Tuple[float, float]],
    metrics: Dict[str, float | str | bool],
    output_path: Path,
    original_filename: str,
) -> None:
    """在裁剪小图上绘制格子油表预测结果"""
    vis_img = crop_image.copy()

    colors = {
        "empty": (255, 0, 0),      # Blue
        "full": (0, 200, 0),       # Green
        "tip": (0, 0, 255),        # Red
    }

    # 绘制关键点
    for name, point in points.items():
        xy = tuple(int(v) for v in point)
        cv2.circle(vis_img, xy, 5, colors[name], -1)
        cv2.putText(
            vis_img, name, (xy[0] + 6, xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1
        )

    # 绘制线段
    empty_xy = tuple(int(v) for v in points["empty"])
    full_xy = tuple(int(v) for v in points["full"])
    tip_xy = tuple(int(v) for v in points["tip"])

    # empty -> full 线段（黄色）
    cv2.line(vis_img, empty_xy, full_xy, (255, 255, 0), 2)

    # empty -> tip 线段（青色）
    cv2.line(vis_img, empty_xy, tip_xy, (0, 255, 255), 2)

    # 添加燃料比例标签
    ratio = float(metrics["fuel_ratio"])
    raw_ratio = float(metrics["raw_fuel_ratio"])
    total_dist = float(metrics["total_distance"])
    current_dist = float(metrics["current_distance"])

    label = f"fuel={ratio * 100:.1f}% raw={raw_ratio * 100:.1f}% (grid)"
    put_label_with_outline(vis_img, label, (10, 25), (0, 255, 255), scale=0.6)

    # 添加距离信息
    dist_label = f"dist: {current_dist:.1f} / {total_dist:.1f}"
    put_label_with_outline(vis_img, dist_label, (10, 50), (255, 255, 0), scale=0.5)

    # 添加原始文件名
    info_text = f"From: {original_filename}"
    h = vis_img.shape[0]
    put_label_with_outline(vis_img, info_text, (10, h - 10), (255, 255, 255), scale=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)


def draw_pointer_prediction_on_crop(
    crop_image: np.ndarray,
    points: Dict[str, Tuple[float, float]],
    metrics: Dict[str, float | str | bool],
    output_path: Path,
    original_filename: str,
) -> None:
    """在裁剪小图上绘制指针油表预测结果"""
    vis_img = crop_image.copy()

    colors = {
        "center": (255, 255, 255),  # White
        "tip": (0, 0, 255),         # Red
        "empty": (255, 0, 0),       # Blue
        "full": (0, 200, 0),        # Green
    }

    center = tuple(int(v) for v in points["center"])

    # 绘制关键点
    for name, point in points.items():
        xy = tuple(int(v) for v in point)
        cv2.circle(vis_img, xy, 5, colors[name], -1)
        cv2.putText(
            vis_img, name, (xy[0] + 6, xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1
        )

    # 从 center 绘制线段到各关键点
    for name in ("tip", "empty", "full"):
        xy = tuple(int(v) for v in points[name])
        cv2.line(vis_img, center, xy, colors[name], 2)

    # 计算角度并绘制弧线
    ratio = float(metrics["fuel_ratio"])
    raw_ratio = float(metrics["raw_fuel_ratio"])
    direction = str(metrics["direction"])
    center_float = points["center"]

    empty_angle = angle_of(points["empty"], center_float)
    tip_angle = angle_of(points["tip"], center_float)
    full_angle = angle_of(points["full"], center_float)

    # 计算弧线半径
    distances = [
        math.hypot(points[name][0] - center_float[0], points[name][1] - center_float[1])
        for name in ("tip", "empty", "full")
    ]
    max_distance = max([d for d in distances if d > 1.0] or [40.0])
    tip_arc_radius = max(18, int(max_distance * 0.38))
    full_arc_radius = max(tip_arc_radius + 12, int(max_distance * 0.58))

    # 绘制 empty->tip 弧线（青色）
    draw_angle_annotation(
        vis_img,
        center,
        empty_angle,
        tip_angle,
        direction,
        tip_arc_radius,
        (0, 255, 255),  # Cyan
        f"empty->tip {float(metrics['offset_deg']):.1f}deg",
        label_offset=(-8, 14),
    )

    # 绘制 empty->full 弧线（黄色）
    draw_angle_annotation(
        vis_img,
        center,
        empty_angle,
        full_angle,
        direction,
        full_arc_radius,
        (255, 255, 0),  # Yellow
        f"empty->full {float(metrics['span_deg']):.1f}deg",
        label_offset=(8, -10),
    )

    # 添加燃料比例标签
    label = f"fuel={ratio * 100:.1f}% raw={raw_ratio * 100:.1f}% {direction}"
    put_label_with_outline(vis_img, label, (10, 25), (0, 255, 255), scale=0.6)

    # 添加原始文件名
    info_text = f"From: {original_filename}"
    h = vis_img.shape[0]
    put_label_with_outline(vis_img, info_text, (10, h - 10), (255, 255, 255), scale=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)


def process_image(
    image_path: Path,
    det_model: YOLO,
    pointer_pose_model: YOLO,
    grid_pose_model: YOLO,
    args: argparse.Namespace,
) -> dict:
    """处理单张图片的二阶段流程

    类别路由：
    - class_id == 0: 指针油表 -> pointer pose 模型 -> 角度计算
    - class_id == 1: 格子油表 -> grid pose 模型 -> 距离计算
    - 其他: unsupported_cls

    Returns:
        包含预测结果和状态的字典
    """
    result = {
        "image": image_path.name,
        "status": "ok",
        "fuel_type": None,
        "det_class": None,
        "det_conf": None,
        "det_x1": None,
        "det_y1": None,
        "det_x2": None,
        "det_y2": None,
        "crop_image": None,
        "pose_conf": None,
        "fuel_ratio": None,
        "raw_fuel_ratio": None,
        "fuel_percent": None,
        "clamped": None,
        "direction": None,
        "span_deg": None,
        "offset_deg": None,
        "total_distance": None,
        "current_distance": None,
        "center_x": None,
        "center_y": None,
        "empty_x": None,
        "empty_y": None,
        "full_x": None,
        "full_y": None,
        "tip_x": None,
        "tip_y": None,
    }

    # Stage 1: 检测
    image = cv2.imread(str(image_path))
    if image is None:
        result["status"] = "read_error"
        return result

    img_height, img_width = image.shape[:2]

    det_results = det_model.predict(
        str(image_path),
        conf=args.det_conf,
        imgsz=args.imgsz,
        verbose=False,
    )

    if not det_results or len(det_results) == 0:
        result["status"] = "no_det"
        return result

    det_result = det_results[0]
    det_idx = best_detection_index(det_result)

    if det_idx is None or det_result.boxes is None or len(det_result.boxes) == 0:
        result["status"] = "no_det"
        return result

    # 获取检测框和类别
    box = det_result.boxes.xyxy[det_idx].cpu().numpy()
    det_conf = float(det_result.boxes.conf[det_idx].cpu().numpy())
    det_class = int(det_result.boxes.cls[det_idx].cpu().numpy())
    x1, y1, x2, y2 = box

    result["det_conf"] = det_conf
    result["det_class"] = det_class
    result["det_x1"] = float(x1)
    result["det_y1"] = float(y1)
    result["det_x2"] = float(x2)
    result["det_y2"] = float(y2)

    # 类别路由：判断油表类型并选择对应模型
    if det_class == 0:
        # 指针油表
        result["fuel_type"] = "pointer"
        pose_model = pointer_pose_model
        is_pointer = True
    elif det_class == 1:
        # 格子油表
        result["fuel_type"] = "grid"
        pose_model = grid_pose_model
        is_pointer = False
    else:
        # 不支持的类别
        result["fuel_type"] = f"class_{det_class}"
        result["status"] = "unsupported_cls"
        return result

    # 扩大检测框并裁剪（按 padding 比例扩大，防止裁掉关键区域）
    crop_x1, crop_y1, crop_x2, crop_y2 = expand_box(
        (x1, y1, x2, y2),
        args.box_padding,
        img_width,
        img_height,
    )

    crop_image = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    if crop_image.size == 0:
        result["status"] = "invalid_crop"
        return result

    # 保存裁剪小图（可选）
    crop_filename = None
    if args.crop_dir:
        crop_dir = Path(args.crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_filename = f"{image_path.stem}_crop{image_path.suffix}"
        crop_path = crop_dir / crop_filename
        cv2.imwrite(str(crop_path), crop_image)
        result["crop_image"] = crop_filename

    # Stage 2: 在裁剪小图上进行 Pose 预测
    pose_results = pose_model.predict(
        crop_image,
        conf=args.pose_conf,
        imgsz=args.imgsz,
        verbose=False,
    )

    if not pose_results or len(pose_results) == 0:
        result["status"] = "no_pose"
        return result

    pose_result = pose_results[0]
    pose_idx = best_detection_index(pose_result)

    if pose_idx is None:
        result["status"] = "no_pose"
        return result

    # 提取关键点并计算燃料比例（根据类型分流）
    try:
        if is_pointer:
            # 指针油表：使用角度计算
            points = extract_points(pose_result, pose_idx)
            metrics = compute_fuel_ratio(points, direction=args.direction)

            # 填充指针专用字段
            result["direction"] = str(metrics["direction"])
            result["span_deg"] = float(metrics["span_deg"])
            result["offset_deg"] = float(metrics["offset_deg"])
            result["center_x"] = float(points["center"][0])
            result["center_y"] = float(points["center"][1])

        else:
            # 格子油表：使用距离计算
            points = extract_grid_points(pose_result, pose_idx)
            metrics = compute_grid_fuel_ratio(points)

            # 验证距离有效性
            if metrics["total_distance"] < 1e-6:
                result["status"] = "invalid_grid_distance"
                return result

            # 填充格子专用字段
            result["total_distance"] = float(metrics["total_distance"])
            result["current_distance"] = float(metrics["current_distance"])

    except (ValueError, IndexError) as e:
        result["status"] = "no_pose"
        return result

    pose_conf = float(pose_result.boxes.conf[pose_idx].cpu().numpy())
    result["pose_conf"] = pose_conf

    # 填充通用字段
    result["fuel_ratio"] = float(metrics["fuel_ratio"])
    result["raw_fuel_ratio"] = float(metrics["raw_fuel_ratio"])
    result["fuel_percent"] = float(metrics["fuel_ratio"]) * 100
    result["clamped"] = bool(metrics["clamped"])

    # 填充关键点坐标（裁剪小图坐标系）
    result["empty_x"] = float(points["empty"][0])
    result["empty_y"] = float(points["empty"][1])
    result["full_x"] = float(points["full"][0])
    result["full_y"] = float(points["full"][1])
    result["tip_x"] = float(points["tip"][0])
    result["tip_y"] = float(points["tip"][1])

    # 保存可视化结果（只在裁剪小图上）
    if args.save_vis:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"{image_path.stem}_vis.jpg"

        if is_pointer:
            draw_pointer_prediction_on_crop(
                crop_image,
                points,
                metrics,
                vis_path,
                image_path.name,
            )
        else:
            draw_grid_prediction_on_crop(
                crop_image,
                points,
                metrics,
                vis_path,
                image_path.name,
            )

    return result


def collect_images(source: Path) -> List[Path]:
    """收集所有图像文件"""
    if source.is_file():
        return [source]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    images = []
    for ext in image_exts:
        images.extend(source.glob(f"*{ext}"))
        images.extend(source.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def write_csv(results: List[dict], output_csv: Path) -> None:
    """将预测结果写入 CSV 文件"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "status",
        "fuel_type",
        "det_class",
        "det_conf",
        "det_x1",
        "det_y1",
        "det_x2",
        "det_y2",
        "crop_image",
        "pose_conf",
        "fuel_ratio",
        "raw_fuel_ratio",
        "fuel_percent",
        "clamped",
        "direction",
        "span_deg",
        "offset_deg",
        "total_distance",
        "current_distance",
        "center_x",
        "center_y",
        "empty_x",
        "empty_y",
        "full_x",
        "full_y",
        "tip_x",
        "tip_y",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    print("=" * 70)
    print("统一二阶段油表预测：指针 + 格子")
    print("=" * 70)
    print(f"检测模型: {args.det_model}")
    print(f"指针 Pose 模型: {args.pointer_pose_model}")
    print(f"格子 Pose 模型: {args.grid_pose_model}")
    print(f"数据源: {source}")
    print(f"检测框扩大比例: {args.box_padding} ({args.box_padding * 100:.1f}%)")
    print(f"指针方向规则: {args.direction}")
    print(f"输出 CSV: {args.output_csv}")
    if args.save_vis:
        print(f"可视化目录: {args.vis_dir}")
    if args.crop_dir:
        print(f"裁剪图目录: {args.crop_dir}")
    print("=" * 70)
    print()

    # 加载模型
    print("正在加载模型...")
    det_model = YOLO(args.det_model)
    pointer_pose_model = YOLO(args.pointer_pose_model)
    grid_pose_model = YOLO(args.grid_pose_model)

    if args.device:
        det_model.to(args.device)
        pointer_pose_model.to(args.device)
        grid_pose_model.to(args.device)

    print("模型加载完成。")
    print()

    # 收集图像
    images = collect_images(source)
    print(f"找到 {len(images)} 张图像。")
    print()

    # 处理图像
    results = []
    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] 处理 {image_path.name}...", end=" ")
        result = process_image(image_path, det_model, pointer_pose_model, grid_pose_model, args)
        results.append(result)

        status = result["status"]
        if status == "ok":
            fuel_type = result["fuel_type"]
            print(f"✓ {fuel_type} fuel={result['fuel_percent']:.1f}%")
        else:
            print(f"✗ {status}")

    print()

    # 写入 CSV
    output_csv = Path(args.output_csv).expanduser().resolve()
    write_csv(results, output_csv)
    print(f"结果已保存到: {output_csv}")

    # 统计汇总
    print()
    print("=" * 70)
    print("统计汇总")
    print("=" * 70)

    status_counts = {}
    type_counts = {"pointer": 0, "grid": 0}

    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        if status == "ok":
            fuel_type = result["fuel_type"]
            if fuel_type in type_counts:
                type_counts[fuel_type] += 1

    print("状态分布:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    ok_count = status_counts.get("ok", 0)
    print(f"\n类型分布:")
    print(f"  pointer: {type_counts['pointer']}")
    print(f"  grid: {type_counts['grid']}")

    print(f"\n成功率: {ok_count}/{len(images)} ({ok_count / len(images) * 100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
