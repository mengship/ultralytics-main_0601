#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一二阶段油表预测（原图版本）：同时识别指针油表和格子油表

与 predict_fuel_two_stage.py 的区别：
- Stage 2 直接在原图上进行 Pose 预测，而不是裁剪小图
- 关键点坐标都在原图空间
- 可视化也在原图上绘制

Stage 1: YOLO 检测模型识别原图中的油表框和类别
Stage 2: 在原图上使用对应的 Pose 模型预测关键点（类别路由）

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

输出坐标均为原图空间。
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
    angle_of,
    angle_in_direction,
    wrap_positive,
    put_label_with_outline,
    draw_angle_annotation,
)


POINTER_KEYPOINT_ORDER = ("center", "tip", "empty", "full")
GRID_KEYPOINT_ORDER = ("empty", "full", "tip")


def extract_points_with_conf(result, index: int) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """从 Pointer Pose 结果中提取关键点和置信度

    Pointer 关键点顺序: 0.center, 1.tip, 2.empty, 3.full

    Returns:
        (points, confidences): 关键点坐标字典和置信度字典
    """
    if result.keypoints is None or result.keypoints.xy is None:
        raise ValueError("No keypoints in result")

    keypoints = result.keypoints.xy[index].detach().cpu().numpy()
    if keypoints.shape[0] < len(POINTER_KEYPOINT_ORDER):
        raise ValueError(f"Expected {len(POINTER_KEYPOINT_ORDER)} keypoints, got {keypoints.shape[0]}")

    # 提取坐标
    points = {
        name: (float(keypoints[i][0]), float(keypoints[i][1]))
        for i, name in enumerate(POINTER_KEYPOINT_ORDER)
    }

    # 提取置信度
    confidences = {}
    if result.keypoints.conf is not None:
        confs = result.keypoints.conf[index].detach().cpu().numpy()
        if len(confs) >= len(POINTER_KEYPOINT_ORDER):
            confidences = {
                name: float(confs[i])
                for i, name in enumerate(POINTER_KEYPOINT_ORDER)
            }
    else:
        # keypoints.conf 不存在，使用默认值
        print(f"警告: keypoints.conf 为 None，无法获取置信度")

    return points, confidences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一二阶段油表预测（原图版本）：指针 + 格子"
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
        "--direction",
        choices=("max_full_span", "tip_side", "auto", "clockwise", "counterclockwise"),
        default="max_full_span",
        help="指针油表角度方向规则（仅用于 pointer）",
    )
    parser.add_argument(
        "--output-csv",
        default="call_entrance_pose_grid/fuel_two_stage_fullimage_predictions.csv",
        help="输出 CSV 文件路径",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="保存可视化结果（在原图上）",
    )
    parser.add_argument(
        "--vis-dir",
        default="call_entrance_pose_grid/fuel_vis_two_stage_fullimage",
        help="可视化结果保存目录",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="推理设备（如 '0', 'cpu', 'mps'）",
    )
    parser.add_argument(
        "--min-keypoint-conf",
        type=float,
        default=0.6,
        help="最小关键点置信度阈值（所有关键点必须 >= 此值）",
    )
    return parser.parse_args()


# ========== Grid 相关函数 ==========

def extract_grid_points(result, index: int) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
    """从 Grid Pose 结果中提取关键点和置信度

    Grid 关键点顺序: 0.empty, 1.full, 2.tip

    Returns:
        (points, confidences): 关键点坐标字典和置信度字典
    """
    if result.keypoints is None or result.keypoints.xy is None:
        raise ValueError("No keypoints in result")

    keypoints = result.keypoints.xy[index].detach().cpu().numpy()
    if keypoints.shape[0] < len(GRID_KEYPOINT_ORDER):
        raise ValueError(f"Expected {len(GRID_KEYPOINT_ORDER)} keypoints, got {keypoints.shape[0]}")

    # 提取坐标
    points = {
        name: (float(keypoints[i][0]), float(keypoints[i][1]))
        for i, name in enumerate(GRID_KEYPOINT_ORDER)
    }

    # 提取置信度
    confidences = {}
    if result.keypoints.conf is not None:
        confs = result.keypoints.conf[index].detach().cpu().numpy()
        if len(confs) >= len(GRID_KEYPOINT_ORDER):
            confidences = {
                name: float(confs[i])
                for i, name in enumerate(GRID_KEYPOINT_ORDER)
            }
    else:
        # keypoints.conf 不存在，使用默认值
        print(f"警告: keypoints.conf 为 None，无法获取置信度")

    return points, confidences


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


def draw_grid_prediction_on_image(
    image: np.ndarray,
    det_box: Tuple[float, float, float, float],
    points: Dict[str, Tuple[float, float]],
    metrics: Dict[str, float | str | bool],
    output_path: Path,
) -> None:
    """在原图上绘制格子油表预测结果"""
    vis_img = image.copy()

    # 绘制检测框（黄色虚线）
    x1, y1, x2, y2 = [int(v) for v in det_box]
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)

    colors = {
        "empty": (255, 0, 0),      # Blue
        "full": (0, 200, 0),       # Green
        "tip": (0, 0, 255),        # Red
    }

    # 绘制关键点
    for name, point in points.items():
        xy = tuple(int(v) for v in point)
        cv2.circle(vis_img, xy, 8, colors[name], -1)
        cv2.circle(vis_img, xy, 10, (255, 255, 255), 2)
        cv2.putText(
            vis_img, name, (xy[0] + 12, xy[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2
        )

    # 绘制线段
    empty_xy = tuple(int(v) for v in points["empty"])
    full_xy = tuple(int(v) for v in points["full"])
    tip_xy = tuple(int(v) for v in points["tip"])

    # empty -> full 线段（黄色）
    cv2.line(vis_img, empty_xy, full_xy, (255, 255, 0), 3)

    # empty -> tip 线段（青色）
    cv2.line(vis_img, empty_xy, tip_xy, (0, 255, 255), 3)

    # 添加燃料比例标签
    ratio = float(metrics["fuel_ratio"])
    raw_ratio = float(metrics["raw_fuel_ratio"])
    total_dist = float(metrics["total_distance"])
    current_dist = float(metrics["current_distance"])

    label = f"GRID fuel={ratio * 100:.1f}% raw={raw_ratio * 100:.1f}%"
    put_label_with_outline(vis_img, label, (x1, y1 - 40), (0, 255, 255), scale=0.8)

    # 添加距离信息
    dist_label = f"dist: {current_dist:.1f} / {total_dist:.1f}"
    put_label_with_outline(vis_img, dist_label, (x1, y1 - 10), (255, 255, 0), scale=0.6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)


def draw_pointer_prediction_on_image(
    image: np.ndarray,
    det_box: Tuple[float, float, float, float],
    points: Dict[str, Tuple[float, float]],
    metrics: Dict[str, float | str | bool],
    output_path: Path,
) -> None:
    """在原图上绘制指针油表预测结果"""
    vis_img = image.copy()

    # 绘制检测框（黄色虚线）
    x1, y1, x2, y2 = [int(v) for v in det_box]
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 255), 2, lineType=cv2.LINE_AA)

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
        cv2.circle(vis_img, xy, 8, colors[name], -1)
        cv2.circle(vis_img, xy, 10, (255, 255, 255), 2)
        cv2.putText(
            vis_img, name, (xy[0] + 12, xy[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[name], 2
        )

    # 从 center 绘制线段到各关键点
    for name in ("tip", "empty", "full"):
        xy = tuple(int(v) for v in points[name])
        cv2.line(vis_img, center, xy, colors[name], 3)

    # 计算角度并绘制弧线
    ratio = float(metrics["fuel_ratio"])
    raw_ratio = float(metrics["raw_fuel_ratio"])
    direction = str(metrics["direction"])
    center_float = points["center"]

    empty_angle = angle_of(points["empty"], center_float)
    tip_angle = angle_of(points["tip"], center_float)
    full_angle = angle_of(points["full"], center_float)

    # 计算弧线半径（基于检测框大小）
    box_width = x2 - x1
    box_height = y2 - y1
    box_size = math.sqrt(box_width * box_height)

    tip_arc_radius = max(30, int(box_size * 0.15))
    full_arc_radius = max(tip_arc_radius + 20, int(box_size * 0.25))

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
        label_offset=(-10, 18),
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
        label_offset=(10, -12),
    )

    # 添加燃料比例标签
    label = f"POINTER fuel={ratio * 100:.1f}% raw={raw_ratio * 100:.1f}% {direction}"
    put_label_with_outline(vis_img, label, (x1, y1 - 10), (0, 255, 255), scale=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)


def process_image(
    image_path: Path,
    det_model: YOLO,
    pointer_pose_model: YOLO,
    grid_pose_model: YOLO,
    args: argparse.Namespace,
) -> dict:
    """处理单张图片的二阶段流程（原图版本）

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
        "center_conf": None,
        "tip_conf": None,
        "empty_conf": None,
        "full_conf": None,
        "min_keypoint_conf": None,
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
        device=args.device,
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

    # Stage 2: 直接在原图上进行 Pose 预测
    pose_results = pose_model.predict(
        str(image_path),
        conf=args.pose_conf,
        imgsz=args.imgsz,
        verbose=False,
        device=args.device,
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
            points, confidences = extract_points_with_conf(pose_result, pose_idx)

            # 填充关键点置信度
            result["center_conf"] = confidences.get("center")
            result["tip_conf"] = confidences.get("tip")
            result["empty_conf"] = confidences.get("empty")
            result["full_conf"] = confidences.get("full")

            # 检查所有关键点置信度
            if confidences:
                min_conf = min(confidences.values())
                result["min_keypoint_conf"] = min_conf

                if min_conf < args.min_keypoint_conf:
                    result["status"] = "low_keypoint_confidence"
                    # 仍然填充坐标，但不计算油量
                    result["center_x"] = float(points["center"][0])
                    result["center_y"] = float(points["center"][1])
                    result["empty_x"] = float(points["empty"][0])
                    result["empty_y"] = float(points["empty"][1])
                    result["full_x"] = float(points["full"][0])
                    result["full_y"] = float(points["full"][1])
                    result["tip_x"] = float(points["tip"][0])
                    result["tip_y"] = float(points["tip"][1])
                    return result

            # 置信度检查通过，计算油量
            metrics = compute_fuel_ratio(points, direction=args.direction)

            # 填充指针专用字段
            result["direction"] = str(metrics["direction"])
            result["span_deg"] = float(metrics["span_deg"])
            result["offset_deg"] = float(metrics["offset_deg"])
            result["center_x"] = float(points["center"][0])
            result["center_y"] = float(points["center"][1])

        else:
            # 格子油表：使用距离计算
            points, confidences = extract_grid_points(pose_result, pose_idx)

            # 填充关键点置信度（格子油表没有 center）
            result["empty_conf"] = confidences.get("empty")
            result["full_conf"] = confidences.get("full")
            result["tip_conf"] = confidences.get("tip")

            # 检查所有关键点置信度
            if confidences:
                min_conf = min(confidences.values())
                result["min_keypoint_conf"] = min_conf

                if min_conf < args.min_keypoint_conf:
                    result["status"] = "low_keypoint_confidence"
                    # 仍然填充坐标，但不计算油量
                    result["empty_x"] = float(points["empty"][0])
                    result["empty_y"] = float(points["empty"][1])
                    result["full_x"] = float(points["full"][0])
                    result["full_y"] = float(points["full"][1])
                    result["tip_x"] = float(points["tip"][0])
                    result["tip_y"] = float(points["tip"][1])
                    return result

            # 置信度检查通过，计算油量
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

    # 填充关键点坐标（原图坐标系）
    result["empty_x"] = float(points["empty"][0])
    result["empty_y"] = float(points["empty"][1])
    result["full_x"] = float(points["full"][0])
    result["full_y"] = float(points["full"][1])
    result["tip_x"] = float(points["tip"][0])
    result["tip_y"] = float(points["tip"][1])

    # 保存可视化结果（在原图上）
    if args.save_vis:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"{image_path.stem}_vis.jpg"

        det_box = (x1, y1, x2, y2)

        if is_pointer:
            draw_pointer_prediction_on_image(
                image,
                det_box,
                points,
                metrics,
                vis_path,
            )
        else:
            draw_grid_prediction_on_image(
                image,
                det_box,
                points,
                metrics,
                vis_path,
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
        "center_conf",
        "tip_conf",
        "empty_conf",
        "full_conf",
        "min_keypoint_conf",
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
    print("统一二阶段油表预测（原图版本）：指针 + 格子")
    print("=" * 70)
    print(f"检测模型: {args.det_model}")
    print(f"指针 Pose 模型: {args.pointer_pose_model}")
    print(f"格子 Pose 模型: {args.grid_pose_model}")
    print(f"数据源: {source}")
    print(f"指针方向规则: {args.direction}")
    print(f"输出 CSV: {args.output_csv}")
    if args.save_vis:
        print(f"可视化目录: {args.vis_dir}")
    print("=" * 70)
    print()

    # 加载模型
    print("正在加载模型...")
    det_model = YOLO(args.det_model)
    pointer_pose_model = YOLO(args.pointer_pose_model)
    grid_pose_model = YOLO(args.grid_pose_model)

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
