#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict fuel ratio from YOLO Pose keypoints.

Keypoint order:
    0 center
    1 tip
    2 empty
    3 full
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


KEYPOINT_ORDER = ("center", "tip", "empty", "full")
TAU = math.tau


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fuel ratio with a trained YOLO Pose model.")
    parser.add_argument("--model", required=True, help="Trained YOLO Pose model path.")
    parser.add_argument("--source", required=True, help="Image file or directory.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--direction",
        choices=("max_full_span", "tip_side", "auto", "clockwise", "counterclockwise"),
        default="max_full_span",
        help="Angle direction rule from empty to tip/full in image coordinates.",
    )
    parser.add_argument("--output-csv", default="call_entrance_pose/pose_fuel_predictions.csv")
    parser.add_argument("--save-vis", action="store_true", help="Save visualized predictions.")
    parser.add_argument("--vis-dir", default="call_entrance_pose/pose_vis")
    return parser.parse_args()


def angle_of(point: Tuple[float, float], center: Tuple[float, float]) -> float:
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


def choose_effective_direction(empty_angle: float, full_angle: float, tip_angle: float) -> Tuple[str, float]:
    """Choose the direction where tip lies between empty and full.

    Business rule: Choose the direction where empty->tip angle < empty->full angle,
    so that the tip is within the valid range [empty, full].

    If both directions satisfy this (unlikely), choose the one with larger full span.
    If neither satisfies, fall back to the direction with larger full span.

    Returns:
        (direction, max_angle): The effective direction and its corresponding angle.
    """
    cw_full = clockwise_angle(empty_angle, full_angle)
    ccw_full = counterclockwise_angle(empty_angle, full_angle)
    cw_tip = clockwise_angle(empty_angle, tip_angle)
    ccw_tip = counterclockwise_angle(empty_angle, tip_angle)

    # Check which directions have tip within [empty, full]
    cw_valid = (cw_full > 1e-6) and (0 <= cw_tip <= cw_full)
    ccw_valid = (ccw_full > 1e-6) and (0 <= ccw_tip <= ccw_full)

    if cw_valid and ccw_valid:
        # Both valid: choose the one with larger full span
        return ("clockwise", cw_full) if cw_full >= ccw_full else ("counterclockwise", ccw_full)
    elif cw_valid:
        return "clockwise", cw_full
    elif ccw_valid:
        return "counterclockwise", ccw_full
    else:
        # Neither valid (tip is outside [empty, full] in both directions)
        # Fall back to the direction with larger full span
        return ("clockwise", cw_full) if cw_full >= ccw_full else ("counterclockwise", ccw_full)


def choose_tip_side_direction(tip: float, empty: float, full: float) -> str:
    """Choose the side where the tip lies between empty and full.

    With image coordinates and atan2(y, x), increasing angle is clockwise.
    """
    candidates = []
    for candidate in ("clockwise", "counterclockwise"):
        span = angle_in_direction(empty, full, candidate)
        offset = angle_in_direction(empty, tip, candidate)
        if span > 1e-6:
            raw_ratio = offset / span
            if 0.0 <= raw_ratio <= 1.0:
                candidates.append((candidate, span, offset))

    if candidates:
        # Usually only one direction is physically valid. If both are valid because
        # tip is exactly at full, keep the direction with the shorter tip arc.
        return min(candidates, key=lambda item: item[2])[0]

    clockwise_offset = clockwise_angle(empty, tip)
    counterclockwise_offset = counterclockwise_angle(empty, tip)
    return "clockwise" if clockwise_offset <= counterclockwise_offset else "counterclockwise"


def compute_fuel_ratio(points: Dict[str, Tuple[float, float]], direction: str = "max_full_span") -> Dict[str, float | str | bool]:
    """Compute fuel ratio from keypoints.

    Business rules:
    1. Calculate empty->full and empty->tip angles in both directions
    2. Choose the direction where tip is between empty and full (ratio in [0, 1])
    3. If both directions are valid, choose the one with larger full span
    4. Calculate fuel_ratio = (empty->tip angle) / (empty->full angle)
    5. Clamp fuel_ratio to [0, 1], but keep raw_fuel_ratio for debugging
    """
    center = points["center"]
    tip_angle = angle_of(points["tip"], center)
    empty_angle = angle_of(points["empty"], center)
    full_angle = angle_of(points["full"], center)

    # Determine effective direction based on mode
    if direction == "max_full_span":
        # Use the direction where tip is between empty and full
        used_direction, max_angle = choose_effective_direction(empty_angle, full_angle, tip_angle)
    elif direction == "tip_side":
        # Use the direction where tip lies between empty and full
        used_direction = choose_tip_side_direction(tip_angle, empty_angle, full_angle)
        max_angle = angle_in_direction(empty_angle, full_angle, used_direction)
    elif direction == "auto":
        # Choose direction where tip is between empty and full (same as max_full_span)
        used_direction, max_angle = choose_effective_direction(empty_angle, full_angle, tip_angle)
    elif direction in ("clockwise", "counterclockwise"):
        # Use the specified fixed direction
        used_direction = direction
        max_angle = angle_in_direction(empty_angle, full_angle, used_direction)
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    # Calculate tip progress along the effective direction
    if max_angle < 1e-6:
        # Degenerate case: empty and full are at the same angle
        fuel_ratio = 0.0
        raw_fuel_ratio = 0.0
        tip_angle_progress = 0.0
        clamped = False
    else:
        tip_angle_progress = angle_in_direction(empty_angle, tip_angle, used_direction)
        raw_fuel_ratio = tip_angle_progress / max_angle
        fuel_ratio = max(0.0, min(1.0, raw_fuel_ratio))
        clamped = (fuel_ratio != raw_fuel_ratio)

    return {
        "fuel_ratio": fuel_ratio,
        "raw_fuel_ratio": raw_fuel_ratio,
        "clamped": clamped,
        "direction": used_direction,
        "span_deg": math.degrees(max_angle),
        "offset_deg": math.degrees(tip_angle_progress),
        "tip_deg": math.degrees(tip_angle),
        "empty_deg": math.degrees(empty_angle),
        "full_deg": math.degrees(full_angle),
    }


def best_detection_index(result) -> int | None:
    if result.boxes is None or len(result.boxes) == 0:
        return None
    conf = result.boxes.conf
    if conf is None:
        return 0
    return int(conf.argmax().item())


def extract_points(result, index: int) -> Dict[str, Tuple[float, float]]:
    if result.keypoints is None or result.keypoints.xy is None:
        raise ValueError("No keypoints in result")

    keypoints = result.keypoints.xy[index].detach().cpu().numpy()
    if keypoints.shape[0] < len(KEYPOINT_ORDER):
        raise ValueError(f"Expected {len(KEYPOINT_ORDER)} keypoints, got {keypoints.shape[0]}")

    return {
        name: (float(keypoints[i][0]), float(keypoints[i][1]))
        for i, name in enumerate(KEYPOINT_ORDER)
    }


def draw_prediction(
    image_path: str,
    points: Dict[str, Tuple[float, float]],
    metrics: Dict[str, float | str | bool],
    output_path: Path,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        return

    colors = {
        "center": (255, 255, 255),
        "tip": (0, 0, 255),
        "empty": (255, 0, 0),
        "full": (0, 200, 0),
    }
    center = tuple(int(v) for v in points["center"])

    for name, point in points.items():
        xy = tuple(int(v) for v in point)
        cv2.circle(image, xy, 5, colors[name], -1)
        cv2.putText(image, name, (xy[0] + 6, xy[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1)

    for name in ("tip", "empty", "full"):
        xy = tuple(int(v) for v in points[name])
        cv2.line(image, center, xy, colors[name], 2)

    ratio = float(metrics["fuel_ratio"])
    raw_ratio = float(metrics["raw_fuel_ratio"])
    direction = str(metrics["direction"])
    label = f"fuel={ratio * 100:.1f}% raw={raw_ratio * 100:.1f}% {direction}"
    cv2.putText(image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def write_csv(rows: Iterable[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "status",
        "confidence",
        "fuel_ratio",
        "raw_fuel_ratio",
        "fuel_percent",
        "clamped",
        "direction",
        "span_deg",
        "offset_deg",
        "tip_deg",
        "empty_deg",
        "full_deg",
        "center_x",
        "center_y",
        "tip_x",
        "tip_y",
        "empty_x",
        "empty_y",
        "full_x",
        "full_y",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    predict_kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
        "stream": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)
    rows: List[dict] = []
    vis_dir = Path(args.vis_dir)

    for result in results:
        row = {
            "image": result.path,
            "status": "no_detection",
            "confidence": "",
            "fuel_ratio": "",
            "raw_fuel_ratio": "",
            "fuel_percent": "",
            "clamped": "",
            "direction": "",
            "span_deg": "",
            "offset_deg": "",
            "tip_deg": "",
            "empty_deg": "",
            "full_deg": "",
            "center_x": "",
            "center_y": "",
            "tip_x": "",
            "tip_y": "",
            "empty_x": "",
            "empty_y": "",
            "full_x": "",
            "full_y": "",
        }

        index = best_detection_index(result)
        if index is None:
            rows.append(row)
            continue

        try:
            points = extract_points(result, index)
            metrics = compute_fuel_ratio(points, args.direction)
            confidence = float(result.boxes.conf[index].item()) if result.boxes.conf is not None else 0.0
            ratio = float(metrics["fuel_ratio"])

            row.update(
                {
                    "status": "ok",
                    "confidence": f"{confidence:.6f}",
                    "fuel_ratio": f"{ratio:.6f}",
                    "raw_fuel_ratio": f"{float(metrics['raw_fuel_ratio']):.6f}",
                    "fuel_percent": f"{ratio * 100:.2f}",
                    "clamped": str(bool(metrics["clamped"])),
                    "direction": metrics["direction"],
                    "span_deg": f"{float(metrics['span_deg']):.3f}",
                    "offset_deg": f"{float(metrics['offset_deg']):.3f}",
                    "tip_deg": f"{float(metrics['tip_deg']):.3f}",
                    "empty_deg": f"{float(metrics['empty_deg']):.3f}",
                    "full_deg": f"{float(metrics['full_deg']):.3f}",
                    "center_x": f"{points['center'][0]:.2f}",
                    "center_y": f"{points['center'][1]:.2f}",
                    "tip_x": f"{points['tip'][0]:.2f}",
                    "tip_y": f"{points['tip'][1]:.2f}",
                    "empty_x": f"{points['empty'][0]:.2f}",
                    "empty_y": f"{points['empty'][1]:.2f}",
                    "full_x": f"{points['full'][0]:.2f}",
                    "full_y": f"{points['full'][1]:.2f}",
                }
            )

            if args.save_vis:
                image_name = Path(result.path).stem
                draw_prediction(result.path, points, metrics, vis_dir / f"{image_name}_pose.jpg")
        except Exception as exc:
            row["status"] = f"error: {exc}"

        rows.append(row)

    output_csv = Path(args.output_csv)
    write_csv(rows, output_csv)
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"Processed {len(rows)} images, ok={ok_count}, output={output_csv}")
    if args.save_vis:
        print(f"Visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
