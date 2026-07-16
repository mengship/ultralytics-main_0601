#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stage fuel gauge prediction: Detection + Pose estimation.

Stage 1: YOLO detector finds the fuel gauge bounding box in the original image
Stage 2: YOLO Pose model predicts keypoints on the cropped gauge image

Keypoint order:
    0 center
    1 tip
    2 empty
    3 full

Output coordinates are in the cropped image space (not mapped back to original).
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

# Import functions from predict_pose_fuel.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
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


KEYPOINT_ORDER = ("center", "tip", "empty", "full")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage fuel gauge prediction: Detection + Pose."
    )
    parser.add_argument(
        "--det-model",
        required=True,
        help="YOLO detection model for stage 1 (find gauge box).",
    )
    parser.add_argument(
        "--pose-model",
        required=True,
        help="YOLO Pose model for stage 2 (predict keypoints).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image file or directory.",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold for stage 1.",
    )
    parser.add_argument(
        "--pose-conf",
        type=float,
        default=0.25,
        help="Pose confidence threshold for stage 2.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for both models.",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=0.08,
        help="Padding ratio to expand detection box before cropping (e.g., 0.08 = 8%%).",
    )
    parser.add_argument(
        "--direction",
        choices=("max_full_span", "tip_side", "auto", "clockwise", "counterclockwise"),
        default="max_full_span",
        help="Angle direction rule for fuel ratio calculation.",
    )
    parser.add_argument(
        "--output-csv",
        default="call_entrance_pose/pose_fuel_two_stage_predictions.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save visualized predictions on cropped images.",
    )
    parser.add_argument(
        "--vis-dir",
        default="call_entrance_pose/pose_vis_two_stage",
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--crop-dir",
        default=None,
        help="Optional directory to save cropped gauge images.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g., '0', 'cpu', 'mps').",
    )
    return parser.parse_args()


def expand_box(
    box_xyxy: Tuple[float, float, float, float],
    padding: float,
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """Expand bounding box by padding ratio and clip to image bounds.

    Args:
        box_xyxy: (x1, y1, x2, y2) in pixel coordinates
        padding: Padding ratio (e.g., 0.08 for 8%)
        img_width: Image width
        img_height: Image height

    Returns:
        (x1, y1, x2, y2) as integers, clipped to image bounds
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

    # Ensure valid crop region
    if ix2 <= ix1:
        ix2 = min(img_width, ix1 + 1)
    if iy2 <= iy1:
        iy2 = min(img_height, iy1 + 1)

    return ix1, iy1, ix2, iy2


def draw_prediction_on_crop(
    crop_image: np.ndarray,
    points: Dict[str, Tuple[float, float]],
    metrics: Dict[str, float | str | bool],
    output_path: Path,
    original_filename: str,
) -> None:
    """Draw pose prediction with angle annotations on cropped image."""
    vis_img = crop_image.copy()

    colors = {
        "center": (255, 255, 255),  # White
        "tip": (0, 0, 255),         # Red
        "empty": (255, 0, 0),       # Blue
        "full": (0, 200, 0),        # Green
    }

    center = tuple(int(v) for v in points["center"])

    # Draw keypoints
    for name, point in points.items():
        xy = tuple(int(v) for v in point)
        cv2.circle(vis_img, xy, 5, colors[name], -1)
        cv2.putText(
            vis_img, name, (xy[0] + 6, xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[name], 1
        )

    # Draw lines from center to keypoints
    for name in ("tip", "empty", "full"):
        xy = tuple(int(v) for v in points[name])
        cv2.line(vis_img, center, xy, colors[name], 2)

    # Calculate angles and draw arcs
    ratio = float(metrics["fuel_ratio"])
    raw_ratio = float(metrics["raw_fuel_ratio"])
    direction = str(metrics["direction"])
    center_float = points["center"]

    empty_angle = angle_of(points["empty"], center_float)
    tip_angle = angle_of(points["tip"], center_float)
    full_angle = angle_of(points["full"], center_float)

    # Calculate arc radii
    distances = [
        math.hypot(points[name][0] - center_float[0], points[name][1] - center_float[1])
        for name in ("tip", "empty", "full")
    ]
    max_distance = max([d for d in distances if d > 1.0] or [40.0])
    tip_arc_radius = max(18, int(max_distance * 0.38))
    full_arc_radius = max(tip_arc_radius + 12, int(max_distance * 0.58))

    # Draw empty->tip arc (cyan)
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

    # Draw empty->full arc (yellow)
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

    # Add fuel ratio label
    label = f"fuel={ratio * 100:.1f}% raw={raw_ratio * 100:.1f}% {direction}"
    put_label_with_outline(vis_img, label, (10, 25), (0, 255, 255), scale=0.6)

    # Add original filename
    info_text = f"From: {original_filename}"
    h = vis_img.shape[0]
    put_label_with_outline(vis_img, info_text, (10, h - 10), (255, 255, 255), scale=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis_img)


def process_image(
    image_path: Path,
    det_model: YOLO,
    pose_model: YOLO,
    args: argparse.Namespace,
) -> dict:
    """Process a single image through two-stage pipeline.

    Class routing:
    - class_id == 0: pointer gauge -> continue to Pose estimation
    - class_id == 1: grid gauge -> return grid_not_ready status
    - others: return unsupported_cls status

    Returns:
        Dictionary with prediction results and status
    """
    result = {
        "image": image_path.name,
        "status": "ok",
        "det_class": None,
        "fuel_type": None,
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
        "tip_deg": None,
        "empty_deg": None,
        "full_deg": None,
        "center_x": None,
        "center_y": None,
        "tip_x": None,
        "tip_y": None,
        "empty_x": None,
        "empty_y": None,
        "full_x": None,
        "full_y": None,
    }

    # Stage 1: Detection
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

    # Get detection box and class
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

    # Class routing: determine fuel type and whether to continue
    if det_class == 0:
        # Pointer gauge: continue to Pose estimation
        result["fuel_type"] = "pointer"
    elif det_class == 1:
        # Grid gauge: not implemented yet
        result["fuel_type"] = "grid"
        result["status"] = "grid_not_ready"
        return result
    else:
        # Unsupported class
        result["fuel_type"] = f"class_{det_class}"
        result["status"] = "unsupported_cls"
        return result

    # Expand box and crop
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

    # Save cropped image if requested
    crop_filename = None
    if args.crop_dir:
        crop_dir = Path(args.crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_filename = f"{image_path.stem}_crop{image_path.suffix}"
        crop_path = crop_dir / crop_filename
        cv2.imwrite(str(crop_path), crop_image)
        result["crop_image"] = crop_filename

    # Stage 2: Pose estimation on cropped image
    # Save crop to temp file for prediction (YOLO needs file path or array)
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

    try:
        points = extract_points(pose_result, pose_idx)
    except (ValueError, IndexError) as e:
        result["status"] = "no_pose"
        return result

    pose_conf = float(pose_result.boxes.conf[pose_idx].cpu().numpy())
    result["pose_conf"] = pose_conf

    # Compute fuel ratio (coordinates are in cropped image space)
    metrics = compute_fuel_ratio(points, direction=args.direction)

    result["fuel_ratio"] = float(metrics["fuel_ratio"])
    result["raw_fuel_ratio"] = float(metrics["raw_fuel_ratio"])
    result["fuel_percent"] = float(metrics["fuel_ratio"]) * 100
    result["clamped"] = bool(metrics["clamped"])
    result["direction"] = str(metrics["direction"])
    result["span_deg"] = float(metrics["span_deg"])
    result["offset_deg"] = float(metrics["offset_deg"])
    result["tip_deg"] = float(metrics["tip_deg"])
    result["empty_deg"] = float(metrics["empty_deg"])
    result["full_deg"] = float(metrics["full_deg"])

    # Store keypoint coordinates (in cropped image space)
    for name in KEYPOINT_ORDER:
        result[f"{name}_x"] = float(points[name][0])
        result[f"{name}_y"] = float(points[name][1])

    # Save visualization if requested
    if args.save_vis:
        vis_dir = Path(args.vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f"{image_path.stem}_vis.jpg"
        draw_prediction_on_crop(
            crop_image,
            points,
            metrics,
            vis_path,
            image_path.name,
        )

    return result


def collect_images(source: Path) -> List[Path]:
    """Collect all image files from source path."""
    if source.is_file():
        return [source]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    images = []
    for ext in image_exts:
        images.extend(source.glob(f"*{ext}"))
        images.extend(source.glob(f"*{ext.upper()}"))

    return sorted(set(images))


def write_csv(results: List[dict], output_csv: Path) -> None:
    """Write prediction results to CSV file."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image",
        "status",
        "det_class",
        "fuel_type",
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
        for row in results:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    print("=" * 70)
    print("Two-Stage Fuel Gauge Prediction: Detection + Pose")
    print("=" * 70)
    print(f"Detection model: {args.det_model}")
    print(f"Pose model: {args.pose_model}")
    print(f"Source: {source}")
    print(f"Box padding: {args.box_padding} ({args.box_padding * 100:.1f}%)")
    print(f"Direction: {args.direction}")
    print(f"Output CSV: {args.output_csv}")
    if args.save_vis:
        print(f"Visualization dir: {args.vis_dir}")
    if args.crop_dir:
        print(f"Crop dir: {args.crop_dir}")
    print("=" * 70)
    print()

    # Load models
    print("Loading models...")
    det_model = YOLO(args.det_model)
    pose_model = YOLO(args.pose_model)

    if args.device:
        det_model.to(args.device)
        pose_model.to(args.device)

    print("Models loaded successfully.")
    print()

    # Collect images
    images = collect_images(source)
    print(f"Found {len(images)} images.")
    print()

    # Process images
    results = []
    for i, image_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Processing {image_path.name}...", end=" ")
        result = process_image(image_path, det_model, pose_model, args)
        results.append(result)

        status = result["status"]
        if status == "ok":
            print(f"✓ fuel={result['fuel_percent']:.1f}%")
        else:
            print(f"✗ {status}")

    print()

    # Write CSV
    output_csv = Path(args.output_csv).expanduser().resolve()
    write_csv(results, output_csv)
    print(f"Results saved to: {output_csv}")

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    ok_count = status_counts.get("ok", 0)
    print(f"\nSuccess rate: {ok_count}/{len(images)} ({ok_count / len(images) * 100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
