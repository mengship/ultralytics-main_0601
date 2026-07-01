#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert LabelMe/X-AnyLabeling JSON files to a YOLO Pose dataset.

Expected annotation per pointer gauge:
    - one rectangle labelled oil_pose or oil
    - four point shapes labelled center, tip, empty, full

If no rectangle exists, the converter can create one from the four keypoints.

Output label row:
    class cx cy w h center_x center_y tip_x tip_y empty_x empty_y full_x full_y
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml


KEYPOINT_ORDER = ("center", "tip", "empty", "full")
BOX_LABELS = {"oil_pose", "oil", "pointer", "gauge", "fuel_gauge"}
GRID_LABELS = {"oil1", "grid"}

KEYPOINT_ALIASES = {
    "center": "center",
    "centre": "center",
    "middle": "center",
    "origin": "center",
    "tip": "tip",
    "needle_tip": "tip",
    "pointer_tip": "tip",
    "empty": "empty",
    "e": "empty",
    "zero": "empty",
    "min": "empty",
    "full": "full",
    "f": "full",
    "max": "full",
}


@dataclass
class BoxShape:
    label: str
    group_id: Optional[int]
    xyxy: Tuple[float, float, float, float]
    fuel_ratio: Optional[float]


@dataclass
class PointShape:
    name: str
    group_id: Optional[int]
    xy: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pointer gauge pose annotations to YOLO Pose format.")
    parser.add_argument("--json-dir", required=True, help="Directory containing JSON files and images.")
    parser.add_argument("--output-dir", default="fuel_pose_dataset", help="Output YOLO Pose dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument("--image-exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"])
    parser.add_argument(
        "--auto-box-padding",
        type=float,
        default=0.35,
        help="If no rectangle exists, create a box from the four keypoints with this relative padding.",
    )
    return parser.parse_args()


def norm_label(label: object) -> str:
    return str(label or "").strip().lower().replace(" ", "_").replace("-", "_")


def canonical_keypoint(label: object) -> Optional[str]:
    return KEYPOINT_ALIASES.get(norm_label(label))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_fuel_ratio(description: object) -> Optional[float]:
    text = str(description or "").strip()
    if not text:
        return None
    try:
        return clamp01(float(text))
    except ValueError:
        return None


def shape_xyxy(points: Iterable[Iterable[float]]) -> Optional[Tuple[float, float, float, float]]:
    pts = list(points)
    if len(pts) < 2:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def clamp_xyxy(
    xyxy: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def find_image_file(json_dir: Path, image_path: str, image_exts: List[str]) -> Optional[Path]:
    if image_path:
        candidates = [json_dir / image_path, json_dir / Path(image_path).name]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    stem = Path(image_path).stem if image_path else ""
    if stem:
        for ext in image_exts:
            candidate = json_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def load_shapes(data: dict) -> Tuple[List[BoxShape], List[PointShape]]:
    boxes: List[BoxShape] = []
    points: List[PointShape] = []

    for shape in data.get("shapes", []):
        label = norm_label(shape.get("label"))
        shape_type = norm_label(shape.get("shape_type"))
        group_id = shape.get("group_id")

        if shape_type == "rectangle":
            if label in GRID_LABELS:
                continue
            if label not in BOX_LABELS and label:
                continue
            xyxy = shape_xyxy(shape.get("points", []))
            if xyxy is None:
                continue
            boxes.append(
                BoxShape(
                    label=label or "oil_pose",
                    group_id=group_id,
                    xyxy=xyxy,
                    fuel_ratio=parse_fuel_ratio(shape.get("description")),
                )
            )
            continue

        if shape_type == "point":
            keypoint_name = canonical_keypoint(label)
            pts = shape.get("points", [])
            if keypoint_name is None or not pts:
                continue
            points.append(
                PointShape(
                    name=keypoint_name,
                    group_id=group_id,
                    xy=(float(pts[0][0]), float(pts[0][1])),
                )
            )

    return boxes, points


def point_inside_box(point: PointShape, box: BoxShape, tolerance: float = 0.03) -> bool:
    x1, y1, x2, y2 = box.xyxy
    bw = x2 - x1
    bh = y2 - y1
    px, py = point.xy
    return (x1 - bw * tolerance) <= px <= (x2 + bw * tolerance) and (y1 - bh * tolerance) <= py <= (y2 + bh * tolerance)


def choose_keypoints(box: BoxShape, points: List[PointShape], single_box: bool) -> Tuple[Dict[str, PointShape], str]:
    selected: Dict[str, PointShape] = {}

    if box.group_id is not None:
        for point in points:
            if point.group_id == box.group_id and point.name not in selected:
                selected[point.name] = point
        if all(name in selected for name in KEYPOINT_ORDER):
            return selected, "group_id"

    candidates = [p for p in points if p.group_id is None and (single_box or point_inside_box(p, box))]
    for point in candidates:
        if point.name not in selected:
            selected[point.name] = point

    if all(name in selected for name in KEYPOINT_ORDER):
        return selected, "inside_box"

    return selected, "missing"


def group_points(points: List[PointShape]) -> Dict[Optional[int], Dict[str, PointShape]]:
    grouped: Dict[Optional[int], Dict[str, PointShape]] = {}
    for point in points:
        grouped.setdefault(point.group_id, {})
        grouped[point.group_id].setdefault(point.name, point)
    return grouped


def make_auto_boxes(points: List[PointShape], width: int, height: int, padding: float) -> List[Tuple[BoxShape, Dict[str, PointShape]]]:
    boxes: List[Tuple[BoxShape, Dict[str, PointShape]]] = []
    for group_id, grouped_points in group_points(points).items():
        if not all(name in grouped_points for name in KEYPOINT_ORDER):
            continue

        xs = [grouped_points[name].xy[0] for name in KEYPOINT_ORDER]
        ys = [grouped_points[name].xy[1] for name in KEYPOINT_ORDER]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad = max(bw, bh) * padding
        xyxy = clamp_xyxy((x1 - pad, y1 - pad, x2 + pad, y2 + pad), width, height)

        boxes.append(
            (
                BoxShape(label="auto_oil_pose", group_id=group_id, xyxy=xyxy, fuel_ratio=None),
                grouped_points,
            )
        )
    return boxes


def yolo_box_from_xyxy(xyxy: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return clamp01(cx), clamp01(cy), clamp01(bw), clamp01(bh)


def yolo_point(point: PointShape, width: int, height: int) -> Tuple[float, float]:
    return clamp01(point.xy[0] / width), clamp01(point.xy[1] / height)


def convert_json(
    json_file: Path,
    json_dir: Path,
    image_exts: List[str],
    auto_box_padding: float,
) -> Tuple[Optional[dict], List[dict]]:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    width = int(data.get("imageWidth") or 0)
    height = int(data.get("imageHeight") or 0)
    image_path = data.get("imagePath", "")
    image_file = find_image_file(json_dir, image_path, image_exts)

    missing: List[dict] = []
    if width <= 0 or height <= 0:
        return None, [{"file": json_file.name, "reason": "missing imageWidth/imageHeight"}]
    if image_file is None:
        return None, [{"file": json_file.name, "reason": f"image not found: {image_path}"}]

    boxes, points = load_shapes(data)

    lines: List[str] = []
    objects: List[dict] = []
    auto_boxes: List[Tuple[BoxShape, Dict[str, PointShape]]] = []
    if not boxes and auto_box_padding >= 0:
        auto_boxes = make_auto_boxes(points, width, height, auto_box_padding)

    if not boxes and not auto_boxes:
        return None, [{"file": json_file.name, "reason": "no pointer rectangle label or complete keypoint group found"}]

    single_box = len(boxes) == 1

    for box_index, box in enumerate(boxes):
        selected, match_mode = choose_keypoints(box, points, single_box)
        missing_names = [name for name in KEYPOINT_ORDER if name not in selected]
        if missing_names:
            missing.append(
                {
                    "file": json_file.name,
                    "box_index": box_index,
                    "group_id": box.group_id,
                    "reason": "missing keypoints",
                    "missing": missing_names,
                }
            )
            continue

        cx, cy, bw, bh = yolo_box_from_xyxy(box.xyxy, width, height)
        values = [0, cx, cy, bw, bh]
        pixel_points = {}
        for name in KEYPOINT_ORDER:
            px, py = yolo_point(selected[name], width, height)
            values.extend([px, py])
            pixel_points[name] = selected[name].xy

        line = " ".join([str(values[0])] + [f"{v:.6f}" for v in values[1:]])
        lines.append(line)
        objects.append(
            {
                "box_index": box_index,
                "group_id": box.group_id,
                "match_mode": match_mode,
                "fuel_ratio": box.fuel_ratio,
                "keypoints": pixel_points,
            }
        )

    for auto_index, (box, selected) in enumerate(auto_boxes):
        cx, cy, bw, bh = yolo_box_from_xyxy(box.xyxy, width, height)
        values = [0, cx, cy, bw, bh]
        pixel_points = {}
        for name in KEYPOINT_ORDER:
            px, py = yolo_point(selected[name], width, height)
            values.extend([px, py])
            pixel_points[name] = selected[name].xy

        line = " ".join([str(values[0])] + [f"{v:.6f}" for v in values[1:]])
        lines.append(line)
        objects.append(
            {
                "box_index": auto_index,
                "group_id": box.group_id,
                "match_mode": "auto_box_from_keypoints",
                "fuel_ratio": None,
                "keypoints": pixel_points,
            }
        )

    if not lines:
        return None, missing

    return {
        "image_file": image_file,
        "image_name": image_file.name,
        "stem": image_file.stem,
        "lines": lines,
        "objects": objects,
        "source_json": json_file.name,
    }, missing


def write_yaml(output_dir: Path) -> None:
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "kpt_shape": [4, 2],
        "flip_idx": [0, 1, 2, 3],
        "names": {0: "oil_pose"},
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    json_dir = Path(args.json_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    converted: List[dict] = []
    missing_report: List[dict] = []

    for json_file in json_files:
        item, missing = convert_json(json_file, json_dir, args.image_exts, args.auto_box_padding)
        missing_report.extend(missing)
        if item is not None:
            converted.append(item)

    random.seed(args.seed)
    random.shuffle(converted)
    val_count = int(round(len(converted) * args.val_ratio))
    if len(converted) > 1 and val_count == 0 and args.val_ratio > 0:
        val_count = 1
    val_items = set(item["source_json"] for item in converted[:val_count])

    metadata = {
        "keypoint_order": KEYPOINT_ORDER,
        "samples": [],
    }

    split_counts = {"train": 0, "val": 0}
    object_counts = {"train": 0, "val": 0}

    for item in converted:
        split = "val" if item["source_json"] in val_items else "train"
        split_counts[split] += 1
        object_counts[split] += len(item["lines"])

        image_dst = output_dir / split / "images" / item["image_name"]
        label_dst = output_dir / split / "labels" / f"{item['stem']}.txt"
        shutil.copy2(item["image_file"], image_dst)
        with open(label_dst, "w", encoding="utf-8") as f:
            f.write("\n".join(item["lines"]) + "\n")

        metadata["samples"].append(
            {
                "split": split,
                "image": item["image_name"],
                "json": item["source_json"],
                "objects": item["objects"],
            }
        )

    write_yaml(output_dir)

    with open(output_dir / "pose_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(output_dir / "missing_pose_report.json", "w", encoding="utf-8") as f:
        json.dump(missing_report, f, ensure_ascii=False, indent=2)

    print("Pose dataset conversion complete")
    print(f"  source: {json_dir}")
    print(f"  output: {output_dir}")
    print(f"  json files: {len(json_files)}")
    print(f"  usable images: {len(converted)}")
    print(f"  train images/objects: {split_counts['train']}/{object_counts['train']}")
    print(f"  val images/objects: {split_counts['val']}/{object_counts['val']}")
    print(f"  skipped or incomplete records: {len(missing_report)}")
    print(f"  yaml: {output_dir / 'data.yaml'}")
    print(f"  report: {output_dir / 'missing_pose_report.json'}")


if __name__ == "__main__":
    main()

# python call_entrance_pose/convert_labelme_pose_dataset.py --json-dir "/Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/ultralytics-main_0601/call_entrance_pose/dataset" --output-dir "/Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/ultralytics-main_0601/call_entrance_pose/dataset_convert"