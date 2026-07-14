#!/usr/bin/env python3
"""Convert LabelMe / X-AnyLabeling JSON annotations into a YOLO OBB dataset.

Only shapes whose ``label`` matches ``--label`` (default ``odometer``) are
converted. Both ``rotation`` (four arbitrary corners) and ``rectangle``
(two diagonal points or four corners) shapes are supported.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

Point = Tuple[float, float]


def two_point_to_four_corners(p1: Point, p2: Point) -> List[Point]:
    """Expand a diagonal two-point rectangle into four axis-aligned corners."""
    x1, y1 = p1
    x2, y2 = p2
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def order_corners_cyclic(points: list[Point]) -> list[Point]:
    """Order four points into a consistent, non-self-crossing cyclic order.

    Sorts by angle around the centroid. This yields a simple (non-bow-tie)
    polygon for any four points that do not already form a valid simple
    quadrilateral in input order, as long as the points are in "convex
    position" (none is inside the triangle formed by the other three).
    """
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    def angle(p: Point) -> float:
        return math.atan2(p[1] - cy, p[0] - cx)

    return sorted(points, key=angle)


def polygon_area(points: list[Point]) -> float:
    """Signed shoelace area (positive = counter-clockwise in image coords... but
    sign only indicates orientation here, not visual winding, since y grows
    downward in image space)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def is_convex(points: list[Point]) -> bool:
    """Return True if the ordered quadrilateral is convex (all cross products
    of consecutive edges share the same sign, allowing for near-zero noise)."""
    n = len(points)
    signs = []
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        x2, y2 = points[(i + 2) % n]
        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1
        cross = dx1 * dy2 - dy1 * dx2
        if abs(cross) > 1e-9:
            signs.append(cross > 0)
    if not signs:
        return False
    return all(s == signs[0] for s in signs)


def segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    """Return True if closed segments a1-a2 and b1-b2 properly cross."""

    def cross(o: Point, p: Point, q: Point) -> float:
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    return False


def is_simple_quadrilateral(points: list[Point]) -> bool:
    """Return True if the ordered quadrilateral's non-adjacent edges do not
    cross (i.e. it is not a bow-tie polygon)."""
    p0, p1, p2, p3 = points
    if segments_intersect(p0, p1, p2, p3):
        return False
    if segments_intersect(p1, p2, p3, p0):
        return False
    return True


class GeometryError(ValueError):
    """Raised when a shape cannot be converted into a valid OBB polygon."""


def build_valid_quadrilateral(points: list[Point]) -> list[Point]:
    """Validate and canonicalize four points into a simple, convex quadrilateral.

    Raises GeometryError with a human-readable reason on failure.
    """
    if len(points) != 4:
        raise GeometryError(f"expected 4 points, got {len(points)}")

    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise GeometryError("non-finite coordinate value")

    # Deduplicate near-identical points (degenerate shape).
    for i in range(4):
        for j in range(i + 1, 4):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if math.hypot(dx, dy) < 1e-6:
                raise GeometryError("duplicate/coincident points")

    ordered = order_corners_cyclic(points)

    area = polygon_area(ordered)
    if abs(area) < 1e-6:
        raise GeometryError("zero or near-zero area")

    if not is_simple_quadrilateral(ordered):
        raise GeometryError("self-intersecting (bow-tie) polygon")

    if not is_convex(ordered):
        raise GeometryError("non-convex polygon")

    return ordered


def normalize_and_clamp(
    points: list[Point], width: float, height: float
) -> tuple[list[Point], bool]:
    """Normalize points by image width/height, clamping into [0, 1].

    Returns (normalized_points, was_clamped).
    """
    normalized = []
    clamped = False
    for x, y in points:
        nx = x / width
        ny = y / height
        cx = min(max(nx, 0.0), 1.0)
        cy = min(max(ny, 0.0), 1.0)
        if cx != nx or cy != ny:
            clamped = True
        normalized.append((cx, cy))
    return normalized, clamped


def validate_in_bounds(points: list[Point], width: float, height: float, tol: float = 1.0) -> None:
    """Raise GeometryError if points fall far outside the image bounds."""
    for x, y in points:
        if x < -tol or x > width + tol or y < -tol or y > height + tol:
            raise GeometryError(
                f"point ({x:.2f}, {y:.2f}) far outside image bounds ({width}x{height})"
            )


# ---------------------------------------------------------------------------
# Conversion pipeline
# ---------------------------------------------------------------------------


@dataclass
class ConvertedSample:
    json_path: Path
    image_path: Path
    stem: str
    label_line: str


@dataclass
class SkipRecord:
    json_path: str
    reason: str


@dataclass
class ConversionStats:
    json_dir: str
    output_dir: str
    seed: int
    val_ratio: float
    discovered_json: int = 0
    converted_total: int = 0
    train_count: int = 0
    val_count: int = 0
    dimension_mismatches: int = 0
    clamped_count: int = 0
    skipped_count: int = 0


def find_image_for_json(
    json_path: Path, image_path_field: str, image_dir: Optional[Path]
) -> Optional[Path]:
    """Resolve the source image referenced by a JSON's imagePath field."""
    candidate = (json_path.parent / image_path_field).resolve()
    if candidate.is_file():
        return candidate

    if image_dir is not None:
        name = Path(image_path_field).name
        for match in image_dir.rglob(name):
            if match.is_file():
                return match

    return None


def stable_stem_for_json(json_path: Path, json_dir: Path) -> str:
    """Derive a filesystem-safe, collision-resistant stem from a JSON's path
    relative to the search root, so same-named files in different
    subdirectories do not collide in the flat output layout."""
    rel = json_path.relative_to(json_dir).with_suffix("")
    return "__".join(rel.parts)


def extract_target_shapes(shapes: list[dict], label: str) -> list[dict]:
    return [s for s in shapes if s.get("label") == label]


def shape_to_quadrilateral(shape: dict) -> list[Point]:
    """Turn a LabelMe shape dict into an unvalidated list of 4 (x, y) points."""
    shape_type = shape.get("shape_type")
    raw_points = shape.get("points")

    if not isinstance(raw_points, list):
        raise GeometryError("missing or malformed 'points'")

    points: list[Point] = []
    for p in raw_points:
        if not (isinstance(p, list) and len(p) == 2):
            raise GeometryError("malformed point entry")
        points.append((float(p[0]), float(p[1])))

    if shape_type == "rotation":
        if len(points) != 4:
            raise GeometryError(f"rotation shape has {len(points)} points, expected 4")
        return points

    if shape_type == "rectangle":
        if len(points) == 2:
            return two_point_to_four_corners(points[0], points[1])
        if len(points) == 4:
            return points
        raise GeometryError(f"rectangle shape has {len(points)} points, expected 2 or 4")

    raise GeometryError(f"unsupported shape_type '{shape_type}'")


def convert_one_json(
    json_path: Path,
    json_dir: Path,
    label: str,
    image_dir: Optional[Path],
) -> tuple[Optional[ConvertedSample], Optional[SkipRecord], bool, bool]:
    """Convert a single JSON file.

    Returns (sample_or_none, skip_or_none, dimension_mismatch, clamped).
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, SkipRecord(str(json_path), f"failed to read/parse JSON: {exc}"), False, False

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        return None, SkipRecord(str(json_path), "missing 'shapes' array"), False, False

    targets = extract_target_shapes(shapes, label)
    if len(targets) == 0:
        return None, SkipRecord(str(json_path), f"no '{label}' shape found"), False, False
    if len(targets) > 1:
        return (
            None,
            SkipRecord(str(json_path), f"multiple '{label}' shapes found ({len(targets)})"),
            False,
            False,
        )

    image_path_field = data.get("imagePath")
    if not image_path_field:
        return None, SkipRecord(str(json_path), "missing 'imagePath'"), False, False

    image_path = find_image_for_json(json_path, image_path_field, image_dir)
    if image_path is None:
        return (
            None,
            SkipRecord(str(json_path), f"could not resolve image '{image_path_field}'"),
            False,
            False,
        )

    img = cv2.imread(str(image_path))
    if img is None:
        return None, SkipRecord(str(json_path), f"failed to read image '{image_path}'"), False, False

    actual_height, actual_width = img.shape[:2]

    dimension_mismatch = False
    json_width = data.get("imageWidth")
    json_height = data.get("imageHeight")
    if json_width is not None and json_height is not None:
        if int(json_width) != actual_width or int(json_height) != actual_height:
            dimension_mismatch = True

    try:
        raw_points = shape_to_quadrilateral(targets[0])
        valid_points = build_valid_quadrilateral(raw_points)
        validate_in_bounds(valid_points, actual_width, actual_height)
        normalized, clamped = normalize_and_clamp(valid_points, actual_width, actual_height)
    except GeometryError as exc:
        return None, SkipRecord(str(json_path), f"invalid geometry: {exc}"), dimension_mismatch, False

    class_id = 0
    coord_str = " ".join(f"{v:.6f}" for pt in normalized for v in pt)
    label_line = f"{class_id} {coord_str}"

    stem = stable_stem_for_json(json_path, json_dir)

    sample = ConvertedSample(
        json_path=json_path, image_path=image_path, stem=stem, label_line=label_line
    )
    return sample, None, dimension_mismatch, clamped


def place_file(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    elif copy_mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"unknown copy mode '{copy_mode}'")


def run_conversion(args: argparse.Namespace) -> ConversionStats:
    json_dir = Path(args.json_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else None

    if not json_dir.is_dir():
        raise SystemExit(f"error: --json-dir '{json_dir}' is not a directory")

    if not (0 <= args.val_ratio < 1):
        raise SystemExit(f"error: --val-ratio must satisfy 0 <= value < 1, got {args.val_ratio}")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"error: --output-dir '{output_dir}' exists and is nonempty; pass --overwrite to proceed"
        )

    json_paths = sorted(json_dir.rglob("*.json"))

    stats = ConversionStats(
        json_dir=str(json_dir),
        output_dir=str(output_dir),
        seed=args.seed,
        val_ratio=args.val_ratio,
        discovered_json=len(json_paths),
    )

    samples: list[ConvertedSample] = []
    skips: list[SkipRecord] = []

    for json_path in json_paths:
        sample, skip, mismatch, clamped = convert_one_json(
            json_path, json_dir, args.label, image_dir
        )
        if mismatch:
            stats.dimension_mismatches += 1
        if clamped:
            stats.clamped_count += 1
        if sample is not None:
            samples.append(sample)
        if skip is not None:
            skips.append(skip)

    stats.converted_total = len(samples)
    stats.skipped_count = len(skips)

    rng = random.Random(args.seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    val_count = round(len(shuffled) * args.val_ratio)
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]

    stats.train_count = len(train_samples)
    stats.val_count = len(val_samples)

    for split_name, split_samples in (("train", train_samples), ("val", val_samples)):
        images_dir = output_dir / split_name / "images"
        labels_dir = output_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for sample in split_samples:
            dst_image = images_dir / f"{sample.stem}{sample.image_path.suffix.lower()}"
            place_file(sample.image_path, dst_image, args.copy_mode)

            dst_label = labels_dir / f"{sample.stem}.txt"
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text(sample.label_line + "\n", encoding="utf-8")

    data_yaml_path = output_dir / "data.yaml"
    data_yaml_content = (
        f"path: {output_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n"
        f"  0: {args.label}\n"
    )
    data_yaml_path.write_text(data_yaml_content, encoding="utf-8")

    report = {
        "json_dir": stats.json_dir,
        "output_dir": stats.output_dir,
        "seed": stats.seed,
        "val_ratio": stats.val_ratio,
        "discovered_json_count": stats.discovered_json,
        "converted_total": stats.converted_total,
        "train_count": stats.train_count,
        "val_count": stats.val_count,
        "image_dimension_mismatches": stats.dimension_mismatches,
        "clamped_boundary_drift_count": stats.clamped_count,
        "skipped_count": stats.skipped_count,
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    skipped_payload = [{"json_path": s.json_path, "reason": s.reason} for s in skips]
    (output_dir / "skipped_samples.json").write_text(
        json.dumps(skipped_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert LabelMe/X-AnyLabeling JSON annotations into a YOLO OBB dataset."
    )
    parser.add_argument("--json-dir", required=True, help="Directory to search recursively for .json files")
    parser.add_argument("--output-dir", required=True, help="Directory to write the YOLO OBB dataset into")
    parser.add_argument("--label", default="odometer", help="Shape label to convert (default: odometer)")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2, help="Validation split ratio, 0 <= value < 1 (default: 0.2)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split (default: 42)")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional fallback directory to search for images not resolvable relative to their JSON",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="How to place source images into the output dataset (default: copy)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a nonempty --output-dir",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stats = run_conversion(args)

    print("Conversion complete.")
    print(f"  JSON dir:            {stats.json_dir}")
    print(f"  Output dir:          {stats.output_dir}")
    print(f"  Discovered JSON:     {stats.discovered_json}")
    print(f"  Converted total:     {stats.converted_total}")
    print(f"  Train / Val:         {stats.train_count} / {stats.val_count}")
    print(f"  Dimension mismatches:{stats.dimension_mismatches}")
    print(f"  Clamped boundary:    {stats.clamped_count}")
    print(f"  Skipped:             {stats.skipped_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
