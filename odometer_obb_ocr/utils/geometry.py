"""Inference-time geometry helpers: quad validation, TL/TR/BR/BL ordering,
padding, and perspective rectification for the odometer OBB pipeline.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_labelme_obb_dataset as conv  # noqa: E402

GeometryError = conv.GeometryError

Point = Tuple[float, float]

MIN_OUTPUT_SIZE_PX = 8


def select_best_detection(conf: np.ndarray, threshold: float) -> Optional[int]:
    """Return the index of the highest-confidence detection if it clears
    ``threshold``, else None."""
    if conf is None or len(conf) == 0:
        return None
    idx = int(np.argmax(conf))
    if float(conf[idx]) < threshold:
        return None
    return idx


def validate_quad_for_inference(points: List[Point]) -> List[Point]:
    """Validate and canonicalize four inference-time points into a simple,
    convex quadrilateral. Raises GeometryError on failure.

    Mirrors convert_labelme_obb_dataset.build_valid_quadrilateral's checks,
    kept as a separate copy so a future change to the converter's dataset-
    build-time validation policy cannot silently change inference behavior.
    """
    points = [(float(x), float(y)) for x, y in points]

    if len(points) != 4:
        raise GeometryError(f"expected 4 points, got {len(points)}")

    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise GeometryError("non-finite coordinate value")

    for i in range(4):
        for j in range(i + 1, 4):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if math.hypot(dx, dy) < 1e-6:
                raise GeometryError("duplicate/coincident points")

    ordered = conv.order_corners_cyclic(points)

    area = conv.polygon_area(ordered)
    if abs(area) < 1e-6:
        raise GeometryError("zero or near-zero area")

    if not conv.is_simple_quadrilateral(ordered):
        raise GeometryError("self-intersecting (bow-tie) polygon")

    if not conv.is_convex(ordered):
        raise GeometryError("non-convex polygon")

    return ordered


def order_tl_tr_br_bl(points: List[Point]) -> List[Point]:
    """Order four valid quad points into top-left, top-right, bottom-right,
    bottom-left order using a sum/diff heuristic.

    TL has the smallest x+y, BR the largest x+y, TR the largest x-y, BL the
    smallest x-y. This is a global min/max over all four points, so it does
    not depend on the input's starting point or winding direction. It assumes
    the quad is not rotated close to 45 degrees, which holds for odometer
    displays (elongated rectangles, rarely rotated more than ~45 degrees).
    """
    if len(points) != 4:
        raise GeometryError(f"expected 4 points, got {len(points)}")

    sums = [x + y for x, y in points]
    diffs = [x - y for x, y in points]

    tl = points[int(np.argmin(sums))]
    br = points[int(np.argmax(sums))]
    tr = points[int(np.argmax(diffs))]
    bl = points[int(np.argmin(diffs))]

    return [tl, tr, br, bl]


def pad_quad(
    points: List[Point], padding_ratio: float, width: int, height: int
) -> List[Point]:
    """Expand a quad outward from its centroid by ``padding_ratio`` of its own
    size, then clamp each point into [0, width] x [0, height]."""
    if padding_ratio < 0:
        raise ValueError(f"padding_ratio must be >= 0, got {padding_ratio}")

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    scale = 1.0 + padding_ratio
    padded = []
    for x, y in points:
        nx = cx + (x - cx) * scale
        ny = cy + (y - cy) * scale
        nx = min(max(nx, 0.0), float(width))
        ny = min(max(ny, 0.0), float(height))
        padded.append((nx, ny))
    return padded


def _edge_length(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def estimate_output_size(
    tl: Point, tr: Point, br: Point, bl: Point
) -> Tuple[int, int]:
    """Estimate rectified output (width, height) from the mean of opposite
    edge lengths of a TL/TR/BR/BL-ordered quad."""
    top = _edge_length(tl, tr)
    bottom = _edge_length(bl, br)
    left = _edge_length(tl, bl)
    right = _edge_length(tr, br)

    width = round((top + bottom) / 2.0)
    height = round((left + right) / 2.0)
    return int(width), int(height)


def rectify_quad(
    image: np.ndarray,
    quad_tl_tr_br_bl: List[Point],
    padding_ratio: float = 0.02,
) -> np.ndarray:
    """Rectify a TL/TR/BR/BL-ordered quad region of ``image`` into a
    horizontal crop via perspective transform.

    Raises GeometryError if the estimated crop size is too small to
    plausibly contain readable digits.
    """
    height, width = image.shape[:2]

    padded = pad_quad(quad_tl_tr_br_bl, padding_ratio, width, height)
    tl, tr, br, bl = padded

    out_w, out_h = estimate_output_size(tl, tr, br, bl)
    if out_w < MIN_OUTPUT_SIZE_PX or out_h < MIN_OUTPUT_SIZE_PX:
        raise GeometryError(
            f"estimated crop size {out_w}x{out_h} is smaller than the "
            f"minimum {MIN_OUTPUT_SIZE_PX}x{MIN_OUTPUT_SIZE_PX}px"
        )

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))


def draw_quad_overlay(
    image: np.ndarray,
    quad: List[Point],
    color: Tuple[int, int, int] = (0, 255, 0),
    det_conf: Optional[float] = None,
) -> np.ndarray:
    """Return a copy of ``image`` with the quad polygon drawn, and an
    optional confidence label near its first point."""
    vis = image.copy()
    pts = np.array(quad, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

    if det_conf is not None:
        x, y = quad[0]
        cv2.putText(
            vis,
            f"{det_conf:.2f}",
            (int(x), max(int(y) - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return vis


class GeometryStatus:
    """Status string constants shared across geometry/ocr/predict modules."""

    OK = "ok"
    NO_DETECTION = "no_detection"
    INVALID_GEOMETRY = "invalid_geometry"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    INVALID_DIGIT_COUNT = "invalid_digit_count"
