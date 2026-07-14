#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end odometer prediction: YOLO OBB detection -> perspective
rectification -> OCR -> validated mileage digits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from utils import geometry
from utils import ocr as ocr_module

ROW_FIELDNAMES = [
    "source_path",
    "status",
    "detail",
    "raw_ocr_text",
    "mileage_digits",
    "det_conf",
    "ocr_conf",
    "quad_points",
    "crop_path",
    "vis_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect an odometer OBB, rectify it, and OCR the mileage digits."
    )
    parser.add_argument("--model", required=True, help="Path to trained YOLO OBB weights")
    parser.add_argument("--source", required=True, help="Image file or directory")
    parser.add_argument("--ocr-engine", choices=["paddle", "easy"], default="paddle")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--ocr-conf", type=float, default=0.70)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.02)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "runs" / "predict"),
    )
    parser.add_argument("--save-crops", action="store_true")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--min-digits", type=int, default=4)
    parser.add_argument("--max-digits", type=int, default=8)
    return parser.parse_args()


def _empty_row(source_path: str) -> Dict[str, Any]:
    return {
        "source_path": source_path,
        "status": "",
        "detail": "",
        "raw_ocr_text": "",
        "mileage_digits": None,
        "det_conf": None,
        "ocr_conf": None,
        "quad_points": None,
        "crop_path": None,
        "vis_path": None,
    }


def process_one_result(
    result: Any,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run the full detect -> rectify -> OCR flow for one prediction result.

    Fully wrapped in try/except so one bad image never aborts a directory batch.
    """
    source_path = str(getattr(result, "path", "unknown"))
    row = _empty_row(source_path)

    try:
        obb = result.obb
        if obb is None or obb.conf is None or len(obb.conf) == 0:
            row["status"] = "no_detection"
            return row

        conf_arr = obb.conf.cpu().numpy()
        idx = geometry.select_best_detection(conf_arr, args.det_conf)
        if idx is None:
            row["status"] = "no_detection"
            return row

        det_conf = float(conf_arr[idx])
        row["det_conf"] = det_conf

        raw_quad = obb.xyxyxyxy[idx].cpu().numpy().tolist()

        try:
            valid = geometry.validate_quad_for_inference(raw_quad)
            ordered = geometry.order_tl_tr_br_bl(valid)
        except geometry.GeometryError as exc:
            row["status"] = "invalid_geometry"
            row["detail"] = str(exc)
            return row

        row["quad_points"] = [list(p) for p in ordered]

        image = cv2.imread(source_path)
        if image is None:
            row["status"] = f"error: failed to read image '{source_path}'"
            return row

        try:
            crop = geometry.rectify_quad(image, ordered, args.crop_padding_ratio)
        except geometry.GeometryError as exc:
            row["status"] = "invalid_geometry"
            row["detail"] = str(exc)
            return row

        stem = Path(source_path).stem

        if args.save_crops:
            crop_path = output_dir / "crops" / f"{stem}.jpg"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop)
            row["crop_path"] = str(crop_path)

        if args.save_vis:
            vis = geometry.draw_quad_overlay(image, ordered, det_conf=det_conf)
            vis_path = output_dir / "vis" / f"{stem}.jpg"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(vis_path), vis)
            row["vis_path"] = str(vis_path)

        try:
            ocr_result = ocr_module.recognize(args.ocr_engine, crop)
        except ocr_module.OcrEngineError as exc:
            row["status"] = f"error: {exc}"
            return row
        except Exception as exc:  # noqa: BLE001 - OCR engine failure must not abort the batch
            row["status"] = f"error: {exc}"
            return row

        row["raw_ocr_text"] = ocr_result.raw_text
        row["ocr_conf"] = ocr_result.confidence

        digits = ocr_module.extract_digits(ocr_result.raw_text)

        if ocr_result.confidence < args.ocr_conf:
            row["status"] = "low_ocr_confidence"
            return row

        if not (args.min_digits <= len(digits) <= args.max_digits):
            row["status"] = "invalid_digit_count"
            return row

        row["status"] = "ok"
        row["mileage_digits"] = digits
        return row

    except Exception as exc:  # noqa: BLE001 - never abort the batch for one bad image
        row["status"] = f"error: {exc}"
        return row


def write_json(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["quad_points"] = json.dumps(out["quad_points"], ensure_ascii=False)
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    predict_kwargs: Dict[str, Any] = dict(source=args.source, conf=args.det_conf, verbose=False)
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)

    rows = [process_one_result(result, args, output_dir) for result in results]

    write_json(rows, output_dir / "predictions.json")
    write_csv(rows, output_dir / "predictions.csv")

    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print(f"Processed {len(rows)} image(s).")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    print(f"\nRecords written to: {output_dir / 'predictions.json'}")
    print(f"Records written to: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
