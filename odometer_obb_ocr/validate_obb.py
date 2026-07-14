#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate a trained YOLO OBB odometer detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO OBB odometer detector.")
    parser.add_argument("--weights", required=True, help="Path to trained weights (e.g. best.pt)")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--project",
        default=str(Path(__file__).resolve().parent / "runs" / "obb_val"),
    )
    parser.add_argument("--name", default="odometer_val")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    val_kwargs = dict(
        data=args.data,
        task="obb",
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        split=args.split,
        save_json=True,
        plots=True,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device

    model = YOLO(args.weights)
    metrics = model.val(**val_kwargs)

    print(f"Box precision (mp): {metrics.box.mp:.4f}")
    print(f"Box recall    (mr): {metrics.box.mr:.4f}")
    print(f"mAP50:              {metrics.box.map50:.4f}")
    print(f"mAP50-95:           {metrics.box.map:.4f}")
    print()
    print(
        "NOTE: these are detection-only metrics (how well the OBB localizes the "
        "odometer region). They do NOT measure end-to-end mileage-reading accuracy, "
        "which also depends on perspective rectification quality and OCR correctness. "
        "See TODO.md section 3."
    )

    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        print(f"\nAnnotated predictions and metrics saved to: {save_dir}")


if __name__ == "__main__":
    main()
