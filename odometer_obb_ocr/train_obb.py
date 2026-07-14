#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a YOLO OBB detector for odometer display detection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO OBB for odometer detection.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument(
        "--model",
        default="yolo11n-obb.pt",
        help="OBB pretrained checkpoint (e.g. yolo11n-obb.pt, yolo11s-obb.pt)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Training image size. Odometer displays are small/elongated; "
        "try 1280 or 1536 if they remain too small after resizing.",
    )
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--project",
        default=str(Path(__file__).resolve().parent / "runs" / "obb"),
    )
    parser.add_argument("--name", default="odometer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing run directory instead of auto-incrementing",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="Rotation augmentation degrees. Keep at 0.0 (default) until baseline "
        "behavior is validated, since annotations already contain strong rotation.",
    )
    parser.add_argument(
        "--perspective",
        type=float,
        default=0.0,
        help="Perspective augmentation strength. Keep at 0.0 (default) for the same reason.",
    )
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_kwargs = dict(
        data=args.data,
        task="obb",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        resume=args.resume,
        exist_ok=args.exist_ok,
        degrees=args.degrees,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=0.0,
        pretrained=True,
        optimizer="AdamW",
        verbose=True,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    print("Augmentation config in use:")
    print(f"  degrees:     {args.degrees}  (rotation augmentation)")
    print(f"  perspective: {args.perspective}")
    print(f"  fliplr:      {args.fliplr}")
    print("  flipud:      0.0  (vertical flip is never used for odometer digits)")

    model = YOLO(args.model)
    model.train(**train_kwargs)

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    if best_path.is_file():
        print(f"\nTraining complete. best.pt: {best_path}")
    else:
        print(
            f"\nTraining finished but expected weights file not found at {best_path}. "
            "Check the actual run directory printed above (Ultralytics may have "
            "auto-incremented the run name)."
        )


if __name__ == "__main__":
    main()
