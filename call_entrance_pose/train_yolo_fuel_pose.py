#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a YOLO Pose model for pointer fuel gauges."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def default_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "fuel_pose_dataset" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO Pose for fuel gauge keypoints.")
    parser.add_argument("--data", default=str(default_data_path()), help="YOLO Pose data.yaml path.")
    parser.add_argument("--model", default="yolo11m-pose.pt", help="Pose pretrained model path/name.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device, mps, or cpu.")
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--project", default="runs/fuel_pose")
    parser.add_argument("--name", default="pose_4kpt")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).expanduser()
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    print("Training YOLO Pose for pointer fuel gauges")
    print(f"  data: {data_path.resolve()}")
    print(f"  model: {args.model}")
    print("  keypoints: center, tip, empty, full")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=0.2,
        close_mosaic=20,
        degrees=15.0,
        translate=0.08,
        scale=0.25,
        perspective=0.0003,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        verbose=True,
    )


if __name__ == "__main__":
    main()
