#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a YOLO Pose model on cropped grid gauge dataset.

This script trains a YOLO Pose model on a pre-cropped grid gauge dataset.
Use `crop_yolo_grid_pose_dataset.py` first to prepare the cropped data.

Pipeline:
    1) convert_labelme_grid_pose_dataset.py: convert LabelMe annotations to YOLO Pose
    2) crop_yolo_grid_pose_dataset.py: crop original images inside annotated boxes
    3) train_yolo_grid_pose.py: train YOLO Pose on cropped images (this script)

Keypoint order:
    0 empty
    1 full
    2 tip

python call_entrance_pose_grid/train_yolo_grid_pose.py \
    --data call_entrance_pose_grid/dataset_convert_crop/data.yaml \
    --model yolo11m-pose.pt \
    --epochs 300 \
    --batch 16 \
    --device 0

Label format (YOLO Pose):
    class cx cy w h empty_x empty_y full_x full_y tip_x tip_y

Training considerations:
    - No horizontal/vertical flipping (directional semantics)
    - Light rotation, scale, color augmentation
    - AdamW optimizer
"""

from __future__ import annotations

import argparse
from pathlib import Path


def default_data_path() -> Path:
    """默认裁剪后的数据集路径。"""
    return Path(__file__).resolve().parent / "dataset_convert_crop" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO Pose on pre-cropped grid gauge boxes."
    )
    parser.add_argument(
        "--data",
        default=str(default_data_path()),
        help="Cropped YOLO Pose data.yaml path (output from crop_yolo_grid_pose_dataset.py).",
    )
    parser.add_argument(
        "--model",
        default="yolo11m-pose.pt",
        help="Pose pretrained model path/name.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device, mps, or cpu.")
    # parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--project", default="runs/grid_pose")
    parser.add_argument("--name", default="grid_pose_3kpt")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--degrees", type=float, default=10.0, help="Rotation augmentation range (+/- deg)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml_path = Path(args.data).expanduser().resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {data_yaml_path}\n"
            f"Please run crop_yolo_grid_pose_dataset.py first to prepare the cropped dataset."
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModuleNotFoundError(
            "ultralytics is required for training. Install it with: pip install ultralytics"
        ) from exc

    print("=" * 60)
    print("Training YOLO Pose on Cropped Grid Gauge Images")
    print("=" * 60)
    print(f"Data: {data_yaml_path}")
    print(f"Model: {args.model}")
    print(f"Keypoints: empty, full, tip (3 keypoints)")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Rotation: ±{args.degrees}°")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print("=" * 60)
    print()

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=0,  # 禁用早停逻辑
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
        # 轻微旋转和缩放（格子油表方向敏感）
        degrees=args.degrees,
        translate=0.08,
        scale=0.2,
        perspective=0.0002,
        # 禁用水平/垂直翻转（empty/full/tip 有方向语义）
        fliplr=0.0,
        flipud=0.0,
        # 颜色增强
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,
        verbose=True,
    )

    print()
    print("=" * 60)
    print("Training completed")
    print("=" * 60)
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")
    print(f"Last model: {args.project}/{args.name}/weights/last.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
