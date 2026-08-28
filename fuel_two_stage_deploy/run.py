#!/usr/bin/env python3
"""Fuel two-stage deployment launcher."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


# 在这里指定每次需要识别的单张图片。
SOURCE_IMAGE = "/home/wang/datasets/output0825/2026-08-25/77912.jpg"



def main() -> None:
    script_dir = Path(__file__).resolve().parent
    source = Path(SOURCE_IMAGE).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"请在 SOURCE_IMAGE 中配置有效的单张图片路径: {source}")

    output_csv = script_dir / "output" / "fuel_two_stage_predictions.csv"
    command = [
        sys.executable,
        str(script_dir / "predict_fuel_two_stage.py"),
        "--det-model",
        str("/home/wang/ultralytics-main_0601/runs/detect/runs/gauge_detect/detect_pointer_grid-4/weights/best.pt"),
        "--pointer-pose-model",
        str("/home/wang/ultralytics-main_0601/runs/pose/runs/fuel_pose/pose_crop_4kpt-16/weights/best.pt"),
        "--grid-pose-model",
        str("/home/wang/ultralytics-main_0601/runs/pose/runs/grid_pose/grid_pose_3kpt-10/weights/best.pt"),
        "--source",
        str(source),
        "--det-conf",
        "0.25",
        "--pose-conf",
        "0.25",
        "--min-keypoint-conf",
        "0.6",
        "--box-padding",
        "0.08",
        "--direction",
        "max_full_span",
        "--output-csv",
        str(output_csv),
        "--device",
        os.environ.get("DEVICE", "cpu"),
    ]
    subprocess.run(command, check=True)

    with output_csv.open("r", encoding="utf-8", newline="") as file:
        result = next(csv.DictReader(file), None)
    if result is None:
        raise RuntimeError(f"没有读取到预测结果: {output_csv}")

    print("\n模型输出结果:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
