#!/usr/bin/env python3
"""Fuel two-stage deployment launcher."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="识别单张油表图片")
    parser.add_argument("--source", required=True, help="待识别的单张图片路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"请输入有效的单张图片路径，不支持目录: {source}")

    command = [
        sys.executable,
        str(script_dir / "predict_fuel_two_stage.py"),
        "--det-model",
        str(script_dir / "models" / "detector.pt"),
        "--pointer-pose-model",
        str(script_dir / "models" / "pointer_pose.pt"),
        "--grid-pose-model",
        str(script_dir / "models" / "grid_pose.pt"),
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
        str(script_dir / "output" / "fuel_two_stage_predictions.csv"),
        "--device",
        os.environ.get("DEVICE", "cpu"),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
