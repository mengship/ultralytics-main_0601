#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练 YOLO 检测模型：识别指针和格子油表

第一阶段检测模型，用于二阶段预测流程：
    Stage 1: 检测模型识别油表框和类别 (pointer / grid)
    Stage 2: Pose 模型预测关键点

类别映射：
    0: pointer (指针油表)
    1: grid (格子油表)

注意：这是普通检测模型，不是 pose 模型。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def default_data_path() -> Path:
    """默认检测数据集路径"""
    return Path(__file__).resolve().parent / "dataset_detect" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 YOLO 检测模型：识别指针和格子油表"
    )
    parser.add_argument(
        "--data",
        default=str(default_data_path()),
        help="YOLO detection data.yaml 路径",
    )
    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="检测模型预训练权重（注意：不是 pose 模型）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="训练轮数",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="训练图像尺寸",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="批次大小",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="训练设备：CUDA 设备号、'mps' 或 'cpu'",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=80,
        help="早停轮数",
    )
    parser.add_argument(
        "--project",
        default="runs/gauge_detect",
        help="项目保存目录",
    )
    parser.add_argument(
        "--name",
        default="detect_pointer_grid",
        help="实验名称",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="允许覆盖已存在的项目",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml_path = Path(args.data).expanduser().resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml 未找到: {data_yaml_path}\n"
            f"请先运行 prepare_yolo_detect_dataset.py 生成检测数据集。"
        )

    # 检查是否使用了错误的 pose 模型
    if "pose" in args.model.lower():
        raise ValueError(
            f"错误：这是检测模型训练脚本，不应使用 pose 模型！\n"
            f"当前模型: {args.model}\n"
            f"正确用法: --model yolo11m.pt (不要用 yolo11m-pose.pt)"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModuleNotFoundError(
            "需要安装 ultralytics 才能训练。请运行: pip install ultralytics"
        ) from exc

    print("=" * 70)
    print("训练 YOLO 检测模型：指针 + 格子油表")
    print("=" * 70)
    print(f"数据集: {data_yaml_path}")
    print(f"模型: {args.model}")
    print(f"类别:")
    print(f"  0: pointer (指针油表)")
    print(f"  1: grid (格子油表)")
    print(f"训练轮数: {args.epochs}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"批次大小: {args.batch}")
    print(f"训练设备: {args.device}")
    print(f"早停轮数: {args.patience}")
    print(f"项目目录: {args.project}")
    print(f"实验名称: {args.name}")
    print("=" * 70)
    print()

    # 加载检测模型（注意：不是 pose 模型）
    model = YOLO(args.model)

    # 训练检测模型
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        # 优化器配置
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # 数据增强配置（轻微增强）
        mosaic=0.2,
        close_mosaic=20,
        translate=0.08,
        scale=0.25,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        # 关闭翻转（方向敏感）
        fliplr=0.0,
        flipud=0.0,
        verbose=True,
    )

    print()
    print("=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"最佳权重: {args.project}/{args.name}/weights/best.pt")
    print(f"最后权重: {args.project}/{args.name}/weights/last.pt")
    print()
    print("使用方法：")
    print(f"python call_entrance_pose_grid/predict_fuel_two_stage.py \\")
    print(f"    --det-model {args.project}/{args.name}/weights/best.pt \\")
    print(f"    --pointer-pose-model <指针pose模型> \\")
    print(f"    --grid-pose-model <格子pose模型> \\")
    print(f"    --source <测试图片>")
    print("=" * 70)


if __name__ == "__main__":
    main()
