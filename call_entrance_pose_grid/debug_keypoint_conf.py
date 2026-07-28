#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试脚本：检查 YOLO Pose 模型是否输出关键点置信度"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="检查 YOLO Pose 关键点置信度")
    parser.add_argument("--model", required=True, help="Pose 模型路径")
    parser.add_argument("--image", required=True, help="测试图像路径")
    args = parser.parse_args()

    print("=" * 70)
    print("YOLO Pose 关键点置信度调试")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"图像: {args.image}")
    print("=" * 70)
    print()

    # 加载模型
    model = YOLO(args.model)

    # 预测
    results = model.predict(args.image, verbose=False)

    if not results or len(results) == 0:
        print("❌ 没有检测结果")
        return

    result = results[0]

    print(f"检测到 {len(result.boxes)} 个目标")
    print()

    if len(result.boxes) == 0:
        print("⚠️  没有检测到目标，请检查：")
        print("   1. 图像中是否有油表")
        print("   2. 模型是否适用于该图像")
        print("   3. 置信度阈值是否过高")
        print()
        print("尝试使用较低的置信度重新预测...")
        results = model.predict(args.image, conf=0.01, verbose=False)
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            result = results[0]
            print(f"✅ 使用 conf=0.01 检测到 {len(result.boxes)} 个目标")
            print()
        else:
            print("❌ 仍然没有检测到目标")
            return

    # 检查 keypoints 结构
    if result.keypoints is None:
        print("❌ result.keypoints 为 None")
        return

    print("✅ result.keypoints 存在")
    print(f"   类型: {type(result.keypoints)}")
    print()

    # 检查 xy
    if result.keypoints.xy is not None:
        print(f"✅ result.keypoints.xy 存在")
        print(f"   形状: {result.keypoints.xy.shape}")
        print(f"   示例数据 [0]: {result.keypoints.xy[0]}")
    else:
        print("❌ result.keypoints.xy 为 None")

    print()

    # 检查 conf
    if result.keypoints.conf is not None:
        print(f"✅ result.keypoints.conf 存在！")
        print(f"   形状: {result.keypoints.conf.shape}")
        print(f"   示例数据 [0]: {result.keypoints.conf[0]}")
        print()
        print("   关键点置信度:")
        for i, conf in enumerate(result.keypoints.conf[0]):
            print(f"     关键点 {i}: {float(conf):.4f}")
    else:
        print("❌ result.keypoints.conf 为 None")
        print()
        print("可能的原因:")
        print("  1. YOLO 版本不支持关键点置信度输出")
        print("  2. 模型训练时配置问题")
        print("  3. ultralytics 版本过旧")
        print()
        print("解决方案:")
        print("  - 升级 ultralytics: pip install -U ultralytics")
        print("  - 使用新版本 YOLO 重新训练模型")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
