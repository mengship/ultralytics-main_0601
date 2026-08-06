#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""油表预测器 - 简洁接口（原图版本）

使用示例:
    from fuel_predictor import predict

    result = predict(
        image_path="test.jpg",
        det_model="path/to/det.pt",
        pointer_model="path/to/pointer.pt",
        grid_model="path/to/grid.pt",
        device="0"
    )

    if result["status"] == "ok":
        print(f"类型: {result['fuel_type']}")
        print(f"比例: {result['fuel_percent']:.1f}%")

说明:
    - 使用 predict_fuel_two_stage_fullimage.py 的处理函数
    - 在原图上进行 Pose 预测，坐标为原图空间
"""

from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO

# 导入原图版本的处理函数
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict_fuel_two_stage_fullimage import process_image


# 全局模型缓存（避免重复加载）
_MODEL_CACHE = {}


def predict(
    image_path: str,
    det_model: str,
    pointer_model: str,
    grid_model: str,
    device: str = "0",
    det_conf: float = 0.25,
    pose_conf: float = 0.25,
    imgsz: int = 640,
    box_padding: float = 0.08,
    direction: str = "max_full_span",
    cache_models: bool = True,
) -> Dict:
    """预测单张油表图片

    Args:
        image_path: 图片路径
        det_model: 检测模型路径
        pointer_model: 指针姿态模型路径
        grid_model: 格子姿态模型路径
        device: 设备 ("0", "cpu", "mps")
        det_conf: 检测置信度阈值 (默认 0.25)
        pose_conf: 姿态置信度阈值 (默认 0.25)
        imgsz: 推理图像尺寸 (默认 640)
        box_padding: 检测框扩大比例 (默认 0.08 = 8%)
        direction: 指针油表角度方向规则 (默认 "max_full_span")
        cache_models: 是否缓存模型，避免重复加载 (默认 True)

    Returns:
        预测结果字典:
        {
            "image": "test.jpg",
            "status": "ok",                  # 状态码
            "fuel_type": "pointer",          # "pointer" 或 "grid"
            "fuel_ratio": 0.45,              # 0.0-1.0
            "fuel_percent": 45.0,            # 0.0-100.0
            "det_conf": 0.95,                # 检测置信度
            "pose_conf": 0.88,               # 姿态置信度
            "empty_x": 50.2,                 # 关键点坐标
            "empty_y": 140.5,
            "full_x": 245.7,
            "full_y": 140.8,
            "tip_x": 165.3,
            "tip_y": 195.2,
            ... (其他字段根据类型而定)
        }

    状态码:
        - "ok": 成功
        - "file_not_found": 图片文件不存在
        - "read_error": 图片读取失败
        - "no_det": 未检测到油表
        - "no_pose": 未检测到关键点
        - "unsupported_cls": 不支持的类别
        - "invalid_crop": 裁剪区域无效
    """
    # 验证图片路径
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        return {
            "image": str(image_path.name),
            "status": "file_not_found",
            "fuel_type": None,
            "fuel_ratio": None,
            "fuel_percent": None,
        }

    # 加载模型（带缓存）
    if cache_models:
        cache_key = (det_model, pointer_model, grid_model, device)
        if cache_key not in _MODEL_CACHE:
            det_model_obj = YOLO(det_model)
            pointer_model_obj = YOLO(pointer_model)
            grid_model_obj = YOLO(grid_model)
            _MODEL_CACHE[cache_key] = (det_model_obj, pointer_model_obj, grid_model_obj)
        else:
            det_model_obj, pointer_model_obj, grid_model_obj = _MODEL_CACHE[cache_key]
    else:
        det_model_obj = YOLO(det_model)
        pointer_model_obj = YOLO(pointer_model)
        grid_model_obj = YOLO(grid_model)

    # 构造参数对象
    class Args:
        pass

    args = Args()
    args.det_conf = det_conf
    args.pose_conf = pose_conf
    args.imgsz = imgsz
    args.box_padding = box_padding
    args.direction = direction
    args.save_vis = False
    args.vis_dir = None
    args.crop_dir = None
    args.device = device

    # 调用原有处理函数
    result = process_image(
        image_path,
        det_model_obj,
        pointer_model_obj,
        grid_model_obj,
        args
    )

    return result


def clear_model_cache():
    """清空模型缓存"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


# ============================================================
# 命令行测试接口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="油表预测器 - 简洁接口")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--det-model", required=True, help="检测模型路径")
    parser.add_argument("--pointer-model", required=True, help="指针姿态模型路径")
    parser.add_argument("--grid-model", required=True, help="格子姿态模型路径")
    parser.add_argument("--device", default="0", help="设备")

    args = parser.parse_args()

    print(f"正在预测: {args.image}\n")

    result = predict(
        image_path=args.image,
        det_model=args.det_model,
        pointer_model=args.pointer_model,
        grid_model=args.grid_model,
        device=args.device,
    )

    print("=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"图片: {result['image']}")
    print(f"状态: {result['status']}")

    if result["status"] == "ok":
        print(f"类型: {result['fuel_type']}")
        print(f"比例: {result['fuel_ratio']:.4f}")
        print(f"百分比: {result['fuel_percent']:.1f}%")
        print(f"检测置信度: {result['det_conf']:.3f}")
        print(f"姿态置信度: {result['pose_conf']:.3f}")
    else:
        print(f"失败原因: {result['status']}")

    print("=" * 60)
