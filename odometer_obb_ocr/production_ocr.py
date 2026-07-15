#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生产环境里程表OCR识别脚本

使用方法:
    python production_ocr.py --image /path/to/image.jpg
    python production_ocr.py --image /path/to/image.jpg --model /path/to/best.pt

返回JSON格式:
    {
        "success": true,
        "mileage": "30594",
        "confidence": 0.853,
        "status": "ok",
        "error": null
    }
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO
from utils import geometry, ocr as ocr_module
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="里程表OCR识别")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument(
        "--model",
        default="runs/obb/odometer/weights/best.pt",
        help="YOLO OBB模型路径"
    )
    parser.add_argument("--det-conf", type=float, default=0.25, help="检测置信度阈值")
    parser.add_argument("--ocr-conf", type=float, default=0.70, help="OCR置信度阈值")
    parser.add_argument("--crop-padding", type=float, default=0.60, help="裁剪padding比例")
    parser.add_argument("--min-digits", type=int, default=4, help="最小数字位数")
    parser.add_argument("--max-digits", type=int, default=8, help="最大数字位数")
    parser.add_argument("--save-crop", help="保存裁剪图路径(可选)")
    return parser.parse_args()


def recognize_odometer(image_path, model_path, det_conf=0.25, ocr_conf=0.70,
                       crop_padding=0.60, min_digits=4, max_digits=8, save_crop=None):
    """
    识别里程表数字

    Args:
        image_path: 输入图片路径
        model_path: YOLO模型路径
        det_conf: 检测置信度阈值
        ocr_conf: OCR置信度阈值
        crop_padding: 裁剪padding比例
        min_digits: 最小数字位数
        max_digits: 最大数字位数
        save_crop: 保存裁剪图路径(可选)

    Returns:
        dict: {
            "success": bool,
            "mileage": str,
            "confidence": float,
            "status": str,
            "error": str
        }
    """
    try:
        # 1. 加载模型
        model = YOLO(model_path)

        # 2. YOLO检测
        results = model.predict(source=image_path, conf=det_conf, verbose=False)

        if len(results) == 0:
            return {
                "success": False,
                "mileage": None,
                "confidence": 0.0,
                "status": "no_detection",
                "error": "未检测到里程表区域"
            }

        result = results[0]
        obb = result.obb

        if obb is None or obb.conf is None or len(obb.conf) == 0:
            return {
                "success": False,
                "mileage": None,
                "confidence": 0.0,
                "status": "no_detection",
                "error": "未检测到里程表区域"
            }

        # 3. 选择最佳检测框
        conf_arr = obb.conf.cpu().numpy()
        idx = geometry.select_best_detection(conf_arr, det_conf)

        if idx is None:
            return {
                "success": False,
                "mileage": None,
                "confidence": 0.0,
                "status": "no_detection",
                "error": "检测置信度过低"
            }

        det_confidence = float(conf_arr[idx])
        raw_quad = obb.xyxyxyxy[idx].cpu().numpy().tolist()

        # 4. 几何验证和排序
        try:
            valid = geometry.validate_quad_for_inference(raw_quad)
            ordered = geometry.order_tl_tr_br_bl(valid)
        except geometry.GeometryError as e:
            return {
                "success": False,
                "mileage": None,
                "confidence": det_confidence,
                "status": "invalid_geometry",
                "error": f"几何验证失败: {e}"
            }

        # 5. 透视矫正
        image = cv2.imread(image_path)
        if image is None:
            return {
                "success": False,
                "mileage": None,
                "confidence": 0.0,
                "status": "error",
                "error": f"无法读取图片: {image_path}"
            }

        try:
            crop = geometry.rectify_quad(image, ordered, crop_padding)
        except geometry.GeometryError as e:
            return {
                "success": False,
                "mileage": None,
                "confidence": det_confidence,
                "status": "invalid_geometry",
                "error": f"透视矫正失败: {e}"
            }

        # 保存裁剪图(可选)
        if save_crop:
            cv2.imwrite(save_crop, crop)

        # 6. OCR识别
        try:
            ocr_result = ocr_module.recognize("paddle", crop)
        except Exception as e:
            return {
                "success": False,
                "mileage": None,
                "confidence": det_confidence,
                "status": "ocr_error",
                "error": f"OCR失败: {e}"
            }

        # 7. 提取数字
        digits = ocr_module.extract_digits(ocr_result.raw_text)

        # 8. 验证
        if ocr_result.confidence < ocr_conf:
            return {
                "success": False,
                "mileage": digits if digits else None,
                "confidence": ocr_result.confidence,
                "status": "low_ocr_confidence",
                "error": f"OCR置信度过低: {ocr_result.confidence:.3f} < {ocr_conf}"
            }

        if not (min_digits <= len(digits) <= max_digits):
            return {
                "success": False,
                "mileage": digits if digits else None,
                "confidence": ocr_result.confidence,
                "status": "invalid_digit_count",
                "error": f"数字位数不符: {len(digits)} 不在 [{min_digits}, {max_digits}]"
            }

        # 9. 成功
        return {
            "success": True,
            "mileage": digits,
            "confidence": ocr_result.confidence,
            "status": "ok",
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "mileage": None,
            "confidence": 0.0,
            "status": "error",
            "error": f"未知错误: {e}"
        }


def main():
    args = parse_args()

    result = recognize_odometer(
        image_path=args.image,
        model_path=args.model,
        det_conf=args.det_conf,
        ocr_conf=args.ocr_conf,
        crop_padding=args.crop_padding,
        min_digits=args.min_digits,
        max_digits=args.max_digits,
        save_crop=args.save_crop
    )

    # 输出JSON
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # 返回状态码
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
