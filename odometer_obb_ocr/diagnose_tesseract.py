#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tesseract 诊断脚本 - 测试不同 PSM 模式和配置"""

import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import pytesseract
except ImportError:
    print("❌ pytesseract 未安装")
    print("运行: pip install pytesseract")
    sys.exit(1)


def test_tesseract_installation():
    """测试 Tesseract 是否正确安装"""
    print("=" * 60)
    print("1. 测试 Tesseract 安装")
    print("=" * 60)
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract 版本: {version}")
        return True
    except Exception as e:
        print(f"❌ Tesseract 未正确安装: {e}")
        return False


def test_crop_image(crop_path):
    """测试 crop 图片"""
    print("\n" + "=" * 60)
    print("2. 测试 Crop 图片")
    print("=" * 60)

    crop = cv2.imread(crop_path)
    if crop is None:
        print(f"❌ 无法读取图片: {crop_path}")
        return None

    h, w = crop.shape[:2]
    print(f"✅ 图片尺寸: {w}×{h}")
    print(f"   宽高比: {h/w:.2f} {'(垂直)' if h > w * 1.2 else '(水平)'}")

    return crop


def test_psm_modes(crop):
    """测试不同的 PSM 模式"""
    print("\n" + "=" * 60)
    print("3. 测试不同 PSM 模式")
    print("=" * 60)

    psm_modes = {
        3: "完全自动分割 (默认)",
        4: "可变大小单列",
        6: "单列垂直文本",
        11: "稀疏文本",
        12: "稀疏文本 + OSD",
    }

    results = {}

    for psm, desc in psm_modes.items():
        print(f"\n测试 PSM {psm}: {desc}")

        config = f'--oem 1 --psm {psm}'

        try:
            text = pytesseract.image_to_string(crop, config=config).strip()
            data = pytesseract.image_to_data(
                crop, config=config, output_type=pytesseract.Output.DICT
            )

            # 提取置信度
            confidences = [int(c) for c in data['conf'] if int(c) > 0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            print(f"  识别文本: '{text}'")
            print(f"  平均置信度: {avg_conf:.2f}%")

            results[psm] = {
                'text': text,
                'conf': avg_conf,
                'desc': desc
            }

        except Exception as e:
            print(f"  ❌ 错误: {e}")
            results[psm] = {'text': '', 'conf': 0, 'desc': desc}

    return results


def test_with_whitelist(crop):
    """测试带字符白名单"""
    print("\n" + "=" * 60)
    print("4. 测试字符白名单")
    print("=" * 60)

    configs = {
        '无白名单': '--oem 1 --psm 6',
        '数字+km': '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789kmKM',
        '仅数字': '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789',
    }

    for name, config in configs.items():
        print(f"\n{name}: {config}")
        try:
            text = pytesseract.image_to_string(crop, config=config).strip()
            print(f"  识别结果: '{text}'")
        except Exception as e:
            print(f"  ❌ 错误: {e}")


def recommend_best_config(results):
    """推荐最佳配置"""
    print("\n" + "=" * 60)
    print("5. 推荐配置")
    print("=" * 60)

    # 找出识别文本最长且置信度较高的
    best = max(results.items(), key=lambda x: (len(x[1]['text']), x[1]['conf']))

    psm, result = best
    print(f"\n✅ 推荐使用 PSM {psm}: {result['desc']}")
    print(f"   识别文本: '{result['text']}'")
    print(f"   置信度: {result['conf']:.2f}%")

    return psm


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tesseract 诊断工具")
    parser.add_argument("--crop", required=True, help="Crop 图片路径")
    args = parser.parse_args()

    # 1. 测试安装
    if not test_tesseract_installation():
        return

    # 2. 加载图片
    crop = test_crop_image(args.crop)
    if crop is None:
        return

    # 3. 测试不同 PSM 模式
    results = test_psm_modes(crop)

    # 4. 测试字符白名单
    test_with_whitelist(crop)

    # 5. 推荐配置
    best_psm = recommend_best_config(results)

    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
    print(f"\n如需修改配置，编辑: utils/ocr.py 第238行")
    print(f"将 '--psm 6' 改为 '--psm {best_psm}'")


if __name__ == "__main__":
    main()
