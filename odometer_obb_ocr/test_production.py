#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生产环境部署测试脚本"""

import sys
from pathlib import Path

def test_imports():
    """测试依赖包导入"""
    print("=" * 60)
    print("1. 测试依赖包")
    print("=" * 60)

    packages = {
        "ultralytics": "YOLO模型",
        "paddlepaddle": "PaddlePaddle",
        "paddleocr": "PaddleOCR",
        "cv2": "OpenCV",
        "numpy": "NumPy"
    }

    failed = []
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"✅ {name:20} OK")
        except ImportError:
            print(f"❌ {name:20} 未安装")
            failed.append(pkg)

    if failed:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(failed)}")
        return False

    return True


def test_project_files():
    """测试项目文件"""
    print("\n" + "=" * 60)
    print("2. 测试项目文件")
    print("=" * 60)

    required_files = [
        "production_ocr.py",
        "utils/__init__.py",
        "utils/geometry.py",
        "utils/ocr.py",
    ]

    failed = []
    for file in required_files:
        path = Path(__file__).parent / file
        if path.exists():
            print(f"✅ {file:30} 存在")
        else:
            print(f"❌ {file:30} 缺失")
            failed.append(file)

    if failed:
        print(f"\n缺失必需文件，请检查项目结构")
        return False

    return True


def test_model():
    """测试模型文件"""
    print("\n" + "=" * 60)
    print("3. 测试模型文件")
    print("=" * 60)

    model_path = Path(__file__).parent / "runs/obb/odometer/weights/best.pt"

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ 模型文件存在: {model_path}")
        print(f"   大小: {size_mb:.2f} MB")
        return True
    else:
        print(f"⚠️  模型文件不存在: {model_path}")
        print(f"   请将模型放置在正确位置，或使用 --model 指定路径")
        return False


def test_ocr_function():
    """测试OCR函数"""
    print("\n" + "=" * 60)
    print("4. 测试OCR函数")
    print("=" * 60)

    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from production_ocr import recognize_odometer
        print(f"✅ OCR函数导入成功")
        return True
    except Exception as e:
        print(f"❌ OCR函数导入失败: {e}")
        return False


def main():
    print("\n🚀 生产环境部署测试\n")

    tests = [
        ("依赖包检查", test_imports),
        ("项目文件检查", test_project_files),
        ("模型文件检查", test_model),
        ("OCR函数检查", test_ocr_function),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} 失败: {e}")
            results.append((name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status:10} {name}")

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有测试通过！可以开始使用生产环境脚本。")
        print("\n快速开始:")
        print("  python production_ocr.py --image your_image.jpg")
    else:
        print("\n⚠️  部分测试失败，请先解决上述问题。")
        sys.exit(1)


if __name__ == "__main__":
    main()
