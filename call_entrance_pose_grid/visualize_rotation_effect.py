#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""可视化旋转增强效果

测试 YOLO 训练时旋转增强对 crop 小图的影响
"""

import cv2
import numpy as np
from pathlib import Path
import argparse


def rotate_image_keep_size(image, angle):
    """旋转图像并保持原始尺寸（模拟 YOLO 训练行为）

    Args:
        image: 输入图像
        angle: 旋转角度（正值为逆时针）

    Returns:
        旋转后的图像（尺寸不变）
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 旋转并用灰色填充（模拟 YOLO 默认行为）
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114)  # YOLO 默认填充值
    )

    return rotated


def rotate_image_expand_canvas(image, angle):
    """旋转图像并扩大画布以容纳完整内容（无裁剪）

    Args:
        image: 输入图像
        angle: 旋转角度（正值为逆时针）

    Returns:
        旋转后的图像（尺寸扩大），新的宽高
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 获取旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的边界框尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵的平移部分
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 旋转并用灰色填充
    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(114, 114, 114)
    )

    return rotated, new_w, new_h


def rotate_keypoints(keypoints, angle, image_width, image_height, new_width=None, new_height=None):
    """旋转关键点坐标

    Args:
        keypoints: [(x, y), ...] 关键点坐标列表
        angle: 旋转角度（正值为逆时针）
        image_width: 原始图像宽度
        image_height: 原始图像高度
        new_width: 旋转后的新宽度（扩大画布模式）
        new_height: 旋转后的新高度（扩大画布模式）

    Returns:
        旋转后的关键点坐标
    """
    center_x = image_width / 2
    center_y = image_height / 2

    angle_rad = np.radians(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    rotated_kpts = []
    for x, y in keypoints:
        # 平移到原点
        x_shifted = x - center_x
        y_shifted = y - center_y

        # 旋转
        x_rot = x_shifted * cos_a - y_shifted * sin_a
        y_rot = x_shifted * sin_a + y_shifted * cos_a

        # 如果画布扩大了，需要调整中心点
        if new_width is not None and new_height is not None:
            x_new = x_rot + new_width / 2
            y_new = y_rot + new_height / 2
        else:
            # 平移回去（原始尺寸）
            x_new = x_rot + center_x
            y_new = y_rot + center_y

        rotated_kpts.append((x_new, y_new))

    return rotated_kpts


def draw_keypoints(image, keypoints, labels, color=(0, 255, 0)):
    """在图像上绘制关键点

    Args:
        image: 图像
        keypoints: [(x, y), ...] 关键点列表
        labels: 关键点标签列表
        color: 绘制颜色
    """
    result = image.copy()

    for (x, y), label in zip(keypoints, labels):
        # 检查关键点是否在图像内
        h, w = image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            # 绘制圆点
            cv2.circle(result, (int(x), int(y)), 5, color, -1)
            # 绘制标签
            cv2.putText(
                result,
                label,
                (int(x) + 8, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        else:
            # 关键点超出边界，绘制红色警告
            print(f"警告：关键点 {label} 超出边界: ({x:.1f}, {y:.1f})")

    return result


def visualize_rotation_effect(image_path, angles=[0, 5, 10, 15], expand_canvas=False):
    """可视化不同旋转角度的效果

    Args:
        image_path: 输入图像路径
        angles: 要测试的旋转角度列表
        expand_canvas: 是否扩大画布以保留完整内容（True=无裁剪，False=保持原尺寸）
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    h, w = image.shape[:2]
    print(f"图像尺寸: {w}x{h}")
    print(f"旋转模式: {'扩大画布（无裁剪）' if expand_canvas else '保持尺寸（会裁剪）'}")

    # 模拟关键点（假设在图像中心附近）
    # 格子油表：empty, full, tip
    keypoints = [
        (w * 0.3, h * 0.5),  # empty (左侧)
        (w * 0.7, h * 0.5),  # full (右侧)
        (w * 0.7, h * 0.5),  # tip (当前位置，这里假设在 full 附近)
    ]
    labels = ['empty', 'full', 'tip']

    # 创建输出目录
    output_dir = Path('rotation_test_output')
    output_dir.mkdir(exist_ok=True)

    results = []

    for angle in angles:
        print(f"\n旋转角度: {angle}°")

        if expand_canvas:
            # 旋转图像（扩大画布）
            rotated_img, new_w, new_h = rotate_image_expand_canvas(image, angle)
            # 旋转关键点
            rotated_kpts = rotate_keypoints(keypoints, angle, w, h, new_w, new_h)
            print(f"  新尺寸: {new_w}x{new_h}")
        else:
            # 旋转图像（保持尺寸）
            rotated_img = rotate_image_keep_size(image, angle)
            # 旋转关键点
            rotated_kpts = rotate_keypoints(keypoints, angle, w, h)
            new_w, new_h = w, h

        # 绘制关键点
        vis_img = draw_keypoints(rotated_img, rotated_kpts, labels)

        # 添加标题
        mode_text = "Expand Canvas" if expand_canvas else "Keep Size"
        title = f"Rotation: {angle}deg ({mode_text})"
        cv2.putText(
            vis_img,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        # 计算黑边占比（近似）
        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 120)  # 接近黑色的像素
        total_pixels = gray.size
        black_ratio = black_pixels / total_pixels * 100

        info = f"Black area: {black_ratio:.1f}%"
        cv2.putText(
            vis_img,
            info,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        print(f"  黑边占比: {black_ratio:.1f}%")

        # 检查关键点是否超出边界
        out_of_bounds = []
        for (x, y), label in zip(rotated_kpts, labels):
            if x < 0 or x >= new_w or y < 0 or y >= new_h:
                out_of_bounds.append(label)

        if out_of_bounds:
            print(f"  ⚠️  超出边界的关键点: {', '.join(out_of_bounds)}")
        else:
            print(f"  ✓ 所有关键点在图像内")

        # 保存结果
        mode_suffix = "_expand" if expand_canvas else "_crop"
        output_path = output_dir / f"rotation_{angle}deg{mode_suffix}.jpg"
        cv2.imwrite(str(output_path), vis_img)
        print(f"  已保存: {output_path}")

        results.append(vis_img)

    # 创建对比图（需要统一尺寸）
    if not expand_canvas and len(results) <= 4:
        # 保持尺寸模式：可以直接拼接
        if len(results) >= 2:
            row1 = np.hstack(results[:2])
        else:
            row1 = results[0]

        if len(results) >= 4:
            row2 = np.hstack(results[2:4])
            comparison = np.vstack([row1, row2])
        elif len(results) >= 3:
            row2 = results[2]
            # 补齐宽度
            if row2.shape[1] < row1.shape[1]:
                pad_width = row1.shape[1] - row2.shape[1]
                row2 = np.hstack([row2, np.zeros((row2.shape[0], pad_width, 3), dtype=np.uint8)])
            comparison = np.vstack([row1, row2])
        else:
            comparison = row1

        mode_suffix = "_expand" if expand_canvas else "_crop"
        comparison_path = output_dir / f"comparison{mode_suffix}.jpg"
        cv2.imwrite(str(comparison_path), comparison)
        print(f"\n对比图已保存: {comparison_path}")

    print(f"\n所有结果保存在: {output_dir.absolute()}")



def main():
    parser = argparse.ArgumentParser(description="可视化旋转增强效果")
    parser.add_argument(
        "--image",
        required=True,
        help="输入图像路径（crop 后的小图）",
    )
    parser.add_argument(
        "--angles",
        nargs="+",
        type=float,
        default=[0, 5, 10, 15],
        help="要测试的旋转角度列表",
    )
    parser.add_argument(
        "--expand-canvas",
        action="store_true",
        help="扩大画布以保留完整内容（无裁剪）",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("可视化旋转增强效果")
    print("=" * 70)
    print(f"输入图像: {args.image}")
    print(f"测试角度: {args.angles}")
    print("=" * 70)
    print()

    visualize_rotation_effect(args.image, args.angles, args.expand_canvas)


if __name__ == "__main__":
    main()
