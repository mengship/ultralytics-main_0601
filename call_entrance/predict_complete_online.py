#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【完整预测流程】YOLO框检测 + ResNet50油量识别

【工作流程】
1. YOLO检测油表框位置
2. 从原图中裁剪框内区域
3. ResNet50识别框内油量
4. 绘制结果并保存

【使用方法】
python predict_complete.py <image_path_or_dir>

【示例】
python predict_complete.py ./test_images
python predict_complete.py image.jpg
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import torchvision.models as models
from torchvision.models import ResNet50_Weights


def expand_box_if_edge_touching(x1, y1, x2, y2, img_h, img_w, expand_ratio=0.1):
    """如果框贴边，自动扩展框以包含更多上下文

    Args:
        x1, y1, x2, y2: 原框坐标
        img_h, img_w: 图片高宽
        expand_ratio: 扩展比例（框大小的比例）

    Returns:
        (x1_new, y1_new, x2_new, y2_new): 扩展后的框坐标
    """

    w = x2 - x1
    h = y2 - y1

    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)

    # 检查并扩展各边界
    x1_new = max(0, x1 - expand_w)
    y1_new = max(0, y1 - expand_h)
    x2_new = min(img_w, x2 + expand_w)
    y2_new = min(img_h, y2 + expand_h)

    # 如果有边贴边，再次扩展对面的边
    if x1_new == 0 and x2_new < img_w:
        x2_new = min(img_w, x2_new + expand_w * 2)
    if x2_new == img_w and x1_new > 0:
        x1_new = max(0, x1_new - expand_w * 2)
    if y1_new == 0 and y2_new < img_h:
        y2_new = min(img_h, y2_new + expand_h * 2)
    if y2_new == img_h and y1_new > 0:
        y1_new = max(0, y1_new - expand_h * 2)

    return x1_new, y1_new, x2_new, y2_new


def is_box_valid(x1, y1, x2, y2, img_h, img_w, min_box_size=50, min_box_ratio=0.3, max_box_ratio=3.0):
    """检查框是否有效

    Args:
        x1, y1, x2, y2: 框的坐标
        img_h, img_w: 图片高度和宽度
        min_box_size: 最小框大小（像素）
        min_box_ratio: 最小框纵横比（高/宽）
        max_box_ratio: 最大框纵横比（高/宽，防止过细的框）

    Returns:
        (is_valid, reason) - 是否有效及原因
    """

    w = x2 - x1
    h = y2 - y1

    # 检查基本尺寸
    if w <= 0 or h <= 0:
        return False, "框尺寸为0"

    if w < min_box_size or h < min_box_size:
        return False, f"框太小 ({w}x{h} < {min_box_size}x{min_box_size})"

    # 检查宽度（框不能太窄）
    aspect_ratio = h / w if w > 0 else float('inf')

    if aspect_ratio < min_box_ratio:
        return False, f"框太宽 (比例 {aspect_ratio:.2f} < {min_box_ratio})"

    if aspect_ratio > max_box_ratio:
        return False, f"框太高 (比例 {aspect_ratio:.2f} > {max_box_ratio})"


    return True, "有效框"


class ResNetFuelNet(nn.Module):
    """ResNet50迁移学习油量识别网络"""

    def __init__(self, pretrained=True):
        super().__init__()

        # 加载预训练的ResNet50（使用新的weights参数）
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = models.resnet50(weights=None)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)





def predict_complete(img_path, result_dir, yolo_conf, min_box_size, min_box_ratio, yolo_weights, resnet_path, img_name=None):
    """完整预测流程：YOLO检测框 + ResNet50识别油量

    Args:
        img_path: 图片路径或目录
        result_dir: 结果保存目录（支持相对路径和绝对路径）
        yolo_conf: YOLO置信度阈值（0-1，默认0.6）
        min_box_size: 最小框大小（像素，默认50）
        min_box_ratio: 最小框纵横比（高/宽，默认0.3，范围0.1-1.0）
    """

    print("\n" + "="*70)
    print("【YOLO + ResNet50 完整预测】")
    print("="*70 + "\n")

    # 获取脚本所在目录
    script_dir = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo'
    if img_name is None:
        img_name = Path(img_path).stem



    # result_dir.mkdir(exist_ok=True)

    # 创建子目录
    # (result_dir / 'detected').mkdir(exist_ok=True)
    # (result_dir / 'crops').mkdir(exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")

    # 打印配置信息
    print(f"⚙️  检测配置:")
    print(f"   YOLO置信度: {yolo_conf:.1%}")
    print(f"   最小框大小: {min_box_size}x{min_box_size} 像素")
    print(f"   最小框比例(高/宽): {min_box_ratio:.1f}\n")

    # ========== 加载YOLO模型 ==========
    print("📦 加载YOLO11m检测模型...\n")

    # yolo_weights = script_dir / 'runs' / 'fuel_yolo' / 'detect' / 'weights' / 'best.pt'


    # if not yolo_weights.exists():
    #     print(f"❌ 找不到YOLO模型: {yolo_weights}")
    #     print("   请先运行: python train_yolo_fuel_resnet.py")
    #     return False

    yolo_model = YOLO(str(yolo_weights))
    print(f"✅ YOLO模型加载成功\n")

    # ========== 加载ResNet模型 ==========
    print("📦 加载ResNet50油量识别模型...\n")





    resnet_model = ResNetFuelNet(pretrained=False)
    resnet_model.load_state_dict(torch.load(str(resnet_path), map_location=device))
    resnet_model.to(device)
    resnet_model.eval()
    print(f"✅ ResNet模型加载成功\n")



    # ========== 预测 ==========
    print("🎯 开始预测...\n")

    results_list = []
    # 读取图片
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  无法读取图片\n")


    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_vis = img_rgb.copy()

    # ========== 第1步：YOLO检测框 ==========
    yolo_results = yolo_model(str(img_path), conf=yolo_conf, verbose=False)

    detected_boxes = []
    fuel_predictions = []

    for result in yolo_results:
        if len(result.boxes) == 0:
            print(f"   ⚠️  未检测到框")
            continue

        # 收集所有检测到的框
        all_boxes = []
        invalid_boxes = []

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            # 限制框在图片范围内
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # 【检查框是否完整】- 传入图片尺寸
            is_valid, reason = is_box_valid(x1, y1, x2, y2, h, w, min_box_size, min_box_ratio)

            if is_valid:
                all_boxes.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'index': i
                })
            else:
                invalid_boxes.append({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'reason': reason
                })

        # 输出被过滤的框
        if invalid_boxes:
            print(f"   ⚠️  过滤掉 {len(invalid_boxes)} 个不完整的框:")
            for invalid in invalid_boxes:
                x1, y1, x2, y2 = invalid['box']
                w_box = x2 - x1
                h_box = y2 - y1
                print(f"      - 置信度 {invalid['conf']:.1%}, 大小 {w_box}x{h_box}, 原因: {invalid['reason']}")

        if not all_boxes:
            print(f"   ⚠️  未检测到有效框\n")
            continue

        # 只保留置信度最高的框
        best_box = max(all_boxes, key=lambda x: x['conf'])
        x1, y1, x2, y2 = best_box['box']
        conf = best_box['conf']
        box_index = best_box['index']

        if len(all_boxes) > 1:
            print(f"   ℹ️  检测到 {len(all_boxes)} 个完整框，仅保留置信度最高的框")

        # ========== 第1.5步：框边界扩展（如果贴边） ==========
        x1, y1, x2, y2 = best_box['box']

        # 检查是否贴边
        margin_left = x1
        margin_top = y1
        margin_right = w - x2
        margin_bottom = h - y2

        is_edge_touch = (margin_left < 10 or margin_top < 10 or
                         margin_right < 10 or margin_bottom < 10)

        if is_edge_touch:
            x1_exp, y1_exp, x2_exp, y2_exp = expand_box_if_edge_touching(x1, y1, x2, y2, h, w, expand_ratio=0.15)
            print(f"   ℹ️  框贴边，已自动扩展")
            print(f"      原框: ({x1}, {y1}, {x2}, {y2})")
            print(f"      扩展后: ({x1_exp}, {y1_exp}, {x2_exp}, {y2_exp})")
            x1, y1, x2, y2 = x1_exp, y1_exp, x2_exp, y2_exp

        # ========== 第2步：ResNet识别油量 ==========
        crop = img_rgb[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (224, 224))

        # 预处理
        img_tensor = torch.from_numpy(crop_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            fuel_pred = resnet_model(img_tensor).item()

        fuel_predictions.append(fuel_pred)
        detected_boxes.append({
            'box': (x1, y1, x2, y2),
            'conf': conf,
            'fuel': fuel_pred
        })

        print(f"   ✅ 框: 置信度 {conf:.1%}, 油量 {fuel_pred:.1%}")

    # 如果没有检测到框
    if not detected_boxes:
        print(f"   ⚠️  未检测到任何框\n")


    # ========== 绘制结果 ==========
    # 绘制检测框和油量信息（上下方都显示）
    for box_info in detected_boxes:
        x1, y1, x2, y2 = box_info['box']
        conf = box_info['conf']
        fuel = box_info['fuel']

        # 绘制框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 准备显示文字
        text_fuel = f"Fuel: {fuel:.1%}"
        text_conf = f"Conf: {conf:.1%}"
        font_scale = 0.9
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 获取文字大小
        (text_w_fuel, text_h), baseline = cv2.getTextSize(text_fuel, font, font_scale, thickness)
        (text_w_conf, _), _ = cv2.getTextSize(text_conf, font, font_scale, thickness)
        text_w = max(text_w_fuel, text_w_conf)
        padding = 5

        # ===== 上方标记 =====
        # 显示两行文字：置信度 + 油量
        total_height = text_h * 2 + padding * 3
        top_bg_y1 = max(0, y1 - total_height - 5)
        top_bg_y2 = y1 - 5
        cv2.rectangle(img_vis, (x1, top_bg_y1), (x1 + text_w + 2 * padding, top_bg_y2), (0, 0, 0), -1)
        # 第一行：置信度（绿色）
        cv2.putText(img_vis, text_conf, (x1 + padding, y1 - text_h - padding - 5), font, font_scale, (0, 255, 0),
                    thickness)
        # 第二行：油量（橙色）
        cv2.putText(img_vis, text_fuel, (x1 + padding, y1 - padding - 5), font, font_scale, (0, 165, 255), thickness)

        # ===== 下方标记 =====
        # 显示两行文字：油量 + 置信度
        bottom_bg_y1 = y2 + 5
        bottom_bg_y2 = min(h, y2 + total_height + 10)
        cv2.rectangle(img_vis, (x1, bottom_bg_y1), (x1 + text_w + 2 * padding, bottom_bg_y2), (0, 0, 0), -1)
        # 第一行：油量（橙色）
        cv2.putText(img_vis, text_fuel, (x1 + padding, y2 + text_h + padding + 5), font, font_scale, (0, 165, 255),
                    thickness)
        # 第二行：置信度（绿色）
        cv2.putText(img_vis, text_conf, (x1 + padding, y2 + text_h * 2 + padding * 2 + 5), font, font_scale,
                    (0, 255, 0), thickness)

    # 保存结果图
    result_img = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    result_path = f"{result_dir}/detected/{img_name}_detected.jpg"
    cv2.imwrite(str(result_path), result_img)
    # print(f"   💾 检测结果已保存: {result_path.name}")

    # 保存裁剪的框内图片
    for i, box_info in enumerate(detected_boxes):
        x1, y1, x2, y2 = box_info['box']
        crop = img[y1:y2, x1:x2]

        fuel = box_info['fuel']
        crop_path =f"{result_dir}/crops/{img_name}_crop_{i + 1}_fuel{fuel:.1%}.jpg"
        cv2.imwrite(str(crop_path), crop)

    # 记录结果
    if fuel_predictions:
        avg_fuel = np.mean(fuel_predictions)
        conf_values = [box['conf'] for box in detected_boxes]
        avg_conf = np.mean(conf_values)
        results_list.append({
            'image': img_name,
            'boxes_count': len(detected_boxes),
            'fuel_values': fuel_predictions,
            'avg_fuel': avg_fuel,
            'conf_values': conf_values,
            'avg_conf': avg_conf,
            'boxes': detected_boxes
        })
        print(f"   📊 平均油量: {avg_fuel:.1%}, 平均置信度: {avg_conf:.1%}\n")




    if not results_list:
        print("没有检测到任何框")
        return False



    return results_list
if __name__ == "__main__":
    image_path = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\image_path\%s.jpg' % pc[9]
    result_dir = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\results_predict'
    yolo_weights = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\models\yolo_best.pt'  # 第一阶段模型
    resnet_path = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\models\fuel_resnet_model.pth'  # 第二阶段模型

    yolo_conf = 0.3  # 默认置信度
    min_box_size = 30  # 默认最小框大小（像素）
    min_box_ratio = 0.4

    try:
        results_list = predict_complete(
            image_path,
            result_dir,
            yolo_conf,
            min_box_size,
            min_box_ratio,
            yolo_weights,
            resnet_path,
            pc[9],
        )
    except NameError:
        results_list = predict_complete(
            image_path,
            result_dir,
            yolo_conf,
            min_box_size,
            min_box_ratio,
            yolo_weights,
            resnet_path,
            Path(image_path).stem,
        )





