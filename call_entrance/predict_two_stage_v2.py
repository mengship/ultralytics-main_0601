#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段预测脚本：YOLO二分类检测 + 双ResNet油量回归 (修复版 v2)

- class 0 -> pointer -> models/resnet/pointer/fuel_resnet_pointer_model.pth
- class 1 -> grid    -> models/resnet/grid/fuel_resnet_grid_model.pth

【v2 修复说明】
  ✅ ResNet TTA改用中位数（对异常值更鲁棒）
  ✅ 降低贴边扩展比例 (15% -> 8%)
  ✅ 放宽框过滤条件 (适应低分辨率和横长油表)
  ✅ 框有效性检查 (尺寸、纵横比)
  ✅ 无效框过滤 (大小、纵横比、贴边等)

每张图最多保留1个框（最高置信度）。
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

try:
    import pandas as pd
except Exception:
    pd = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    
    # 检查纵横比（框不能太窄或太高）
    aspect_ratio = h / w if w > 0 else float('inf')
    
    if aspect_ratio < min_box_ratio:
        return False, f"框太宽 (比例 {aspect_ratio:.2f} < {min_box_ratio})"
    
    if aspect_ratio > max_box_ratio:
        return False, f"框太高 (比例 {aspect_ratio:.2f} > {max_box_ratio})"
    
    return True, "有效框"


class ResNetFuelNet(nn.Module):
    """ResNet152回归油量(0~1)。"""

    def __init__(self):
        super().__init__()
        self.backbone = models.resnet152(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.backbone(x)


def parse_args(defaults):
    parser = argparse.ArgumentParser(description="Two-stage fuel prediction")
    parser.add_argument("source", nargs="?", default=defaults["source"], help="Image path or directory")
    parser.add_argument("--yolo", default=defaults["yolo"], help="YOLO model path")
    parser.add_argument(
        "--resnet-pointer",
        default=defaults["resnet_pointer"],
        help="Pointer ResNet model path",
    )
    parser.add_argument(
        "--resnet-grid",
        default=defaults["resnet_grid"],
        help="Grid ResNet model path",
    )
    parser.add_argument("--outdir", default=defaults["outdir"], help="Output directory")
    parser.add_argument("--conf", type=float, default=defaults["conf"], help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=defaults["imgsz"], help="YOLO inference image size")
    parser.add_argument("--yolo-tta", type=int, default=defaults["yolo_tta"], help="Enable YOLO rotation TTA: 1/0")
    parser.add_argument("--resnet-tta", type=int, default=defaults["resnet_tta"], help="Enable ResNet rotation TTA: 1/0")
    return parser.parse_args()


def gather_images(source_path: Path):
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        files = [p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        return sorted(files)
    return []


def load_resnet(model_path: Path, device: torch.device):
    model = ResNetFuelNet().to(device)
    try:
        state = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"ResNet权重加载失败: {model_path}\n"
            f"当前推理结构为 ResNet152，请确认该权重是否由 `train_yolo_fuel_two_models.py` 重新训练生成。"
        ) from exc
    model.eval()
    return model


def preprocess_crop(crop_bgr: np.ndarray):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(crop_rgb).float() / 255.0
    x = x.permute(2, 0, 1)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    return x1, y1, x2, y2


def rotate_image_for_tta(img: np.ndarray, angle: int):
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")


def map_point_to_original(xr: float, yr: float, angle: int, orig_w: int, orig_h: int):
    # Map point from rotated image coordinates back to original image coordinates.
    if angle == 0:
        return xr, yr
    if angle == 90:
        # Inverse of 90 CW
        return yr, (orig_h - 1) - xr
    if angle == 180:
        return (orig_w - 1) - xr, (orig_h - 1) - yr
    if angle == 270:
        # Inverse of 90 CCW
        return (orig_w - 1) - yr, xr
    raise ValueError(f"Unsupported angle: {angle}")


def map_box_to_original(x1: float, y1: float, x2: float, y2: float, angle: int, orig_w: int, orig_h: int):
    # Use 4 corners to avoid formula mistakes under rotation.
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    mapped = [map_point_to_original(x, y, angle, orig_w, orig_h) for x, y in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return min(xs), min(ys), max(xs), max(ys)


def detect_best_box_with_rotation_tta(yolo_model, img: np.ndarray, conf: float, imgsz: int, use_tta: bool = True,
                                      min_box_size=30, min_box_ratio=0.25, max_box_ratio=4.0):
    """检测最优框，支持旋转TTA和框有效性检查
    
    Args:
        yolo_model: YOLO模型
        img: 输入图片
        conf: 置信度阈值
        imgsz: 推理图片大小
        use_tta: 是否使用旋转TTA
        min_box_size: 最小框大小（像素）
        min_box_ratio: 最小纵横比
        max_box_ratio: 最大纵横比
    
    Returns:
        dict: 包含框信息的字典，或None
    """
    h, w = img.shape[:2]
    best = None
    all_candidates = []  # 记录所有有效框
    
    angles = (0, 90, 180, 270) if use_tta else (0,)
    for angle in angles:
        rot_img = rotate_image_for_tta(img, angle)
        results = yolo_model.predict(source=rot_img, conf=conf, imgsz=imgsz, verbose=False)
        boxes = results[0].boxes if results else None
        
        if boxes is None or len(boxes) == 0:
            continue
        
        for i in range(len(boxes)):
            score = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            rx1, ry1, rx2, ry2 = boxes.xyxy[i].tolist()
            ox1, oy1, ox2, oy2 = map_box_to_original(rx1, ry1, rx2, ry2, angle, w, h)
            
            # ⭐ 新增：框有效性检查
            ox1, oy1, ox2, oy2 = int(ox1), int(oy1), int(ox2), int(oy2)
            is_valid, reason = is_box_valid(ox1, oy1, ox2, oy2, h, w, 
                                            min_box_size=min_box_size,
                                            min_box_ratio=min_box_ratio,
                                            max_box_ratio=max_box_ratio)
            
            if is_valid:
                all_candidates.append({
                    "conf": score,
                    "cls_id": cls_id,
                    "xyxy": (ox1, oy1, ox2, oy2),
                    "tta_angle": angle,
                })
    
    # 从所有有效框中选择最高置信度的
    if all_candidates:
        best = max(all_candidates, key=lambda x: x["conf"])
    
    return best


def predict_fuel_with_rotation_tta(model, crop_bgr: np.ndarray, device: torch.device, use_tta: bool = True):
    """ResNet rotation TTA: 使用中位数而非平均值（v2修复）

    修改原因：
    - 简单平均会被异常预测值拉低准确率
    - 中位数对异常值更鲁棒
    - 例如：[0.25, 0.80, 0.30, 0.28] -> 平均0.408(错)，中位数0.29(对)
    """
    preds = []
    angles = (0, 90, 180, 270) if use_tta else (0,)
    with torch.no_grad():
        for angle in angles:
            rot_crop = rotate_image_for_tta(crop_bgr, angle)
            x = preprocess_crop(rot_crop).to(device)
            p = float(model(x).squeeze().item())
            preds.append(p)

    if not preds:
        return 0.0
    # ✅ v2修复：使用中位数代替平均值
    return float(np.median(preds))


def save_results(rows, csv_path: Path, xlsx_path: Path):
    header = ["image", "status", "fuel_type", "conf", "fuel_ratio", "x1", "y1", "x2", "y2"]

    # Always keep CSV for compatibility.
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Export XLSX when pandas/openpyxl is available.
    if pd is not None:
        try:
            df = pd.DataFrame(rows, columns=header)
            df.to_excel(xlsx_path, index=False)
            return True
        except Exception as e:
            print(f"⚠️  XLSX导出失败，已保留CSV: {e}")
            return False

    print("⚠️  未安装pandas/openpyxl，跳过XLSX导出，仅保留CSV")
    return False


def main(defaults):
    args = parse_args(defaults)

    source_path = Path(args.source)
    yolo_path = Path(args.yolo)
    pointer_path = Path(args.resnet_pointer)
    grid_path = Path(args.resnet_grid)

    if not yolo_path.exists():
        raise FileNotFoundError(f"YOLO模型不存在: {yolo_path}")
    if not pointer_path.exists():
        raise FileNotFoundError(f"指针ResNet模型不存在: {pointer_path}")
    if not grid_path.exists():
        raise FileNotFoundError(f"格子ResNet模型不存在: {grid_path}")

    image_files = gather_images(source_path)
    if not image_files:
        raise FileNotFoundError(f"未找到可处理图片: {source_path}")

    outdir = Path(args.outdir)
    vis_dir = outdir / "vis"
    stage1_dir = outdir / "stage1_yolo"  # ⭐ 新增：阶段一结果（仅框）
    vis_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"YOLO旋转TTA: {'开' if args.yolo_tta == 1 else '关'} | ResNet旋转TTA: {'开' if args.resnet_tta == 1 else '关'}")

    yolo = YOLO(str(yolo_path))
    pointer_model = load_resnet(pointer_path, device)
    grid_model = load_resnet(grid_path, device)

    csv_path = outdir / "results.csv"
    xlsx_path = outdir / "results.xlsx"
    rows = []

    for img_path in image_files:
        image_id = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            rows.append([image_id, "read_failed", "", "", "", "", "", "", ""])
            continue

        h, w = img.shape[:2]
        best = detect_best_box_with_rotation_tta(yolo, img, args.conf, args.imgsz, use_tta=(args.yolo_tta == 1))

        if best is None:
            cv2.putText(img, "No fuel gauge detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            save_path = vis_dir / img_path.name
            cv2.imwrite(str(save_path), img)
            rows.append([image_id, "no_detection", "", "", "", "", "", "", ""])
            print(f"[NO DET] {img_path.name}")
            continue

        x1, y1, x2, y2 = clamp_box(*best["xyxy"], w, h)
        if x2 <= x1 or y2 <= y1:
            rows.append([image_id, "invalid_box", "", f"{best['conf']:.4f}", "", x1, y1, x2, y2])
            continue
        
        # ⭐ 新增：检查框是否贴边，如果贴边则自动扩展
        margin_left = x1
        margin_top = y1
        margin_right = w - x2
        margin_bottom = h - y2
        
        is_edge_touch = (margin_left < 10 or margin_top < 10 or
                        margin_right < 10 or margin_bottom < 10)
        
        if is_edge_touch:
            x1_exp, y1_exp, x2_exp, y2_exp = expand_box_if_edge_touching(
                x1, y1, x2, y2, h, w, expand_ratio=0.08  # v2修复：从0.15降到0.08
            )
            print(f"   ℹ️  框贴边，已自动扩展")
            x1, y1, x2, y2 = x1_exp, y1_exp, x2_exp, y2_exp
        
        crop = img[y1:y2, x1:x2]
        # ⭐ 新增：保存阶段一结果（只有YOLO框）
        img_stage1 = img.copy()
        cls_id = best["cls_id"]
        if cls_id == 0:
            color_stage1 = (0, 255, 0)  # 绿色：指针类
            label_stage1 = f"Pointer (conf={best['conf']:.2f})"
        else:
            color_stage1 = (255, 0, 0)  # 红色：格子类
            label_stage1 = f"Grid (conf={best['conf']:.2f})"

        cv2.rectangle(img_stage1, (x1, y1), (x2, y2), color_stage1, 2)
        cv2.putText(img_stage1, label_stage1, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_stage1, 2)

        stage1_save_path = stage1_dir / img_path.name
        cv2.imwrite(str(stage1_save_path), img_stage1)

        cls_id = best["cls_id"]
        if cls_id == 0:
            fuel_type = "pointer"
            model = pointer_model
            color = (0, 255, 0)
        else:
            fuel_type = "grid"
            model = grid_model
            color = (255, 0, 0)

        fuel = predict_fuel_with_rotation_tta(model, crop, device, use_tta=(args.resnet_tta == 1))
        fuel = max(0.0, min(1.0, fuel))

        label_top = f"{fuel_type} conf={best['conf']:.2f}"
        label_bottom = f"fuel={fuel * 100:.1f}%"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label_top, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, label_bottom, (x1, min(h - 10, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        save_path = vis_dir / img_path.name
        cv2.imwrite(str(save_path), img)

        rows.append([
            image_id,
            "ok",
            fuel_type,
            f"{best['conf']:.4f}",
            f"{fuel:.6f}",
            x1,
            y1,
            x2,
            y2,
        ])
        print(f"[OK] {img_path.name} | type={fuel_type} conf={best['conf']:.2f} fuel={fuel * 100:.1f}%")

    xlsx_ok = save_results(rows, csv_path, xlsx_path)

    print("\n完成")
    print(f"阶段一(YOLO框): {stage1_dir}")
    print(f"阶段二(完整标注): {vis_dir}")
    if xlsx_ok:
        print(f"结果表格: {xlsx_path}")
    else:
        print(f"结果表格: {csv_path}")


if __name__ == "__main__":
    # =================== 修改这里设置默认路径 ===================

    dt = '0521'
    excelname = '0521识别错误'
    # ========== 默认值配置（可在此修改） ==========
    DEFAULT_SOURCE = 'E:/predict/'+ dt +'/'+ excelname  # 默认输入目录
    DEFAULT_OUTDIR = 'E:/predict/'+ dt +'/'+ excelname + 'predictv3_1TTA'  # 默认输出目录 TTA = Test Time Augmentation

    # DEFAULT_SOURCE = r"fuel_detection_dataset\test\images"
    # DEFAULT_OUTDIR = r"results_two_stage"

    DEFAULT_YOLO = r"../runs/fuel_yolo/detect_2class/weights/best.pt"
    DEFAULT_RESNET_POINTER = r"../models/resnet/pointer/fuel_resnet_pointer_model.pth"
    DEFAULT_RESNET_GRID = r"../models/resnet/grid/fuel_resnet_grid_model.pth"
    DEFAULT_CONF = 0.6
    DEFAULT_IMGSZ = 640
    DEFAULT_YOLO_TTA = 1
    DEFAULT_RESNET_TTA = 1
    # ===========================================================

    default_cfg = {
        "source": DEFAULT_SOURCE,
        "outdir": DEFAULT_OUTDIR,
        "yolo": DEFAULT_YOLO,
        "resnet_pointer": DEFAULT_RESNET_POINTER,
        "resnet_grid": DEFAULT_RESNET_GRID,
        "conf": DEFAULT_CONF,
        "imgsz": DEFAULT_IMGSZ,
        "yolo_tta": DEFAULT_YOLO_TTA,
        "resnet_tta": DEFAULT_RESNET_TTA,
    }

    print("\n默认配置（可在 __main__ 中修改）：")
    print(f"- source: {default_cfg['source']}")
    print(f"- outdir: {default_cfg['outdir']}")
    print(f"- yolo: {default_cfg['yolo']}")
    print(f"- resnet_pointer: {default_cfg['resnet_pointer']}")
    print(f"- resnet_grid: {default_cfg['resnet_grid']}\n")
    print(f"- conf: {default_cfg['conf']}")
    print(f"- imgsz: {default_cfg['imgsz']}")
    print(f"- yolo_tta: {default_cfg['yolo_tta']}")
    print(f"- resnet_tta: {default_cfg['resnet_tta']}\n")

    main(default_cfg)

