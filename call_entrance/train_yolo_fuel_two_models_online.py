#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段线上推理脚本：YOLO二分类检测 + 双ResNet油量回归

说明：
- 这是 `train_yolo_fuel_two_models.py` 对应的线上版本
- 仅保留推理流程，不包含训练逻辑
- 目录下直接运行即可，默认读取项目根目录下的模型文件
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


def expand_box_if_edge_touching(x1, y1, x2, y2, img_h, img_w, expand_ratio=0.15):
    w = x2 - x1
    h = y2 - y1
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)
    x1_new = max(0, x1 - expand_w)
    y1_new = max(0, y1 - expand_h)
    x2_new = min(img_w, x2 + expand_w)
    y2_new = min(img_h, y2 + expand_h)
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
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False, "框尺寸为0"
    if w < min_box_size or h < min_box_size:
        return False, f"框太小 ({w}x{h})"
    aspect_ratio = h / w if w > 0 else float('inf')
    if aspect_ratio < min_box_ratio:
        return False, f"框太宽 ({aspect_ratio:.2f})"
    if aspect_ratio > max_box_ratio:
        return False, f"框太高 ({aspect_ratio:.2f})"
    return True, "有效框"


class ResNetFuelNet(nn.Module):
    """ResNet152油量识别网络"""

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
    parser = argparse.ArgumentParser(description="Two-stage fuel online prediction")
    parser.add_argument("source", nargs="?", default=defaults["source"], help="Image path or directory")
    parser.add_argument("--yolo", default=defaults["yolo"], help="YOLO model path")
    parser.add_argument("--resnet-pointer", default=defaults["resnet_pointer"], help="Pointer ResNet model path")
    parser.add_argument("--resnet-grid", default=defaults["resnet_grid"], help="Grid ResNet model path")
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
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess_crop(crop_bgr: np.ndarray):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(crop_rgb).float() / 255.0
    x = x.permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    x = (x - mean) / std
    return x.unsqueeze(0)


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
    if angle == 0:
        return xr, yr
    if angle == 90:
        return yr, (orig_h - 1) - xr
    if angle == 180:
        return (orig_w - 1) - xr, (orig_h - 1) - yr
    if angle == 270:
        return (orig_w - 1) - yr, xr
    raise ValueError(f"Unsupported angle: {angle}")


def map_box_to_original(x1: float, y1: float, x2: float, y2: float, angle: int, orig_w: int, orig_h: int):
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    mapped = [map_point_to_original(x, y, angle, orig_w, orig_h) for x, y in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return min(xs), min(ys), max(xs), max(ys)


def detect_best_box_with_rotation_tta(yolo_model, img: np.ndarray, conf: float, imgsz: int, use_tta: bool = True,
                                      min_box_size=50, min_box_ratio=0.3, max_box_ratio=3.0):
    h, w = img.shape[:2]
    all_candidates = []
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
            ox1, oy1, ox2, oy2 = int(ox1), int(oy1), int(ox2), int(oy2)
            is_valid, _ = is_box_valid(ox1, oy1, ox2, oy2, h, w,
                                       min_box_size=min_box_size,
                                       min_box_ratio=min_box_ratio,
                                       max_box_ratio=max_box_ratio)
            if is_valid:
                all_candidates.append({"conf": score, "cls_id": cls_id, "xyxy": (ox1, oy1, ox2, oy2), "tta_angle": angle})
    return max(all_candidates, key=lambda x: x["conf"]) if all_candidates else None


def predict_fuel_with_rotation_tta(model, crop_bgr: np.ndarray, device: torch.device, use_tta: bool = True):
    preds = []
    angles = (0, 90, 180, 270) if use_tta else (0,)
    with torch.no_grad():
        for angle in angles:
            rot_crop = rotate_image_for_tta(crop_bgr, angle)
            x = preprocess_crop(rot_crop).to(device)
            preds.append(float(model(x).squeeze().item()))
    return float(np.mean(preds)) if preds else 0.0


def save_results(rows, csv_path: Path, xlsx_path: Path):
    header = ["image", "status", "fuel_type", "conf", "fuel_ratio", "x1", "y1", "x2", "y2"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    if pd is not None:
        try:
            pd.DataFrame(rows, columns=header).to_excel(xlsx_path, index=False)
            return True
        except Exception:
            return False
    return False


def predict_complete(image_path, result_dir, yolo_conf, min_box_size, min_box_ratio, yolo_weights, resnet_pointer_path, resnet_grid_path, img_name,
                     imgsz=640, yolo_tta=1, resnet_tta=1):
    """兼容旧调用方式的线上入口。"""
    defaults = {
        "source": image_path,
        "outdir": result_dir,
        "yolo": yolo_weights,
        "resnet_pointer": resnet_pointer_path,
        "resnet_grid": resnet_grid_path,
        "conf": yolo_conf,
        "imgsz": imgsz,
        "yolo_tta": yolo_tta,
        "resnet_tta": resnet_tta,
    }
    return main(defaults)


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
    stage1_dir = outdir / "stage1_yolo"
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
    results_list = []

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
            cv2.imwrite(str(vis_dir / img_path.name), img)
            rows.append([image_id, "no_detection", "", "", "", "", "", "", ""])
            print(f"[NO DET] {img_path.name}")
            continue

        x1, y1, x2, y2 = best["xyxy"]
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))
        if x2 <= x1 or y2 <= y1:
            rows.append([image_id, "invalid_box", "", f"{best['conf']:.4f}", "", x1, y1, x2, y2])
            continue

        margin_left = x1
        margin_top = y1
        margin_right = w - x2
        margin_bottom = h - y2
        if margin_left < 10 or margin_top < 10 or margin_right < 10 or margin_bottom < 10:
            x1, y1, x2, y2 = expand_box_if_edge_touching(x1, y1, x2, y2, h, w, expand_ratio=0.15)

        crop = img[y1:y2, x1:x2]
        stage1_img = img.copy()
        cls_id = best["cls_id"]
        if cls_id == 0:
            fuel_type = "pointer"
            model = pointer_model
            color = (0, 255, 0)
            stage1_label = f"Pointer conf={best['conf']:.2f}"
        else:
            fuel_type = "grid"
            model = grid_model
            color = (255, 0, 0)
            stage1_label = f"Grid conf={best['conf']:.2f}"

        cv2.rectangle(stage1_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(stage1_img, stage1_label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(stage1_dir / img_path.name), stage1_img)

        fuel = predict_fuel_with_rotation_tta(model, crop, device, use_tta=(args.resnet_tta == 1))
        fuel = max(0.0, min(1.0, fuel))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{fuel_type} conf={best['conf']:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, f"fuel={fuel * 100:.1f}%", (x1, min(h - 10, y2 + 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(str(vis_dir / img_path.name), img)

        rows.append([image_id, "ok", fuel_type, f"{best['conf']:.4f}", f"{fuel:.6f}", x1, y1, x2, y2])
        results_list.append({
            'image': image_id,
            'boxes_count': 1,
            'fuel_values': [fuel],
            'avg_fuel': fuel,
            'conf_values': [best['conf']],
            'avg_conf': best['conf'],
            'boxes': [{
                'box': (x1, y1, x2, y2),
                'conf': best['conf'],
                'fuel': fuel,
                'fuel_type': fuel_type,
            }],
        })
        print(f"[OK] {img_path.name} | type={fuel_type} conf={best['conf']:.2f} fuel={fuel * 100:.1f}%")

    xlsx_ok = save_results(rows, csv_path, xlsx_path)
    print("\n完成")
    print(f"阶段一(YOLO框): {stage1_dir}")
    print(f"阶段二(完整标注): {vis_dir}")
    print(f"结果表格: {xlsx_path if xlsx_ok else csv_path}")
    return results_list


def build_defaults():
    project_root = Path(__file__).resolve().parents[1]
    return {
        "source": str(project_root / "fuel_detection_dataset" / "test" / "images"),
        "outdir": str(project_root / "results_two_stage_online"),
        "yolo": str(project_root / "runs" / "fuel_yolo" / "detect_2class" / "weights" / "best.pt"),
        "resnet_pointer": str(project_root / "models" / "resnet" / "pointer" / "fuel_resnet_pointer_model.pth"),
        "resnet_grid": str(project_root / "models" / "resnet" / "grid" / "fuel_resnet_grid_model.pth"),
        "conf": 0.6,
        "imgsz": 640,
        "yolo_tta": 1,
        "resnet_tta": 1,
    }


if __name__ == "__main__":
    pc = globals().get("pc", ["sample"] * 10)

    image_path = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\image_path\%s.jpg' % pc[9]
    result_dir = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\results_predict'
    yolo_weights = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\models\yolo_best.pt'  # 第一阶段模型
    resnet_pointer_path = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\models\fuel_resnet_model1.pth'  # 第二阶段模型
    resnet_grid_path = r'C:\Users\wangy\Desktop\车辆费用\20260514\yolo\models\fuel_resnet_model2.pth'  # 第二阶段模型

    yolo_conf = 0.6  # 默认置信度
    min_box_size = 30  # 默认最小框大小（像素）
    min_box_ratio = 0.4

    results_list = predict_complete(
        image_path,
        result_dir,
        yolo_conf,
        min_box_size,
        min_box_ratio,
        yolo_weights,
        resnet_pointer_path,
        resnet_grid_path,
        pc[9],
    )

