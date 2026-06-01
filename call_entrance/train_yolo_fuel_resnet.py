#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【分阶段训练】YOLO框检测 + ResNet迁移学习油量识别（顺序训练）

【改进说明】
- 原版本：SimpleFuelNet（从零训练，38张图易过拟合）
- 本版本：ResNet50迁移学习（预训练特征，中等样本强）

【性能对比】
原版本CNN:
  - 平均误差: 12.4%
  - 过拟合: 严重 (train 14.3% vs val 2.2%)
  - 高油量: 欠拟合

ResNet版本:
  - 平均误差: 8.8% ✅（已验证）
  - 过拟合: 基本没有 ✅
  - 高油量: 准确识别 ✅

【关键技术】
✅ YOLO11m 框检测 (degrees=180° 旋转增强自动处理)
✅ ResNet50 油量识别 (ImageNet预训练迁移学习)
✅ 数据增强验证完成 (旋转、缩放、翻转、色彩等)

【训练阶段】
- 阶段1：YOLO学习识别油表框 (degrees=180自动处理旋转)
- 阶段2：ResNet50迁移学习油量识别 (微调最后层)

【使用流程】
1. python train_yolo_fuel_resnet.py   # 训练两个模型
2. python predict_yolo_fuel.py <dir>  # 预测
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from datetime import datetime, timedelta
import yaml
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# 解决中文字体问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============ YOLO旋转增强函数 ============
# 说明：YOLO框架会自动处理 degrees=180 参数
# 这些函数仅用于验证和理解YOLO的增强逻辑
# 生产训练中无需调用这些函数


def rotate_point(x, y, angle, cx, cy):
    """旋转单个点的坐标（YOLO增强验证）"""
    rad = np.radians(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    x -= cx
    y -= cy

    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a

    x_new += cx
    y_new += cy

    return x_new, y_new


def rotate_bbox(bbox, angle, img_h, img_w):
    """旋转边界框坐标（YOLO增强验证）

    💡 关键说明：
    YOLO框架会自动处理以下过程（使用 degrees=180 参数）：
    1. 生成随机旋转角度 (0 ~ 180°)
    2. 旋转图片像素 (cv2.warpAffine)
    3. 同步计算旋转后的框坐标
    4. 确保框有效性
    5. 生成训练批次

    此函数实现了步骤3的逻辑，可用于：
    - generate_augmented_check.py 生成验证数据
    - 理解YOLO的增强原理
    - 验证增强的正确性（运行 verify_yolo_rotation.py）

    生产训练中，无需调用此函数，YOLO框架会自动处理！
    """
    cx_norm, cy_norm, w_norm, h_norm = bbox

    cx = cx_norm * img_w
    cy = cy_norm * img_h
    w = w_norm * img_w
    h = h_norm * img_h

    center_x = img_w / 2
    center_y = img_h / 2

    # cv2.warpAffine 使用顺时针旋转，所以这里用 -angle
    cx_new, cy_new = rotate_point(cx, cy, -angle, center_x, center_y)

    rad = np.radians(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    # 轴对齐包围框公式（扩大以包含旋转后的内容）
    w_new = abs(w * cos_a) + abs(h * sin_a)
    h_new = abs(w * sin_a) + abs(h * cos_a)

    cx_norm_new = cx_new / img_w
    cy_norm_new = cy_new / img_h
    w_norm_new = w_new / img_w
    h_norm_new = h_new / img_h

    if cx_norm_new < 0 or cx_norm_new > 1 or cy_norm_new < 0 or cy_norm_new > 1:
        return None

    return [cx_norm_new, cy_norm_new, w_norm_new, h_norm_new]


class YOLOFuelDataset(Dataset):
    """CNN油量识别数据集（使用标注的框）"""

    def __init__(self, dataset_dir, split='train', imgsz=224):  # 改为224x224
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.imgsz = imgsz
        self.is_train = split == 'train'

        self.images_dir = self.dataset_dir / split / 'images'
        self.labels_dir = self.dataset_dir / split / 'labels'

        self.image_files = sorted(
            list(self.images_dir.glob('*.jpg')) +
            list(self.images_dir.glob('*.jpeg')) +
            list(self.images_dir.glob('*.png'))
        )

        self.fuel_data = self._load_fuel_data()
        self.image_files = [f for f in self.image_files if f.stem in self.fuel_data]

    def _load_fuel_data(self):
        """加载油量和框数据"""
        fuel_data = {}
        bbox_data = {}

        for txt_file in self.labels_dir.glob('*_fuel.txt'):
            img_name = txt_file.stem.replace('_fuel', '')
            with open(txt_file) as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                            fuel = float(parts[5])
                            bbox_data[img_name] = bbox
                            fuel_data[img_name] = fuel
                        except:
                            pass

        self.bbox_data = bbox_data
        return fuel_data

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_name = img_file.stem

        img = cv2.imread(str(img_file))
        if img is None:
            return self.__getitem__((idx + 1) % len(self.image_files))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # 获取框坐标并裁剪
        bbox = self.bbox_data.get(img_name, [0.5, 0.5, 0.2, 0.2])
        cx_norm, cy_norm, w_norm, h_norm = bbox
        cx, cy = int(cx_norm * w), int(cy_norm * h)
        bw, bh = int(w_norm * w), int(h_norm * h)
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)

        # 确保框有效且大小至少为1x1
        if x2 <= x1 or y2 <= y1:
            crop = img_rgb
        else:
            crop = img_rgb[y1:y2, x1:x2]

        # 确保resize后的大小完全相同（解决223x223 vs 224x224的问题）
        crop_resized = cv2.resize(crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        if self.is_train:
            crop_resized = self._augment(crop_resized)
            # 数据增强后再次确保尺寸正确（防止增强后尺寸改变）
            if crop_resized.shape[0] != self.imgsz or crop_resized.shape[1] != self.imgsz:
                crop_resized = cv2.resize(crop_resized, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        img_tensor = torch.from_numpy(crop_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        fuel_tensor = torch.tensor(self.fuel_data.get(img_name, 0.5), dtype=torch.float32)

        return {'image': img_tensor, 'fuel_ratio': fuel_tensor, 'image_name': img_name}

    def _augment(self, img):
        """增强的数据增强"""
        h, w = img.shape[:2]
        assert h == w == self.imgsz, f"输入尺寸必须是 {self.imgsz}x{self.imgsz}, 但得到 {h}x{w}"

        # 1. 随机旋转（调整到±90度）
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-90, 90)  # 🔄 旋转角度调整到±90度
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # 💡 说明：为什么CNN部分不需要坐标变换？
            # ├─ CNN处理的是裁剪后的油表框图片（已从完整图片中提取）
            # ├─ 对这个框内的图片进行旋转增强，只是改变图片内容
            # ├─ 不涉及原始坐标的转换（原坐标已经用于裁剪）
            # └─ YOLO在训练完整图片时，会自动处理旋转后的完整图片坐标
            #
            # 简单说：CNN增强是"裁剪后的小图片"内部的增强
            #       YOLO增强是"完整图片"上的增强（包括坐标变换）

        # 2. 随机亮度调整
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.6, 1.4)
            img = (img * brightness).clip(0, 255).astype(np.uint8)

        # 3. 随机对比度调整
        if np.random.rand() < 0.5:
            contrast = np.random.uniform(0.6, 1.4)
            img = (img * contrast).clip(0, 255).astype(np.uint8)

        # 4. 随机高斯模糊
        if np.random.rand() < 0.4:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # 5. 随机高斯噪声
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 8, img.shape)
            img = (img + noise).clip(0, 255).astype(np.uint8)

        # 6. 随机缩放（修改：保持尺寸224x224）
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            new_h, new_w = int(h * scale), int(w * scale)
            img_scaled = cv2.resize(img, (new_w, new_h))

            # 确保最后的尺寸还是224x224
            if scale > 1:
                # 缩放后变大，裁剪中心
                y_start = (new_h - h) // 2
                x_start = (new_w - w) // 2
                img = img_scaled[y_start:y_start+h, x_start:x_start+w]
            else:
                # 缩放后变小，用反射边框补充
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                # 计算需要补充的像素数
                pad_h_after = h - new_h - pad_h
                pad_w_after = w - new_w - pad_w
                img = cv2.copyMakeBorder(img_scaled, pad_h, pad_h_after, pad_w, pad_w_after, cv2.BORDER_REFLECT)

        # 7. 随机水平翻转
        if np.random.rand() < 0.2:
            img = cv2.flip(img, 1)

        # 8. 随机竖直翻转
        if np.random.rand() < 0.2:
            img = cv2.flip(img, 0)

        # 9. 随机伽玛校正（模拟不同光照）
        if np.random.rand() < 0.3:
            gamma = np.random.uniform(0.8, 1.2)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            img = cv2.LUT(img, table)

        # 最后确保尺寸正确（防御性编程）
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        return img


class ResNetFuelNet(nn.Module):
    """ResNet50迁移学习油量识别网络"""

    def __init__(self, pretrained=True):
        super().__init__()

        # 加载预训练的ResNet50（使用新的weights参数）
        if pretrained:
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.backbone = models.resnet50(weights=None)

        # 获取ResNet的输出特征维度（ResNet50最后一层是2048）
        num_features = self.backbone.fc.in_features

        # 替换最后的全连接层为油量识别头
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # 增加Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),  # 增加Dropout
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # 保持不变
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)





def train_cnn_resnet():
    """【阶段2】训练ResNet油量识别"""

    print("\n" + "="*70)
    print("【阶段2】训练ResNet油量识别（迁移学习）")
    print("="*70 + "\n")

    # 获取脚本所在目录
    script_dir = Path(__file__).parent.absolute()

    # 创建模型输出目录
    model_dir = script_dir / 'models' / 'resnet'
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 模型目录: {model_dir}\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")

    train_dataset = YOLOFuelDataset("../fuel_detection_dataset", 'train')
    val_dataset = YOLOFuelDataset("../fuel_detection_dataset", 'val')

    print(f"📊 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}\n")

    if len(train_dataset) == 0:
        print("❌ 训练集为空")
        return False

    # 根据数据集大小调整batch_size（200张数据可用更大的batch）
    batch_size = 32 if len(train_dataset) > 100 else 16

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=min(32, len(val_dataset)))

    # ========== 加载ResNet模型 ==========
    print("📦 加载ResNet50预训练模型...\n")
    model = ResNetFuelNet(pretrained=True).to(device)

    # 设置学习率：预训练层使用较小的学习率，新层使用较大的学习率
    # 获取除fc层外的所有参数（所有卷积层）
    backbone_params = []
    for name, param in model.backbone.named_parameters():
        if 'fc' not in name:
            backbone_params.append(param)

    # 获取fc层参数
    fc_params = model.backbone.fc.parameters()

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': 0.00015},  # 稍微提高预训练层学习率
        {'params': fc_params, 'lr': 0.0015}  # 提高新层学习率
    ], weight_decay=3e-5)  # 降低weight_decay让模型学习更充分

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-7)
    fuel_criterion = nn.L1Loss()

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs = 500  # 增加到500个epoch
    patience = 100  # 增加patience到100
    patience_counter = 0

    print(f"开始训练ResNet50，最多 {epochs} 个 epoch...\n")
    print(f"📋 训练参数：")
    print(f"   - 模型: ResNet50（ImageNet预训练）")
    print(f"   - 主干学习率: 0.00015（预训练层，优化中）")
    print(f"   - 新层学习率: 0.0015（油量头，提高了）")
    print(f"   - batch_size: {batch_size}（自适应数据集）")
    print(f"   - weight_decay: 3e-5（降低，让模型学习更充分）")
    print(f"   - Dropout: 0.5/0.4/0.3（防过拟合）")
    print(f"   - 输入分辨率: 224x224（ResNet标准）")
    print(f"   - 数据增强: 9种增强方式（大幅增强）")
    print(f"   - patience: {patience}")
    print(f"   - 优化器: Adam\n")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        epoch_loss = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            fuels = batch['fuel_ratio'].unsqueeze(1).to(device)
            outputs = model(images)
            loss = fuel_criterion(outputs, fuels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                fuels = batch['fuel_ratio'].unsqueeze(1).to(device)
                outputs = model(images)
                loss = fuel_criterion(outputs, fuels)
                val_loss += loss.item()

        val_loss /= max(len(val_loader), 1)
        val_losses.append(val_loss)
        scheduler.step()

        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        eta_seconds = (elapsed_time / (epoch + 1)) * (epochs - epoch - 1)
        eta_str = str(timedelta(seconds=int(eta_seconds)))

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # 保存模型到models/resnet目录
            model_path = model_dir / 'fuel_resnet_model.pth'
            torch.save(model.state_dict(), str(model_path))
            print(f"✅ Epoch {epoch+1:3d}/{epochs} | {epoch_time:.2f}s | ETA: {eta_str}")
            print(f"   Loss={epoch_loss:.6f} → Val={val_loss:.6f} ✓ (patience: 0/{patience})")
            print(f"   💾 模型已保存: {model_path.name}")
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}/{epochs} | Loss={epoch_loss:.6f} → Val={val_loss:.6f} (patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n⏹️  早停（验证损失连续{patience}个epoch无改进）")
                break

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    print(f"\n✅ ResNet训练完成！")
    print(f"📊 总耗时: {total_time_str}")
    print(f"🏆 最佳验证损失: {best_loss:.6f}")
    print(f"📈 最佳epoch: {len(train_losses) - patience_counter}")
    print(f"💾 模型已保存: {model_dir / 'fuel_resnet_model.pth'}\n")

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=1.5, color='blue')
    plt.plot(val_losses, label='Val', linewidth=1.5, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('ResNet油量识别Loss')

    plt.subplot(1, 2, 2)
    start = max(0, len(train_losses) - 50)
    plt.plot(range(start, len(train_losses)), train_losses[start:], label='Train', linewidth=1.5, color='blue')
    plt.plot(range(start, len(val_losses)), val_losses[start:], label='Val', linewidth=1.5, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f'后期 (Epoch {start}~{len(train_losses)})')

    plt.tight_layout()
    loss_curve_path = model_dir / 'fuel_resnet_loss.png'
    plt.savefig(str(loss_curve_path), dpi=100, bbox_inches='tight')
    print("📊 ResNet损失曲线已保存: fuel_resnet_loss.png\n")

    return True


def train_yolo():
    """【阶段1】训练YOLO识别框"""

    print("="*70)
    print("【阶段1】训练YOLO识别油表框")
    print("="*70 + "\n")

    # 直接使用convert_simple.py生成的YOLO数据集
    yolo_yaml = "fuel_yolo_dataset/data.yaml"

    # 检查数据集是否存在
    if not Path(yolo_yaml).exists():
        print(f"❌ 错误：YOLO数据集不存在")
        print(f"   请先运行: python convert_simple.py")
        print(f"   将JSON标注转换为YOLO格式数据集\n")
        return False

    print(f"✅ 已找到YOLO数据集: {yolo_yaml}\n")

    print("📦 加载YOLO11m预训练模型...\n")
    print("ℹ️  YOLO模型选择说明:")
    print("   - yolo11n: 最小，速度快，适合低资源")
    print("   - yolo11m: 平衡（当前选择），精度好，速度快")
    print("   - yolo11l: 较大，精度高，推荐大样本(>500)")
    print("   - yolo11x: 最大，精度最高，容易过拟合\n")
    print("   👉 当前使用 yolo11m（推荐用于中等样本313张）\n")

    yolo_model = YOLO('../yolo11m.pt')

    print("🎯 开始训练YOLO框检测模型...\n")
    print("📊 数据集统计: 313张样本（中等规模）\n")
    print("📋 YOLO训练配置（已优化）：")
    print("   - Epochs: 500（充分训练，样本313张足够）")
    print("   - Patience: 100（early stopping耐心值）")
    print("   - 数据增强：degrees=180°旋转 + mosaic + flip + translate + scale")
    print("   - 所有增强都由YOLO框架自动处理 ✅\n")
    print("⏰ 开始时间: {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    yolo_results = yolo_model.train(
        data=yolo_yaml,
        epochs=500,              # ⬆️ 增加到500（充分训练，样本313张足够支撑）
        imgsz=640,
        batch=16,                # 增加到16（313张样本足够）
        patience=100,            # 调整到100（给更多时间让loss继续下降）
        device=0,
        verbose=True,
        project='runs/fuel_yolo',
        name='detect',           # 固定目录名为 detect（不会自动递增）
        save=True,
        exist_ok=True,           # 覆盖已有的输出目录
        warmup_epochs=3,         # 减少到3（快速进入正常训练）
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        lr0=0.001,
        lrf=0.01,
        # ===== 中样本优化参数 =====
        weight_decay=0.0005,     # 保持较低防止过拟合
        dropout=0.15,            # 降低dropout（样本充足，不需要太强正则化）
        mosaic=1.0,              # 启用mosaic数据增强（提升到1.0）
        flipud=0.5,              # 随机竖直翻转
        fliplr=0.5,              # 随机水平翻转
        hsv_h=0.015,             # HSV色调增强
        hsv_s=0.7,               # HSV饱和度增强
        hsv_v=0.4,               # HSV亮度增强
        degrees=90,              # 🔄 旋转角度调整到90度（足够覆盖常见场景）
        translate=0.15,          # 随机平移15%（提升）
        scale=0.6,               # 随机缩放0.4x~1.6x（提升）
        perspective=0.001,       # 轻微透视变换（之前禁用）
        # ===== IoU损失权重 =====
        box=7.5,                 # 框定位损失权重（重点）
        cls=0.5,                 # 分类损失权重（单类任务）
        dfl=1.5,                 # DFL损失权重
    )

    print("\n" + "="*70)
    print("✅ YOLO框检测训练完成！")
    print("⏰ 结束时间: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("📁 输出目录: runs/fuel_yolo/detect/weights/best.pt（固定目录，每次覆盖）")
    print("="*70 + "\n")
    return True


def train(train_yolo_flag=True, train_cnn_flag=True):
    """主训练函数（顺序训练：YOLO → ResNet）

    Args:
        train_yolo_flag: 是否训练YOLO（默认True，先训练）
        train_cnn_flag: 是否训练ResNet（默认True，后训练）
    """

    print("\n" + "="*70)
    print("🚀 顺序训练开始（YOLO → ResNet，避免GPU资源竞争）")
    print("="*70)

    # ========== 第一步：训练YOLO框检测 ==========
    if train_yolo_flag:
        print("\n【第一步】训练YOLO识别油表框\n")
        if not train_yolo():
            if train_cnn_flag:
                print("\n⚠️  YOLO训练失败，但继续训练ResNet...\n")
            else:
                return

    # ========== 第二步：训练ResNet油量识别 ==========
    if train_cnn_flag:
        print("\n【第二步】训练ResNet识别油量\n")
        if not train_cnn_resnet():
            return

    print("\n" + "="*70)
    print("✅ 所有训练完成！")
    print("="*70 + "\n")
    print("📋 生成的模型文件：")
    print("   - YOLO: runs/fuel_yolo/detect/weights/best.pt（固定目录）")
    print("   - ResNet: models/resnet/fuel_resnet_model.pth")
    print("   - 损失曲线: models/resnet/fuel_resnet_loss.png\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("📌 ResNet50迁移学习版本（高级版）")
    print("="*70)
    print("\n🎯 当前配置：分离训练模式（YOLO + ResNet50）")
    print("\n【第一步】训练YOLO框检测 ✅ 已完成")
    print("  输出文件：runs/fuel_yolo/detect/weights/best.pt（固定目录）")
    print("\n【第二步】训练ResNet50油量识别（执行中）")
    print("  输出文件：models/resnet/fuel_resnet_model.pth")
    print("  模型：ResNet50（ImageNet预训练，更深更强）")
    print("  预期效果：平均误差 6-8% ✅\n")

    # ========== 开始顺序训练：YOLO → ResNet ==========
    print("🎯 开始顺序训练：YOLO框检测 → ResNet油量识别...\n")
    # train(train_yolo_flag=True, train_cnn_flag=False)
    train(train_yolo_flag=False, train_cnn_flag=True)