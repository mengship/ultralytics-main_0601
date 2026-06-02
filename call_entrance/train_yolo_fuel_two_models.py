#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【两阶段两模型训练】YOLO分类 + ResNet双模型油量识别

【核心思路】
第一阶段：YOLO分类油表类型
  ├─ 类别0：指针类油表（4种扇形变体合并）
  └─ 类别1：格子类油表

第二阶段：两个独立的ResNet模型
  ├─ ResNet-指针：检测指针角度 → 计算油量比例
  └─ ResNet-格子：计算格子填充比例 → 油量比例

【预测流程】
1. YOLO识别框 + 分类油表类型（指针/格子）
2. 根据类型选择对应的ResNet模型
3. ResNet识别油量，返回结果

【优势】
✅ 指针类和格子类使用不同的识别逻辑，互不干扰
✅ 每个模型专注于一种油表风格，准确率更高
✅ 样本少的风格有独立的模型优化空间
✅ 可以针对不同类型调整参数和数据增强策略

【关键配置】
- YOLO分类：2分类（指针 vs 格子）
- ResNet-指针：针对指针类4种变体优化
- ResNet-格子：针对格子类油表优化
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from datetime import datetime, timedelta
import torchvision.models as models
from torchvision.models import ResNet152_Weights
# 解决中文字体问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


class FuelTypeDataset(Dataset):
    """根据油表类型裁剪和加载数据（指针类或格子类）"""

    def __init__(self, dataset_dir, fuel_type, split='train', imgsz=224):
        """
        Args:
            dataset_dir: 数据集根目录
            fuel_type: 'pointer'（指针类）或 'grid'（格子类）
            split: 'train', 'val', 'test'
            imgsz: 输入图片大小
        """
        self.dataset_dir = Path(dataset_dir)
        self.fuel_type = fuel_type
        self.split = split
        self.imgsz = imgsz
        self.is_train = split == 'train'
        # ImageNet normalization (matches ResNet50 pretraining)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

        self.images_dir = self.dataset_dir / split / 'images'
        self.labels_dir = self.dataset_dir / split / 'labels'

        self.image_files = sorted(
            list(self.images_dir.glob('*.jpg')) +
            list(self.images_dir.glob('*.jpeg')) +
            list(self.images_dir.glob('*.png'))
        )

        # 加载油表框、油量和类型数据
        self.fuel_data = {}  # {img_name: fuel_ratio}
        self.bbox_data = {}  # {img_name: [cx, cy, w, h]}
        self.type_data = {}  # {img_name: fuel_type}
        self._load_fuel_data()

        # 只保留指定类型的数据
        self.image_files = [
            f for f in self.image_files
            if f.stem in self.fuel_data and
            self.type_data.get(f.stem) == self.fuel_type
        ]

        if len(self.image_files) == 0:
            print(f"⚠️  警告：找不到 {fuel_type} 类型的数据")

    def _load_fuel_data(self):
        """加载油量、框和类型数据

        格式：class cx cy w h fuel_ratio
        class: 0=pointer, 1=grid
        """
        for txt_file in self.labels_dir.glob('*_fuel.txt'):
            img_name = txt_file.stem.replace('_fuel', '')
            with open(txt_file) as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 6:  # class cx cy w h fuel_ratio
                        try:
                            class_id = int(parts[0])
                            bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                            fuel = float(parts[5])
                            fuel_type_str = 'pointer' if class_id == 0 else 'grid'
                            self.bbox_data[img_name] = bbox
                            self.fuel_data[img_name] = fuel
                            self.type_data[img_name] = fuel_type_str
                        except:
                            pass

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

        if x2 <= x1 or y2 <= y1:
            crop = img_rgb
        else:
            crop = img_rgb[y1:y2, x1:x2]

        # 确保大小完全相同
        crop_resized = cv2.resize(crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        if self.is_train:
            crop_resized = self._augment(crop_resized)
            if crop_resized.shape[0] != self.imgsz or crop_resized.shape[1] != self.imgsz:
                crop_resized = cv2.resize(crop_resized, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        img_tensor = torch.from_numpy(crop_resized).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = (img_tensor - self.mean) / self.std
        fuel_tensor = torch.tensor(self.fuel_data.get(img_name, 0.5), dtype=torch.float32)

        return {
            'image': img_tensor,
            'fuel_ratio': fuel_tensor,
            'image_name': img_name
        }

    def _augment(self, img):
        """数据增强（格子类强化版本）"""
        h, w = img.shape[:2]
        
        # ⭐ 格子类增强：对格子类使用更强的增强参数
        is_grid = self.fuel_type == 'grid'
        
        # 1. 随机旋转
        if np.random.rand() < (0.7 if is_grid else 0.5):  # 格子类概率更高
            angle = np.random.uniform(-30 if is_grid else -20, 30 if is_grid else 20)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 2. 亮度调整
        if np.random.rand() < (0.6 if is_grid else 0.5):
            brightness = np.random.uniform(0.75 if is_grid else 0.8, 1.3 if is_grid else 1.2)
            img = (img * brightness).clip(0, 255).astype(np.uint8)

        # 3. 对比度调整
        if np.random.rand() < (0.6 if is_grid else 0.5):
            contrast = np.random.uniform(0.75 if is_grid else 0.8, 1.3 if is_grid else 1.2)
            img = (img * contrast).clip(0, 255).astype(np.uint8)

        # 4. 高斯模糊
        if np.random.rand() < (0.5 if is_grid else 0.4):
            img = cv2.GaussianBlur(img, (3, 3), 0)

        # 5. 高斯噪声
        if np.random.rand() < (0.5 if is_grid else 0.3):
            noise = np.random.normal(0, 8 if is_grid else 5, img.shape)
            img = (img + noise).clip(0, 255).astype(np.uint8)

        # 6. 缩放
        if np.random.rand() < (0.4 if is_grid else 0.3):
            scale = np.random.uniform(0.9 if is_grid else 0.95, 1.1 if is_grid else 1.05)
            new_h, new_w = int(h * scale), int(w * scale)
            img_scaled = cv2.resize(img, (new_w, new_h))

            if scale > 1:
                y_start = (new_h - h) // 2
                x_start = (new_w - w) // 2
                img = img_scaled[y_start:y_start+h, x_start:x_start+w]
            else:
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                pad_h_after = h - new_h - pad_h
                pad_w_after = w - new_w - pad_w
                img = cv2.copyMakeBorder(img_scaled, pad_h, pad_h_after, pad_w, pad_w_after, cv2.BORDER_REFLECT)

        # 7. 水平翻转（格子类优先）
        if np.random.rand() < (0.5 if is_grid else 0.2):
            img = cv2.flip(img, 1)

        # 8. 竖直翻转（格子类优先）
        if np.random.rand() < (0.5 if is_grid else 0.2):
            img = cv2.flip(img, 0)

        # 9. 伽玛校正
        if np.random.rand() < (0.5 if is_grid else 0.3):
            gamma = np.random.uniform(0.75 if is_grid else 0.8, 1.25 if is_grid else 1.2)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            img = cv2.LUT(img, table)

        # 10. 随机仿射变换（格子类专用）
        if is_grid and np.random.rand() < 0.3:
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
            M = cv2.getAffineTransform(pts1, pts2)
            img = cv2.warpAffine(img, M, (w, h))

        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        return img


class ResNetFuelNet(nn.Module):
    """ResNet152油量识别网络"""

    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        else:
            self.backbone = models.resnet152(weights=None)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)


def train_resnet_for_fuel_type(fuel_type, fuel_type_name, detection_dataset_dir="fuel_detection_dataset"):
    """训练指定油表类型的ResNet模型

    Args:
        fuel_type: 'pointer' 或 'grid'
        fuel_type_name: '指针' 或 '格子'
        detection_dataset_dir: 检测数据集目录路径
    """

    print(f"\n{'='*70}")
    print(f"【训练{fuel_type_name}类油表的ResNet模型】")
    print(f"{'='*70}\n")

    script_dir = Path(__file__).parent.absolute()
    model_dir = script_dir / 'models' / 'resnet' / fuel_type
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 模型目录: {model_dir}\n")
    print(f"📂 数据集: {detection_dataset_dir}\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}\n")

    # 加载对应类型的数据集
    train_dataset = FuelTypeDataset(detection_dataset_dir, fuel_type, 'train')
    val_dataset = FuelTypeDataset(detection_dataset_dir, fuel_type, 'val')

    print(f"📊 {fuel_type_name}类训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}\n")

    if len(train_dataset) == 0:
        print(f"❌ {fuel_type_name}类训练集为空，跳过训练")
        return False

    batch_size = 32 if len(train_dataset) > 100 else 16 if len(train_dataset) > 50 else 8

    # ⭐ 改进：添加 WeightedRandomSampler 进行类别平衡采样
    # 计算每个样本的权重，使得数据充分利用
    num_samples = len(train_dataset)
    weights = torch.ones(num_samples)  # 默认权重相等
    
    train_sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=True)
    
    # ⭐ 改进：添加 drop_last=True 确保每个 batch 大小一致，避免小 batch 导致的 batch_norm 问题
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # 使用加权采样器
        drop_last=True,         # 确保batch大小一致
        num_workers=0
    )
    
    val_batch_size = min(32, max(1, len(val_dataset)))
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)

    print(f"📦 加载ResNet152预训练模型（{fuel_type_name}类）...\n")

    model = ResNetFuelNet(pretrained=True).to(device)

    def build_optimizer(current_model, base_lr=1e-3):
        """构建分组学习率优化器
        
        ⭐ 改进：
        - Backbone使用较小学习率（迁移学习）
        - 新添加的FC层使用较大学习率
        - 明确添加weight_decay进行L2正则化
        """
        backbone_params = []
        for name, param in current_model.backbone.named_parameters():
            if 'fc' not in name and param.requires_grad:
                backbone_params.append(param)
        fc_params = [p for p in current_model.backbone.fc.parameters() if p.requires_grad]

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': base_lr * 0.1})  # 0.0001
        if fc_params:
            param_groups.append({'params': fc_params, 'lr': base_lr})  # 0.001

        return optim.Adam(param_groups, weight_decay=3e-4, betas=(0.9, 0.999))  # ⭐ 增加weight_decay

    # ⭐ 改进：对超小样本禁用 backbone 冻结（根据样本数自适应）
    if len(train_dataset) > 50:
        freeze_warmup_epochs = 10  # 样本充足：使用冻结策略
    else:
        freeze_warmup_epochs = 0   # 样本稀少：跳过冻结，直接全量微调
    
    if freeze_warmup_epochs > 0:
        for name, param in model.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    else:
        print(f"   ℹ️  {fuel_type_name}类样本少({len(train_dataset)}张)，跳过冻结热启动，直接全量微调\n")

    optimizer = build_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-7)
    fuel_criterion = nn.L1Loss()

    best_loss = float('inf')
    train_losses = []
    val_losses = []
    epochs = 500
    patience = 100
    patience_counter = 0

    print(f"开始训练 ResNet152-{fuel_type_name}，最多 {epochs} 个 epoch...\n")
    print(f"📋 训练参数：")
    print(f"   - 模型: ResNet152（{fuel_type_name}类油表）")
    print(f"   - batch_size: {batch_size}（自适应数据量）")
    current_optimizer = build_optimizer(model)
    backbone_lr = next((group['lr'] for group in current_optimizer.param_groups if group['lr'] < 1e-3), None)
    fc_lr = max(group['lr'] for group in current_optimizer.param_groups)
    if backbone_lr is not None:
        print(f"   - 主干学习率: {backbone_lr:.6f}")
    else:
        print(f"   - 主干学习率: 冻结中")
    print(f"   - 新层学习率: {fc_lr:.6f}")
    print(f"   - weight_decay: 3e-4 ✅（改进）")
    print(f"   - Dropout: 0.5/0.4/0.3 ✅")
    if freeze_warmup_epochs > 0:
        print(f"   - 冻结热身轮次: {freeze_warmup_epochs} ✅（大样本策略）")
    else:
        print(f"   - 冻结热身轮次: 禁用 ✅（小样本策略）")
    print(f"   - drop_last: True ✅（改进）")
    print(f"   - WeightedRandomSampler: True ✅（改进）")
    print(f"   - patience: {patience}")
    print(f"   - 优化器: Adam（带weight_decay）\n")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # ⭐ 改进：只在启用冻结时才执行解冻
        if freeze_warmup_epochs > 0 and epoch == freeze_warmup_epochs:
            print(f"\n🔓 第 {epoch+1} 轮开始解冻backbone，进入全量微调")
            for name, param in model.backbone.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = True
            optimizer = build_optimizer(model)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - epoch), eta_min=1e-7)

        model.train()
        epoch_loss = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            fuels = batch['fuel_ratio'].unsqueeze(1).to(device)
            outputs = model(images)
            loss = fuel_criterion(outputs, fuels)
            optimizer.zero_grad()
            loss.backward()
            # ⭐ 改进：梯度裁剪防止梯度爆炸
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
            model_path = model_dir / f'fuel_resnet_{fuel_type}_model.pth'
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

    print(f"\n✅ {fuel_type_name}类ResNet训练完成！")
    print(f"📊 总耗时: {total_time_str}")
    print(f"🏆 最佳验证损失: {best_loss:.6f}")
    print(f"📈 最佳epoch: {len(train_losses) - patience_counter}")
    print(f"💾 模型已保存: {model_dir / f'fuel_resnet_{fuel_type}_model.pth'}\n")

    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train', linewidth=1.5, color='blue')
    plt.plot(val_losses, label='Val', linewidth=1.5, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f'ResNet-{fuel_type_name} Loss')

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
    loss_curve_path = model_dir / f'fuel_resnet_{fuel_type}_loss.png'
    plt.savefig(str(loss_curve_path), dpi=100, bbox_inches='tight')
    print(f"📊 {fuel_type_name}类损失曲线已保存: {loss_curve_path.name}\n")

    return True


def train_yolo_classifier(yolo_dataset_dir="fuel_yolo_dataset"):
    """【阶段1】训练YOLO检测油表框（2分类：指针 vs 格子）"""

    print("\n" + "="*70)
    print("【阶段1】训练YOLO检测油表框类型")
    print("="*70 + "\n")

    yolo_yaml = Path(yolo_dataset_dir) / "data.yaml"

    if not yolo_yaml.exists():
        print(f"❌ 错误：YOLO数据集不存在")
        print(f"   数据集路径: {Path(yolo_dataset_dir).absolute()}")
        print(f"   请先运行: python convert_simple.py")
        return False

    print(f"✅ 已找到YOLO数据集: {yolo_dataset_dir}\n")
    print(f"📦 加载YOLO11m预训练模型...\n")
    print(f"📋 YOLO检测配置：")
    print(f"   - 任务: 2分类检测（指针 vs 格子）")
    print(f"   - 数据集: fuel_yolo_dataset")
    print(f"   - 类别0: 指针类油表")
    print(f"   - 类别1: 格子类油表")
    print(f"   - Epochs: 500")
    print(f"   - Patience: 50\n")

    yolo_model = YOLO('../yolo11m.pt')

    print(f"🎯 开始训练YOLO检测模型...\n")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    yolo_results = yolo_model.train(
        data=str(yolo_yaml),
        epochs=500,
        imgsz=640,
        batch=16,
        patience=100,
        device=0,
        verbose=True,
        project='runs/fuel_yolo',
        name='detect_2class',
        save=True,
        exist_ok=True,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        dropout=0.15,
        mosaic=0.5,
        flipud=0.1,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        degrees=15,
        translate=0.08,
        scale=0.3,
        perspective=0.0003,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )

    print("\n" + "="*70)
    print("✅ YOLO检测训练完成！")
    print("📁 输出目录: runs/fuel_yolo/detect_2class/weights/best.pt")
    print("="*70 + "\n")
    return True


def train(yolo_dataset_dir="fuel_yolo_dataset", detection_dataset_dir="fuel_detection_dataset", train_yolo=True):
    """主训练函数

    Args:
        yolo_dataset_dir: YOLO数据集目录（5列YOLO格式）
        detection_dataset_dir: 检测数据集目录（6列CNN格式）
        train_yolo: 是否训练YOLO检测模型
    """

    print("\n" + "="*70)
    print("🚀 两阶段训练：YOLO 2分类检测 + ResNet双模型油量识别")
    print("="*70)
    print("\n📂 数据集配置：")
    print(f"   - YOLO数据集: {yolo_dataset_dir}")
    print(f"   - ResNet数据集: {detection_dataset_dir}\n")

    # 第一步：YOLO检测
    if train_yolo:
        print("【第一步】训练YOLO检测 + 2分类油表类型\n")
        if not train_yolo_classifier(yolo_dataset_dir):
            return
    else:
        print("【第一步】跳过YOLO检测训练，使用已训练好的模型\n")

    # 第二步：指针类ResNet
    print("\n【第二步】训练指针类油表的ResNet模型\n")
    train_resnet_for_fuel_type('pointer', '指针', detection_dataset_dir)

    # 第三步：格子类ResNet
    print("\n【第三步】训练格子类油表的ResNet模型\n")
    train_resnet_for_fuel_type('grid', '格子', detection_dataset_dir)

    print("\n" + "="*70)
    print("✅ 所有训练完成！")
    print("="*70 + "\n")
    print("📋 生成的模型文件：")
    print("   - YOLO检测: runs/fuel_yolo/detect_2class/weights/best.pt")
    print("   - ResNet-指针: models/resnet/pointer/fuel_resnet_pointer_model.pth")
    print("   - ResNet-格子: models/resnet/grid/fuel_resnet_grid_model.pth")
    print("\n🚀 预测流程：")
    print("   1. YOLO检测框 + 分类类型")
    print("   2. 根据class_id选择ResNet模型")
    print("   3. ResNet识别油量\n")


if __name__ == '__main__':
    # =================== 修改这里设置数据集路径 ===================
    YOLO_DATASET_DIR = "../fuel_yolo_dataset"  # YOLO训练数据集
    DETECTION_DATASET_DIR = "../fuel_detection_dataset"  # ResNet训练数据集
    TRAIN_YOLO = False  # YOLO已训练完成时设为False，只训练ResNet
    # ============================================================

    print("\n" + "="*70)
    print("📌 两阶段训练：YOLO 2分类检测 + ResNet双模型")
    print("="*70)
    print("\n🎯 数据集配置：")
    print(f"   ├─ YOLO数据集: {YOLO_DATASET_DIR}")
    print(f"   │  └─ 格式：5列YOLO标准格式")
    print(f"   └─ ResNet数据集: {DETECTION_DATASET_DIR}")
    print(f"      └─ 格式：6列CNN格式（class cx cy w h fuel）")
    print("\n💡 核心思想：")
    print("   ✅ YOLO既检测框又分类类型，高效合一")
    print("   ✅ ResNet根据YOLO的分类结果，选择合适的识别策略")
    print("   ✅ 指针类和格子类各有专用模型\n")

    train(YOLO_DATASET_DIR, DETECTION_DATASET_DIR, train_yolo=TRAIN_YOLO)
