# 油表姿态检测（YOLO Pose）

这是一个面向指针式油表的姿态检测版本，用于识别油表指针位置并计算油量比例。

## 整体流程

```text
LabelMe / X-AnyLabeling JSON
  ↓ (convert_labelme_pose_dataset.py)
YOLO Pose 数据集（完整图）
  ↓ (crop_yolo_fuel_pose_dataset.py)
裁剪后的 YOLO Pose 数据集
  ↓ (train_yolo_fuel_pose_crop.py)
训练好的 YOLO Pose 模型
  ↓ (predict_pose_fuel.py)
油量比例 (fuel_ratio)
```

---

## 标签格式

### 关键点顺序

固定使用 4 个关键点，顺序不能改：

```text
0 center  - 表盘中心
1 tip     - 指针尖端
2 empty   - 空刻度位置
3 full    - 满刻度位置
```

### YOLO Pose 标签格式

每行一个目标，格式如下：

```text
class cx cy w h center_x center_y tip_x tip_y empty_x empty_y full_x full_y
```

- `class`: 类别 ID（通常为 0）
- `cx cy w h`: 归一化的 bbox 中心坐标和宽高 `[0, 1]`
- `center_x center_y`: 归一化的 center 关键点坐标 `[0, 1]`
- `tip_x tip_y`: 归一化的 tip 关键点坐标 `[0, 1]`
- `empty_x empty_y`: 归一化的 empty 关键点坐标 `[0, 1]`
- `full_x full_y`: 归一化的 full 关键点坐标 `[0, 1]`

### data.yaml 配置

```yaml
path: /path/to/dataset
train: train/images
val: val/images
kpt_shape: [4, 2]  # 4 个关键点，每个 2 维 (x, y)
flip_idx: [0, 1, 2, 3]  # 不做左右翻转映射
names:
  0: oil_pose
```

---

## 步骤 1：数据标注与转换

### LabelMe 标注规范

每个指针油表建议这样标：

```text
rectangle: oil_pose   或 oil
point:     center
point:     tip
point:     empty
point:     full
```

- 建议保留矩形框。如果没有框，转换脚本可以根据四个点自动生成一个带 padding 的框
- 如果一张图里有多个油表，请给对应的矩形和四个点设置相同的 `group_id`

### 转换 LabelMe 数据集

#### 单目录转换

```bash
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir "/path/to/json_and_images" \
  --output-dir call_entrance_pose/dataset_convert \
  --val-ratio 0.2
```

#### 按类型子目录转换

如果源目录下面有类型子目录（例如 `left/`, `lower_left/` 等），直接传父目录即可：

```bash
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir "/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/0702" \
  --output-dir call_entrance_pose/dataset_convert \
  --val-ratio 0.2
```

转换器会把每个包含 JSON 的一级子目录当作一个 source group，并按同样比例分别切分 train / val。

#### 转换输出结构

```text
dataset_convert/
  ├── data.yaml                    # YOLO 数据集配置
  ├── train/
  │   ├── images/                  # 训练图片
  │   └── labels/                  # 训练标签
  ├── val/
  │   ├── images/                  # 验证图片
  │   └── labels/                  # 验证标签
  ├── pose_metadata.json           # 每个样本的来源记录
  └── missing_pose_report.json     # 缺失关键点的样本报告
```

---

## 步骤 2：裁剪数据集

在训练前，建议先把完整图像裁剪成只包含油表框的小图，这样：

- **减少背景干扰**：模型只学习油表框内的特征
- **匹配推理场景**：推理时先检测油表框，再对裁剪图运行 Pose 模型
- **提高精度**：关键点相对位置更稳定

### 裁剪命令

```bash
python call_entrance_pose/crop_yolo_fuel_pose_dataset.py \
  --data call_entrance_pose/dataset_convert/data.yaml \
  --output-dir call_entrance_pose/dataset_convert_crop \
  --crop-padding 0.05
```

### 参数说明

- `--data`: 原始 YOLO Pose 数据集的 `data.yaml` 路径（转换脚本的输出）
- `--output-dir`: 裁剪后数据集的输出目录
- `--crop-padding`: 裁剪时额外保留的 padding 比例
  - `0.0` = 严格按框裁剪
  - `0.05` = 保留 5% 的 padding（推荐，防止边缘信息被裁掉）
  - `0.1` = 保留 10% 的 padding

### 裁剪规则

1. **每张图只有一个油表框**：标签文件只包含 1 行
2. **读取 bbox**：`cx cy w h`（归一化坐标）
3. **还原成像素坐标**：`xyxy` 格式
4. **扩展 padding**：根据 `--crop-padding` 参数扩大裁剪框
5. **裁剪图像**：用 OpenCV 裁剪
6. **重映射坐标**：
   - 把 bbox 从原图坐标转换到 crop 坐标
   - 把 4 个关键点从原图坐标转换到 crop 坐标
   - 重新归一化到 `[0, 1]`

### 裁剪输出结构

```text
dataset_convert_crop/
  ├── data.yaml                    # 新的 YOLO 数据集配置
  ├── train/
  │   ├── images/                  # 裁剪后的训练图片
  │   └── labels/                  # 重映射后的训练标签
  ├── val/
  │   ├── images/                  # 裁剪后的验证图片
  │   └── labels/                  # 重映射后的验证标签
  └── crop_summary.json            # 裁剪统计摘要
```

### 异常处理

脚本会自动跳过以下情况并统计：

- **缺失图片**：label 文件存在但对应图片找不到
- **标签不是单行**：不符合"每张图一个油表框"的假设
- **无效裁剪**：裁剪区域为空（例如 bbox 超出图像边界）

---

## 步骤 3：训练 YOLO Pose 模型

### 准备预训练模型

下载 YOLO Pose 预训练权重，例如：

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt
```

### 训练命令

```bash
python call_entrance_pose/train_yolo_fuel_pose_crop.py \
  --data call_entrance_pose/dataset_convert_crop/data.yaml \
  --model yolo11m-pose.pt \
  --epochs 300 \
  --batch 16 \
  --device 0
```

### 参数说明

- `--data`: 裁剪后数据集的 `data.yaml` 路径
- `--model`: 预训练模型路径或名称
- `--epochs`: 训练轮数（默认 300）
- `--batch`: 批大小（默认 16）
- `--device`: 设备（`0` 表示 GPU 0，`cpu` 表示 CPU，`mps` 表示 Mac GPU）
- `--imgsz`: 输入图像尺寸（默认 640）
- `--patience`: 早停耐心值（默认 80）
- `--project`: 训练输出项目目录（默认 `runs/fuel_pose`）
- `--name`: 训练输出实验名称（默认 `pose_crop_4kpt`）

### 训练输出

```text
runs/fuel_pose/pose_crop_4kpt/
  ├── weights/
  │   ├── best.pt                  # 最佳模型
  │   └── last.pt                  # 最后一轮模型
  ├── results.png                  # 训练曲线
  └── ...
```

---

## 步骤 4：预测油量比例

训练完成后，使用 `predict_pose_fuel.py` 进行推理：

```bash
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/images \
  --direction max_full_span \
  --save-vis
```

### 参数说明

- `--model`: 训练好的模型路径
- `--source`: 输入图片目录或单张图片
- `--direction`: 角度计算方向（见下文）
- `--save-vis`: 保存可视化结果
- `--vis-dir`: 可视化结果输出目录
- `--output-csv`: 输出 CSV 结果文件

### 角度计算规则

脚本支持多种角度计算方向：

#### 1. `max_full_span`（推荐，默认值）

自动选择让指针落在 `[empty, full]` 范围内的方向：

1. 分别计算 `empty->full` 和 `empty->tip` 在顺时针、逆时针两个方向上的角度
2. 选择能让 `tip` 落在 `empty` 和 `full` 之间的方向（即 `empty->tip` 角度 < `empty->full` 角度）
3. 如果两个方向都满足，选择 `empty->full` 角度更大的方向
4. 如果两个方向都不满足，回退到 `empty->full` 角度更大的方向
5. `fuel_ratio = (empty->tip 角度) / (empty->full 角度)`
6. 如果比例超出 `[0, 1]`，截断到这个范围，但保留 `raw_fuel_ratio` 便于调试

```bash
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/images \
  --direction max_full_span
```

#### 2. `tip_side`

选择 `tip` 落在 `empty -> full` 扫描范围内的方向（主要用于和旧实验对比）：

```bash
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/images \
  --direction tip_side
```

#### 3. 固定方向

如果油表本身是固定方向的，可以直接指定：

```bash
# 顺时针
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/images \
  --direction clockwise

# 逆时针
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/images \
  --direction counterclockwise
```

注意：在图像坐标系里，`clockwise` 表示 `atan2(y - cy, x - cx)` 递增的方向。

### 输出示例

```text
predict_result.csv:
image,fuel_ratio,raw_fuel_ratio,direction
img1.jpg,0.75,0.75,clockwise
img2.jpg,0.42,0.42,counterclockwise
```

---

## 完整两阶段流程示例

### 场景：从 LabelMe 标注到训练完成

```bash
# 1. 转换 LabelMe 数据集为 YOLO Pose 格式
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir /path/to/labelme_json \
  --output-dir call_entrance_pose/dataset_convert \
  --val-ratio 0.2

# 2. 裁剪数据集（只保留油表框内区域）
python call_entrance_pose/crop_yolo_fuel_pose_dataset.py \
  --data call_entrance_pose/dataset_convert/data.yaml \
  --output-dir call_entrance_pose/dataset_convert_crop \
  --crop-padding 0.05

# 3. 训练 YOLO Pose 模型
python call_entrance_pose/train_yolo_fuel_pose_crop.py \
  --data call_entrance_pose/dataset_convert_crop/data.yaml \
  --model yolo11m-pose.pt \
  --epochs 300 \
  --batch 16 \
  --device 0

# 4. 预测油量比例
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/test_images \
  --direction max_full_span \
  --save-vis \
  --output-csv results.csv
```

---

## 不使用裁剪的训练方式

如果不想裁剪，可以直接在完整图上训练（不推荐，除非油表占比很大）：

```bash
python call_entrance_pose/train_yolo_fuel_pose.py \
  --data call_entrance_pose/dataset_convert/data.yaml \
  --model yolo11m-pose.pt
```

---

## 常见问题

### 1. 裁剪脚本报错"not single line"

**原因**：某些标签文件包含多行（多个目标），不符合"每张图一个油表框"的假设。

**解决**：检查标签文件，确保每个文件只有 1 行。如果确实有多个油表，需要在转换阶段就分开。

### 2. 裁剪后图像太小

**原因**：原始标注框太小，或者 `--crop-padding` 设置为 0。

**解决**：增大 `--crop-padding` 参数，例如从 `0.0` 改为 `0.1`。

### 3. 训练时 keypoint loss 不下降

**原因**：关键点标注质量差，或者数据量不足。

**解决**：
- 检查裁剪后的数据集，确认关键点是否正确重映射
- 增加数据量
- 调整数据增强参数（`degrees`, `translate`, `scale` 等）

### 4. 预测的 fuel_ratio 经常超出 [0, 1]

**原因**：模型预测的关键点位置偏差较大，或者角度计算方向选择不当。

**解决**：
- 使用 `--direction max_full_span`（默认）
- 检查可视化结果，确认关键点预测是否准确
- 增加训练数据或调整训练参数

---

## 相关脚本

- `convert_labelme_pose_dataset.py`: LabelMe JSON → YOLO Pose 数据集
- `crop_yolo_fuel_pose_dataset.py`: 裁剪 YOLO Pose 数据集
- `train_yolo_fuel_pose_crop.py`: 训练 YOLO Pose 模型（裁剪版）
- `train_yolo_fuel_pose.py`: 训练 YOLO Pose 模型（完整图版）
- `predict_pose_fuel.py`: 预测油量比例
- `prefix_type_filenames.py`: 给文件名添加类型前缀（可选）

---

## 注意事项

1. **关键点顺序不能改**：必须按 `center, tip, empty, full` 顺序标注和训练
2. **裁剪时宁可大一点**：建议 `--crop-padding` 至少设为 `0.05`，防止边缘信息丢失
3. **验证集划分**：转换脚本会按 source group 分别划分 train / val，确保每种类型都有验证样本
4. **推理时的两阶段**：推理时需要先检测油表框，再对裁剪图运行 Pose 模型，这样才能匹配训练时的数据分布

---

## 实验服务器示例（参考）

```bash
# 转换
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir /home/wang/datasets/yolopose_dataset \
  --output-dir /home/wang/datasets/yolopose_dataset_convert \
  --val-ratio 0.2

# 裁剪
python call_entrance_pose/crop_yolo_fuel_pose_dataset.py \
  --data /home/wang/datasets/yolopose_dataset_convert/data.yaml \
  --output-dir /home/wang/datasets/yolopose_dataset_crop \
  --crop-padding 0.05

# 训练
python call_entrance_pose/train_yolo_fuel_pose_crop.py \
  --data /home/wang/datasets/yolopose_dataset_crop/data.yaml \
  --model yolo11m-pose.pt \
  --epochs 300 \
  --batch 16 \
  --device 0

# 预测
python call_entrance_pose/predict_pose_fuel.py \
  --model /home/wang/ultralytics-main_0601/runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /home/wang/datasets/yolopose_dataset_convert/val/images \
  --direction max_full_span \
  --output-csv /home/wang/datasets/yolopose_dataset_result/predict_result.csv \
  --save-vis \
  --vis-dir /home/wang/datasets/yolopose_dataset_result/predict_vis
```
