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

## 步骤 2：裁剪数据集你是资深 Python / Ultralytics 工程师，请在仓库 `ultralytics-main_0601` 里重构 `call_entrance_pose`，把“裁剪”和“训练”拆成两个脚本。

目标：
1. 新增一个只负责裁剪数据集的脚本
2. 把 `call_entrance_pose/train_yolo_fuel_pose_crop.py` 改成只负责训练
3. 更新 `call_entrance_pose/README.md`，用中文说明新流程

请不要改 `predict_pose_fuel.py`，不要改关键点顺序和标签格式。

---

## 现有标签格式

YOLO Pose 标签格式固定为：

`class cx cy w h center_x center_y tip_x tip_y empty_x empty_y full_x full_y`

关键点顺序固定为：

- 0 center
- 1 tip
- 2 empty
- 3 full

---

## 任务 1：新增裁剪脚本

新增一个脚本，例如：

`call_entrance_pose/crop_yolo_fuel_pose_dataset.py`

它只做裁剪，不训练，不依赖 `ultralytics`。

### 输入
- 读取现有 YOLO Pose 数据集的 `data.yaml`
- 支持 `train` / `val`
- 支持常见 YOLO 目录结构

### 裁剪规则
- 每张图只有一个油表框，不会有多个
- 每个标签文件只有 1 行
- 读取 bbox `cx cy w h`
- 还原成像素坐标 `xyxy`
- 裁剪时要稍微裁大一点，默认保留一点 padding，防止把重要信息裁掉
- `--crop-padding` 建议默认 0.05 或 0.1
- 将 bbox 和 `center / tip / empty / full` 全部重映射到 crop 坐标系
- 再重新归一化到 `[0, 1]`

### 输出
- 输出新的数据集目录，例如 `call_entrance_pose/dataset_convert_crop`
- 目录结构：
  - `train/images`
  - `train/labels`
  - `val/images`
  - `val/labels`
- 生成新的 `data.yaml`
- 生成 `crop_summary.json`

### 异常处理
- 如果图片缺失、标签异常、标签不是 1 行，直接跳过并统计
- 输出统计信息：
  - 裁了多少
  - 缺图多少
  - 无效标签多少
  - 无效 crop 多少

---

## 任务 2：训练脚本改成纯训练

把 `call_entrance_pose/train_yolo_fuel_pose_crop.py` 改成纯训练脚本。

### 要求
- 只读取裁剪后的数据集 `data.yaml`
- 只调用 `YOLO(...).train(...)`
- 删除其中所有裁剪、重映射、生成数据集的逻辑
- 参数风格可以沿用当前 `train_yolo_fuel_pose.py`
- 默认 `--data` 可以指向 crop 后的数据集，例如：
  - `call_entrance_pose/dataset_convert_crop/data.yaml`

---

## 任务 3：更新 README

把 `call_entrance_pose/README.md` 更新成中文，重点写清楚：

- pose 数据的标签格式
- 裁剪脚本怎么用
- 训练脚本怎么用
- 两阶段流程：
  1. 先按标注框裁剪
  2. 再训练
- 命令示例要完整

---

## 代码要求
- 只改 `call_entrance_pose` 相关文件
- 不要回滚别的无关改动
- 代码里把这些步骤写清楚注释：
  - bbox 还原
  - crop 裁剪
  - bbox 重映射
  - keypoint 重映射
- 保持仓库现有风格

---

## 验证要求
完成后请至少做这些检查：

1. `python -m py_compile` 通过
2. 用现有样例数据跑一次裁剪 smoke test
3. 确认输出目录里有：
   - `data.yaml`
   - `train/images`
   - `train/labels`
   - `val/images`
   - `val/labels`
   - `crop_summary.json`

---

## 额外说明
- 裁剪时宁可稍微大一点，也不要严格贴框
- 不要把“裁剪”和“训练”混在一个脚本里
- 我希望最后结构清楚、职责单一、后面好维护

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

python call_entrance_pose/crop_yolo_fuel_pose_dataset.py \
  --data /home/wang/datasets/yolopose_dataset_convert/data.yaml \
  --output-dir /home/wang/datasets/yolopose_dataset_convert_crop \
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

远程云坏境的地址
python call_entrance_pose/train_yolo_fuel_pose_crop.py \
  --data /home/wang/datasets/yolopose_dataset_convert_crop/data.yaml \
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

### 方式 1：单阶段预测（直接在裁剪图上）

如果你的图片已经是裁剪好的油表图，使用 `predict_pose_fuel.py`：

```bash
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/cropped_images \
  --direction max_full_span \
  --save-vis
```

### 方式 2：二阶段预测（检测 + 姿态估计）★ 推荐

**适用场景**：原图包含完整场景，需要先检测油表位置再预测关键点。

**流程**：
1. **第一阶段**：YOLO 检测模型找到原图中的油表框
   - `class_id == 0`：指针式油表，继续第二阶段
   - `class_id == 1`：格子式油表，返回 `grid_not_ready`（格子识别暂未实现）
   - 其他类别：返回 `unsupported_cls`
2. **裁剪扩展**：将检测框四周扩大一点（默认 8%），防止裁掉指针或刻度
3. **第二阶段**：YOLO Pose 模型在裁剪小图上预测 4 个关键点（仅针对指针式）
4. **计算油量**：根据关键点角度计算油量比例

```bash
python call_entrance_pose/predict_pose_fuel_two_stage.py \
  --det-model runs/detect/oil_detector/weights/best.pt \
  --pose-model runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /path/to/original_images \
  --box-padding 0.08 \
  --direction max_full_span \
  --save-vis \
  --crop-dir call_entrance_pose/crops

云服务器实际路径
python call_entrance_pose/predict_pose_fuel_two_stage.py \
  --det-model /home/wang/ultralytics-main_0601/call_entrance/runs/detect/runs/fuel_yolo/detect_2class/weights/best.pt \
  --pose-model /home/wang/ultralytics-main_0601/runs/pose/runs/fuel_pose/pose_crop_4kpt/weights/best.pt \
  --source /home/wang//datasets/yolopose_dataset_convert/val/images/ \
  --box-padding 0.08 \
  --direction max_full_span \
  --save-vis \
  --crop-dir call_entrance_pose/crops
```

#### 二阶段预测参数说明

- `--det-model`: 第一阶段 YOLO 检测模型路径（用于找油表框）
- `--pose-model`: 第二阶段 YOLO Pose 模型路径（用于预测关键点）
- `--source`: 输入图片目录或单张图片（原图）
- `--box-padding`: 裁剪时在检测框四周的扩展比例（默认 0.08 = 8%）
  - 建议值：0.05 ~ 0.1，防止边缘关键点被裁掉
- `--det-conf`: 检测置信度阈值（默认 0.25）
- `--pose-conf`: 姿态置信度阈值（默认 0.25）
- `--direction`: 角度计算方向（默认 `max_full_span`）
- `--save-vis`: 保存可视化结果（绘制在裁剪小图上）
- `--vis-dir`: 可视化结果输出目录
- `--crop-dir`: 可选，保存裁剪后的小图
- `--output-csv`: 输出 CSV 文件路径

#### 二阶段预测输出

CSV 文件包含以下字段：

```text
image,status,det_class,fuel_type,det_conf,det_x1,det_y1,det_x2,det_y2,crop_image,pose_conf,
fuel_ratio,raw_fuel_ratio,fuel_percent,clamped,direction,span_deg,offset_deg,
tip_deg,empty_deg,full_deg,center_x,center_y,tip_x,tip_y,empty_x,empty_y,full_x,full_y
```

**字段说明**：
- `det_class`: 第一阶段检测到的类别 ID（0=指针, 1=格子, 其他）
- `fuel_type`: 油表类型（`pointer` / `grid` / `class_{id}`）

**状态说明**：
- `ok`: 成功预测（仅指针式油表）
- `no_det`: 第一阶段未检测到油表框
- `grid_not_ready`: 检测到格子式油表，但格子识别暂未实现
- `unsupported_cls`: 检测到不支持的类别
- `invalid_crop`: 裁剪区域无效
- `no_pose`: 第二阶段未检测到关键点
- `read_error`: 图片读取失败

**注意**：
- 只有 `class_id == 0`（指针式油表）会继续执行 Pose 预测和角度计算
- 输出的关键点坐标（`center_x`, `tip_x` 等）是在**裁剪小图**坐标系下的，不是原图坐标

### 单阶段预测参数说明

- `--model`: 训练好的 YOLO Pose 模型路径
- `--source`: 输入图片目录或单张图片（已裁剪的油表图）
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
