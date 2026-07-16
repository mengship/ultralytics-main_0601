# 格子油表 Pose 版本需求说明

这个目录用于处理格子类型油表的数据。当前样例数据在：

```text
call_entrance_pose_grid/datasets/
```

其中 `260622_NGC7836.json` 是一份 LabelMe / X-AnyLabeling 标注文件。标注里包含两类框和三个关键点：

```text
oil1      格子油表框，需要用于训练
odometer  里程数框，暂时不参与格子油量识别
empty     空油位置点
full      满油位置点
tip       当前油量位置点
```

格子油表的目标是识别 `empty / full / tip` 三个关键点，然后用距离比例计算油量：

```text
总距离 = distance(empty, full)
当前距离 = distance(empty, tip)
fuel_ratio = 当前距离 / 总距离
```

## 标注理解

### 需要提取的标签

只提取下面这些标签：

```text
oil1
empty
full
tip
```

`oil1` 是格子油表区域框，相当于它包住了 `empty / full / tip` 三个点。后续训练和预测都围绕 `oil1` 这块区域进行。

### 暂时忽略的标签

```text
odometer
```

`odometer` 是里程数框，里面是里程数信息。当前格子油量识别不处理它，转换数据时应该跳过。

### oil1 的形状

样例数据里 `oil1` 可能是：

```text
rectangle
rotation
```

后续转换脚本需要同时兼容这两种形状。无论是 `rectangle` 还是 `rotation`，最终都要得到一个可裁剪的外接框 `xyxy`。

## YOLO Pose 标签格式

格子油表也走 YOLO Pose 思路，但关键点只有 3 个：

```yaml
names:
  0: grid_pose
kpt_shape: [3, 2]
```

关键点顺序固定为：

```text
0 empty
1 full
2 tip
```

转换后的 YOLO Pose 标签格式：

```text
class cx cy w h empty_x empty_y full_x full_y tip_x tip_y
```

所有坐标都归一化到对应图片尺寸。

## 三步流程

### 第一步：提取标注并转换成 YOLO Pose 数据集

输入：

```text
call_entrance_pose_grid/datasets/*.json
call_entrance_pose_grid/datasets/*.jpg
```

处理逻辑：

1. 读取每个 JSON。
2. 找到 `oil1` 框。
3. 找到 `empty / full / tip` 三个点。
4. 忽略 `odometer`。
5. 生成 YOLO Pose 标签。
6. 按 8:2 比例划分训练集和验证集。

输出建议：

```text
call_entrance_pose_grid/dataset_convert/
  data.yaml
  train/images/
  train/labels/
  val/images/
  val/labels/
  missing_grid_pose_report.json
  grid_pose_metadata.json
```

如果缺少 `oil1` 或缺少 `empty / full / tip` 任一点，应该跳过该样本，并写入 report。

### 第二步：按 oil1 裁剪格子油表并更新标签

在第一步得到完整图的 YOLO Pose 数据集后，再按 `oil1` 框裁剪出格子油表区域。

处理逻辑：

1. 读取第一步输出的 YOLO Pose 数据集。
2. 根据 `oil1` 对应的 bbox 裁剪图片。
3. 裁剪时可以稍微放大一点，例如 `--crop-padding 0.05`，避免裁掉边缘格子。
4. 将 bbox 和 `empty / full / tip` 三个关键点重映射到裁剪图坐标系。
5. 保存新的裁剪版 YOLO Pose 数据集。

输出建议：

```text
call_entrance_pose_grid/dataset_convert_crop/
  data.yaml
  train/images/
  train/labels/
  val/images/
  val/labels/
  crop_summary.json
```

裁剪后的标签仍然保持：

```text
class cx cy w h empty_x empty_y full_x full_y tip_x tip_y
```

### 第三步：训练格子油表 Pose 模型

使用第二步生成的裁剪数据集训练 YOLO Pose 模型。

训练目标：

```text
输入：裁剪后的格子油表图片
输出：empty / full / tip 三个关键点
```

训练脚本建议后续放在：

```text
call_entrance_pose_grid/train_yolo_grid_pose.py
```

训练命令示例：

```bash
python call_entrance_pose_grid/train_yolo_grid_pose.py \
  --data call_entrance_pose_grid/dataset_convert_crop/data.yaml \
  --model yolo11m-pose.pt
```

## 预测时的油量计算

模型预测出三个关键点后，按距离比例计算：

```text
empty = 空油点
full  = 满油点
tip   = 当前油量点

total_distance = distance(empty, full)
current_distance = distance(empty, tip)
fuel_ratio = current_distance / total_distance
```

最后将 `fuel_ratio` 截断到 `[0, 1]`，同时保留原始比例 `raw_fuel_ratio` 便于调试。

## 与指针油表的区别

指针油表：

```text
center / tip / empty / full
按角度比例计算油量
```

格子油表：

```text
empty / full / tip
按距离比例计算油量
```

所以格子油表不需要 `center` 点，也不需要角度方向规则。

## 当前先做的事情

当前阶段先明确需求和数据流：

1. 从 JSON 中提取 `oil1 / empty / full / tip`。
2. 忽略 `odometer`。
3. 转成 3 关键点 YOLO Pose 数据集，并按 8:2 切分。
4. 按 `oil1` 裁剪格子油表并重映射标签。
5. 训练格子油表 Pose 模型。

后续再补对应的转换、裁剪、训练和预测脚本。

## TODO 执行清单

按下面顺序一步一步推进。

### 1. 标注数据确认

- [ ] 检查 `datasets/` 下所有 JSON 是否都包含 `oil1 / empty / full / tip`
- [ ] 确认 `oil1` 的 `rectangle` 和 `rotation` 都能转成外接框
- [ ] 确认 `odometer` 只作为里程数框，当前全部忽略
- [ ] 统计缺少关键标签的样本，形成 report

### 2. 转换成完整图 YOLO Pose 数据集

- [x] 新建转换脚本，例如 `convert_labelme_grid_pose_dataset.py`
- [x] 从 JSON 中提取 `oil1 / empty / full / tip`
- [x] 将 `oil1` 转成 bbox：`cx cy w h`
- [x] 将三个点转成关键点：`empty_x empty_y full_x full_y tip_x tip_y`
- [x] 输出 YOLO Pose 标签格式：`class cx cy w h empty_x empty_y full_x full_y tip_x tip_y`
- [x] 按 8:2 切分 train / val
- [x] 生成 `data.yaml`
- [x] 生成 `missing_grid_pose_report.json`
- [x] 生成 `grid_pose_metadata.json`
- [x] 跑一次小样本转换 smoke test

**转换命令：**

```bash
# 默认设置
python call_entrance_pose_grid/convert_labelme_grid_pose_dataset.py

# 自定义路径
python call_entrance_pose_grid/convert_labelme_grid_pose_dataset.py \
    --json-dir call_entrance_pose_grid/datasets \
    --output-dir call_entrance_pose_grid/dataset_convert \
    --val-ratio 0.2
```

### 3. 按 oil1 裁剪格子油表并更新标签

- [x] 新建裁剪脚本，例如 `crop_yolo_grid_pose_dataset.py`
- [x] 读取第二步生成的完整图 YOLO Pose 数据集
- [x] 按 `oil1` bbox 裁剪图片
- [x] 默认保留一点 padding，例如 `--crop-padding 0.05`
- [x] 将 bbox 重映射到裁剪图坐标系
- [x] 将 `empty / full / tip` 重映射到裁剪图坐标系
- [x] 输出裁剪版 YOLO Pose 数据集
- [x] 生成 `crop_summary.json`
- [x] 抽样可视化裁剪后的标签，确认点和框没有错位

**裁剪命令：**

```bash
# 默认设置
python call_entrance_pose_grid/crop_yolo_grid_pose_dataset.py

# 自定义参数
python call_entrance_pose_grid/crop_yolo_grid_pose_dataset.py \
    --data call_entrance_pose_grid/dataset_convert/data.yaml \
    --output-dir call_entrance_pose_grid/dataset_convert_crop \
    --crop-padding 0.05
```

**裁剪结果：**
- Train: 4/4 样本成功裁剪
- Val: 1/1 样本成功裁剪
- 成功率：100%

### 4. 训练格子油表 Pose 模型

- [x] 新建训练脚本，例如 `train_yolo_grid_pose.py`
- [x] 默认读取 `dataset_convert_crop/data.yaml`
- [x] 使用 YOLO Pose 预训练模型训练 3 个关键点
- [ ] 检查训练日志和验证集效果
- [ ] 保存最优权重路径

**训练命令：**

```bash
# 默认设置
python call_entrance_pose_grid/train_yolo_grid_pose.py

# 自定义参数
python call_entrance_pose_grid/train_yolo_grid_pose.py \
    --data call_entrance_pose_grid/dataset_convert_crop/data.yaml \
    --model yolo11m-pose.pt \
    --epochs 300 \
    --batch 16 \
    --device 0

# 快速测试（小 epoch）
python call_entrance_pose_grid/train_yolo_grid_pose.py \
    --epochs 10 \
    --batch 4 \
    --device cpu
```

**训练配置：**
- 模型：yolo11m-pose.pt
- 关键点数：3 个（empty, full, tip）
- 增强策略：
  - ✓ 轻微旋转（10°）
  - ✓ 缩放和平移
  - ✓ 颜色增强
  - ✗ 禁用水平/垂直翻转（方向敏感）
- 优化器：AdamW
- 默认输出：`runs/grid_pose/grid_pose_3kpt/`

### 5. 格子油量预测

- [ ] 新建预测脚本，例如 `predict_grid_pose_fuel.py`
- [ ] 预测裁剪小图上的 `empty / full / tip`
- [ ] 计算 `distance(empty, tip) / distance(empty, full)`
- [ ] 输出 `fuel_ratio / raw_fuel_ratio / fuel_percent`
- [ ] 保存可视化图片，画出三个点和距离线
- [ ] 将比例截断到 `[0, 1]`

### 6. 接入二阶段总预测

- [ ] 在二阶段检测中保留类别分流
- [ ] `class_id == 0` 走指针 Pose 逻辑
- [ ] `class_id == 1` 走格子 Pose 逻辑
- [ ] 格子逻辑没完成前保持 `grid_not_ready`
- [ ] 完成后将 `grid_not_ready` 替换为真实格子预测结果

### 7. 最终验证

- [ ] `python -m py_compile` 检查所有新增脚本
- [ ] 用样例 JSON 完成从转换到训练前数据集的完整流程
- [ ] 检查 train / val 比例是否接近 8:2
- [ ] 检查裁剪图是否保留完整格子油表
- [ ] 检查关键点顺序是否始终为 `empty / full / tip`
- [ ] 检查预测比例是否符合人工观察
