## 生产环境部署指南

### 📦 需要部署的文件清单

#### 必需的Python脚本文件
- `predict_complete.py` - 主预测脚本

#### 必需的模型文件
- `models/using_model/yolo_best.pt` - YOLO油表框检测模型（需要提前训练或拷贝）
- `models/using_model/fuel_resnet_model.pth` - ResNet50油量识别模型（需要提前训练或拷贝）

#### 依赖配置文件
- `requirements.txt` - Python包依赖清单

### 🔧 部署步骤

#### 1️⃣ 运维服务器环境准备
```bash
# 创建项目目录
mkdir -p /opt/fuel_detection
cd /opt/fuel_detection

# 创建Python虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖包
pip install -r requirements.txt
```

#### 2️⃣ 创建目录结构
```
/opt/fuel_detection/
├── predict_complete.py          # 主脚本
├── requirements.txt             # 依赖清单
├── models/
│   └── using_model/
│       ├── yolo_best.pt         # YOLO模型 (~40MB)
│       └── fuel_resnet_model.pth # ResNet模型 (~100MB)
└── results/                     # 输出结果目录（自动创建）
```

#### 3️⃣ 复制文件
```bash
# 从本地复制脚本
scp predict_complete.py user@server:/opt/fuel_detection/
scp requirements.txt user@server:/opt/fuel_detection/

# 从本地复制模型
scp -r models/using_model/* user@server:/opt/fuel_detection/models/using_model/
```

### 🚀 使用方式

#### 基础使用（使用脚本内默认值）
```bash
python predict_complete.py
```

#### 指定输入目录
```bash
python predict_complete.py /path/to/images
```

#### 指定输入/输出目录
```bash
python predict_complete.py /path/to/images /path/to/results
```

#### 指定全部参数
```bash
python predict_complete.py /path/to/images /path/to/results 0.5 80 0.4
参数说明：
  - 0.5       : YOLO置信度阈值（0.0-1.0）
  - 80        : 最小框大小（像素）
  - 0.4       : 最小框比例(高/宽)
```

### 📋 需要修改的地方

如果生产环境的路径不同，需要修改脚本中的默认值：

**文件: predict_complete.py 第 647-650 行**
```python
# ========== 默认值配置（可在此修改） ==========
DEFAULT_IMAGE_DIR = '/opt/fuel_detection/input'      # 输入目录
DEFAULT_RESULT_DIR = '/opt/fuel_detection/output'     # 输出目录
DEFAULT_YOLO_CONF = 0.3                               # 默认置信度
```

### ⚠️ 注意事项

1. **GPU支持**
   - 脚本会自动检测GPU（CUDA），如果有GPU会加速推理
   - 如果服务器无GPU，脚本会自动使用CPU（性能会降低10倍）

2. **显存需求**
   - 建议GPU显存 ≥ 4GB（3060 12GB最佳）
   - 无GPU时需要至少 8GB 内存

3. **磁盘空间**
   - 模型文件 ~150MB
   - 输出结果取决于输入图片数量和大小

4. **Python版本**
   - 推荐 Python 3.8-3.10

### ✅ 部署验证

部署完成后运行测试：
```bash
# 查看脚本帮助信息
python predict_complete.py -h

# 用测试图片验证
python predict_complete.py ./test_images

# 检查输出目录
ls -la results/detected/
ls -la results/crops/
```

### 📊 输出文件说明

运行后会生成：
- `results/detected/` - 标注了框和油量的原图
- `results/crops/` - 裁剪出的油表框图片
- `results/prediction_results.xlsx` - Excel格式的预测结果报告

### 🔗 模型获取

如果没有模型文件，需要先运行训练脚本生成：
- YOLO模型: `train_yolo_fuel_resnet.py`
- ResNet模型: `train_yolo_fuel_resnet.py`

生成的模型会保存到默认位置，然后拷贝到生产服务器。

---

**最后检查清单：**
- [ ] Python版本 ≥ 3.8
- [ ] 安装了 requirements.txt 中的所有包
- [ ] 模型文件存在于 `models/using_model/` 目录
- [ ] 有可读的输入目录权限
- [ ] 有可写的输出目录权限
- [ ] GPU可用（可选但推荐）

