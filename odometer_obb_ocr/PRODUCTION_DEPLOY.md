# 生产环境部署指南

## 📦 依赖安装

### 必需依赖

```bash
pip install ultralytics paddlepaddle paddleocr opencv-python numpy
```

### 可选依赖（仅开发用）

```bash
pip install easyocr pytesseract  # 其他OCR引擎（生产环境不需要）
```

---

## 🚀 快速使用

### 1. 单张图片识别

```bash
python production_ocr.py --image /path/to/image.jpg
```

**输出示例：**
```json
{
  "success": true,
  "mileage": "30594",
  "confidence": 0.853,
  "status": "ok",
  "error": null
}
```

### 2. 自定义参数

```bash
python production_ocr.py \
  --image /path/to/image.jpg \
  --model /path/to/best.pt \
  --det-conf 0.25 \
  --ocr-conf 0.70 \
  --crop-padding 0.60 \
  --save-crop /path/to/crop.jpg
```

---

## 📝 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | *必需* | 输入图片路径 |
| `--model` | `runs/obb/odometer/weights/best.pt` | YOLO模型路径 |
| `--det-conf` | 0.25 | 检测置信度阈值 |
| `--ocr-conf` | 0.70 | OCR置信度阈值 |
| `--crop-padding` | 0.60 | 裁剪padding比例（推荐0.60） |
| `--min-digits` | 4 | 最小数字位数 |
| `--max-digits` | 8 | 最大数字位数 |
| `--save-crop` | *(可选)* | 保存裁剪图路径 |

---

## 🔧 Python集成

### 方式1：命令行调用

```python
import subprocess
import json

result = subprocess.run(
    ["python", "production_ocr.py", "--image", "test.jpg"],
    capture_output=True,
    text=True
)

data = json.loads(result.stdout)
if data["success"]:
    print(f"识别结果: {data['mileage']}")
else:
    print(f"识别失败: {data['error']}")
```

### 方式2：直接导入函数

```python
from production_ocr import recognize_odometer

result = recognize_odometer(
    image_path="test.jpg",
    model_path="runs/obb/odometer/weights/best.pt",
    crop_padding=0.60
)

if result["success"]:
    print(f"里程数: {result['mileage']}")
    print(f"置信度: {result['confidence']:.3f}")
else:
    print(f"错误: {result['error']}")
```

---

## 📊 返回状态说明

| status | 说明 | success |
|--------|------|---------|
| `ok` | 识别成功 | ✅ true |
| `no_detection` | 未检测到里程表 | ❌ false |
| `invalid_geometry` | 检测框几何验证失败 | ❌ false |
| `low_ocr_confidence` | OCR置信度过低 | ❌ false |
| `invalid_digit_count` | 数字位数不符 | ❌ false |
| `ocr_error` | OCR执行错误 | ❌ false |
| `error` | 其他错误 | ❌ false |

---

## 🏭 生产环境部署

### 方式1：独立服务器

```bash
# 1. 复制文件
odometer_obb_ocr/
├── production_ocr.py
├── utils/
│   ├── __init__.py
│   ├── geometry.py
│   └── ocr.py
└── runs/obb/odometer/weights/best.pt

# 2. 安装依赖
pip install ultralytics paddlepaddle paddleocr opencv-python numpy

# 3. 运行
python production_ocr.py --image test.jpg
```

### 方式2：Web API（Flask示例）

```python
from flask import Flask, request, jsonify
from production_ocr import recognize_odometer
import tempfile
import os

app = Flask(__name__)

@app.route('/api/ocr', methods=['POST'])
def ocr_endpoint():
    if 'image' not in request.files:
        return jsonify({"error": "未上传图片"}), 400
    
    file = request.files['image']
    
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        result = recognize_odometer(
            image_path=tmp_path,
            model_path="runs/obb/odometer/weights/best.pt",
            crop_padding=0.60
        )
        return jsonify(result)
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 方式3：Docker部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN pip install ultralytics paddlepaddle paddleocr opencv-python-headless numpy

# 复制代码
COPY production_ocr.py .
COPY utils/ ./utils/
COPY runs/obb/odometer/weights/best.pt ./runs/obb/odometer/weights/

# 运行
ENTRYPOINT ["python", "production_ocr.py"]
```

---

## ⚡ 性能优化

### GPU加速

如果有GPU，安装GPU版本的PaddlePaddle：

```bash
pip install paddlepaddle-gpu
```

### 批量处理

```python
from production_ocr import recognize_odometer
from concurrent.futures import ThreadPoolExecutor
import glob

images = glob.glob("images/*.jpg")

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(
        lambda img: recognize_odometer(img, model_path="best.pt"),
        images
    ))

for img, result in zip(images, results):
    if result["success"]:
        print(f"{img}: {result['mileage']}")
```

---

## 🐛 故障排查

### 问题1：识别失败率高

**检查：**
```bash
# 保存裁剪图查看质量
python production_ocr.py --image test.jpg --save-crop crop.jpg
```

**解决：**
- 增大 `--crop-padding 0.80`
- 降低 `--ocr-conf 0.60`

### 问题2：模型找不到

**错误：** `FileNotFoundError: runs/obb/odometer/weights/best.pt`

**解决：**
```bash
python production_ocr.py --image test.jpg --model /absolute/path/to/best.pt
```

### 问题3：PaddleOCR下载模型慢

**解决：** 预先下载模型到 `~/.paddlex/official_models/`

---

## 📞 支持

生产环境问题请提供：
1. 完整的错误JSON输出
2. 输入图片（如果可以）
3. `--save-crop` 保存的裁剪图

---

**部署完成后，记得在真实数据上测试准确率！** 🚀
