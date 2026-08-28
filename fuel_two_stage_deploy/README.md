# Fuel Two-stage Deployment

批量识别指针油表和格子油表。第一阶段检测油表类型与位置，第二阶段在裁剪区域预测关键点并计算油量。

## 目录结构

```text
fuel_two_stage_deploy/
├── predict_fuel_two_stage.py
├── predict_pose_fuel.py
├── requirements.txt
├── run.py
├── models/
│   ├── detector.pt
│   ├── pointer_pose.pt
│   └── grid_pose.pt
└── output/
```

每次运行前，在 `run.py` 顶部的 `SOURCE_IMAGE` 中填写单张图片路径。权重文件较大，不包含在部署目录中，需要按上述名称放入线上目录。

## 安装与运行

建议使用 Python 3.10 或 3.11：

```bash
cd fuel_two_stage_deploy
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
mkdir -p models output
python3 run.py
```

默认使用 CPU。需要使用第一张 GPU 时：

```bash
DEVICE=0 python3 run.py
```

结果 CSV 写入 `output/fuel_two_stage_predictions.csv`。部署入口不保存可视化图片和裁剪图片。

状态 `low_keypoint_confidence` 表示至少一个关键点置信度低于 `0.6`。该记录会保留关键点坐标，但不计算油量。
