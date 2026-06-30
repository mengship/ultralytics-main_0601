#!/usr/bin/env python3
from predict_two_stage_v2 import predict_single_image

# 模型路径配置
YOLO_MODEL = './runs/detect/runs/fuel_yolo/detect_2class/weights/best.pt'
POINTER_MODEL = './models/resnet/pointer/fuel_resnet_pointer_model.pth'
GRID_MODEL = './models/resnet/grid/fuel_resnet_grid_model.pth'

# 预测单张图片
image_path = '/home/wang/datasets/20260602油量人工识别/AI识别结果不准确/2026-06-02_CCC5629.jpg'

result = predict_single_image(
    image_path=image_path,
    yolo_model_path=YOLO_MODEL,
    resnet_pointer_path=POINTER_MODEL,
    resnet_grid_path=GRID_MODEL,
    conf_threshold=0.5,
    device=None  # 自动检测：有GPU用GPU，没有则用CPU
)

print(result)