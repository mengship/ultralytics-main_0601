#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版：X-AnyLabeling JSON 转 YOLO 格式
一键生成两个数据集目录：
  1. fuel_detection_dataset - 原始标注格式（5列+6列）
  2. fuel_yolo_dataset - YOLO训练格式（5列 + data.yaml）

【数据分配规则】
- 指针类：80% 训练 / 20% 验证
- 格子类：80% 训练 / 20% 验证
分别按油表类型切分，确保两个集合都包含两种类型的数据
"""
import json
from pathlib import Path
import shutil
import random
import yaml


def prepare_yolo_dataset_from_converted():
    """从转换后的fuel_detection_dataset生成YOLO格式的训练数据集

    生成2分类YOLO数据集：
    - class 0: 指针类油表框
    - class 1: 格子类油表框
    """

    dataset_dir = Path("../fuel_detection_dataset")
    yolo_dir = Path("../fuel_yolo_dataset")

    # 🗑️  清空目标目录（确保数据干净）
    if yolo_dir.exists():
        print("🗑️  清空旧YOLO数据集...\n")
        shutil.rmtree(yolo_dir)

    # 创建YOLO格式的train/val目录
    yolo_dir.mkdir(exist_ok=True)

    for split in ['train', 'val']:
        (yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    print("📝 准备YOLO框检测数据集（2分类：指针 + 格子）...\n")

    total_files = 0
    type_stats = {'pointer': 0, 'grid': 0}

    for split in ['train', 'val']:
        src_images = dataset_dir / split / 'images'
        src_labels = dataset_dir / split / 'labels'

        dst_images = yolo_dir / split / 'images'
        dst_labels = yolo_dir / split / 'labels'

        split_count = 0

        # 处理每个标签文件
        for txt_file in src_labels.glob('*_fuel.txt'):
            img_name = txt_file.stem.replace('_fuel', '')

            # 找到对应的图片
            img_files = list(src_images.glob(f'{img_name}.*'))
            if not img_files:
                continue

            img_file = img_files[0]

            # 复制图片
            shutil.copy(str(img_file), str(dst_images / img_file.name))

            # 转换标签格式：读取油表类型，转换为正确的class_id
            with open(txt_file) as f:
                line = f.readline().strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 1:
                        class_id = int(parts[0])  # 直接从第1列读取class_id
                    else:
                        class_id = 0

                    # 统计类型
                    if class_id == 0:
                        type_stats['pointer'] += 1
                    else:
                        type_stats['grid'] += 1

                    # 生成YOLO标注（5列，class_id正确分配）
                    yolo_label = f"{class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
                    with open(dst_labels / f'{img_name}.txt', 'w') as out_f:
                        out_f.write(yolo_label)
                    split_count += 1

        total_files += split_count
        print(f"  ✅ {split.upper()}: {split_count} 张图片")

    # 创建data.yaml（2分类：指针类 + 格子类）
    data_yaml = {
        'path': str(yolo_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 2,
        'names': {0: 'pointer', 1: 'grid'}
    }

    yaml_path = yolo_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ YOLO数据集已准备: {yolo_dir}")
    print(f"   - 总计: {total_files} 张图片")
    print(f"   - 指针类(class_id=0): {type_stats['pointer']} 张")
    print(f"   - 格子类(class_id=1): {type_stats['grid']} 张\n")

    return str(yaml_path)


def convert_json_to_yolo():
    """一键转换"""

    # =================== 修改这里 ===================
    # JSON_DIR = r"E:\data"                    # 你的 JSON + 图像文件目录
    JSON_DIR = r"/home/wang/datasets/data_relabel"                    # 你的 JSON + 图像文件目录
    OUTPUT_DIR = r"../fuel_detection_dataset"  # 输出目录
    RANDOM_SEED = 42                         # 随机种子（保证可复现）
    # ================================================

    # 设置随机种子
    random.seed(RANDOM_SEED)

    json_dir = Path(JSON_DIR)
    output_dir = Path(OUTPUT_DIR)

    # 检查输入目录
    if not json_dir.exists():
        print(f"❌ 错误：目录不存在 {json_dir}")
        return

    # 清空目标目录（如果存在）
    if output_dir.exists():
        print(f"🗑️  清空目标目录: {output_dir}")
        shutil.rmtree(output_dir)
        print(f"✅ 目标目录已清空\n")

    # 创建输出目录（只需train和val）
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # 获取所有 JSON 文件
    json_files = sorted(json_dir.glob("*.json"))
    print(f"✅ 找到 {len(json_files)} 个标注文件\n")

    if not json_files:
        print("❌ 没有找到 JSON 文件！")
        return

    # 【第一步】先遍历所有文件，确定每个图片的油表类型
    print("📋 第一步：识别数据集中的油表类型...\n")
    file_types = {}  # {json_filename: fuel_type}
    pointer_files = []  # 指针类文件
    grid_files = []  # 格子类文件

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        # 确定这个文件的油表类型（以第一个框的类型为准）
        fuel_type = 'pointer'  # 默认
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'rectangle':
                label = shape.get('label', '').strip().lower()
                if label == 'oil1':
                    fuel_type = 'grid'
                elif label == 'oil' or label == '':
                    fuel_type = 'pointer'
                else:
                    fuel_type = 'pointer'
                break  # 只看第一个框

        file_types[json_file.name] = fuel_type
        if fuel_type == 'pointer':
            pointer_files.append(json_file)
        else:
            grid_files.append(json_file)

    print(f"  ✅ 指针类文件: {len(pointer_files)} 个")
    print(f"  ✅ 格子类文件: {len(grid_files)} 个\n")

    # 【第二步】分别对指针类和格子类进行 8:2 切分
    print("📊 第二步：按类型分别进行 8:2 切分...\n")
    
    # 打乱顺序
    random.shuffle(pointer_files)
    random.shuffle(grid_files)

    # 指针类切分
    pointer_train_count = int(len(pointer_files) * 0.8)
    pointer_train = pointer_files[:pointer_train_count]
    pointer_val = pointer_files[pointer_train_count:]

    # 格子类切分
    grid_train_count = int(len(grid_files) * 0.8)
    grid_train = grid_files[:grid_train_count]
    grid_val = grid_files[grid_train_count:]

    print(f"  【指针类】")
    print(f"    - 训练集: {len(pointer_train)} 张 ({len(pointer_train)/len(pointer_files)*100:.1f}%)")
    print(f"    - 验证集: {len(pointer_val)} 张 ({len(pointer_val)/len(pointer_files)*100:.1f}%)")
    print(f"  【格子类】")
    print(f"    - 训练集: {len(grid_train)} 张 ({len(grid_train)/len(grid_files)*100:.1f}%)")
    print(f"    - 验证集: {len(grid_val)} 张 ({len(grid_val)/len(grid_files)*100:.1f}%)\n")

    # 创建分配字典
    split_assignment = {}
    for json_file in pointer_train:
        split_assignment[json_file.name] = 'train'
    for json_file in pointer_val:
        split_assignment[json_file.name] = 'val'
    for json_file in grid_train:
        split_assignment[json_file.name] = 'train'
    for json_file in grid_val:
        split_assignment[json_file.name] = 'val'

    # 统计最终的集合数量
    n_train = len(pointer_train) + len(grid_train)
    n_val = len(pointer_val) + len(grid_val)
    n_total = len(json_files)

    fuel_data = {}
    failed_fuel_images = []  # 记录解析油量失败的图片
    image_type_stats = {'pointer': 0, 'grid': 0}  # 按图片统计油表类型
    frame_type_stats = {'pointer': 0, 'grid': 0}  # 按框统计油表类型

    # 【第三步】处理数据
    print("🔄 第三步：处理数据和生成标注...\n")

    for idx, json_file in enumerate(json_files):
        # 获取分配的集合
        split = split_assignment[json_file.name]

        print(f"[{idx+1}/{n_total}] 处理 {json_file.name} -> {split}")

        try:
            # 读取 JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ❌ 读取失败: {e}")
            continue

        # 获取图像信息
        image_path = data.get('imagePath', '')
        image_height = data.get('imageHeight', 0)
        image_width = data.get('imageWidth', 0)

        if not image_path:
            print(f"  ⚠️  没有图像路径")
            continue

        # 查找图像文件
        image_candidates = [
            json_dir / image_path,
            json_dir / Path(image_path).name,
        ]

        image_file = None
        for candidate in image_candidates:
            if candidate.exists():
                image_file = candidate
                break

        if not image_file:
            print(f"  ⚠️  找不到图像 {image_path}")
            continue

        # 复制图像
        image_name = image_file.stem
        image_ext = image_file.suffix
        dest_image = output_dir / split / 'images' / f"{image_name}{image_ext}"
        shutil.copy(image_file, dest_image)

        # 处理标注
        yolo_lines = []
        yolo_fuel_lines = []
        fuel_ratio = 0.5  # 默认值
        fuel_type = 'pointer'  # 默认油表类型（用于图片级统计）
        image_has_frames = False

        for shape in data.get('shapes', []):
            if shape['shape_type'] != 'rectangle':
                continue

            points = shape.get('points', [])
            if len(points) != 4:
                continue

            image_has_frames = True

            # 提取油表类型（从label字段）
            # label: "oil" = pointer类， "oil1" = grid类
            label = shape.get('label', '')
            if label:
                label = str(label).strip().lower()

            # 为这个shape确定油表类型
            shape_fuel_type = 'pointer'  # 这个shape的类型
            if label == 'oil1':
                shape_fuel_type = 'grid'
            elif label == 'oil' or label == '':
                shape_fuel_type = 'pointer'
            else:
                # 未知类型，也默认为指针类
                shape_fuel_type = 'pointer'
                print(f"  ⚠️  未知的油表类型: '{label}' (默认为指针类)")

            # 更新图片级别的fuel_type（以最后一个框为准）
            fuel_type = shape_fuel_type

            # 提取油量（从描述字段）
            # 格式: "0.5" 或其他数字
            description = shape.get('description', '').strip()
            if description:
                try:
                    fuel_ratio = float(description)
                    # 限制范围在 [0, 1]
                    fuel_ratio = max(0.0, min(1.0, fuel_ratio))
                except ValueError:
                    # 解析失败，记录失败信息
                    failed_fuel_images.append({
                        'image': image_name,
                        'label': label,
                        'description': description,
                        'error': '油量无法转换为数字'
                    })
                    print(f"  ⚠️  油量解析失败: '{description}' (使用默认值 0.50)")
                    fuel_ratio = 0.5
            else:
                # 没有描述，使用默认值
                failed_fuel_images.append({
                    'image': image_name,
                    'label': label,
                    'description': '(无描述)',
                    'error': '缺少油量信息'
                })
                print(f"  ⚠️  缺少油量信息 (使用默认值 0.50)")
                fuel_ratio = 0.5

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # YOLO 格式（归一化）
            cx = (x_min + x_max) / 2 / image_width
            cy = (y_min + y_max) / 2 / image_height
            w = (x_max - x_min) / image_width
            h = (y_max - y_min) / image_height

            # 根据这个shape的油表类型转换class_id
            class_id = 0 if shape_fuel_type == 'pointer' else 1

            # 5 列格式（YOLO 检测）
            yolo_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)

            # 6 列格式（含油量，用于CNN训练）
            # 格式: class cx cy w h fuel_ratio
            # class: 0=指针类, 1=格子类
            yolo_fuel_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {fuel_ratio:.2f}"
            yolo_fuel_lines.append(yolo_fuel_line)

            # 统计框数量
            frame_type_stats[shape_fuel_type] += 1

        # 保存标注文件（5列，YOLO 标准格式）
        label_file = output_dir / split / 'labels' / f"{image_name}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # 保存含油量的标注文件（6列，用于CNN训练）
        label_fuel_file = output_dir / split / 'labels' / f"{image_name}_fuel.txt"
        with open(label_fuel_file, 'w') as f:
            f.write('\n'.join(yolo_fuel_lines))

        # 统计图片级别的油表类型（每张图片只算一次，以最后一个框的类型为准）
        if image_has_frames:
            image_type_stats[fuel_type] += 1

        # 保存油量信息
        if fuel_ratio is not None and fuel_ratio != 0.5:
            fuel_data[image_name] = {'fuel': fuel_ratio, 'type': fuel_type}
            print(f"  ✅ 油量: {fuel_ratio:.2f} | 类型: {fuel_type} (label识别)")

    # 保存油量信息
    with open(output_dir / 'fuel_ratios.json', 'w') as f:
        json.dump(fuel_data, f, indent=2)

    # 创建 YAML（2分类：指针类 + 格子类）
    yaml_content = f"""path: {output_dir.resolve()}
train: train/images
val: val/images
nc: 2
names:
  0: pointer
  1: grid
"""
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    print(f"\n{'='*50}")
    print(f"✅ 转换完成！")
    print(f"{'='*50}")
    print(f"📁 输出目录: {output_dir.resolve()}")
    print(f"📊 数据统计:")
    print(f"   - 训练集: {n_train} 张（80%）")
    print(f"   - 验证集: {n_val} 张（20%）")
    print(f"🔀 分配方式: 按油表类型分别切分（指针 8:2 + 格子 8:2）")
    print(f"   ✨ 确保训练集和验证集都包含两种类型的数据")
    print(f"🔍 已保存 {len(fuel_data)} 个油量信息")
    print(f"\n🎯 油表类型统计:")
    print(f"   【图片级别】")
    print(f"   - 指针类: {image_type_stats['pointer']} 张")
    print(f"   - 格子类: {image_type_stats['grid']} 张")
    print(f"   【框级别】")
    print(f"   - 指针类: {frame_type_stats['pointer']} 个框")
    print(f"   - 格子类: {frame_type_stats['grid']} 个框")

    # 输出失败的图片列表
    if failed_fuel_images:
        print(f"\n⚠️  油量解析失败的图片（共 {len(failed_fuel_images)} 张）:")
        print(f"{'='*50}")
        for failed in failed_fuel_images:
            print(f"  ❌ {failed['image']}")
            print(f"     描述: {failed['description']}")
            print(f"     原因: {failed['error']}")
        print(f"{'='*50}\n")
    else:
        print(f"\n✅ 所有图片油量信息解析成功！\n")

    # 【第二步】生成YOLO训练数据集
    print("\n" + "="*60)
    print("【第二步】生成YOLO训练数据集")
    print("="*60 + "\n")
    prepare_yolo_dataset_from_converted()

    print(f"\n{'='*60}")
    print(f"✨ 所有转换完成！")
    print(f"{'='*60}")
    print(f"📁 生成的数据集目录：")
    print(f"   1️⃣  fuel_detection_dataset - 原始标注（用于CNN训练）")
    print(f"   2️⃣  fuel_yolo_dataset - YOLO训练数据（用于YOLO训练）")
    print(f"\n下一步: python train_yolo_fuel_resnet.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    convert_json_to_yolo()