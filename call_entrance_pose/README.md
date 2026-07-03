# Fuel Gauge Pose Version

This directory is a new pose-based version for pointer fuel gauges.

Pipeline:

```text
LabelMe / X-AnyLabeling JSON
-> YOLO Pose dataset
-> YOLO Pose predicts center/tip/empty/full
-> angle mapping
-> fuel_ratio
```

## Keypoint Order

Use one object class:

```yaml
names:
  0: oil_pose
kpt_shape: [4, 2]
```

The four keypoints must use this fixed order:

```text
0 center
1 tip
2 empty
3 full
```

YOLO label format:

```text
class cx cy w h center_x center_y tip_x tip_y empty_x empty_y full_x full_y
```

All coordinates are normalized to the full image size.

## Recommended LabelMe Labels

For each pointer gauge:

```text
rectangle: oil_pose   or oil
point:     center
point:     tip
point:     empty
point:     full
```

The rectangle is recommended. If it is missing, the converter can generate a box from the four points with padding.

If an image contains more than one gauge, set the same `group_id` on the rectangle and its four points.

Grid gauges are not handled by this pose version. Keep using the existing ResNet/grid path for grid gauges.

## Convert Dataset

### Optional: Prefix Filenames With Type

If the raw data is organized by gauge type, for example:

```text
0702/
  left/
  lower left/
  lower right/
  top right/
```

you can prefix every image/JSON filename with its type directory name. This is only for easier checking and later traceability; YOLO does not use filenames as labels.

```bash
# Dry run first
python call_entrance_pose/prefix_type_filenames.py \
  --root "/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/0702"

python call_entrance_pose/prefix_type_filenames.py \
  --root "/home/wang/datasets/yolopose_dataset"

# Apply rename and update JSON imagePath
python call_entrance_pose/prefix_type_filenames.py \
  --root "/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/0702" \
  --apply
```

The prefixes are:

```text
left/        -> left_
lower left/  -> lower_left_
lower right/ -> lower_right_
top right/   -> top_right_
```

The script also updates each JSON `imagePath` so the converter can still find the renamed image.

### Single Directory

From repository root, a plain JSON/image directory still works:

```bash
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir "/path/to/json_and_images" \
  --output-dir fuel_pose_dataset \
  --val-ratio 0.2
```

### Type Subdirectories

If the source directory contains type subdirectories:

```text
0702/
  left/
  lower left/
  lower right/
  top right/
```

pass the parent directory. The converter reads each immediate child directory that contains JSON files as one source group, then splits each group by the same ratio. This keeps all four types represented in both train and val.

```bash
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir "/Users/flash/Documents/Data_Work/99_临时中转站/9 潘杰/0702" \
  --output-dir "/Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/ultralytics-main_0601/call_entrance_pose/dataset_convert" \
  --val-ratio 0.2

python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir "/home/wang/datasets/yolopose_dataset" \
  --output-dir "/home/wang/datasets/yolopose_dataset_convert" \
  --val-ratio 0.2
```

The converter also accepts multiple source directories:

```bash
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir "/path/to/left" "/path/to/lower left" "/path/to/lower right" "/path/to/top right" \
  --output-dir fuel_pose_dataset \
  --val-ratio 0.2
```

Conversion outputs:

```text
dataset_convert/
  data.yaml
  train/images/
  train/labels/
  val/images/
  val/labels/
  pose_metadata.json
  missing_pose_report.json
```

`pose_metadata.json` records each sample's source group and split. `missing_pose_report.json` records incomplete samples.

The converter skips samples that do not have all four keypoints and writes a report:

```text
fuel_pose_dataset/missing_pose_report.json
```

## Train

Place a pose pretrained model in the repo root, for example:

```text
yolo11m-pose.pt
```

Then run:

```bash
python call_entrance_pose/train_yolo_fuel_pose.py \
  --data fuel_pose_dataset/data.yaml \
  --model yolo11m-pose.pt

python call_entrance_pose/train_yolo_fuel_pose.py \
  --data /home/wang/datasets/yolopose_dataset_convert/data.yaml \
  --model yolo11m-pose.pt
```

## Predict Fuel Ratio

```bash
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_4kpt/weights/best.pt \
  --source /path/to/images \
  --direction max_full_span \
  --save-vis

python call_entrance_pose/predict_pose_fuel.py \
  --model /home/wang/ultralytics-main_0601/runs/pose/runs/fuel_pose/pose_4kpt-3/weights/best.pt \
  --source "/home/wang/datasets/data_relabel" \
  --direction max_full_span \
  --output-csv /home/wang/datasets/data_relabel_result/predict_result.csv \
  --save-vis \
  --vis-dir /home/wang/datasets/data_relabel_result/predict_vis
```

The angle calculation rule is:

```text
1. Calculate empty->full and empty->tip angles in both directions (clockwise and counterclockwise)
2. Choose the direction where tip is between empty and full (i.e., empty->tip angle < empty->full angle)
3. If both directions are valid, choose the one with larger full span
4. If neither is valid, fall back to the direction with larger full span
5. fuel_ratio = (empty->tip angle) / (empty->full angle)
6. Clamp to [0, 1], but also report raw_fuel_ratio for debugging
```

So `--direction max_full_span` is the default. It ensures the tip is within the valid range [empty, full] by choosing the direction where the pointer angle is less than the full span angle. Both the clamped `fuel_ratio` and the raw `raw_fuel_ratio` are reported for debugging.

`--direction tip_side` chooses the direction where `tip` lies inside the `empty -> full` sweep. Use it only for comparison with old experiments:

```bash
--direction tip_side
```

For fixed-direction gauges, use:

```bash
--direction clockwise
--direction counterclockwise
```

In image coordinates, `clockwise` means increasing `atan2(y - cy, x - cx)`.
