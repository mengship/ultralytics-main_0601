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

From repository root:

```bash
python call_entrance_pose/convert_labelme_pose_dataset.py \
  --json-dir "/path/to/json_and_images" \
  --output-dir fuel_pose_dataset
```

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
```

## Predict Fuel Ratio

```bash
python call_entrance_pose/predict_pose_fuel.py \
  --model runs/fuel_pose/pose_4kpt/weights/best.pt \
  --source /path/to/images \
  --direction tip_side \
  --save-vis
```

The dataset rule is:

```text
direction = the side from the empty line toward the tip line
fuel_ratio = angle(empty -> tip, direction) / angle(empty -> full, direction)
```

So `--direction tip_side` is the default. In code this chooses the direction where `tip` lies inside the `empty -> full` sweep, then reports both the clamped `fuel_ratio` and the raw `raw_fuel_ratio` for debugging.

`--direction auto` chooses the shorter arc from `empty` to `full`. Use it only for quick experiments with old annotations:

```bash
--direction auto
```

For fixed-direction gauges, use:

```bash
--direction clockwise
--direction counterclockwise
```

In image coordinates, `clockwise` means increasing `atan2(y - cy, x - cx)`.
