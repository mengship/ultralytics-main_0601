# Prompt for Claude

Copy the following request to Claude together with access to this repository.

```text
You are working in the repository:
  /Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/ultralytics-main_0601

Implement a production-oriented odometer-reading pipeline in this NEW directory only:
  odometer_obb_ocr/

Do not modify the Ultralytics package source, `call_entrance_pose/`, or unrelated files. You may
create subdirectories and files under `odometer_obb_ocr/`. Use Python 3.8+ compatible syntax.

## Problem

Given photos of vehicle dashboards, find the digital odometer display, including heavily rotated
or perspective-skewed cases, then read the mileage number. The required flow is:

  source image
  -> YOLO OBB detects the odometer text display as a rotated quadrilateral
  -> order the four corners and perform a four-point perspective transform
  -> run OCR on the rectified crop
  -> return digits only, plus confidence and an auditable result record

The detector answers "where is the odometer?". OCR answers "what number is in it?".

## Existing annotation format

Raw annotations are LabelMe / X-AnyLabeling JSON files. A representative source directory is:

  /Users/flash/Downloads/07.09油表数据标注

Each JSON file has `imagePath`, `imageWidth`, `imageHeight`, and `shapes`. The desired region uses
the label `odometer`, for example:

  {
    "label": "odometer",
    "points": [[1053.08, 35.13], [1363.92, 241.65],
               [1318.18, 310.49], [1007.35, 103.97]],
    "shape_type": "rotation",
    "direction": 0.5864
  }

Other labels in the same JSON, such as `oil`, `center`, `tip`, `empty`, and `full`, must be ignored.

The annotation policy is:

- one `odometer` object per source image;
- the quadrilateral tightly covers the mileage digit string, normally including its adjacent `km`;
- exclude the left-side `ODO` label and unrelated screen elements;
- all new labels should use `shape_type: "rotation"`, but the converter must also accept LabelMe
  `shape_type: "rectangle"` with two diagonal points and expand it to a four-point horizontal box.

For `rotation`, use the four `points` values; ignore `direction`. Do not approximate a rotation
annotation with an axis-aligned bounding box.

## Required deliverables

Create a clear, self-contained project with this approximate layout:

  odometer_obb_ocr/
    README.md
    requirements-optional.txt
    convert_labelme_obb_dataset.py
    train_obb.py
    validate_obb.py
    predict_odometer.py
    utils/
      geometry.py
      ocr.py
    tests/
      test_geometry.py
      test_labelme_conversion.py

Small, focused changes to this structure are acceptable, but keep the command-line workflow simple.
Do not require a GUI or an online service.

## 1. LabelMe to YOLO OBB conversion

Implement `convert_labelme_obb_dataset.py`.

Required CLI arguments:

  --json-dir PATH        Required. Directory containing JSON/image pairs; recurse through children.
  --output-dir PATH      Required. New YOLO OBB dataset root.
  --label odometer       Default `odometer`; only this label is converted.
  --val-ratio FLOAT      Default 0.2.
  --seed INT             Default 42.
  --image-dir PATH       Optional fallback location when `imagePath` cannot be resolved relative to JSON.
  --copy-mode {copy,symlink}
  --overwrite            Required to replace an existing output directory.

Expected output:

  <output-dir>/
    data.yaml
    train/images/
    train/labels/
    val/images/
    val/labels/
    conversion_report.json
    skipped_samples.json

Output labels must use the Ultralytics YOLO OBB polygon format, one object per line:

  0 x1 y1 x2 y2 x3 y3 x4 y4

All eight coordinates are normalized to [0, 1] using source image width and height. The four corners
must describe a valid convex quadrilateral in a consistent clockwise or counter-clockwise order,
without a crossing "bow-tie" polygon. Do not append an angle value. The `data.yaml` must declare one
class named `odometer` and use paths that work when commands run from the repository root.

Converter behavior and validation:

- Resolve `imagePath` relative to the JSON file first; then try `--image-dir`; report missing images.
- Support `rotation` (four points) and `rectangle` (two diagonal points) as described above.
- Validate finite coordinates, nonzero polygon area, convexity, image bounds, and exactly one target
  shape per input image. Skip invalid or ambiguous samples and record the reason in
  `skipped_samples.json`; do not silently repair invalid labels.
- Clamp only harmless floating point boundary drift, and record any clamping in the report.
- Preserve each source image exactly once. Avoid filename collisions for same-named images in
  different directories, e.g. derive a stable unique name from the relative source path.
- Split at source-image level with the provided seed. A source image must never appear in both splits.
- Print a concise summary: discovered JSON files, converted train/val counts, missing image count,
  invalid annotation count, and skipped samples.

## 2. OBB training and validation

Implement `train_obb.py` and `validate_obb.py` using the local `ultralytics` package.

Training requirements:

- Use an OBB pretrained checkpoint, never a normal detection checkpoint. Default to
  `yolo11n-obb.pt`, but expose `--model` so `yolo11s-obb.pt` can be used for the production run.
- Use `YOLO(model_path).train(...)` with task `obb` and an explicit `project` directory under
  `odometer_obb_ocr/runs` by default.
- Expose practical arguments at least for `--data`, `--model`, `--epochs`, `--imgsz`, `--batch`,
  `--device`, `--workers`, `--project`, `--name`, `--seed`, `--patience`, and `--resume`.
- Choose conservative defaults suitable for small, elongated odometer targets: default `imgsz=1024`;
  document trying 1280 or 1536 if the ODO display remains too small after resizing.
- Save the complete run configuration and print the path to `best.pt`.
- Keep augmentation choices visible and configurable. Do not use vertical flips. Avoid aggressive
  perspective/rotation augmentation until baseline behavior is validated because annotations already
  contain strong rotation.

Validation requirements:

- Run `YOLO(weights).val(data=..., task="obb")`.
- Save metrics and annotated validation predictions under the supplied output directory.
- Print box precision, recall, mAP50, and mAP50-95, but do not claim that detection mAP equals
  end-to-end mileage accuracy.

## 3. Geometry and perspective rectification

Implement reusable geometry functions in `utils/geometry.py`.

Input during prediction will be an Ultralytics result OBB quadrilateral from:

  result.obb.xyxyxyxy

Implement and test the following behavior:

- Select the highest-confidence detection of class `odometer`; expose `--det-conf` and record the
  detection score. If no detection passes threshold, return status `no_detection`.
- Robustly order a valid quadrilateral into top-left, top-right, bottom-right, bottom-left order.
  Do not assume the predictor's point order is stable.
- Reject self-crossing, degenerate, non-convex, or too-small quadrilaterals with an explicit status.
- Add an optional small fractional border around the quadrilateral before warping, controlled by
  `--crop-padding-ratio` with default 0.02, while keeping coordinates within image bounds.
- Estimate output width from the average top/bottom edge lengths and output height from the average
  left/right edge lengths. Apply `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to produce
  a horizontal crop. Never use just an axis-aligned outer rectangle for a rotated detection.
- Preserve the original aspect ratio; do not force an arbitrary square crop.

Provide a visualization helper that draws predicted polygons on the original image and saves the
rectified crop. This is required for diagnosing failures.

## 4. OCR and end-to-end prediction

Implement `predict_odometer.py` with PaddleOCR as the primary engine and EasyOCR as an optional
fallback. OCR dependencies must remain optional, documented in `requirements-optional.txt`, and
imports should raise an actionable installation message only when that engine is selected.

Required CLI arguments:

  --model PATH                 Required YOLO OBB weights.
  --source PATH                Required image file or directory.
  --ocr-engine {paddle,easy}   Default paddle.
  --det-conf FLOAT             Default 0.25.
  --ocr-conf FLOAT             Default 0.70.
  --crop-padding-ratio FLOAT   Default 0.02.
  --device DEVICE              Default auto / empty.
  --output-dir PATH            Default odometer_obb_ocr/runs/predict.
  --save-crops
  --save-vis
  --min-digits INT             Default 4.
  --max-digits INT             Default 8.

OCR requirements:

- Because the crop is already perspective-rectified, do not rely on the OCR orientation classifier
  for arbitrary-angle correction.
- Restrict recognition to `0123456789kmKM` when the engine supports an allowlist.
- Retain raw OCR text and OCR confidence. Extract the odometer by removing non-digits from the
  selected OCR text. Do not silently replace ambiguous characters such as O with 0 in the first
  implementation.
- A result is `ok` only when a detection exists, OCR confidence is at least `--ocr-conf`, and digit
  count is within `[min_digits, max_digits]`. Otherwise return `low_ocr_confidence`,
  `invalid_digit_count`, or the appropriate geometry/detection error.
- For every source image produce one machine-readable record in both JSON and CSV with: source path,
  status, raw OCR text, mileage digits or null, detection confidence, OCR confidence, predicted four
  points, rectified crop path, and visualization path.
- Do not let one bad image abort a directory-level run.

## 5. Tests and documentation

Add focused unit tests that run without model weights or OCR packages:

- quad point ordering is stable for differently ordered rectangle corners;
- invalid and degenerate quadrilaterals are rejected;
- a synthetic rotated rectangle is rectified to a horizontal crop;
- a tiny temporary LabelMe dataset with one `rotation` annotation and one `rectangle` annotation
  converts to valid OBB labels;
- ignored non-odometer shapes are not emitted.

Write `README.md` with exact commands for conversion, training, validation, and prediction. Include
these examples, adjusted for final file names:

  python odometer_obb_ocr/convert_labelme_obb_dataset.py \\
    --json-dir "/Users/flash/Downloads/07.09油表数据标注" \\
    --output-dir odometer_obb_ocr/datasets/odometer_obb

  python odometer_obb_ocr/train_obb.py \\
    --data odometer_obb_ocr/datasets/odometer_obb/data.yaml \\
    --model yolo11n-obb.pt --epochs 100 --imgsz 1024

  python odometer_obb_ocr/predict_odometer.py \\
    --model odometer_obb_ocr/runs/obb/odometer/weights/best.pt \\
    --source "/path/to/dashboard_images" --ocr-engine paddle \\
    --save-crops --save-vis

Before finishing, run unit tests and `--help` for every CLI. Report changed files, commands run,
and any dependency/model-download steps that the user must perform separately.
```

## Notes for Review

- The input JSON `rotation` annotation already stores four corners. `direction` is annotation-tool
  metadata, not a YOLO label value.
- `yolo11n-obb.pt` is a quick baseline. Evaluate `yolo11s-obb.pt` for the final model if speed and
  GPU memory allow it.
- The OCR recognizer is intentionally off-the-shelf in this phase. Save low-confidence crops and
  corrected mileage text later to build an OCR fine-tuning dataset.
