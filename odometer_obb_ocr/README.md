# Odometer OBB OCR

This directory is reserved for the odometer-reading pipeline. It is intentionally separate from
`call_entrance_pose`, which estimates pointer fuel levels.

Target pipeline:

```text
LabelMe / X-AnyLabeling rotation annotation
-> YOLO OBB dataset
-> YOLO OBB locates the odometer display
-> four-point perspective rectification
-> PaddleOCR or EasyOCR reads the mileage digits
```

The planned implementation is described in `CLAUDE_IMPLEMENTATION_PROMPT.md`. Work through
`TODO.md` before treating a model as production-ready.

Scope for the first implementation:

- Detect the single `odometer` region in each source image.
- Support `shape_type: "rotation"` with four points, plus ordinary `rectangle` annotations.
- Produce a rectified crop before OCR.
- Extract and validate the numeric odometer value; preserve raw OCR output and confidence for review.

Custom OCR recognition training is not part of the first implementation. Start with PaddleOCR as
the primary recognizer and collect failed or low-confidence rectified crops for the next iteration.

## Dataset conversion

`convert_labelme_obb_dataset.py` converts LabelMe / X-AnyLabeling JSON+image pairs into a YOLO OBB
dataset. Only shapes whose `label` matches `--label` (default `odometer`) are converted; other
shapes (e.g. `oil`, `center`, `tip`, `empty`, `full`) are ignored. Both `rotation` shapes (four
arbitrary corners) and `rectangle` shapes (two diagonal points or four corners) are supported.

```bash
python odometer_obb_ocr/convert_labelme_obb_dataset.py \
  --json-dir odometer_obb_ocr/case \
  --output-dir odometer_obb_ocr/datasets/case_obb \
  --val-ratio 0.5 \
  --seed 42 \
  --overwrite
```

Output structure:

```text
<output-dir>/
  data.yaml
  train/images/  train/labels/
  val/images/    val/labels/
  conversion_report.json
  skipped_samples.json
```

Each label line is `0 x1 y1 x2 y2 x3 y3 x4 y4`, with coordinates normalized to `[0, 1]` by the
actual (OpenCV-read) image dimensions — not `xywhr` or an axis-aligned box. `data.yaml` points at
the output directory itself so training works regardless of the current working directory:

```python
from ultralytics import YOLO

YOLO("yolo11n-obb.pt").train(data="odometer_obb_ocr/datasets/case_obb/data.yaml", task="obb")
```

## Training

`train_obb.py` trains a YOLO OBB detector on the converted dataset. It always uses an OBB
pretrained checkpoint (never a plain detection checkpoint) and defaults to `imgsz=1024` since
odometer displays are small and elongated relative to the full dashboard photo.

```bash
python odometer_obb_ocr/train_obb.py \
  --data /home/wang//datasets/odometer_obb_ocr_obb/data.yaml \
  --model yolo11n-obb.pt --epochs 300 --imgsz 1024
```

Use `--model yolo11s-obb.pt` for the production run once the `yolo11n-obb.pt` baseline looks
reasonable. Vertical flip (`flipud`) is always `0.0` and is not exposed as a flag, since odometer
digits are never upside-down. `--degrees` and `--perspective` default to `0.0` — annotations
already contain strong rotation, so avoid adding rotation/perspective augmentation on top until a
baseline model has been validated. `best.pt` is written under
`<--project>/<--name>/weights/best.pt` (default `odometer_obb_ocr/runs/obb/odometer/weights/best.pt`)
and its path is printed at the end of the run.

## Validation

`validate_obb.py` runs Ultralytics' own OBB validation and prints detection metrics:

```bash
python odometer_obb_ocr/validate_obb.py \
  --weights odometer_obb_ocr/runs/obb/odometer/weights/best.pt \
  --data odometer_obb_ocr/datasets/odometer_obb/data.yaml
```

This prints box precision, recall, mAP50, and mAP50-95, and saves annotated validation predictions
under `<--project>/<--name>/`. These are detection-only metrics — they measure how well the OBB
localizes the odometer region, not end-to-end mileage-reading accuracy, which also depends on
perspective rectification quality and OCR correctness. See `TODO.md` section 3 for how to measure
that separately.

## Prediction

`predict_odometer.py` runs the full pipeline: YOLO OBB detection, perspective rectification to a
horizontal crop, then OCR of the mileage digits.

```bash
python odometer_obb_ocr/predict_odometer.py \
  --model odometer_obb_ocr/runs/obb/odometer/weights/best.pt \
  --source "/home/wang/datasets/odometer_obb_ocr_obb/val/images" --ocr-engine paddle \
  --save-crops --save-vis
```

For each source image, the highest-confidence `odometer` detection above `--det-conf` (default
0.25) is selected, its four corners are ordered into top-left/top-right/bottom-right/bottom-left
regardless of the model's raw point order, validated (rejecting self-crossing, non-convex,
degenerate, or too-small quads), padded by `--crop-padding-ratio` (default 0.02), and rectified via
a perspective transform into a horizontal crop that preserves its original aspect ratio. OCR then
runs on the rectified crop (recognition, not the OCR engine's own orientation classifier, since the
crop is already rectified) and extracts digits by stripping non-digit characters from the raw OCR
text — ambiguous characters like `O` are never auto-corrected to `0`.

A result is `ok` only when a detection exists, OCR confidence is at least `--ocr-conf` (default
0.70), and the digit count falls within `[--min-digits, --max-digits]` (default `[4, 8]`).
Otherwise the status is `no_detection`, `invalid_geometry`, `low_ocr_confidence`,
`invalid_digit_count`, or `error: <message>` for unexpected per-image failures — one bad image
never aborts a directory-level run. Every image produces one record (in both
`predictions.json` and `predictions.csv` under `--output-dir`) with its source path, status, raw
OCR text, extracted mileage digits (or null), detection confidence, OCR confidence, the predicted
four points, and the rectified crop / visualization paths if `--save-crops` / `--save-vis` were
passed.

PaddleOCR (`--ocr-engine paddle`, the default) and EasyOCR (`--ocr-engine easy`) are optional
dependencies — see `requirements-optional.txt`. Importing `predict_odometer.py` or `utils/ocr.py`
never requires either package; only actually running OCR with a given engine does, and an
actionable install message is raised if that engine isn't installed. PaddleOCR's constructor kwargs
(e.g. `use_angle_cls` vs `use_textline_orientation`) vary by installed version, so `utils/ocr.py`
tries a few known-good spellings and falls back to a bare `PaddleOCR(lang="en")`. There is no
reliable cross-version PaddleOCR kwarg for restricting recognition to a charset, so `0123456789kmKM`
restriction is applied as post-processing via `extract_digits` rather than at the engine level.

## Tests

```bash
python3 -m unittest discover -s odometer_obb_ocr/tests -v
```

`tests/test_geometry.py` covers corner ordering, quad validation/rejection, perspective
rectification, and OCR result parsing (using fake reader objects, so no `paddleocr`/`easyocr`
install is required). `tests/test_labelme_conversion.py` covers end-to-end conversion of a
synthetic dataset with both `rotation` and `rectangle` annotations, confirming non-`odometer`
shapes are never emitted. None of the tests require model weights, a GPU, or OCR packages.

## Environment notes

This project was developed without internet access in the development environment, so the
following steps must be performed separately, in an environment with connectivity:

- `yolo11n-obb.pt` / `yolo11s-obb.pt` auto-download on first use of `YOLO(...)` if not already
  present locally.
- `pip install paddleocr paddlepaddle` and/or `pip install easyocr`, per `requirements-optional.txt`,
  for whichever `--ocr-engine` you plan to use.
- Actual training, validation, and prediction runs against real data — verified here only via
  `--help` output, unit tests, and import-safety checks.
