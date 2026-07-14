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

Run the tests with:

```bash
python3 -m unittest discover -s odometer_obb_ocr/tests -v
```
