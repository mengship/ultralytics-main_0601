# Odometer OBB OCR TODO

This checklist tracks the delivery path from annotations to a production candidate. The model should
not be considered ready solely because YOLO validation mAP looks good; the end-to-end value is the
exact mileage-string accuracy after OCR.

## 1. Annotation and Dataset

- [ ] Confirm every source image has exactly one `odometer` label.
- [ ] Use `rotation` labels consistently. Tighten the quadrilateral around the digit string and
      adjacent `km`; exclude `ODO`, `TRIP`, voltage, and UI labels.
- [ ] Review rotated and perspective-heavy samples at 100% zoom. The four corners must trace the
      actual text-display region, not its axis-aligned outer rectangle.
- [ ] Add representative hard cases: strong rotation, strong perspective, reflections, night/low
      light, blur, overexposure, varied dashboard models, and small ODO targets.
- [ ] Set aside a fixed hold-out test set before model selection. Do not train on or tune against it.
- [ ] Run the LabelMe-to-YOLO-OBB converter and resolve every item in `skipped_samples.json`.
- [ ] Visually inspect at least 30 converted train/val image-label pairs to catch point-order or
      coordinate-normalization errors.

## 2. Detection Baseline

- [ ] Train `yolo11n-obb.pt` for a fast baseline with `imgsz=1024`.
- [ ] Inspect validation visualizations, especially missed small odometer displays and incorrect
      detections on other digital text.
- [ ] Record OBB precision, recall, mAP50, mAP50-95, training configuration, dataset revision, and
      checkpoint path.
- [ ] Compare `imgsz=1024` with 1280 or 1536 if the ODO display is too small after preprocessing.
- [ ] Compare `yolo11n-obb.pt` with `yolo11s-obb.pt` after the baseline is stable.
- [ ] Add difficult false positives and false negatives back into the next annotation/training round.

## 3. Crop Rectification and OCR

- [ ] Run OBB prediction on the fixed validation and hold-out sets with saved polygon overlays and
      rectified crops.
- [ ] Check that rectified crops are horizontal and that no digit is cut off. Tune only the crop
      padding ratio when needed; do not revert to axis-aligned crops for rotated images.
- [ ] Run PaddleOCR with an allowlist restricted to digits and `km`.
- [ ] Save raw OCR output, OCR confidence, final digit string, and failure status for every image.
- [ ] Calculate exact mileage-string accuracy, digit-level accuracy, no-detection rate, invalid-output
      rate, and low-confidence review rate on the hold-out set.
- [ ] Review confusion cases such as `0/6/8`, glare, LED blooming, motion blur, and partially
      occluded digits.

## 4. Production Guardrails

- [ ] Set detection and OCR confidence thresholds from hold-out data, not intuition.
- [ ] Enforce basic output rules: digits only, expected digit length, and explicit failure statuses.
- [ ] If vehicle history is available, add an optional business check: odometer cannot decrease and
      unexpectedly large increments require review.
- [ ] Route `no_detection`, geometry errors, low OCR confidence, and invalid digit counts to manual
      review instead of guessing a mileage value.
- [ ] Persist source image path, model version, detector confidence, OCR confidence, raw text,
      predicted quadrilateral, and final result for traceability.
- [ ] Measure CPU/GPU latency, memory use, and throughput using representative production image sizes.

## 5. Next Iteration: Custom OCR

- [ ] Collect rectified crops from real production failures and manually correct their full text.
- [ ] Build an OCR recognition dataset from the corrected crops. Full-string labels such as
      `010980km` are sufficient; per-character boxes are not required.
- [ ] Fine-tune a PaddleOCR recognition model or CRNN+CTC using only the limited character set
      `0123456789km`.
- [ ] Re-evaluate end-to-end exact mileage accuracy against the frozen hold-out set before replacing
      the off-the-shelf OCR engine.
