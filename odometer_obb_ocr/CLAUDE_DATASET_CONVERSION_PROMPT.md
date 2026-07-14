# Prompt for Claude: Convert Odometer Annotations to YOLO OBB

Copy the request below to Claude. Its scope is only the data-conversion step; do not ask it to train
or download model weights yet.

```text
You are working in this repository:
  /Users/flash/Documents/Data_Work/07_学习积累/果壳/projectcode/ultralytics-main_0601

Implement only the dataset conversion stage for odometer OBB detection. Work exclusively under:
  odometer_obb_ocr/

Do not modify `ultralytics/`, `call_entrance_pose/`, or unrelated files.

## Input samples

The directory below contains two actual LabelMe / X-AnyLabeling JSON-image pairs:

  odometer_obb_ocr/case/
    260709_CCF1234.json
    260709_CCF1234.jpg
    260709_CCF3122.json
    260709_CCF3122.jpg

Each JSON has several unrelated fuel-gauge annotations (`oil`, `center`, `tip`, `empty`, `full`) that
must be ignored. Convert only shapes where `label == "odometer"`.

The two samples intentionally demonstrate both annotation forms you must support:

1. `260709_CCF3122.json`: `odometer` has `shape_type: "rotation"` and four rotated corners.
2. `260709_CCF1234.json`: `odometer` has `shape_type: "rectangle"` but its annotation tool writes
   four axis-aligned corners, not merely two diagonal corners.

Future `rectangle` annotations may use either two diagonal points or four corners, so support both.
For `rotation`, use its four `points` values. Ignore the `direction` field; it is tool metadata and
must not be written to the YOLO label.

## Objective

Create a robust CLI script:

  odometer_obb_ocr/convert_labelme_obb_dataset.py

that converts LabelMe-style JSON/image pairs into a dataset accepted by the local Ultralytics OBB
task. Add a short conversion section to `odometer_obb_ocr/README.md`, and add focused tests under
`odometer_obb_ocr/tests/`.

## Required CLI

The script must support:

  python odometer_obb_ocr/convert_labelme_obb_dataset.py \\
    --json-dir odometer_obb_ocr/case \\
    --output-dir odometer_obb_ocr/datasets/case_obb \\
    --val-ratio 0.5 \\
    --seed 42 \\
    --overwrite

Arguments:

- `--json-dir PATH`: required; recursively search for `.json` files.
- `--output-dir PATH`: required; write the YOLO OBB dataset here.
- `--label NAME`: default `odometer`; convert only this label.
- `--val-ratio FLOAT`: default `0.2`; require `0 <= value < 1`.
- `--seed INT`: default `42`; deterministic source-image-level split.
- `--image-dir PATH`: optional fallback search directory if `imagePath` is not resolvable relative
  to the JSON file.
- `--copy-mode {copy,symlink}`: default `copy`.
- `--overwrite`: fail if output exists and is nonempty unless this flag is supplied.

## Output structure

Create exactly this usable structure:

  <output-dir>/
    data.yaml
    train/images/
    train/labels/
    val/images/
    val/labels/
    conversion_report.json
    skipped_samples.json

`data.yaml` must contain one class:

  names:
    0: odometer

Use paths that remain valid when the training command runs from the repository root. The output
dataset must be accepted by:

  YOLO("yolo11n-obb.pt").train(data=".../data.yaml", task="obb")

## Label format and geometry requirements

For each source image, write one YOLO OBB label line:

  0 x1 y1 x2 y2 x3 y3 x4 y4

The eight values are the quadrilateral corner coordinates normalized by source image width and
height. They must be written as floats in `[0, 1]`. Do not output `xywhr`, an angle, or an
axis-aligned bounding box.

Implement reusable helpers to:

- turn a two-point rectangle into four corners;
- accept a four-point rectangle unchanged as a polygon before normalizing;
- robustly order any four valid corners into a non-self-crossing cyclic order;
- validate finite values, nonzero area, convexity, and image bounds;
- reject malformed, degenerate, non-convex, out-of-bounds, or ambiguous annotations rather than
  silently generating invalid data.

The orientation (clockwise vs counterclockwise) may be chosen freely, but must be consistent. The
points must not form a bow-tie polygon. Do not calculate or rely on `direction`.

## Source-image handling

- Resolve the image named by `imagePath` relative to the JSON file first.
- If that fails, search the optional `--image-dir` fallback.
- Verify the actual image dimensions with OpenCV/Pillow; compare them with JSON metadata and report
  a mismatch. Use actual image dimensions to normalize coordinates.
- Ignore non-target shapes completely.
- Require exactly one valid target `odometer` shape per source image. Record a skip reason for zero,
  multiple, invalid, or missing-image cases.
- Prevent name collisions when nested input folders contain same-named images, by deriving a stable
  name from the JSON's relative path.
- Do not duplicate a source image across train and validation sets.

## Reports and usability

`conversion_report.json` must include source directory, output directory, seed, val ratio,
discovered JSON count, converted total, train count, val count, image-dimension mismatches, clamped
boundary-drift count, and skipped count. `skipped_samples.json` must list source JSON paths and
human-readable reasons.

The CLI must print a concise equivalent summary. It must not delete or overwrite the source
annotations/images.

## Tests and verification

Add tests requiring no model weights and no external downloads. At minimum test:

- four corners of a rotated rectangle in shuffled order normalize into a valid non-crossing polygon;
- a two-point rectangle becomes four normalized points;
- the four-point `rectangle` case is accepted;
- a degenerate quadrilateral is rejected;
- the two supplied `odometer_obb_ocr/case` examples generate labels with nine whitespace-separated
  fields (class id + eight normalized values), with all coordinate values in `[0, 1]`;
- labels for `oil`, `center`, `tip`, `empty`, and `full` are not emitted.

Run the converter against `odometer_obb_ocr/case` with `--val-ratio 0.5 --seed 42 --overwrite`,
inspect the output labels, run the tests, and run `--help`. In your final response, report the
created files, the resulting train/val counts, the exact label contents for the two sample images,
and any assumptions you made.
```

## Expected Result for the Current Samples

After Claude implements and runs the converter, there should be two images and two label files split
between `train` and `val`; every label file contains exactly one `odometer` OBB polygon. The numeric
coordinate values depend on the point-ordering convention, so do not compare their literal order;
instead verify that the polygon is valid and all values are normalized.
