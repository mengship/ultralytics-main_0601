import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_labelme_obb_dataset as conv  # noqa: E402


def _write_synthetic_image(path: Path, width: int, height: int) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


class TestSyntheticLabelmeConversion(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.json_dir = self.tmp_path / "src"
        self.output_dir = self.tmp_path / "out"
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _write_json(self, name: str, shapes: list, width: int, height: int) -> Path:
        image_name = f"{name}.jpg"
        _write_synthetic_image(self.json_dir / image_name, width, height)

        payload = {
            "version": "4.0.0-beta.2",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        json_path = self.json_dir / f"{name}.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")
        return json_path

    def test_rotation_and_rectangle_annotations_both_convert(self):
        rotation_shapes = [
            {
                "label": "odometer",
                "points": [[50.0, 50.0], [150.0, 60.0], [140.0, 120.0], [40.0, 110.0]],
                "shape_type": "rotation",
                "direction": 0.5,
            },
            {
                "label": "oil",
                "points": [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
                "shape_type": "rectangle",
            },
            {
                "label": "center",
                "points": [[15.0, 15.0]],
                "shape_type": "point",
            },
        ]
        self._write_json("sample_rotation", rotation_shapes, width=200, height=200)

        rectangle_shapes = [
            {
                "label": "odometer",
                "points": [[30.0, 30.0], [170.0, 170.0]],
                "shape_type": "rectangle",
            },
            {
                "label": "tip",
                "points": [[5.0, 5.0]],
                "shape_type": "point",
            },
            {
                "label": "empty",
                "points": [[6.0, 6.0]],
                "shape_type": "point",
            },
            {
                "label": "full",
                "points": [[7.0, 7.0]],
                "shape_type": "point",
            },
        ]
        self._write_json("sample_rectangle", rectangle_shapes, width=200, height=200)

        parser = conv.build_arg_parser()
        args = parser.parse_args(
            [
                "--json-dir",
                str(self.json_dir),
                "--output-dir",
                str(self.output_dir),
                "--val-ratio",
                "0.5",
                "--seed",
                "42",
                "--overwrite",
            ]
        )
        stats = conv.run_conversion(args)

        self.assertEqual(stats.discovered_json, 2)
        self.assertEqual(stats.converted_total, 2)
        self.assertEqual(stats.skipped_count, 0)

        label_files = list((self.output_dir / "train" / "labels").glob("*.txt")) + list(
            (self.output_dir / "val" / "labels").glob("*.txt")
        )
        self.assertEqual(len(label_files), 2)

        for label_file in label_files:
            lines = label_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1, "exactly one odometer OBB line per image")
            fields = lines[0].split()
            self.assertEqual(len(fields), 9)
            self.assertEqual(fields[0], "0")
            for value in fields[1:]:
                v = float(value)
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_non_odometer_shapes_never_emitted(self):
        shapes = [
            {
                "label": "odometer",
                "points": [[20.0, 20.0], [80.0, 20.0], [80.0, 60.0], [20.0, 60.0]],
                "shape_type": "rectangle",
            },
            {
                "label": "oil",
                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                "shape_type": "rectangle",
            },
            {"label": "center", "points": [[5.0, 5.0]], "shape_type": "point"},
            {"label": "tip", "points": [[6.0, 6.0]], "shape_type": "point"},
            {"label": "empty", "points": [[7.0, 7.0]], "shape_type": "point"},
            {"label": "full", "points": [[8.0, 8.0]], "shape_type": "point"},
        ]
        self._write_json("mixed", shapes, width=100, height=100)

        parser = conv.build_arg_parser()
        args = parser.parse_args(
            [
                "--json-dir",
                str(self.json_dir),
                "--output-dir",
                str(self.output_dir),
                "--val-ratio",
                "0.0",
                "--seed",
                "42",
                "--overwrite",
            ]
        )
        conv.run_conversion(args)

        label_files = list((self.output_dir / "train" / "labels").glob("*.txt"))
        self.assertEqual(len(label_files), 1)
        lines = label_files[0].read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].split()[0], "0")


if __name__ == "__main__":
    unittest.main()
