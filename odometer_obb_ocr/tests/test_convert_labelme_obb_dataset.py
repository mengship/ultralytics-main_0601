import json
import random
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import convert_labelme_obb_dataset as conv  # noqa: E402


CASE_DIR = Path(__file__).resolve().parents[1] / "case"


class TestGeometryHelpers(unittest.TestCase):
    def test_rotation_shuffled_order_normalizes_to_valid_polygon(self):
        points = [
            (492.8444553040469, 555.567907257399),
            (525.1679441590084, 510.75028465874493),
            (542.1962366027091, 523.0314765004449),
            (509.8727477477477, 567.8490990990991),
        ]
        shuffled = list(points)
        rng = random.Random(0)
        rng.shuffle(shuffled)

        valid = conv.build_valid_quadrilateral(shuffled)
        self.assertEqual(len(valid), 4)
        self.assertTrue(conv.is_simple_quadrilateral(valid))
        self.assertTrue(conv.is_convex(valid))
        self.assertNotAlmostEqual(conv.polygon_area(valid), 0.0)

        normalized, clamped = conv.normalize_and_clamp(valid, 768, 1024)
        self.assertFalse(clamped)
        for x, y in normalized:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 1.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 1.0)

    def test_two_point_rectangle_becomes_four_points(self):
        corners = conv.two_point_to_four_corners((10.0, 20.0), (110.0, 220.0))
        self.assertEqual(len(corners), 4)
        xs = {c[0] for c in corners}
        ys = {c[1] for c in corners}
        self.assertEqual(xs, {10.0, 110.0})
        self.assertEqual(ys, {20.0, 220.0})

        valid = conv.build_valid_quadrilateral(corners)
        self.assertTrue(conv.is_convex(valid))
        self.assertTrue(conv.is_simple_quadrilateral(valid))

    def test_four_point_rectangle_shape_is_accepted(self):
        shape = {
            "label": "odometer",
            "shape_type": "rectangle",
            "points": [
                [224.62962962962956, 250.62962962962933],
                [539.4444444444443, 250.62962962962933],
                [539.4444444444443, 356.1851851851849],
                [224.62962962962956, 356.1851851851849],
            ],
        }
        points = conv.shape_to_quadrilateral(shape)
        self.assertEqual(len(points), 4)
        valid = conv.build_valid_quadrilateral(points)
        self.assertTrue(conv.is_convex(valid))

    def test_degenerate_quadrilateral_is_rejected(self):
        collinear = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        with self.assertRaises(conv.GeometryError):
            conv.build_valid_quadrilateral(collinear)

        duplicate = [(0.0, 0.0), (0.0, 0.0), (10.0, 10.0), (10.0, 0.0)]
        with self.assertRaises(conv.GeometryError):
            conv.build_valid_quadrilateral(duplicate)


class TestCaseSamplesConversion(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(__file__).resolve().parents[1] / "datasets" / "test_case_obb_tmp"
        if self.output_dir.exists():
            import shutil

            shutil.rmtree(self.output_dir)

    def tearDown(self):
        if self.output_dir.exists():
            import shutil

            shutil.rmtree(self.output_dir)

    def _run(self, val_ratio=0.5):
        parser = conv.build_arg_parser()
        args = parser.parse_args(
            [
                "--json-dir",
                str(CASE_DIR),
                "--output-dir",
                str(self.output_dir),
                "--val-ratio",
                str(val_ratio),
                "--seed",
                "42",
                "--overwrite",
            ]
        )
        return conv.run_conversion(args)

    def test_case_samples_produce_valid_nine_field_labels(self):
        stats = self._run(val_ratio=0.5)
        self.assertEqual(stats.converted_total, 2)
        self.assertEqual(stats.skipped_count, 0)

        label_files = list((self.output_dir / "train" / "labels").glob("*.txt")) + list(
            (self.output_dir / "val" / "labels").glob("*.txt")
        )
        self.assertEqual(len(label_files), 2)

        for label_file in label_files:
            line = label_file.read_text(encoding="utf-8").strip()
            fields = line.split()
            self.assertEqual(len(fields), 9)
            self.assertEqual(fields[0], "0")
            for value in fields[1:]:
                v = float(value)
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 1.0)

    def test_only_odometer_label_is_emitted(self):
        self._run(val_ratio=0.5)
        label_files = list((self.output_dir / "train" / "labels").glob("*.txt")) + list(
            (self.output_dir / "val" / "labels").glob("*.txt")
        )
        for label_file in label_files:
            lines = label_file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0].split()[0], "0")

    def test_data_yaml_and_reports_created(self):
        self._run(val_ratio=0.5)
        self.assertTrue((self.output_dir / "data.yaml").is_file())
        report = json.loads((self.output_dir / "conversion_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["converted_total"], 2)
        skipped = json.loads((self.output_dir / "skipped_samples.json").read_text(encoding="utf-8"))
        self.assertEqual(skipped, [])


class TestShapeFiltering(unittest.TestCase):
    def test_non_target_shapes_are_ignored(self):
        shapes = [
            {"label": "oil", "shape_type": "rectangle", "points": [[0, 0], [1, 1]]},
            {"label": "center", "shape_type": "point", "points": [[0, 0]]},
            {
                "label": "odometer",
                "shape_type": "rotation",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            },
        ]
        targets = conv.extract_target_shapes(shapes, "odometer")
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["label"], "odometer")


if __name__ == "__main__":
    unittest.main()
