import random
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import geometry as geo  # noqa: E402
from utils import ocr  # noqa: E402


class TestOrderTlTrBrBl(unittest.TestCase):
    def test_stable_for_shuffled_and_rotated_starting_points(self):
        # A mildly rotated rectangle (~15 degrees), known corner identities.
        tl = (100.0, 100.0)
        tr = (300.0, 130.0)
        br = (280.0, 230.0)
        bl = (80.0, 200.0)
        base = [tl, tr, br, bl]

        for shift in range(4):
            rotated = base[shift:] + base[:shift]
            ordered = geo.order_tl_tr_br_bl(rotated)
            self.assertEqual(ordered[0], tl)
            self.assertEqual(ordered[1], tr)
            self.assertEqual(ordered[2], br)
            self.assertEqual(ordered[3], bl)

        reversed_order = list(reversed(base))
        ordered = geo.order_tl_tr_br_bl(reversed_order)
        self.assertEqual(ordered[0], tl)
        self.assertEqual(ordered[1], tr)
        self.assertEqual(ordered[2], br)
        self.assertEqual(ordered[3], bl)

        shuffled = list(base)
        rng = random.Random(1)
        rng.shuffle(shuffled)
        ordered = geo.order_tl_tr_br_bl(shuffled)
        self.assertEqual(ordered[0], tl)
        self.assertEqual(ordered[1], tr)
        self.assertEqual(ordered[2], br)
        self.assertEqual(ordered[3], bl)


class TestValidateQuadForInference(unittest.TestCase):
    def test_valid_quad_passes(self):
        points = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
        result = geo.validate_quad_for_inference(points)
        self.assertEqual(len(result), 4)

    def test_collinear_points_rejected(self):
        points = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0)]
        with self.assertRaises(geo.GeometryError):
            geo.validate_quad_for_inference(points)

    def test_duplicate_points_rejected(self):
        points = [(0.0, 0.0), (0.0, 0.0), (10.0, 10.0), (10.0, 0.0)]
        with self.assertRaises(geo.GeometryError):
            geo.validate_quad_for_inference(points)

    def test_point_inside_triangle_of_others_rejected(self):
        # (5, 3) lies strictly inside the triangle formed by the other three
        # points (not on any edge/diagonal), so no cyclic reordering can make
        # this a convex quadrilateral.
        points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 3.0)]
        with self.assertRaises(geo.GeometryError):
            geo.validate_quad_for_inference(points)


class TestPadQuad(unittest.TestCase):
    def test_padding_stays_within_image_bounds(self):
        width, height = 100, 100
        points = [(2.0, 2.0), (98.0, 2.0), (98.0, 98.0), (2.0, 98.0)]
        padded = geo.pad_quad(points, padding_ratio=0.5, width=width, height=height)
        for x, y in padded:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, float(width))
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, float(height))

    def test_zero_padding_leaves_points_unchanged(self):
        points = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
        padded = geo.pad_quad(points, padding_ratio=0.0, width=100, height=100)
        for (x1, y1), (x2, y2) in zip(points, padded):
            self.assertAlmostEqual(x1, x2)
            self.assertAlmostEqual(y1, y2)


class TestRectifyQuad(unittest.TestCase):
    def test_synthetic_rotated_rectangle_rectifies_horizontal(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        # A wide, mildly rotated rectangle roughly in the middle of the image.
        tl = (80.0, 150.0)
        tr = (320.0, 130.0)
        br = (330.0, 190.0)
        bl = (90.0, 210.0)

        crop = geo.rectify_quad(image, [tl, tr, br, bl], padding_ratio=0.0)

        self.assertEqual(crop.shape[2], 3)
        out_h, out_w = crop.shape[:2]
        self.assertGreater(out_w, out_h)

        expected_w, expected_h = geo.estimate_output_size(tl, tr, br, bl)
        self.assertEqual(out_w, expected_w)
        self.assertEqual(out_h, expected_h)

    def test_too_small_quad_rejected(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        tiny = [(10.0, 10.0), (12.0, 10.0), (12.0, 12.0), (10.0, 12.0)]
        with self.assertRaises(geo.GeometryError):
            geo.rectify_quad(image, tiny, padding_ratio=0.0)


class TestSelectBestDetection(unittest.TestCase):
    def test_returns_argmax_when_above_threshold(self):
        conf = np.array([0.1, 0.9, 0.4])
        self.assertEqual(geo.select_best_detection(conf, 0.5), 1)

    def test_returns_none_when_below_threshold(self):
        conf = np.array([0.1, 0.2, 0.3])
        self.assertIsNone(geo.select_best_detection(conf, 0.5))

    def test_returns_none_for_empty(self):
        conf = np.array([])
        self.assertIsNone(geo.select_best_detection(conf, 0.5))


class FakePaddleReader:
    def __init__(self, raw_output):
        self._raw_output = raw_output

    def ocr(self, crop_bgr, cls=False):
        return self._raw_output


class FakeEasyReader:
    def __init__(self, results):
        self._results = results

    def readtext(self, crop_bgr, allowlist=None, detail=1, paragraph=False):
        return self._results


class TestOcrHelpers(unittest.TestCase):
    def test_extract_digits_strips_non_digits_without_substitution(self):
        self.assertEqual(ocr.extract_digits("012345km"), "012345")
        self.assertEqual(ocr.extract_digits("O12345"), "12345")
        self.assertEqual(ocr.extract_digits(""), "")

    def test_run_paddle_ocr_parses_fake_result(self):
        fake_box = [[0, 0], [10, 0], [10, 10], [0, 10]]
        raw_output = [[[fake_box, ("012345km", 0.91)]]]
        reader = FakePaddleReader(raw_output)
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)

        result = ocr.run_paddle_ocr(reader, dummy)
        self.assertEqual(result.raw_text, "012345km")
        self.assertAlmostEqual(result.confidence, 0.91)

    def test_run_paddle_ocr_handles_empty_result(self):
        reader = FakePaddleReader([None])
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)
        result = ocr.run_paddle_ocr(reader, dummy)
        self.assertEqual(result.raw_text, "")
        self.assertEqual(result.confidence, 0.0)

    def test_run_easy_ocr_parses_fake_result(self):
        fake_box = [[0, 0], [10, 0], [10, 10], [0, 10]]
        results = [(fake_box, "012345km", 0.85)]
        reader = FakeEasyReader(results)
        dummy = np.zeros((10, 10, 3), dtype=np.uint8)

        result = ocr.run_easy_ocr(reader, dummy)
        self.assertEqual(result.raw_text, "012345km")
        self.assertAlmostEqual(result.confidence, 0.85)


if __name__ == "__main__":
    unittest.main()
