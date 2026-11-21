import json
import os
import unittest

import numpy as np

from vision.identification import (
    clamp_box,
    crop_box,
    format_identification,
    identify_detection,
    select_best_detection,
)
from vision.products.catalog import ProductCatalog
from vision.products.matcher import ProductMatcher

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")


def detection(class_name, confidence, box=(10, 10, 50, 50)):
    return {"class_id": 0, "class_name": class_name, "confidence": confidence, "box": box}


class StubRecognizer:
    """Stands in for TextRecognizer; records the image it was given."""

    def __init__(self, results):
        self.results = results
        self.images = []

    def read(self, image):
        self.images.append(image)
        return self.results


class SelectionTests(unittest.TestCase):
    def test_highest_confidence_bottle(self):
        detections = [detection("person", 0.99), detection("bottle", 0.6), detection("bottle", 0.8, (1, 2, 3, 4))]
        self.assertEqual(select_best_detection(detections)["box"], (1, 2, 3, 4))

    def test_no_bottle(self):
        self.assertIsNone(select_best_detection([detection("person", 0.99)]))
        self.assertIsNone(select_best_detection([]))


class CropTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3)

    def test_crop_uses_box(self):
        crop = crop_box(self.frame, (10, 20, 60, 70))
        self.assertEqual(crop.shape, (50, 50, 3))
        self.assertTrue(np.array_equal(crop, self.frame[20:70, 10:60]))

    def test_box_clamped_to_frame(self):
        self.assertEqual(clamp_box((-5, -5, 250, 150), 200, 100), (0, 0, 200, 100))
        self.assertEqual(crop_box(self.frame, (150, 50, 400, 400)).shape, (50, 50, 3))

    def test_empty_box(self):
        self.assertIsNone(clamp_box((50, 50, 50, 80), 200, 100))
        self.assertIsNone(clamp_box((300, 10, 400, 50), 200, 100))
        self.assertIsNone(crop_box(self.frame, (10, 10, 5, 5)))


class IdentifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.matcher = ProductMatcher(ProductCatalog.from_openfda_results(json.load(f)["results"]))

    def setUp(self):
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)

    def test_no_bottle_skips_ocr(self):
        recognizer = StubRecognizer([])
        self.assertIsNone(identify_detection(self.frame, [detection("person", 0.9)], recognizer, self.matcher))
        self.assertEqual(recognizer.images, [])

    def test_ocr_runs_on_best_bottle_crop(self):
        recognizer = StubRecognizer([{"text": "ADVIL", "confidence": 0.9, "box": (0, 0, 5, 5)}])
        detections = [detection("bottle", 0.5, (0, 0, 10, 10)), detection("bottle", 0.9, (20, 30, 120, 90))]
        found, ocr_results, result = identify_detection(self.frame, detections, recognizer, self.matcher)
        self.assertEqual(found["confidence"], 0.9)
        self.assertEqual(recognizer.images[0].shape, (60, 100, 3))
        self.assertEqual(result.status.value, "UNCERTAIN")

    def test_format_outputs(self):
        cases = [
            ([], "NO_MATCH\nNo supported product identity found."),
            ([{"text": "ADVIL", "confidence": 0.9}], "UNCERTAIN"),
            ([{"text": "ADVIL", "confidence": 0.9}, {"text": "NDC 0573-0150-20", "confidence": 0.9}],
             "VERIFIED / PACKAGE"),
        ]
        for ocr_results, expected in cases:
            text = format_identification(detection("bottle", 0.8), ocr_results, self.matcher.match(ocr_results))
            self.assertIn("--- PRODUCT IDENTIFICATION ---", text)
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
