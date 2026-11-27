"""main.py end to end without a camera: mocked frames, detections and OCR.

Runs the real main.py (runpy) with the camera, YOLO, EasyOCR and the OpenCV window
replaced by fakes, presses "i" once and "q", and checks which pixels reach OCR."""
import contextlib
import inspect
import io
import json
import os
import runpy
import unittest
from unittest import mock

import cv2
import numpy as np

import vision.camera
import vision.detector
import vision.ocr
from vision.products.catalog import ProductCatalog

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")
MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
FDA_CACHE = "vision/products/data/otc_ndc_2026-10-30.json"
W, H = 1920, 1080
WATER_BOX = (1490, 280, 1710, 800)        # right of centre, more confident
MEDICINE_BOX = (850, 280, 1070, 800)      # centred, less confident
DETECTIONS = [
    {"class_id": 39, "class_name": "bottle", "confidence": 0.75, "box": WATER_BOX},
    {"class_id": 39, "class_name": "bottle", "confidence": 0.70, "box": MEDICINE_BOX},
    {"class_id": 0, "class_name": "person", "confidence": 0.90, "box": (0, 0, 400, 1080)},
]


def camera_frame():
    """A clean frame whose two bottle regions have distinct pixel values."""
    frame = np.full((H, W, 3), 30, np.uint8)
    x1, y1, x2, y2 = WATER_BOX
    frame[y1:y2, x1:x2] = (200, 120, 40)
    x1, y1, x2, y2 = MEDICINE_BOX
    frame[y1:y2, x1:x2] = (240, 240, 240)
    return frame


class FakeCamera:
    def __init__(self):
        self.frames = 0

    def read(self):
        self.frames += 1
        return (True, camera_frame()) if self.frames <= 5 else (False, None)

    def release(self):
        pass


class FakeResult:
    def __init__(self, frame):
        self.frame = frame

    def plot(self):
        return self.frame.copy()


class Run:
    """One run of main.py with the given environment; records everything."""
    def __init__(self, env):
        self.env, self.ocr_inputs, self.detector_args, self.camera_opened = env, [], [], False
        self.output = io.StringIO()

    def go(self):
        run = self

        class FakeDetector:
            def __init__(self, *args, **kwargs):
                run.detector_args.append((args, kwargs))

            def detect(self, frame):
                return FakeResult(frame)

            def get_detections(self, result):
                return [dict(d) for d in DETECTIONS]

        class FakeRecognizer:
            def __init__(self, *args, **kwargs):
                pass

            def read(self, image):
                run.ocr_inputs.append(image.copy())
                return [{"text": "Advil", "confidence": 0.9}, {"text": "Ibuprofen 200 mg", "confidence": 0.9}]

        def open_camera(*args, **kwargs):
            run.camera_opened = True
            return FakeCamera()

        with open(FIXTURE) as f:
            catalog = ProductCatalog.from_openfda_results(json.load(f)["results"])
        keys = iter([ord("i"), ord("q")])
        real_exists = os.path.exists
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, self.env, clear=False))
            if "VA_TARGET_SELECTION" not in self.env:
                os.environ.pop("VA_TARGET_SELECTION", None)
            stack.enter_context(mock.patch.object(vision.camera, "open_camera", open_camera))
            stack.enter_context(mock.patch.object(vision.detector, "ObjectDetector", FakeDetector))
            stack.enter_context(mock.patch.object(vision.ocr, "TextRecognizer", FakeRecognizer))
            stack.enter_context(mock.patch.object(ProductCatalog, "from_file", classmethod(lambda cls, path: catalog)))
            stack.enter_context(mock.patch("os.path.exists", lambda p: p == FDA_CACHE or real_exists(p)))
            stack.enter_context(mock.patch.object(cv2, "imshow", lambda *a: None))
            stack.enter_context(mock.patch.object(cv2, "waitKey", lambda *a: next(keys)))
            stack.enter_context(mock.patch.object(cv2, "destroyAllWindows", lambda *a: None))
            stack.enter_context(contextlib.redirect_stdout(self.output))
            runpy.run_path(MAIN, run_name="__main__")
        return self


def region(box):
    x1, y1, x2, y2 = box
    return camera_frame()[y1:y2, x1:x2]


class MainIntegrationTests(unittest.TestCase):
    def test_default_reads_the_highest_confidence_bottle_as_before(self):
        run = Run({}).go()
        self.assertEqual(len(run.ocr_inputs), 1)
        self.assertTrue(np.array_equal(run.ocr_inputs[0], region(WATER_BOX)))     # clean pixels, exact crop
        text = run.output.getvalue()
        self.assertNotIn("target selection:", text)                              # nothing new printed
        self.assertIn("bottle 0.75 RIGHT box=(1490, 280, 1710, 800)", text)

    def test_explicit_original_baseline_is_the_same_as_the_default(self):
        run = Run({"VA_TARGET_SELECTION": "confidence"}).go()
        self.assertTrue(np.array_equal(run.ocr_inputs[0], region(WATER_BOX)))

    def test_weighted_opt_in_passes_the_selected_crop_to_ocr(self):
        run = Run({"VA_TARGET_SELECTION": "weighted"}).go()
        self.assertEqual(len(run.ocr_inputs), 1)
        self.assertTrue(np.array_equal(run.ocr_inputs[0], region(MEDICINE_BOX)))
        text = run.output.getvalue()
        self.assertIn("target selection: weighted (wc=0.40, wa=0.25, wp=0.35), frame 1920x1080", text)
        self.assertIn("-> #1", text)
        self.assertIn("bottle 0.70 CENTER box=(850, 280, 1070, 800)", text)       # reported position follows the choice

    def test_validated_confidence_opt_in_matches_the_original_on_valid_boxes(self):
        run = Run({"VA_TARGET_SELECTION": "confidence_validated"}).go()
        self.assertTrue(np.array_equal(run.ocr_inputs[0], region(WATER_BOX)))

    def test_unknown_method_stops_before_the_camera_opens(self):
        run = Run({"VA_TARGET_SELECTION": "largest"})
        with self.assertRaises(ValueError):
            run.go()
        self.assertFalse(run.camera_opened)


class DetectionThresholdTests(unittest.TestCase):
    def test_main_uses_the_detector_default_threshold_of_0_5(self):
        run = Run({"VA_TARGET_SELECTION": "weighted"}).go()
        self.assertEqual(run.detector_args, [((), {})])                         # ObjectDetector() with defaults
        self.assertEqual(inspect.signature(vision.detector.ObjectDetector.__init__).parameters["conf"].default, 0.5)

    def test_the_threshold_is_passed_to_yolo(self):
        calls = []

        class FakeYolo:
            names = {39: "bottle"}

            def __init__(self, path):
                pass

            def __call__(self, frame, **kwargs):
                calls.append(kwargs)
                return [object()]

        with mock.patch.object(vision.detector, "YOLO", FakeYolo):
            vision.detector.ObjectDetector().detect(np.zeros((4, 4, 3), np.uint8))
        self.assertEqual(calls, [{"verbose": False, "conf": 0.5}])

    def test_selection_never_adds_or_rethresholds_detections(self):
        from vision.target_selection import select_target
        above = [dict(d) for d in DETECTIONS]
        for method in ("confidence_validated", "center", "area", "weighted"):
            selection = select_target(above, (W, H), method)
            self.assertTrue(any(selection.detection is d for d in above))      # always one of the inputs
            self.assertEqual(len(selection.candidates), len(above))
        self.assertIsNone(select_target([], (W, H), "weighted").detection)


if __name__ == "__main__":
    unittest.main()
