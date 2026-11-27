"""The local latency benchmark: statistics, percentile gates and real (fast) component runs."""
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from research import benchmark


class StatisticsTests(unittest.TestCase):
    def test_summary(self):
        values = [float(v) for v in range(1, 101)]           # 1..100
        self.assertEqual(benchmark.summarize(values),
                         {"n": 100, "median": 50.5, "p90": 90.1, "p95": 95.05, "min": 1.0, "max": 100.0})
        self.assertEqual(benchmark.summarize([]), {"n": 0})

    def test_percentiles_are_gated_by_sample_size(self):
        self.assertEqual(benchmark.summarize([7.0, 9.0]), {"n": 2, "min": 7.0, "max": 9.0})
        self.assertEqual(benchmark.summarize([1.0, 2.0, 9.0]), {"n": 3, "median": 2.0, "min": 1.0, "max": 9.0})
        nine, ten = benchmark.summarize([float(v) for v in range(9)]), benchmark.summarize([float(v) for v in range(10)])
        self.assertNotIn("p90", nine)
        self.assertIn("p90", ten)
        self.assertNotIn("p95", ten)
        self.assertNotIn("p95", benchmark.summarize([float(v) for v in range(19)]))
        self.assertIn("p95", benchmark.summarize([float(v) for v in range(20)]))

    def test_percentile_interpolates(self):
        self.assertEqual(benchmark.percentile([10, 20], 0.5), 15)
        self.assertEqual(benchmark.percentile([30, 10, 20], 0.5), 20)
        self.assertIsNone(benchmark.percentile([], 0.9))

    def test_resized_keeps_shape_and_never_enlarges(self):
        image = np.zeros((1080, 1920, 3), np.uint8)
        self.assertEqual(benchmark.resized(image, 640).shape[:2], (360, 640))
        self.assertIs(benchmark.resized(image, 0), image)
        self.assertIs(benchmark.resized(image, 4000), image)

    def test_component_lists_are_consistent(self):
        for group in (benchmark.NO_IMAGE, benchmark.RESIZABLE, benchmark.DEFAULT):
            self.assertLessEqual(set(group), set(benchmark.COMPONENTS))

    def test_no_web_or_vision_language_components(self):
        self.assertEqual(set(benchmark.COMPONENTS),
                         {"decode", "yolo", "crop", "ocr", "matcher", "barcode_decode", "barcode_lookup"})
        with open(benchmark.__file__) as f:
            source = f.read()
        for forbidden in ("from web", "import web", "import flask", "ollama", "qwen"):
            self.assertNotIn(forbidden, source.lower())


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cv2.imwrite(os.path.join(cls.folder.name, "a.png"), np.full((120, 160, 3), 200, np.uint8))
        cv2.imwrite(os.path.join(cls.folder.name, "b.jpg"), np.full((90, 160, 3), 90, np.uint8))
        with open(os.path.join(cls.folder.name, "notes.txt"), "w") as f:
            f.write("not an image")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def args(self, **changes):
        return SimpleNamespace(**dict({"images": self.folder.name, "trials": 3, "timeout": 120}, **changes))

    def test_only_images_are_listed(self):
        self.assertEqual([os.path.basename(p) for p in benchmark.list_images(self.folder.name)], ["a.png", "b.jpg"])

    def test_component_runs_in_its_own_process(self):
        result = benchmark.run_component(self.args(), "crop", 0)
        self.assertEqual((result["status"], result["failures"]), ("ok", []))
        self.assertEqual(result["warm_ms"]["n"], 6)           # 3 trials x 2 images; the cold call is separate
        self.assertIn("cold_ms", result)
        self.assertIn("load_ms", result)
        self.assertGreater(result["peak_rss_mb"], 1)
        self.assertEqual(result["image_sizes"], ["160x120", "160x90"])

    def test_resolution_limit_is_applied(self):
        result = benchmark.run_component(self.args(trials=1), "barcode_decode", 80)
        self.assertEqual((result["status"], result["max_side"]), ("ok", 80))
        self.assertEqual(result["image_sizes"], ["80x45", "80x60"])
        self.assertEqual(result["warm_outcomes"][0]["result"], {"barcodes": 0, "decoded": 0})

    def test_timeout_is_recorded_not_raised(self):
        result = benchmark.run_component(self.args(timeout=0.01), "crop", 0)
        self.assertEqual(result, {"component": "crop", "max_side": None, "status": "timeout", "timeout_s": 0.01})

    def test_csv_summary(self):
        results = [benchmark.run_component(self.args(trials=1), "crop", 0),
                   {"component": "ocr", "status": "timeout", "timeout_s": 1}]
        path = os.path.join(self.folder.name, "out.csv")
        benchmark.write_csv(path, results)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["component"], r["status"]) for r in rows], [("crop", "ok"), ("ocr", "timeout")])
        self.assertEqual(rows[0]["warm_n"], "2")
        self.assertEqual(rows[0]["warm_median_ms"], "")       # n=2 is below the median gate
        self.assertEqual(rows[1]["warm_median_ms"], "")       # nothing measured, nothing invented


def benchmark_without_resource():
    """A fresh copy of research.benchmark imported as if the platform had no `resource`
    module (Windows). sys.modules["resource"] = None makes `import resource` fail."""
    spec = importlib.util.spec_from_file_location("benchmark_no_resource", benchmark.__file__)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"resource": None}):
        spec.loader.exec_module(module)
    return module


class PlatformWithoutResourceTests(unittest.TestCase):
    """Regression: the benchmark imported the Unix-only `resource` module unconditionally,
    so it (and these tests) failed on Windows."""

    def test_imports_and_reports_memory_as_unavailable(self):
        module = benchmark_without_resource()
        self.assertIsNone(module.resource)
        self.assertIsNone(module.peak_rss_mb())
        self.assertIsNotNone(benchmark.peak_rss_mb())                 # this platform still reports it

    def test_component_run_records_unavailable_memory(self):
        module = benchmark_without_resource()
        with tempfile.TemporaryDirectory() as folder:
            cv2.imwrite(os.path.join(folder, "a.png"), np.full((20, 30, 3), 100, np.uint8))
            args = SimpleNamespace(child="crop", child_max_side=0, images=folder, trials=1)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                module.run_child(args)
            line = next(l for l in output.getvalue().splitlines() if l.startswith("##RESULT##"))
            result = json.loads(line[len("##RESULT##"):])
            self.assertEqual(result["status"], "ok")
            self.assertIsNone(result["peak_rss_mb"])
            self.assertIn("unavailable", result["peak_rss_note"])
            path = os.path.join(folder, "out.csv")
            module.write_csv(path, [result])
            with open(path) as f:
                row = next(csv.DictReader(f))
            self.assertEqual(row["peak_rss_mb"], "")                      # blank, not an invented number


if __name__ == "__main__":
    unittest.main()
