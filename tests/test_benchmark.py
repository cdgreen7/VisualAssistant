"""The research benchmark runner: statistics and one real (fast) component run."""
import csv
import os
import tempfile
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from research import benchmark


class StatisticsTests(unittest.TestCase):
    def test_summary(self):
        values = [float(v) for v in range(1, 101)]           # 1..100
        self.assertEqual(benchmark.summarize(values),
                         {"n": 100, "median": 50.5, "p90": 90.1, "p95": 95.05, "min": 1.0, "max": 100.0})
        self.assertEqual(benchmark.summarize([7.0]), {"n": 1, "median": 7.0, "p90": 7.0, "p95": 7.0, "min": 7.0, "max": 7.0})
        self.assertEqual(benchmark.summarize([]), {"n": 0})

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
        for group in (benchmark.VLM, benchmark.NO_IMAGE, benchmark.RESIZABLE, benchmark.DEFAULT):
            self.assertLessEqual(set(group), set(benchmark.COMPONENTS))
        self.assertNotIn("research", benchmark.DEFAULT)      # nothing online runs by default


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
        return SimpleNamespace(**dict({"images": self.folder.name, "trials": 3, "vlm_trials": 1, "timeout": 120}, **changes))

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
        results = [benchmark.run_component(self.args(trials=1), "tracking", 0),
                   {"component": "qwen_scene", "status": "unavailable", "reason": "Ollama is not reachable"}]
        path = os.path.join(self.folder.name, "out.csv")
        benchmark.write_csv(path, results)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["component"], r["status"]) for r in rows], [("tracking", "ok"), ("qwen_scene", "unavailable")])
        self.assertEqual(rows[0]["warm_n"], "1")
        self.assertEqual(rows[1]["warm_median_ms"], "")       # nothing measured, nothing invented


if __name__ == "__main__":
    unittest.main()
