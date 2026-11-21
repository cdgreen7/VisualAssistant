"""DailyMed pilot tooling. Synthetic files and catalog only: no network, no
models, and the real dataset directory is never written (ROOT is patched)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from research import dailymed_pilot as pilot
from vision.products.catalog import ProductCatalog

SET_A, SET_B = "aaaaaaaa-0000-0000-0000-000000000001", "bbbbbbbb-0000-0000-0000-000000000002"


def raw(product_ndc, set_id, ingredients, package_suffix="01", upc=None):
    return {"product_id": f"{product_ndc}_{set_id}", "product_ndc": product_ndc, "brand_name": "TESTBRAND",
            "generic_name": None, "labeler_name": "Test Labs", "product_type": "HUMAN OTC DRUG",
            "dosage_form": "TABLET", "active_ingredients": [{"name": n, "strength": s} for n, s in ingredients],
            "packaging": [{"package_ndc": f"{product_ndc}-{package_suffix}", "description": "24 TABLET in 1 BOTTLE"}],
            "openfda": {"spl_set_id": [set_id], "upc": [upc] if upc else []}}


CATALOG = ProductCatalog.from_openfda_results([
    raw("1111-2222", SET_A, [("IBUPROFEN", "200 mg/1")], upc="0305730150200"),
    raw("3333-4444", SET_B, [("ACETAMINOPHEN", "500 mg/1")]),
])

LABEL = {
    "id": "t-one", "image": "research/datasets/dailymed-pilot/images/t-one.jpg", "product": "Test",
    "brand_words": ["testbrand"], "ingredients": [["IBUPROFEN", "200 mg"]],
    "formulation": [["IBUPROFEN", "200 mg/1"]], "package_ndc": "1111-2222-01",
    "visible_text": ["TESTBRAND", "Ibuprofen 200 mg"], "roi": [1, 2, 6, 8],
    "barcodes": [{"type": "UPC_A", "digits": "305730150200"}],
    "provenance": {"spl_set_id": SET_A, "media_name": "x.jpg",
                   "source_url": pilot.SOURCE_PREFIX + SET_A + "&name=x.jpg"},
}


def jpeg(width=10, height=10):
    ok, data = cv2.imencode(".jpg", np.full((height, width, 3), 200, np.uint8))
    return data.tobytes()


class FakeResponse:
    def __init__(self, data):
        self.data, self.headers = data, {"Content-Type": "image/jpeg"}

    def read(self, limit=-1):
        return self.data if limit < 0 else self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Sandbox(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(pilot, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.retrieval = self.root / "retrieval.json"

    def write_labels(self, labels):
        path = self.root / "labels.json"
        path.write_text(json.dumps(labels))
        return path


class RepositoryLabelsTest(unittest.TestCase):
    def test_pilot_labels_are_valid_and_cover_both_categories(self):
        labels = pilot.load_labels()
        self.assertGreaterEqual(len(labels), 5)
        self.assertLessEqual(len(labels), 10)
        categories = {label["category"] for label in labels}
        self.assertIn("combination", categories)
        self.assertIn("single_ingredient", categories)
        for label in labels:
            self.assertEqual(len(label["ingredients"]), len(label["formulation"]), label["id"])


class LoadLabelsTest(Sandbox):
    def rejects(self, change):
        label = copy.deepcopy(LABEL)
        change(label)
        with self.assertRaises(ValueError):
            pilot.load_labels(self.write_labels([label]))

    def test_accepts_valid_label(self):
        self.assertEqual(pilot.load_labels(self.write_labels([LABEL]))[0]["id"], "t-one")

    def test_rejects_non_dailymed_or_mismatched_source(self):
        self.rejects(lambda l: l["provenance"].update(source_url="https://example.com/x.jpg"))
        self.rejects(lambda l: l["provenance"].update(spl_set_id=SET_B))

    def test_rejects_image_outside_dataset(self):
        self.rejects(lambda l: l.update(image="vision/advil_front.jpg"))
        self.rejects(lambda l: l.update(image="/tmp/x.jpg"))

    def test_rejects_bad_roi_and_duplicates(self):
        self.rejects(lambda l: l.update(roi=[5, 5, 2, 9]))
        self.rejects(lambda l: l.update(roi=[0, 0, 1.5, 2]))
        with self.assertRaises(ValueError):
            pilot.load_labels(self.write_labels([LABEL, LABEL]))

    def test_rejects_missing_field(self):
        self.rejects(lambda l: l.pop("visible_text"))


class FetchTest(Sandbox):
    def opener(self, data):
        calls = []

        def open_url(request, timeout):
            calls.append(request.full_url)
            return FakeResponse(data)
        return open_url, calls

    def test_downloads_once_and_records_provenance(self):
        open_url, calls = self.opener(jpeg())
        self.assertEqual(pilot.fetch([LABEL], self.retrieval, open_url, pause=0), ["t-one"])
        record = json.loads(self.retrieval.read_text())["t-one"]
        self.assertEqual(record["spl_set_id"], SET_A)
        self.assertEqual(record["source_url"], LABEL["provenance"]["source_url"])
        self.assertEqual(record["bytes"], len(jpeg()))
        self.assertTrue((self.root / LABEL["image"]).exists())

        self.assertEqual(pilot.fetch([LABEL], self.retrieval, open_url, pause=0), [])
        self.assertEqual(len(calls), 1)

    def test_changed_file_is_an_error_and_not_overwritten(self):
        open_url, _ = self.opener(jpeg())
        pilot.fetch([LABEL], self.retrieval, open_url, pause=0)
        image = self.root / LABEL["image"]
        image.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            pilot.fetch([LABEL], self.retrieval, open_url, pause=0)
        self.assertEqual(image.read_bytes(), b"tampered")

    def test_existing_file_without_record_is_an_error(self):
        image = self.root / LABEL["image"]
        image.parent.mkdir(parents=True)
        image.write_bytes(jpeg())
        with self.assertRaises(ValueError):
            pilot.fetch([LABEL], self.retrieval, self.opener(jpeg())[0], pause=0)

    def test_rejects_non_jpeg_and_oversize(self):
        with self.assertRaises(ValueError):
            pilot.fetch([LABEL], self.retrieval, self.opener(b"<html>error</html>")[0], pause=0)
        with mock.patch.object(pilot, "MAX_BYTES", 8):
            with self.assertRaises(ValueError):
                pilot.fetch([LABEL], self.retrieval, self.opener(jpeg())[0], pause=0)
        self.assertFalse((self.root / LABEL["image"]).exists())


class VerifyTest(Sandbox):
    def test_consistent_label_has_no_problems(self):
        problems, notes = pilot.verify([LABEL], CATALOG, check_files=False)
        self.assertEqual((problems, notes), ([], []))

    def test_wrong_formulation_package_or_set(self):
        wrong = copy.deepcopy(LABEL)
        wrong["formulation"] = [["IBUPROFEN", "400 mg/1"]]
        wrong["package_ndc"] = "3333-4444-01"   # a package of the other set
        problems, _ = pilot.verify([wrong], CATALOG, check_files=False)
        self.assertEqual(len(problems), 2)

        unknown = copy.deepcopy(LABEL)
        unknown["provenance"]["spl_set_id"] = "cccccccc-0000-0000-0000-000000000003"
        self.assertEqual(len(pilot.verify([unknown], CATALOG, check_files=False)[0]), 1)

    def test_unlisted_upc_is_a_note_not_a_problem(self):
        label = copy.deepcopy(LABEL)
        label["barcodes"] = [{"type": "UPC_A", "digits": "041100080165"}]
        problems, notes = pilot.verify([label], CATALOG, check_files=False)
        self.assertEqual(problems, [])
        self.assertEqual(len(notes), 1)

    def test_file_checks(self):
        problems, _ = pilot.verify([LABEL], CATALOG, self.retrieval)
        self.assertIn("image missing", problems[0])

        pilot.fetch([LABEL], self.retrieval, lambda request, timeout: FakeResponse(jpeg(10, 10)), pause=0)
        self.assertEqual(pilot.verify([LABEL], CATALOG, self.retrieval)[0], [])

        big_roi = copy.deepcopy(LABEL)
        big_roi["roi"] = [0, 0, 20, 5]
        self.assertIn("exceeds", pilot.verify([big_roi], CATALOG, self.retrieval)[0][0])


class ReplayTest(unittest.TestCase):
    def test_replay_reuses_saved_ocr_and_keeps_previous_outcome(self):
        from vision.products.matcher import ProductMatcher
        matcher = ProductMatcher(CATALOG)
        previous = {"rows": [
            {"id": "t-one", "product": "Test", "input": "front_panel", "outcome": "UNCERTAIN", "level": "NONE",
             "formulations": 3, "reason": "old reason",
             "ocr": [["TESTBRAND", 0.9], ["Ibuprofen 200 mg", 0.9], ["NDC 1111-2222-01", 0.9]]},
            {"id": "t-one", "product": "Test", "input": "oracle", "outcome": "UNCERTAIN", "level": "NONE",
             "formulations": 1},
        ]}
        rows = pilot.replay(previous, [LABEL], matcher)
        self.assertEqual([r["input"] for r in rows], ["front_panel", "oracle"])
        front = rows[0]
        self.assertEqual((front["previous_outcome"], front["previous_reason"]), ("UNCERTAIN", "old reason"))
        self.assertEqual(front["ocr"], previous["rows"][0]["ocr"])           # OCR text carried, not re-read
        self.assertEqual(front["outcome"], "CORRECT_VERIFIED")               # package NDC + full brand
        self.assertEqual(front["package"]["package_ndc"], "1111-2222-01")


class RegionAndOutcomeTest(unittest.TestCase):
    def test_region(self):
        image = np.zeros((10, 12, 3), np.uint8)
        self.assertEqual(pilot.region(image, LABEL, "front_panel").shape[:2], (6, 5))
        self.assertIs(pilot.region(image, LABEL, "full"), image)
        self.assertIs(pilot.region(image, dict(LABEL, roi=None), "front_panel"), image)

    def test_uncertain_is_never_a_verified_outcome(self):
        base = {"false_verified": False, "package_unlabeled": False}
        self.assertEqual(pilot.classify(dict(base, status="VERIFIED")), "CORRECT_VERIFIED")
        self.assertEqual(pilot.classify(dict(base, status="VERIFIED", false_verified=True)), "FALSE_VERIFIED")
        self.assertEqual(pilot.classify(dict(base, status="VERIFIED", package_unlabeled=True)),
                         "VERIFIED_PACKAGE_UNSCORED")
        self.assertEqual(pilot.classify(dict(base, status="UNCERTAIN")), "UNCERTAIN")
        self.assertEqual(pilot.classify(dict(base, status="NO_MATCH")), "NO_MATCH")

    def test_summary_counts_outcomes_separately(self):
        rows = [{"input": "full", "outcome": o, "truth_rank": r}
                for o, r in [("CORRECT_VERIFIED", 1), ("UNCERTAIN", 2), ("UNCERTAIN", None), ("FALSE_VERIFIED", None)]]
        summary = pilot.summarize(rows)["full"]
        self.assertEqual((summary["n"], summary["CORRECT_VERIFIED"], summary["UNCERTAIN"],
                          summary["FALSE_VERIFIED"], summary["truth_in_candidates"]), (4, 1, 2, 1, 2))


if __name__ == "__main__":
    unittest.main()
