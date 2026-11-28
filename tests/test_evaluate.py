import datetime
import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from vision.products import evaluate
from vision.products.catalog import ProductCatalog
from vision.products.evaluate import PREPROCESSORS, blur_score, oracle_result, score_match, score_ocr
from vision.products.matcher import ProductMatcher

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")

ADVIL_LABEL = {
    "image": "unused.jpg",
    "product": "Advil 200 mg tablets",
    "brand_words": ["Advil"],
    "ingredients": [["IBUPROFEN", "200 mg"]],
    "formulation": [["IBUPROFEN", "200 mg/1"]],
    "package_ndc": "0573-0150-20",
    "visible_text": ["Advil", "Ibuprofen Tablets 200 mg", "NDC 0573-0150-20"],
}


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.matcher = ProductMatcher(ProductCatalog.from_openfda_results(json.load(f)["results"]))

    def test_ocr_scores_separate_plain_and_corrected_strengths(self):
        scores = score_ocr(ADVIL_LABEL, [{"text": "ADVIL", "confidence": 0.9},
                                         {"text": "Ibuprofen 2OOmg", "confidence": 0.9}], self.matcher)
        self.assertEqual(scores, {"brand": True, "ingredients": True,
                                  "strengths_plain": False, "strengths_with_correction": True})

    def test_correct_package_is_not_false_verified(self):
        result = oracle_result(ADVIL_LABEL, self.matcher)
        scores = score_match(ADVIL_LABEL, result)
        self.assertEqual((scores["status"], scores["level"]), ("VERIFIED", "PACKAGE"))
        self.assertFalse(scores["false_verified"])
        self.assertEqual(scores["truth_rank"], 1)

    def test_wrong_label_counts_as_false_verified(self):
        result = oracle_result(ADVIL_LABEL, self.matcher)
        wrong = dict(ADVIL_LABEL, formulation=[["IBUPROFEN", "200 mg/1"], ["DIPHENHYDRAMINE CITRATE", "38 mg/1"]])
        self.assertTrue(score_match(wrong, result)["false_verified"])
        self.assertIsNone(score_match(wrong, result)["truth_rank"])
        other_package = dict(ADVIL_LABEL, package_ndc="0573-0161-35")
        self.assertTrue(score_match(other_package, result)["false_verified"])

    def test_preprocessors_and_blur(self):
        image = (np.random.default_rng(0).random((40, 60, 3)) * 255).astype(np.uint8)
        for name, process in PREPROCESSORS.items():
            self.assertIsNotNone(process(image), name)
        self.assertEqual(PREPROCESSORS["upscale2x"](image).shape[:2], (80, 120))
        self.assertGreater(blur_score(image), blur_score(np.full((40, 60, 3), 128, np.uint8)))


def reviewed(label, status="REVIEWED", reviewers=("RJ", "TM"), **truth):
    """A copy of label with a review block (as reconcile writes it) and truth fields."""
    out = dict(label, **truth)
    out.setdefault("physical_product_id", "P1")
    out.setdefault("negative", False)
    out["review"] = {"status": status, "reviewers": list(reviewers), "date": "2026-11-26"}
    return out


MEDICINE_BOX, WATER_BOX = [10, 10, 40, 90], [60, 5, 95, 95]
PHOTO = dict(ADVIL_LABEL, id="p1-front", group="photo",
             target_boxes=[{"role": "medicine", "box": MEDICINE_BOX}, {"role": "other_bottle", "box": WATER_BOX}])
FULL_OCR = [{"text": "Advil", "confidence": 0.95}, {"text": "Ibuprofen Tablets 200 mg", "confidence": 0.95},
            {"text": "NDC 0573-0150-20", "confidence": 0.95}]


def detection(box, confidence, class_name="bottle"):
    return {"class_id": 39, "class_name": class_name, "confidence": confidence, "box": tuple(box)}


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return image

    def get_detections(self, result):
        return list(self.detections)


class FakeRecognizer:
    """Whole-image OCR reads the full label; any crop reads only the ingredient line."""
    def __init__(self, width):
        self.width, self.calls = width, []

    def read(self, image):
        self.calls.append(image.shape[:2])
        return FULL_OCR if image.shape[1] == self.width else FULL_OCR[1:2]


class ReviewGatingTests(unittest.TestCase):
    def test_label_validation(self):
        self.assertEqual(evaluate.validate_label(ADVIL_LABEL), [])                     # old labels stay valid
        self.assertEqual(evaluate.review_status(ADVIL_LABEL), "UNREVIEWED")
        self.assertEqual(evaluate.validate_label(reviewed(ADVIL_LABEL)), [])
        bad = {
            "one reviewer called REVIEWED": reviewed(ADVIL_LABEL, reviewers=("RJ",)),
            "same person twice": reviewed(ADVIL_LABEL, reviewers=("RJ", "rj")),
            "software as reviewer": reviewed(ADVIL_LABEL, status="SINGLE_REVIEWED", reviewers=("assistant",)),
            "AI reviewer among people": reviewed(ADVIL_LABEL, reviewers=("RJ", "Claude")),
            "UNREVIEWED naming reviewers": dict(ADVIL_LABEL, review={"status": "UNREVIEWED", "reviewers": ["RJ"]}),
            "unknown level": dict(ADVIL_LABEL, review={"status": "VERIFIED"}),
            "medicine without formulation": reviewed(ADVIL_LABEL, formulation=[]),
            "no product id": reviewed(ADVIL_LABEL, physical_product_id=None),
            "negative unknown": reviewed(ADVIL_LABEL, negative=None),
            "bad box": dict(ADVIL_LABEL, target_boxes=[{"role": "medicine", "box": [5, 5, 2, 9]}]),
            "bad role": dict(ADVIL_LABEL, target_boxes=[{"role": "cup", "box": [1, 1, 2, 2]}]),
        }
        for name, label in bad.items():
            self.assertTrue(evaluate.validate_label(label), name)

    def test_repository_label_files_load_and_are_all_unreviewed(self):
        for path in ("evaluation/labels.json", "research/datasets/dailymed-pilot/labels.json",
                     "research/datasets/photos-v1/labels.json"):
            labels = evaluate.load_labels(path)
            self.assertTrue(labels, path)
            self.assertEqual({evaluate.review_status(l) for l in labels}, {"UNREVIEWED"}, path)

    def test_prepared_photo_labels_contain_no_truth(self):
        for label in evaluate.load_labels("research/datasets/photos-v1/labels.json"):
            self.assertEqual(label["group"], "photo")
            self.assertEqual(len(label["sha256"]), 64)
            for field in ("physical_product_id", "negative", "readable", "product", "package_ndc", "upc", "target_boxes"):
                self.assertIsNone(label[field], (label["id"], field))
            for field in ("brand_words", "ingredients", "formulation", "visible_text"):
                self.assertEqual(label[field], [], (label["id"], field))

    def test_load_labels_reports_every_problem(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "labels.json")
            with open(path, "w") as f:
                json.dump([dict(ADVIL_LABEL, id="a"), dict(ADVIL_LABEL, id="a", review={"status": "REVIEWED"})], f)
            with self.assertRaises(ValueError) as caught:
                evaluate.load_labels(path)
            self.assertIn("duplicate id", str(caught.exception))
            self.assertIn("two different reviewers", str(caught.exception))


class OutcomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.matcher = ProductMatcher(ProductCatalog.from_openfda_results(json.load(f)["results"]))

    def test_unreviewed_rows_are_descriptive_only(self):
        result = oracle_result(ADVIL_LABEL, self.matcher)
        row = evaluate.make_row(ADVIL_LABEL, "full", "none", result, self.matcher, FULL_OCR)
        self.assertEqual((row["tier"], row["status"]), ("descriptive", "VERIFIED"))
        for field in ("outcome", "truth_rank", "false_verified", "ocr_scores"):
            self.assertNotIn(field, row)
        self.assertEqual(row["ocr"][0], ["Advil", 0.95])                          # raw OCR is kept

    def test_reviewed_rows_are_scored(self):
        result = oracle_result(ADVIL_LABEL, self.matcher)
        row = evaluate.make_row(reviewed(ADVIL_LABEL), "full", "none", result, self.matcher, FULL_OCR)
        self.assertEqual((row["tier"], row["outcome"], row["truth_rank"]), ("primary", "CORRECT_VERIFIED", 1))
        self.assertTrue(row["ocr_scores"]["brand"])
        single = evaluate.make_row(reviewed(ADVIL_LABEL, "SINGLE_REVIEWED", ("RJ",)), "full", "none", result, self.matcher)
        self.assertEqual(single["tier"], "exploratory")

    def test_any_verified_on_a_negative_is_false(self):
        negative = reviewed(dict(ADVIL_LABEL, formulation=[], ingredients=[]), negative=True)
        result = oracle_result(ADVIL_LABEL, self.matcher)
        row = evaluate.make_row(negative, "full", "none", result, self.matcher, FULL_OCR)
        self.assertEqual(row["outcome"], "FALSE_VERIFIED")
        self.assertTrue(row["false_verified"])
        self.assertNotIn("ocr_scores", row)
        no_match = self.matcher.match([{"text": "UIC Construction", "confidence": 0.9}])
        self.assertEqual(evaluate.make_row(negative, "full", "none", no_match, self.matcher)["outcome"], "NO_MATCH")
        self.assertEqual(evaluate.classify({"status": "VERIFIED", "false_verified": False, "package_unlabeled": False},
                                           negative=True), "FALSE_VERIFIED")


class DetectionTests(unittest.TestCase):
    def test_iou_and_one_to_one_matching(self):
        self.assertEqual(evaluate.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(evaluate.iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        self.assertEqual(evaluate.iou([0, 0, 1, 1], [5, 5, 6, 6]), 0.0)
        # one prediction can't satisfy two annotated boxes
        self.assertEqual(len(evaluate.match_boxes([[0, 0, 10, 10], [0, 0, 10, 11]], [[0, 0, 10, 10]])), 1)
        self.assertEqual(evaluate.match_boxes([[0, 0, 10, 10]], [[5, 0, 15, 10]]), [])     # IoU 0.33 < 0.5

    def test_highest_confidence_bottle_that_is_not_the_medicine(self):
        label = reviewed(PHOTO)
        detections = [detection(WATER_BOX, 0.75), detection([11, 12, 41, 88], 0.70), detection([0, 0, 5, 5], 0.9, "cup")]
        score = evaluate.score_detection(label, detections)
        self.assertEqual((score["bottles_found"], score["bottle_targets"]), (2, 2))
        self.assertEqual((score["medicine_found"], score["medicine_targets"]), (1, 1))
        self.assertTrue(score["multiple_bottles"])
        self.assertEqual(score["selection"], "WRONG_OBJECT_SELECTED")
        detections[1] = detection([11, 12, 41, 88], 0.80)
        self.assertEqual(evaluate.score_detection(label, detections)["selection"], "MEDICINE_SELECTED")

    def test_missed_medicine_and_unannotated_labels(self):
        label = reviewed(PHOTO)
        score = evaluate.score_detection(label, [detection(WATER_BOX, 0.7)])
        self.assertEqual((score["medicine_found"], score["bottles_found"]), (0, 1))
        self.assertEqual(score["selection"], "WRONG_OBJECT_SELECTED")
        self.assertEqual(evaluate.score_detection(label, [])["selection"], "NO_BOTTLE_DETECTED")
        self.assertIsNone(evaluate.score_detection(ADVIL_LABEL, [detection(WATER_BOX, 0.7)]))
        negative = reviewed(dict(PHOTO, target_boxes=[{"role": "other_bottle", "box": WATER_BOX}]), negative=True)
        self.assertEqual(evaluate.target_selection(negative, [detection(WATER_BOX, 0.7)]), "NON_MEDICINE_SELECTED")
        self.assertEqual(evaluate.target_selection(negative, []), "NOTHING_SELECTED")

    def test_truth_crop_box(self):
        self.assertEqual(evaluate.truth_crop_box(PHOTO), MEDICINE_BOX)
        self.assertEqual(evaluate.truth_crop_box(dict(ADVIL_LABEL, roi=[1, 2, 3, 4])), [1, 2, 3, 4])
        two = dict(PHOTO, target_boxes=[{"role": "medicine", "box": MEDICINE_BOX}] * 2)
        self.assertIsNone(evaluate.truth_crop_box(two))
        self.assertIsNone(evaluate.truth_crop_box(ADVIL_LABEL))


class RunAndReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.matcher = ProductMatcher(ProductCatalog.from_openfda_results(json.load(f)["results"]))
        cls.image = np.full((100, 100, 3), 200, np.uint8)

    def run_label(self, label, detections, inputs=("full", "truth_crop", "yolo_crop")):
        recognizer = FakeRecognizer(width=100)
        rows, record = evaluate.evaluate_label(label, self.image, recognizer, self.matcher,
                                               FakeDetector(detections), inputs)
        return rows, record, recognizer

    def test_three_inputs_and_oracle(self):
        rows, record, recognizer = self.run_label(reviewed(PHOTO), [detection(WATER_BOX, 0.75), detection(MEDICINE_BOX, 0.7)])
        by_input = {row["input"]: row for row in rows}
        self.assertEqual(set(by_input), {"oracle", "full", "truth_crop", "yolo_crop"})
        self.assertEqual(by_input["oracle"]["outcome"], "CORRECT_VERIFIED")
        self.assertEqual(by_input["full"]["outcome"], "CORRECT_VERIFIED")
        self.assertEqual(by_input["truth_crop"]["crop_box"], MEDICINE_BOX)
        self.assertEqual(by_input["truth_crop"]["region_size"], "30x80")
        self.assertEqual(by_input["yolo_crop"]["crop_box"], WATER_BOX)              # what main.py would read
        self.assertEqual(by_input["truth_crop"]["outcome"], "UNCERTAIN")
        self.assertEqual(record["detection"]["selection"], "WRONG_OBJECT_SELECTED")
        self.assertEqual(len(record["detections"]), 2)
        self.assertEqual(len(recognizer.calls), 3)                                  # oracle needs no OCR
        for row in rows:
            if row["input"] != "oracle":
                self.assertIsNotNone(row["ocr"])
                self.assertIn("ocr_ms", row)

    def test_no_bottle_is_no_region_not_skipped(self):
        rows, record, _ = self.run_label(reviewed(PHOTO), [], inputs=("yolo_crop",))
        yolo = [r for r in rows if r["input"] == "yolo_crop"]
        self.assertEqual([(r["status"], r["outcome"], r["ocr"]) for r in yolo], [("NO_REGION", "NO_REGION", None)])
        self.assertEqual(record["detection"]["selection"], "NO_BOTTLE_DETECTED")

    def test_truth_crop_is_skipped_without_a_box(self):
        rows, record, _ = self.run_label(reviewed(ADVIL_LABEL), [], inputs=("truth_crop",))
        self.assertEqual([r["input"] for r in rows], ["oracle"])
        self.assertEqual(record["skipped"][0]["input"], "truth_crop")
        self.assertNotIn("detections", record)                                       # no boxes, no yolo_crop: no YOLO

    def test_unreviewed_detections_are_recorded_not_scored(self):
        rows, record, _ = self.run_label(PHOTO, [detection(MEDICINE_BOX, 0.7)])
        self.assertIsNone(record["detection"])
        self.assertTrue(all("outcome" not in r for r in rows))

    def test_replay_rescores_saved_ocr_after_review(self):
        rows, record, _ = self.run_label(PHOTO, [detection(WATER_BOX, 0.75), detection(MEDICINE_BOX, 0.7)])
        saved = json.loads(json.dumps({"rows": rows, "images": [record]}, default=str))
        new_rows, images = evaluate.replay(saved, [reviewed(PHOTO)], self.matcher)
        self.assertEqual([r["input"] for r in new_rows], [r["input"] for r in rows])
        full = next(r for r in new_rows if r["input"] == "full")
        self.assertEqual((full["outcome"], full["previous"]["review_status"]), ("CORRECT_VERIFIED", "UNREVIEWED"))
        self.assertEqual(full["ocr"], next(r for r in rows if r["input"] == "full")["ocr"])
        self.assertEqual(images[0]["detection"]["selection"], "WRONG_OBJECT_SELECTED")   # from saved detections
        crop = next(r for r in new_rows if r["input"] == "truth_crop")
        self.assertNotIn("stale", crop)
        self.assertEqual(crop["crop_box"], MEDICINE_BOX)
        with self.assertRaises(ValueError):
            evaluate.replay(saved, [], self.matcher)

    def saved_run(self, label):
        rows, record, _ = self.run_label(label, [detection(WATER_BOX, 0.75), detection(MEDICINE_BOX, 0.7)])
        record["sha256"] = "a" * 64
        return json.loads(json.dumps({"rows": rows, "images": [record]}, default=str))

    def test_replay_never_reuses_ocr_from_a_different_crop(self):
        saved = self.saved_run(reviewed(PHOTO))
        moved = reviewed(dict(PHOTO, target_boxes=[{"role": "medicine", "box": [12, 10, 40, 90]},
                                                   {"role": "other_bottle", "box": WATER_BOX}]))
        rows, images = evaluate.replay(saved, [moved], self.matcher)
        crop = next(r for r in rows if r["input"] == "truth_crop")
        self.assertEqual(crop["status"], "STALE")
        self.assertIn("crop changed", crop["stale"])
        self.assertEqual((crop["crop_box"], crop["current_crop_box"]), (MEDICINE_BOX, [12, 10, 40, 90]))
        for field in ("ocr", "outcome", "truth_rank", "ocr_scores", "formulations"):
            self.assertFalse(crop.get(field), field)                       # nothing reused or scored
        # full-frame and YOLO-crop OCR depend only on the file, so they are still re-scored
        self.assertEqual(next(r for r in rows if r["input"] == "full")["outcome"], "CORRECT_VERIFIED")
        section = evaluate.summarize(rows, images)["photo"]["REVIEWED"]
        self.assertEqual(section["stale_rows_excluded"], 1)
        self.assertNotIn("truth_crop/none", section)                        # excluded from every metric

    def test_crop_removed_or_added_after_the_saved_run(self):
        saved = self.saved_run(reviewed(PHOTO))
        no_box = reviewed(dict(PHOTO, target_boxes=[{"role": "other_bottle", "box": WATER_BOX}]))
        rows, _ = evaluate.replay(saved, [no_box], self.matcher)
        self.assertEqual(next(r for r in rows if r["input"] == "truth_crop")["status"], "STALE")
        # a run made before any box existed has no truth_crop OCR; replay asks for a new run
        unboxed = self.saved_run(dict(PHOTO, target_boxes=None))
        rows, images = evaluate.replay(unboxed, [reviewed(PHOTO)], self.matcher)
        self.assertNotIn("truth_crop", [r["input"] for r in rows])
        self.assertEqual(images[0]["needs_ocr"], ["truth_crop"])

    def test_replay_never_reuses_ocr_or_detections_from_a_different_file(self):
        saved = self.saved_run(reviewed(PHOTO))
        same_file = reviewed(dict(PHOTO, sha256="a" * 64))
        rows, images = evaluate.replay(saved, [same_file], self.matcher)
        self.assertFalse(any(r.get("stale") for r in rows))
        self.assertIsNotNone(images[0]["detection"])
        other_file = reviewed(dict(PHOTO, sha256="b" * 64))
        rows, images = evaluate.replay(saved, [other_file], self.matcher)
        self.assertEqual({r["input"] for r in rows if r.get("stale")}, {"full", "truth_crop", "yolo_crop"})
        self.assertTrue(all(r["ocr"] is None for r in rows if r.get("stale")))
        self.assertEqual(next(r for r in rows if r["input"] == "oracle")["outcome"], "CORRECT_VERIFIED")
        self.assertIsNone(images[0]["detection"])                          # detections were on other pixels
        report = evaluate.summarize(rows, images)["photo"]["REVIEWED"]
        self.assertEqual(report["stale_rows_excluded"], 3)
        self.assertIsNone(report["detection"])


def row(product, outcome, status="REVIEWED", negative=False, truth_rank=1, image=None, group="photo", input_name="full"):
    return {"id": image or f"{product}-{outcome}-{random_id()}", "group": group, "review_status": status,
            "tier": evaluate.TIERS[status], "physical_product_id": product, "negative": negative,
            "input": input_name, "variant": "none", "status": "VERIFIED" if "VERIFIED" in outcome else outcome,
            "formulations": 1, "outcome": outcome, "truth_rank": truth_rank}


_counter = [0]


def random_id():
    _counter[0] += 1
    return str(_counter[0])


class AggregationTests(unittest.TestCase):
    def test_views_of_one_product_are_one_product(self):
        rows = [row("P1", "CORRECT_VERIFIED"), row("P1", "UNCERTAIN"), row("P1", "UNCERTAIN"), row("P2", "NO_MATCH", truth_rank=None)]
        m = evaluate.summarize_matcher(rows)["medicine"]
        self.assertEqual(m["verification_coverage"]["images"], [1, 4])
        self.assertEqual(m["verification_coverage"]["products"], [1, 2])
        self.assertEqual(m["candidate_recall"]["products"], [1, 2])
        self.assertEqual(m["outcomes_products"]["CORRECT_VERIFIED"], 1)
        self.assertEqual(m["outcomes_products"]["NO_MATCH"], 1)
        self.assertIsNone(m["verification_coverage"]["product_ci95"])              # 2 products < minimum
        self.assertIsNone(m["verification_coverage"]["image_ci95"])

    def test_a_false_verified_view_makes_the_product_false_verified(self):
        rows = [row("P1", "CORRECT_VERIFIED"), row("P1", "FALSE_VERIFIED")]
        m = evaluate.summarize_matcher(rows)["medicine"]
        self.assertEqual(m["outcomes_products"]["FALSE_VERIFIED"], 1)
        self.assertEqual(m["outcomes_products"]["CORRECT_VERIFIED"], 0)
        self.assertEqual(m["verification_precision_images"]["images"], [1, 2])
        self.assertEqual(m["verification_precision_products"], [0, 1])

    def test_intervals_over_products(self):
        rows = [row(f"P{i}", "CORRECT_VERIFIED" if i < 4 else "UNCERTAIN") for i in range(6) for _ in range(3)]
        coverage = evaluate.summarize_matcher(rows, seed=1)["medicine"]["verification_coverage"]
        self.assertEqual(coverage["products"], [4, 6])
        self.assertEqual(coverage["product_ci95"], evaluate.wilson(4, 6))          # n = products, not 18 images
        self.assertEqual(coverage["image_ci95"], evaluate.summarize_matcher(rows, seed=1)["medicine"]["verification_coverage"]["image_ci95"])
        low, high = coverage["image_ci95"]
        self.assertLessEqual(low, 12 / 18)
        self.assertGreaterEqual(high, 12 / 18)

    def test_negatives_and_descriptive_rows(self):
        rows = [row("N1", "NO_MATCH", negative=True, truth_rank=None), row("N2", "FALSE_VERIFIED", negative=True, truth_rank=None)]
        n = evaluate.summarize_matcher(rows)["non_medicine"]
        self.assertEqual((n["correct_rejection"]["images"], n["false_verified"]["images"]), ([1, 2], [1, 2]))
        descriptive = [dict(row("P1", "UNCERTAIN", status="UNREVIEWED"), formulations=3)]
        for r in descriptive:
            del r["outcome"], r["truth_rank"]
        s = evaluate.summarize_matcher(descriptive)
        self.assertNotIn("medicine", s)
        self.assertEqual(s["status_counts"]["UNCERTAIN"], 1)

    def test_levels_and_groups_are_never_pooled(self):
        rows = [row("P1", "CORRECT_VERIFIED"), row("P2", "FALSE_VERIFIED", status="SINGLE_REVIEWED"),
                row("A1", "CORRECT_VERIFIED", group="artwork")]
        report = evaluate.summarize(rows, [])
        self.assertEqual(set(report), {"photo", "artwork"})
        self.assertEqual(set(report["photo"]), {"REVIEWED", "SINGLE_REVIEWED"})
        self.assertEqual(report["photo"]["REVIEWED"]["full/none"]["medicine"]["false_verified"]["images"], [0, 1])
        self.assertEqual(report["photo"]["SINGLE_REVIEWED"]["tier"], "exploratory")

    def test_detection_summary_counts_boxes_and_selection(self):
        images = [{"id": f"i{k}", "group": "photo", "review_status": "REVIEWED", "physical_product_id": "P1",
                   "detection": {"bottle_targets": 2, "bottles_found": found, "medicine_targets": 1,
                                 "medicine_found": int(found == 2), "bottle_detections": found,
                                 "unmatched_bottle_detections": 0, "multiple_bottles": True, "selection": selection}}
                  for k, (found, selection) in enumerate([(2, "MEDICINE_SELECTED"), (1, "WRONG_OBJECT_SELECTED")])]
        d = evaluate.summarize_detection(images)
        self.assertEqual(d["bottle_recall"]["images"], [3, 4])
        self.assertEqual(d["medicine_recall"]["images"], [1, 2])
        self.assertEqual(d["medicine_recall"]["products"], [1, 1])
        self.assertEqual(d["target_selection_multiple_bottles"]["images"], [1, 2])

    def test_wilson(self):
        self.assertIsNone(evaluate.wilson(0, 0))
        low, high = evaluate.wilson(5, 10)
        self.assertAlmostEqual(low, 0.2366, places=3)
        self.assertAlmostEqual(high, 0.7634, places=3)


class ReproducibilityTests(unittest.TestCase):
    def test_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            when = datetime.datetime(2026, 11, 26, 12, 0, 0)
            path = evaluate.save_results({"rows": []}, folder, now=when)
            self.assertTrue(path.endswith("evaluation-20261126-120000.json"))
            with self.assertRaises(FileExistsError):
                evaluate.save_results({"rows": []}, folder, now=when)

    def test_image_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "a.png")
            cv2.imwrite(path, np.zeros((4, 4, 3), np.uint8))
            image, digest = evaluate.read_image({"image": path})
            self.assertEqual(image.shape, (4, 4, 3))
            self.assertEqual(evaluate.read_image({"image": path, "sha256": digest})[1], digest)
            with self.assertRaises(ValueError):
                evaluate.read_image({"image": path, "sha256": "0" * 64})
            self.assertEqual(evaluate.read_image({"image": os.path.join(folder, "missing.png")}), (None, None))

    def test_environment_records_code_data_and_versions(self):
        with tempfile.TemporaryDirectory() as folder:
            labels, cache = os.path.join(folder, "labels.json"), os.path.join(folder, "cache.json")
            for path in (labels, cache):
                with open(path, "w") as f:
                    f.write("[]")
            env = evaluate.environment(labels, cache)
        self.assertEqual(env["labels"]["sha256"], env["catalog"]["sha256"])
        self.assertEqual(len(env["labels"]["sha256"]), 64)
        self.assertIn("commit", env["git"])
        self.assertIn("easyocr", env["versions"])
        self.assertIsNone(env["detector"])


if __name__ == "__main__":
    unittest.main()
