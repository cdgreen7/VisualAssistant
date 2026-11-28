"""Human annotation review: independent reviewers, agreement rules and review levels.
Temporary files only; the real dataset is never written, and no window is opened."""
import contextlib
import io
import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from research import annotation_review as review
from vision.products import evaluate

TODAY = __import__("datetime").date(2026, 11, 26)
MEDICINE = {"negative": False, "physical_product_id": "P1", "checked_package": True, "readable": True,
            "product": "Ibuprofen Tablets", "brand_words": "equate", "ingredients": ["Ibuprofen | 200 mg"],
            "package_ndc": "", "upc": "6 81131 69937 2", "visible_text": ["equate", "Ibuprofen", "Tablets USP, 200 mg"]}
BOXES = [{"role": "medicine", "box": [100, 200, 300, 600]}, {"role": "other_bottle", "box": [400, 50, 600, 650]}]


def prepared(image_id, path, sha):
    return {"id": image_id, "image": path, "sha256": sha, "group": "photo", "prepared_by": "assistant: metadata only",
            "review": {"status": "UNREVIEWED"}, "physical_product_id": None, "negative": None, "readable": None,
            "product": None, "brand_words": [], "ingredients": [], "formulation": [], "package_ndc": None,
            "upc": None, "visible_text": [], "target_boxes": None}


class Dataset:
    """A temporary labels.json with two small images."""
    def __init__(self):
        self.folder = tempfile.TemporaryDirectory()
        self.labels_path = os.path.join(self.folder.name, "labels.json")
        labels = []
        for name in ("front", "side"):
            path = os.path.join(self.folder.name, f"{name}.png")
            cv2.imwrite(path, np.full((700, 700, 3), 128 if name == "front" else 90, np.uint8))
            labels.append(prepared(name, path, evaluate.file_sha256(path)))
        self.write(labels)

    def write(self, labels):
        with open(self.labels_path, "w") as f:
            json.dump(labels, f)

    def labels(self):
        return evaluate.load_labels(self.labels_path)

    def sha(self, image_id):
        return next(l for l in self.labels() if l["id"] == image_id)["sha256"]

    def add(self, reviewer, image_id, answers=MEDICINE, boxes=BOXES, **kw):
        annotation = review.build_annotation(answers, boxes, self.sha(image_id), today=TODAY)
        return review.save_annotation(self.labels_path, reviewer, image_id, annotation, **kw)

    def reconcile(self):
        return review.reconcile(self.labels(), review.load_reviews(self.labels_path))

    def close(self):
        self.folder.cleanup()


class AnnotationTests(unittest.TestCase):
    def test_reviewers_must_be_people(self):
        self.assertEqual(review.reviewer_slug("R. J."), "r-j")
        for name in ("assistant", "Claude", "AI", "", "  ", "qwen"):
            with self.assertRaises(ValueError):
                review.reviewer_slug(name)

    def test_ingredient_parsing(self):
        self.assertEqual(review.parse_ingredient("Acetaminophen |  160 mg/5 mL"), ["Acetaminophen", "160 mg/5 mL"])
        for bad in ("Ibuprofen 200 mg", "| 200 mg", "Ibuprofen | many"):
            with self.assertRaises(ValueError):
                review.parse_ingredient(bad)

    def test_annotation_rules(self):
        a = review.build_annotation(MEDICINE, BOXES, "x" * 64, today=TODAY)
        self.assertEqual((a["upc"], a["confirmation_source"], a["date"]), ("681131699372", "physical package", "2026-11-26"))
        self.assertEqual(a["brand_words"], ["equate"])
        problems = {
            "no ingredient": (dict(MEDICINE, ingredients=[]), BOXES),
            "negative with ingredient": (dict(MEDICINE, negative=True), [BOXES[1]]),
            "no medicine box": (MEDICINE, [BOXES[1]]),
            "two medicine boxes": (MEDICINE, [BOXES[0], BOXES[0]]),
            "medicine box on a negative": (dict(MEDICINE, negative=True, ingredients=[]), BOXES),
            "bad NDC": (dict(MEDICINE, package_ndc="12-34"), BOXES),
            "no product id": (dict(MEDICINE, physical_product_id=" "), BOXES),
        }
        for name, (answers, boxes) in problems.items():
            with self.assertRaises(ValueError, msg=name):
                review.build_annotation(answers, boxes, "x" * 64)
        # artwork without boxes
        self.assertIsNone(review.build_annotation(MEDICINE, None, "x" * 64)["target_boxes"])

    def test_saving_is_per_reviewer_and_never_silent_overwrite(self):
        data = Dataset()
        try:
            path = data.add("RJ", "front")
            self.assertTrue(path.endswith(os.path.join("reviews", "rj.json")))
            with self.assertRaises(ValueError):
                data.add("RJ", "front")
            data.add("RJ", "front", answers=dict(MEDICINE, readable=False), replace=True)
            self.assertFalse(review.load_reviews(data.labels_path)["RJ"]["front"]["readable"])
            with open(os.path.join(review.reviews_dir(data.labels_path), "assistant.json"), "w") as f:
                json.dump({"reviewer": "assistant", "annotations": {"front": {}}}, f)
            self.assertEqual(set(review.load_reviews(data.labels_path)), {"RJ"})       # software file skipped
        finally:
            data.close()


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        self.data = Dataset()

    def tearDown(self):
        self.data.close()

    def test_no_annotation_leaves_labels_unchanged(self):
        labels, report = self.data.reconcile()
        self.assertEqual(labels, self.data.labels())
        self.assertEqual(report, [])

    def test_one_reviewer_is_single_reviewed(self):
        self.data.add("RJ", "front")
        labels, _ = self.data.reconcile()
        front = labels[0]
        self.assertEqual(front["review"]["status"], "SINGLE_REVIEWED")
        self.assertEqual(front["review"]["reviewers"], ["RJ"])
        self.assertEqual(front["formulation"], [["Ibuprofen", "200 mg"]])
        self.assertEqual(evaluate.formulation_key(front["formulation"]), evaluate.formulation_key([["IBUPROFEN", "200 mg/1"]]))
        self.assertEqual(evaluate.validate_label(front), [])
        self.assertEqual(labels[1]["review"]["status"], "UNREVIEWED")
        self.assertEqual(front["prepared_by"], "assistant: metadata only")              # metadata kept

    def test_two_agreeing_reviewers_are_reviewed_with_mean_boxes(self):
        self.data.add("RJ", "front")
        shifted = [{"role": "medicine", "box": [104, 196, 306, 604]}, {"role": "other_bottle", "box": [398, 52, 600, 650]}]
        other = dict(MEDICINE, physical_product_id="bottle-A", brand_words="EQUATE",
                     ingredients=["IBUPROFEN | 200mg"], visible_text=["Tablets USP 200 mg", "EQUATE", "ibuprofen"])
        self.data.add("TM", "front", answers=other, boxes=shifted)
        labels, report = self.data.reconcile()
        front = labels[0]
        self.assertEqual(report, [])
        self.assertEqual((front["review"]["status"], front["review"]["reviewers"]), ("REVIEWED", ["RJ", "TM"]))
        self.assertEqual(front["target_boxes"][0], {"role": "medicine", "box": [102, 198, 303, 602]})
        self.assertEqual(evaluate.tier(front), "primary")
        self.data.write(labels)
        self.assertEqual(evaluate.review_status(self.data.labels()[0]), "REVIEWED")      # loads and validates

    def test_disagreement_is_unreviewed_and_names_the_fields(self):
        self.data.add("RJ", "front")
        self.data.add("TM", "front", answers=dict(MEDICINE, ingredients=["Ibuprofen | 400 mg"]),
                      boxes=[{"role": "medicine", "box": [100, 200, 200, 600]}, BOXES[1]])
        labels, report = self.data.reconcile()
        review_block = labels[0]["review"]
        self.assertEqual(review_block["status"], "UNREVIEWED")
        self.assertEqual(review_block["conflicts"], ["formulation", "target_boxes"])
        self.assertEqual(review_block["conflicting_reviewers"], ["RJ", "TM"])
        self.assertEqual(labels[0]["formulation"], [])                                    # no truth on a conflict
        self.assertIn("disagree", report[0])
        self.assertEqual(evaluate.validate_label(labels[0]), [])

    def test_a_later_conflict_clears_earlier_reviewed_truth(self):
        self.data.add("RJ", "front")
        labels, _ = self.data.reconcile()
        self.data.write(labels)
        self.data.add("TM", "front", answers=dict(MEDICINE, negative=True, ingredients=[]), boxes=[BOXES[1]])
        labels, _ = self.data.reconcile()
        self.assertEqual(labels[0]["review"]["status"], "UNREVIEWED")
        self.assertIsNone(labels[0]["negative"])
        self.assertEqual(labels[0]["formulation"], [])

    def test_unreviewed_prepared_fields_are_kept(self):
        labels = self.data.labels()
        labels[1].update(formulation=[["IBUPROFEN", "200 mg/1"]], ingredients=[["IBUPROFEN", "200 mg"]])
        self.data.write(labels)
        out, _ = self.data.reconcile()
        self.assertEqual(out[1]["formulation"], [["IBUPROFEN", "200 mg/1"]])            # e.g. the DailyMed transcription
        self.assertEqual(evaluate.tier(out[1]), "descriptive")

    def test_annotation_of_a_changed_file_is_ignored(self):
        self.data.add("RJ", "front")
        labels = self.data.labels()
        labels[0]["sha256"] = "f" * 64
        self.data.write(labels)
        out, report = self.data.reconcile()
        self.assertEqual(out[0]["review"]["status"], "UNREVIEWED")
        self.assertIn("different file", report[0])

    def test_product_grouping_must_match_but_ids_may_differ(self):
        for reviewer, ids in (("RJ", ("P1", "P1")), ("TM", ("A", "A"))):
            for image_id, pid in zip(("front", "side"), ids):
                self.data.add(reviewer, image_id, answers=dict(MEDICINE, physical_product_id=pid))
        labels, _ = self.data.reconcile()
        self.assertEqual([l["review"]["status"] for l in labels], ["REVIEWED", "REVIEWED"])
        self.assertEqual({l["physical_product_id"] for l in labels}, {"P1"})
        self.data.add("TM", "side", answers=dict(MEDICINE, physical_product_id="B"), replace=True)
        labels, _ = self.data.reconcile()
        self.assertTrue(all("physical_product_id grouping" in l["review"]["conflicts"] for l in labels))


class IntendedTargetReviewTests(unittest.TestCase):
    """Intent is recorded separately from the medicine box and compared by box geometry."""
    def setUp(self):
        self.data = Dataset()

    def tearDown(self):
        self.data.close()

    def test_intent_rules(self):
        self.assertIsNone(review.build_annotation(MEDICINE, BOXES, "x" * 64)["intended_target"])   # not known
        self.assertEqual(review.build_annotation(dict(MEDICINE, intended_target=1), BOXES, "x" * 64)["intended_target"], 1)
        self.assertEqual(review.build_annotation(dict(MEDICINE, intended_target="none"), BOXES, "x" * 64)
                         ["intended_target"], "none")
        for bad in (2, -1, True, "medicine"):
            with self.assertRaises(ValueError):
                review.build_annotation(dict(MEDICINE, intended_target=bad), BOXES, "x" * 64)
        with self.assertRaises(ValueError):                                      # artwork has no boxes to point at
            review.build_annotation(dict(MEDICINE, intended_target=0), None, "x" * 64)

    def test_same_intended_box_in_a_different_order_agrees(self):
        self.data.add("RJ", "front", answers=dict(MEDICINE, intended_target=1))    # the other bottle
        # TM drew the boxes in the opposite order, so the same physical box is number 0 for TM
        swapped = [{"role": "other_bottle", "box": [401, 51, 600, 650]}, {"role": "medicine", "box": [102, 198, 302, 602]}]
        self.data.add("TM", "front", answers=dict(MEDICINE, intended_target=0), boxes=swapped)
        labels, report = self.data.reconcile()
        self.assertEqual(report, [])
        front = labels[0]
        self.assertEqual(front["review"]["status"], "REVIEWED")
        self.assertEqual(front["target_boxes"][front["intended_target"]]["role"], "other_bottle")
        self.assertEqual(evaluate.validate_label(front), [])

    def test_different_or_unknown_intent_is_a_conflict(self):
        self.data.add("RJ", "front", answers=dict(MEDICINE, intended_target=0))
        self.data.add("TM", "front", answers=dict(MEDICINE, intended_target=1))
        labels, _ = self.data.reconcile()
        self.assertEqual(labels[0]["review"]["conflicts"], ["intended_target"])
        self.data.add("TM", "front", answers=MEDICINE, replace=True)              # TM doesn't know the intent
        labels, _ = self.data.reconcile()
        self.assertEqual(labels[0]["review"]["conflicts"], ["intended_target"])

    def test_single_reviewer_without_intent_leaves_it_unannotated(self):
        self.data.add("RJ", "front")
        labels, _ = self.data.reconcile()
        self.assertIsNone(labels[0]["intended_target"])


class InteractiveTests(unittest.TestCase):
    def test_annotate_with_scripted_answers_and_boxes(self):
        data = Dataset()
        try:
            answers = iter(["y", "P1", "y", "y", "Ibuprofen Tablets", "equate", "Ibuprofen | 200 mg", "",
                            "", "681131699372", "equate", "Ibuprofen", "", "7", "0", "y"])   # "7": not a box, asked again
            drawn = iter([[[100, 200, 300, 600]], [[400, 50, 600, 650]]])
            with contextlib.redirect_stdout(io.StringIO()):
                path = review.annotate(data.labels_path, "front", "RJ", ask=lambda prompt: next(answers),
                                       drawer=lambda image, title: next(drawn))
            saved = review.load_reviews(data.labels_path)["RJ"]["front"]
            self.assertTrue(path.endswith("rj.json"))
            self.assertEqual(saved["target_boxes"], BOXES)
            self.assertEqual(saved["ingredients"], [["Ibuprofen", "200 mg"]])
            self.assertEqual(saved["visible_text"], ["equate", "Ibuprofen"])
            self.assertEqual(saved["image_sha256"], data.sha("front"))
            self.assertEqual(saved["intended_target"], 0)
            preview = review.preview(data.labels_path, "front", "RJ", out_dir=os.path.join(data.folder.name, "p"))
            self.assertEqual(cv2.imread(preview).shape, (700, 700, 3))
        finally:
            data.close()

    def test_annotate_refuses_a_changed_image_and_unknown_ids(self):
        data = Dataset()
        try:
            labels = data.labels()
            labels[0]["sha256"] = "0" * 64
            data.write(labels)
            with self.assertRaises(ValueError):
                review.annotate(data.labels_path, "front", "RJ", ask=lambda p: "y", drawer=lambda i, t: [])
            with self.assertRaises(ValueError):
                review.annotate(data.labels_path, "nope", "RJ", ask=lambda p: "y", drawer=lambda i, t: [])
        finally:
            data.close()

    def test_cli_reconcile_writes_only_valid_labels(self):
        data = Dataset()
        try:
            data.add("RJ", "front")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(review.main(["reconcile", "--labels", data.labels_path, "--dry-run"]), 0)
            self.assertEqual(evaluate.review_status(data.labels()[0]), "UNREVIEWED")          # dry run: unchanged
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(review.main(["reconcile", "--labels", data.labels_path]), 0)
            self.assertEqual(evaluate.review_status(data.labels()[0]), "SINGLE_REVIEWED")
        finally:
            data.close()


if __name__ == "__main__":
    unittest.main()
