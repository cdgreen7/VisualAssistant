"""Phase 3 target selection: scoring, edge cases, backward compatibility and integration.
Synthetic boxes only; these tests show algorithm behaviour, not real-world accuracy."""
import math
import random
import unittest

import numpy as np

from vision import target_selection as ts
from vision.identification import identify_detection, select_best_detection
from vision.products import evaluate

W, H = 1920, 1080


def bottle(confidence, box, class_name="bottle"):
    return {"class_id": 39, "class_name": class_name, "confidence": confidence, "box": box}


def centred(width, height, dx=0, dy=0, frame=(W, H)):
    cx, cy = frame[0] // 2 + dx, frame[1] // 2 + dy
    return (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2)


WATER = bottle(0.75, centred(220, 520, dx=620))      # more confident, off to the right
MEDICINE = bottle(0.70, centred(220, 520))           # less confident, centred


class ScoringTests(unittest.TestCase):
    def test_empty_and_missing_detections(self):
        for detections in ([], None):
            selection = ts.select_target(detections, (W, H), "weighted")
            self.assertIsNone(selection.detection)
            self.assertEqual(selection.candidates, [])
        only_others = [bottle(0.9, centred(100, 100), "cup"), bottle(0.8, centred(200, 300), "person")]
        selection = ts.select_target(only_others, (W, H), "weighted")
        self.assertIsNone(selection.detection)
        self.assertTrue(all(c.reason == "class is not 'bottle'" for c in selection.candidates))

    def test_one_eligible_detection(self):
        for method in ts.METHODS:
            selection = ts.select_target([WATER], (W, H), method)
            self.assertIs(selection.detection, WATER)

    def test_components_follow_the_formulas(self):
        selection = ts.select_target([bottle(0.6, (0, 0, W, H))], (W, H), "weighted")
        c = selection.candidates[0]
        self.assertEqual((c.confidence, c.area, c.center), (0.6, 1.0, 1.0))
        self.assertAlmostEqual(c.score, 0.40 * 0.6 + 0.25 * 1.0 + 0.35 * 1.0)
        corner = ts.select_target([bottle(0.6, (0, 0, 2, 2))], (W, H), "center").candidates[0]
        self.assertAlmostEqual(corner.center, 1 - math.hypot(959 / 960, 539 / 540) / math.sqrt(2))
        edge = ts.select_target([bottle(0.6, (0, 539, 2, 541))], (W, H), "center").candidates[0]
        self.assertAlmostEqual(edge.center, 1 - (959 / 960) / math.sqrt(2))     # ~ 0.29 at an edge midpoint
        quarter = ts.select_target([bottle(0.6, (0, 0, W // 2, H // 2))], (W, H), "area").candidates[0]
        self.assertAlmostEqual(quarter.area, 0.25)

    def test_higher_confidence_versus_more_centred(self):
        detections = [WATER, MEDICINE]
        self.assertIs(ts.select_target(detections, (W, H), "confidence_validated").detection, WATER)
        self.assertIs(ts.select_target(detections, (W, H), "center").detection, MEDICINE)
        weighted = ts.select_target(detections, (W, H), "weighted")
        self.assertIs(weighted.detection, MEDICINE)
        water, medicine = sorted(weighted.candidates, key=lambda c: c.index)
        self.assertGreater(medicine.score - water.score, 0.1)                   # breakdown is inspectable

    def test_larger_off_centre_versus_smaller_centred(self):
        large = bottle(0.70, centred(420, 900, dx=-650))
        small = bottle(0.70, centred(160, 360))
        self.assertIs(ts.select_target([large, small], (W, H), "area").detection, large)
        self.assertIs(ts.select_target([large, small], (W, H), "center").detection, small)
        self.assertIs(ts.select_target([large, small], (W, H), "weighted").detection, small)

    def test_multiple_detections_are_ranked_and_rejected_ones_listed(self):
        detections = [WATER, bottle(0.9, (5, 5, 5, 9)), MEDICINE, bottle(0.8, centred(50, 50), "cup")]
        selection = ts.select_target(detections, (W, H), "weighted")
        self.assertEqual([c.index for c in selection.candidates if c.eligible], [2, 0])
        self.assertEqual({c.index: c.reason for c in selection.candidates if not c.eligible},
                         {1: "box has zero or negative area", 3: "class is not 'bottle'"})
        self.assertIs(selection.chosen.detection, MEDICINE)
        text = ts.describe(selection)
        self.assertIn("-> #2", text)
        self.assertIn("not eligible (box has zero or negative area)", text)

    def test_ties_are_deterministic(self):
        left, right = bottle(0.7, centred(200, 400, dx=-300)), bottle(0.7, centred(200, 400, dx=300))
        self.assertIs(ts.select_target([left, right], (W, H), "weighted").detection, left)    # earlier detection
        self.assertIs(ts.select_target([right, left], (W, H), "weighted").detection, right)
        # equal S, different confidence: the more confident detection wins
        a = bottle(0.8, centred(200, 400, dx=500))
        b = bottle(0.6, centred(200, 400, dx=500))
        weights = ts.Weights(0.0, 0.5, 0.5)                   # confidence has no weight, so S is equal
        self.assertIs(ts.select_target([b, a], (W, H), weights=weights).detection, a)
        for _ in range(3):
            self.assertIs(ts.select_target([left, right, b, a], (W, H), "center").detection, left)

    def test_invalid_boxes_are_rejected_safely(self):
        cases = {
            "box is not four numbers": [None, (1, 2, 3), ("a", 1, 2, 3), "1234", 5, (1, 2, 3, 4, 5)],
            "box has a non-finite coordinate": [(0, 0, float("nan"), 10), (0, 0, float("inf"), 10)],
            "box has zero or negative area": [(10, 10, 10, 50), (10, 50, 40, 20)],
            "box is outside the frame": [(W + 10, 10, W + 90, 90), (-200, -200, -10, -10), (0, H, 50, H + 40)],
        }
        for reason, boxes in cases.items():
            for box in boxes:
                selection = ts.select_target([bottle(0.9, box), MEDICINE], (W, H), "weighted")
                self.assertIs(selection.detection, MEDICINE, box)
                self.assertEqual(selection.candidates[-1].reason, reason, box)
        partly_outside = ts.select_target([bottle(0.9, (W - 100, 100, W + 300, 500))], (W, H), "area").candidates[0]
        self.assertEqual(partly_outside.clipped_box, (W - 100, 100, W, 500))      # only the visible part counts
        self.assertAlmostEqual(partly_outside.area, 100 * 400 / (W * H))

    def test_invalid_confidence_is_rejected(self):
        for value in (None, float("nan"), float("inf"), -0.1, 1.2, True, "0.9"):
            selection = ts.select_target([bottle(value, centred(100, 200)), WATER], (W, H), "weighted")
            self.assertIs(selection.detection, WATER, value)
            self.assertEqual(selection.candidates[-1].reason, "confidence is not a number in [0, 1]")

    def test_yolo_detections_are_not_modified(self):
        detections = [dict(WATER), dict(MEDICINE)]
        before = [dict(d) for d in detections]
        selection = ts.select_target(detections, (W, H), "weighted")
        self.assertEqual(detections, before)
        self.assertIs(selection.detection, detections[1])                 # the input dict itself, not a copy

    def test_same_layout_at_different_resolutions(self):
        def layout(frame):
            w, h = frame
            return [bottle(0.75, (int(0.78 * w), int(0.25 * h), int(0.90 * w), int(0.75 * h))),
                    bottle(0.70, (int(0.44 * w), int(0.25 * h), int(0.56 * w), int(0.75 * h)))]
        results = []
        # sizes where every box edge falls on a whole pixel, so the layout is exactly the same
        for frame in ((500, 400), (2000, 1000), (4300, 5700), (1200, 1600)):
            selection = ts.select_target(layout(frame), frame, "weighted")
            results.append((selection.candidates[0].index,
                            [round(c.center, 9) for c in sorted(selection.candidates, key=lambda c: c.index)]))
        # 640x480 and the photos' 4284x5712 truncate to whole pixels: same choice, P within a pixel's worth
        for frame in ((640, 480), (4284, 5712)):
            selection = ts.select_target(layout(frame), frame, "weighted")
            self.assertEqual(selection.candidates[0].index, 1)
            self.assertAlmostEqual(min(c.center for c in selection.candidates), results[0][1][0], places=2)
        self.assertEqual({index for index, _ in results}, {1})
        self.assertEqual(len({tuple(p) for _, p in results}), 1)        # P depends only on relative position

    def test_invalid_weights_methods_and_frames(self):
        for bad in ((0.5, 0.5, 0.5), (-0.1, 0.6, 0.5), (float("nan"), 0.5, 0.5), (True, 0, 0), (1.0, 0.0)):
            with self.assertRaises((ValueError, TypeError)):
                ts.Weights(*bad)
        self.assertEqual(ts.Weights(0.2, 0.3, 0.5).center, 0.5)
        with self.assertRaises(ValueError):
            ts.select_target([MEDICINE], (W, H), "largest")
        with self.assertRaises(ValueError):
            ts.make_selector("largest")
        for frame in ((0, 1080), (1920,), None, (-5, 5)):
            with self.assertRaises(ValueError):
                ts.select_target([MEDICINE], frame, "weighted")
        for name, weights in ts.METHODS.items():
            self.assertAlmostEqual(weights.confidence + weights.area + weights.center, 1.0, msg=name)
        self.assertEqual(ts.METHODS["weighted"], ts.Weights(0.40, 0.25, 0.35))


class BackwardCompatibilityTests(unittest.TestCase):
    def test_confidence_method_equals_select_best_detection(self):
        rng = random.Random(3)
        for _ in range(500):
            detections = []
            for _ in range(rng.randint(0, 6)):
                x, y = rng.randint(0, W - 50), rng.randint(0, H - 50)
                confidence = rng.choice([0.5, 0.6, 0.7, round(rng.uniform(0.5, 1.0), 3)])     # frequent ties
                detections.append(bottle(confidence, (x, y, x + rng.randint(1, 50), y + rng.randint(1, 50)),
                                         rng.choice(["bottle", "bottle", "cup"])))
            self.assertIs(ts.select_target(detections, (W, H), "confidence_validated").detection,
                          select_best_detection(detections))

    def test_default_identification_path_is_unchanged(self):
        frame = np.zeros((H, W, 3), np.uint8)

        class Recorder:
            def __init__(self):
                self.shapes = []

            def read(self, image):
                self.shapes.append(image.shape[:2])
                return []

        class Matcher:
            def match(self, ocr):
                return "result"

        default = identify_detection(frame, [WATER, MEDICINE], Recorder(), Matcher())
        self.assertIs(default[0], WATER)                                  # highest confidence, as before
        seen = []
        selector = ts.make_selector("weighted")
        chosen = identify_detection(frame, [WATER, MEDICINE], Recorder(), Matcher(),
                                    selector=lambda d, size: seen.append(size) or selector(d, size))
        self.assertIs(chosen[0], MEDICINE)
        self.assertEqual(seen, [(W, H)])                                  # (width, height) of the frame
        self.assertIsNone(identify_detection(frame, [], Recorder(), Matcher(), selector=selector))

    def test_main_keeps_confidence_selection_by_default(self):
        with open("main.py") as f:
            source = f.read()
        self.assertIn('TARGET_SELECTION = os.environ.get("VA_TARGET_SELECTION", BASELINE)', source)
        self.assertIn("selector = None if TARGET_SELECTION == BASELINE else make_selector(TARGET_SELECTION)", source)
        self.assertEqual(ts.BASELINE, "confidence")
        self.assertNotIn(ts.BASELINE, ts.METHODS)                         # the original is not a scored method
        with self.assertRaises(ValueError):
            ts.select_target([MEDICINE], (W, H), ts.BASELINE)

    def test_original_and_validated_confidence_differ_only_by_validation(self):
        invalid_first = [bottle(0.95, (900, 300, 900, 800)), MEDICINE]          # zero-area box, top confidence
        self.assertIs(select_best_detection(invalid_first), invalid_first[0])  # original picks the invalid box
        self.assertIs(ts.select_target(invalid_first, (W, H), "confidence_validated").detection, MEDICINE)
        frame = np.zeros((H, W, 3), np.uint8)
        # with the original, identification then has nothing to read ("No bottle detected")
        self.assertIsNone(identify_detection(frame, invalid_first, None, None))


class EvaluatorIntegrationTests(unittest.TestCase):
    def label(self, status="REVIEWED", boxes=None):
        label = {"id": "s1", "image": "unused.jpg", "group": "photo", "product": "synthetic",
                 "brand_words": [], "ingredients": [["IBUPROFEN", "200 mg"]], "formulation": [["IBUPROFEN", "200 mg/1"]],
                 "package_ndc": None, "visible_text": [], "negative": False, "physical_product_id": "P1",
                 "target_boxes": boxes if boxes is not None else [
                     {"role": "medicine", "box": list(MEDICINE["box"])}, {"role": "other_bottle", "box": list(WATER["box"])}]}
        if status != "UNREVIEWED":
            label["review"] = {"status": status, "reviewers": ["RJ", "TM"], "date": "2026-11-27"}
        return label

    def test_all_methods_scored_on_identical_detections(self):
        score = evaluate.score_detection(self.label(), [WATER, MEDICINE], frame_size=(W, H))
        self.assertEqual(score["selection"], "WRONG_OBJECT_SELECTED")    # production default, unchanged
        self.assertEqual(score["selection_by_method"], {"confidence_original": "WRONG_OBJECT_SELECTED",
                                                        "confidence_validated": "WRONG_OBJECT_SELECTED",
                                                        "center": "MEDICINE_SELECTED", "area": "WRONG_OBJECT_SELECTED",
                                                        "weighted": "MEDICINE_SELECTED"})
        self.assertIsNone(evaluate.score_detection(self.label(), [WATER], frame_size=None)["selection_by_method"])

    def test_summary_separates_several_single_and_no_medicine_images(self):
        def image(i, label, detections, product):
            return {"id": f"i{i}", "group": "photo", "review_status": "REVIEWED", "physical_product_id": product,
                    "detection": evaluate.score_detection(label, detections, frame_size=(W, H))}
        single = self.label(boxes=[{"role": "medicine", "box": list(MEDICINE["box"])}])
        empty = dict(self.label(boxes=[]), negative=True, formulation=[], ingredients=[])
        images = [image(0, self.label(), [WATER, MEDICINE], "P1"), image(1, single, [MEDICINE], "P2"),
                  image(2, empty, [], "N1"), image(3, empty, [WATER], "N2")]
        methods = evaluate.summarize_detection(images)["selection_methods"]
        self.assertEqual(methods["confidence_original"]["medicine_selected"]["images"], [1, 2])
        self.assertEqual(methods["weighted"]["medicine_selected"]["images"], [2, 2])
        self.assertEqual(methods["confidence_original"]["wrong_object_selected"]["images"], [1, 2])
        self.assertEqual(methods["weighted"]["medicine_selected_several_bottles"]["images"], [1, 1])
        self.assertEqual(methods["weighted"]["medicine_selected_single_bottle"]["images"], [1, 1])
        # no reviewer recorded intent here, so intended-target metrics are not computed
        self.assertEqual(methods["weighted"]["intent_annotated_images"], 0)
        self.assertIsNone(methods["weighted"]["intended_target"])
        self.assertEqual(methods["weighted"]["no_medicine_images"], {"NOTHING_SELECTED": 1, "NON_MEDICINE_SELECTED": 1})

    def test_unreviewed_images_get_choices_but_no_outcomes(self):
        class Detector:
            def detect(self, image):
                return image

            def get_detections(self, result):
                return [WATER, MEDICINE]

        image = np.zeros((H, W, 3), np.uint8)
        rows, record = evaluate.evaluate_label(self.label("UNREVIEWED"), image, None, None, Detector(), inputs=())
        self.assertIsNone(record["detection"])
        self.assertEqual(record["selection_choices"]["confidence_original"]["index"], 0)
        self.assertEqual(record["selection_choices"]["confidence_validated"]["index"], 0)
        self.assertEqual(record["selection_choices"]["weighted"]["index"], 1)
        # replay rebuilds the same choices from the saved detections and frame size
        saved = {"rows": rows, "images": [dict(record, detections=[dict(d, box=list(d["box"])) for d in record["detections"]])]}
        _, images = evaluate.replay(saved, [self.label()], None)
        self.assertEqual(images[0]["detection"]["selection_by_method"]["weighted"], "MEDICINE_SELECTED")
        self.assertEqual(images[0]["selection_choices"], record["selection_choices"])


class IntendedTargetTests(unittest.TestCase):
    """The intended target (what the person meant) is annotated separately from medicine boxes."""
    OTHER_MEDICINE = bottle(0.72, centred(220, 520, dx=-620))

    def label(self, intended, boxes=None):
        boxes = boxes or [{"role": "medicine", "box": list(MEDICINE["box"])},
                          {"role": "medicine", "box": list(self.OTHER_MEDICINE["box"])},
                          {"role": "other_bottle", "box": list(WATER["box"])}]
        return {"id": "t", "image": "x.jpg", "group": "photo", "negative": False, "physical_product_id": "P1",
                "formulation": [["IBUPROFEN", "200 mg/1"]], "ingredients": [["IBUPROFEN", "200 mg"]],
                "brand_words": [], "target_boxes": boxes, "intended_target": intended,
                "review": {"status": "REVIEWED", "reviewers": ["RJ", "TM"], "date": "2026-11-27"}}

    def test_two_medicines_selecting_a_medicine_is_not_selecting_the_intended_one(self):
        label = self.label(intended=1)                       # the person meant the left-hand medicine
        score = evaluate.score_detection(label, [WATER, MEDICINE, self.OTHER_MEDICINE], frame_size=(W, H))
        self.assertEqual(score["selection_by_method"]["weighted"], "MEDICINE_SELECTED")
        self.assertEqual(score["intent_by_method"]["weighted"], "UNINTENDED_SELECTED")
        self.assertTrue(score["intent_annotated"])

    def test_intent_outcomes(self):
        self.assertEqual(evaluate.intent_outcome(self.label(0), {"box": list(MEDICINE["box"])}), "INTENDED_SELECTED")
        self.assertEqual(evaluate.intent_outcome(self.label(0), None), "NOTHING_SELECTED")
        self.assertEqual(evaluate.intent_outcome(self.label("none"), None), "CORRECTLY_NOTHING")
        self.assertEqual(evaluate.intent_outcome(self.label("none"), {"box": list(WATER["box"])}), "UNINTENDED_SELECTED")
        self.assertIsNone(evaluate.intent_outcome(self.label(None), {"box": list(MEDICINE["box"])}))

    def test_intent_is_validated_and_never_inferred(self):
        self.assertEqual(evaluate.validate_label(self.label(2)), [])
        for bad in (3, -1, "medicine", True, 1.0):
            self.assertTrue(evaluate.validate_label(self.label(bad)), bad)
        score = evaluate.score_detection(self.label(None), [WATER, MEDICINE], frame_size=(W, H))
        self.assertFalse(score["intent_annotated"])
        self.assertIsNone(score["intent_by_method"])         # medicine boxes are not used as intent

    def test_summary_reports_intent_only_where_annotated(self):
        images = [{"id": f"i{k}", "group": "photo", "review_status": "REVIEWED", "physical_product_id": f"P{k}",
                   "detection": evaluate.score_detection(label, [WATER, MEDICINE, self.OTHER_MEDICINE], frame_size=(W, H))}
                  for k, label in enumerate([self.label(0), self.label(1), self.label(None)])]
        weighted = evaluate.summarize_detection(images)["selection_methods"]["weighted"]
        self.assertEqual(weighted["medicine_selected"]["images"], [3, 3])
        self.assertEqual(weighted["intent_annotated_images"], 2)
        self.assertEqual(weighted["intended_target"]["images"], [1, 2])
        self.assertEqual(weighted["intent_counts"]["UNINTENDED_SELECTED"], 1)


class StudyTests(unittest.TestCase):
    def test_synthetic_scenarios_and_weight_grid(self):
        from research import target_selection_study as study
        grid = study.weight_grid()
        self.assertEqual(len(grid), 231)
        self.assertTrue(all(abs(w.confidence + w.area + w.center - 1) < 1e-9 for w in grid))
        for name, _, detections, intended in study.SCENARIOS:
            self.assertTrue(intended is None or detections[intended]["class_name"] == "bottle", name)
        rows = {r["scenario"]: r for r in study.synthetic_results()}
        self.assertFalse(rows["confidence_vs_centre"]["correct"]["confidence_original"])
        # the only scenario where original and validated confidence differ is the invalid box
        differ = [r["scenario"] for r in rows.values()
                  if r["correct"]["confidence_original"] != r["correct"]["confidence_validated"]]
        self.assertEqual(differ, ["invalid_box_present"])
        self.assertTrue(rows["confidence_vs_centre"]["correct"]["weighted"])
        self.assertFalse(rows["medicine_held_to_the_side"]["correct"]["weighted"])     # a failure case is kept
        self.assertEqual(study.sensitivity(), study.sensitivity())                    # deterministic


if __name__ == "__main__":
    unittest.main()
