import json
import os
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
