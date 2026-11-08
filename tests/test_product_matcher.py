"""ProductMatcher tests with synthetic OCR results.

RealCatalogMatcherTests use the full dated OTC cache (gitignored; build it with
`python -m vision.products.build_cache`) and are skipped if it is missing.
FixtureMatcherTests use the small real-record fixture and always run.
"""
import json
import os
import unittest

from vision.products.catalog import ProductCatalog
from vision.products.matcher import Level, ProductMatcher, Status

ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE = os.path.join(ROOT, "vision", "products", "data", "otc_ndc_2026-10-30.json")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")


def ocr(*items, confidence=0.95):
    """("ADVIL", 0.4) or "ADVIL" -> TextRecognizer-style result dicts."""
    results = []
    for item in items:
        text, conf = item if isinstance(item, tuple) else (item, confidence)
        results.append({"text": text, "confidence": conf, "box": (0, 0, 10, 10)})
    return results


def ingredient_names(candidate):
    return {name for name, _ in candidate.ingredients}


@unittest.skipUnless(os.path.exists(CACHE), "full openFDA OTC cache not built")
class RealCatalogMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = ProductMatcher(ProductCatalog.from_file(CACHE))

    def match(self, *items, **kwargs):
        return self.matcher.match(ocr(*items, **kwargs))

    # A
    def test_advil_ibuprofen_200mg_is_uncertain_with_superset_competitors(self):
        result = self.match("ADVIL", "IBUPROFEN", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual(result.level, Level.NONE)
        self.assertGreater(len(result.candidates), 1)

        by_ingredients = {frozenset(ingredient_names(c)): c for c in result.candidates}
        self.assertIn(frozenset({"IBUPROFEN"}), by_ingredients)
        pm = by_ingredients[frozenset({"IBUPROFEN", "DIPHENHYDRAMINE CITRATE"})]
        self.assertEqual(pm.missing_ingredients, ["DIPHENHYDRAMINE CITRATE"])
        self.assertIn("PM", pm.missing_brand_words["ADVIL PM"])
        self.assertTrue(any("PSEUDOEPHEDRINE HYDROCHLORIDE" in ingredient_names(c) for c in result.candidates))

        # The observation stays what OCR read
        brand = [e for e in result.evidence if e.kind == "brand"]
        self.assertEqual([e.value for e in brand], ["ADVIL"])
        self.assertEqual(brand[0].observations[0].text, "ADVIL")

    # B
    def test_advil_pm_with_both_ingredients_and_strengths_is_verified(self):
        result = self.match("ADVIL PM", "IBUPROFEN", "DIPHENHYDRAMINE CITRATE", "38 mg", "200 mg")
        self.assertEqual(result.status, Status.VERIFIED)
        self.assertEqual(result.level, Level.FORMULATION)
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.ingredients, (("DIPHENHYDRAMINE CITRATE", "38 mg/1"), ("IBUPROFEN", "200 mg/1")))
        self.assertGreater(len(candidate.records), 1)  # several companies, one formulation
        self.assertEqual(candidate.missing_ingredients, [])
        self.assertIsNone(result.package)

    def test_base_ingredient_name_without_salt_is_not_verified(self):
        result = self.match("ADVIL PM", "IBUPROFEN", "DIPHENHYDRAMINE", "38 mg", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual(result.candidates[0].base_only_ingredients, ["DIPHENHYDRAMINE CITRATE"])

    def test_hcl_matches_hydrochloride_but_observation_keeps_hcl(self):
        result = self.match("ADVIL PM", "IBUPROFEN", "DIPHENHYDRAMINE HCL", "25 mg", "200 mg")
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))
        self.assertIn("DIPHENHYDRAMINE HCL", [e.value for e in result.evidence if e.kind == "ingredient"])
        self.assertIn(("DIPHENHYDRAMINE HYDROCHLORIDE", "25 mg/1"), result.candidates[0].ingredients)

    def test_whole_label_line(self):
        result = self.match("Advil PM Ibuprofen 200 mg / Diphenhydramine citrate 38 mg")
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    # C
    def test_generic_ibuprofen_200mg_is_uncertain(self):
        result = self.match("IBUPROFEN", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertIn("no distinctive brand", result.reason)
        self.assertGreater(len(result.candidates), 1)

    # D
    def test_motrin_ib_is_verified_across_labelers(self):
        result = self.match("MOTRIN IB", "IBUPROFEN", "200 mg")
        self.assertEqual(result.status, Status.VERIFIED)
        self.assertEqual(result.level, Level.FORMULATION)
        self.assertEqual(len(result.candidates), 1)
        records = result.candidates[0].records
        self.assertGreater(len({r["labeler_name"] for r in records}), 1)
        self.assertGreater(len({r["product_ndc"] for r in records}), 1)

    def test_motrin_without_ib_is_not_verified(self):
        # "MOTRIN" alone is not a complete FDA brand, so it is not brand evidence
        result = self.match("MOTRIN", "IBUPROFEN", "200 mg")
        self.assertNotEqual(result.status, Status.VERIFIED)

    # E
    def test_ocr_typo_is_not_corrected(self):
        self.assertEqual(self.match("ADV1L").status, Status.NO_MATCH)
        result = self.match("ADV1L", "IBUPROFEN", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertFalse(any(e.kind == "brand" for e in result.evidence))

    # F
    def test_low_confidence_is_not_verified(self):
        result = self.match("MOTRIN IB", "IBUPROFEN", "200 mg", confidence=0.45)
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertIn("below verify confidence", result.reason)

        result = self.match(("MOTRIN IB", 0.95), ("IBUPROFEN", 0.5), ("200 mg", 0.95))
        self.assertEqual(result.status, Status.UNCERTAIN)

        result = self.match("MOTRIN IB", "IBUPROFEN", "200 mg", confidence=0.2)
        self.assertEqual(result.status, Status.NO_MATCH)
        self.assertEqual(len(result.ignored), 3)

    # G
    def test_conflicting_ingredient_is_reported(self):
        result = self.match("MOTRIN IB", "ACETAMINOPHEN", "500 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertTrue(any(c.evidence.value == "ACETAMINOPHEN" for c in result.conflicts))
        self.assertIn("conflicting evidence", result.reason)

    def test_conflicting_brands_are_reported(self):
        result = self.match("ADVIL", "MOTRIN IB", "IBUPROFEN", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertTrue(any(c.evidence.value == "MOTRIN IB" for c in result.conflicts))

    def test_conflicting_salt_is_reported(self):
        result = self.match("ADVIL PM", "IBUPROFEN", "DIPHENHYDRAMINE CITRATE", "25 mg", "200 mg")
        self.assertNotEqual(result.status, Status.VERIFIED)
        self.assertTrue(result.conflicts)

    # H
    def test_advil_pm_with_diphenhydramine_missed_is_uncertain(self):
        result = self.match("ADVIL PM", "IBUPROFEN", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual(len(result.candidates), 2)
        for candidate in result.candidates:
            self.assertTrue(any(n.startswith("DIPHENHYDRAMINE") for n in candidate.missing_ingredients))

    def test_plain_advil_never_verified_from_ocr_alone(self):
        for items in (("ADVIL",), ("ADVIL", "IBUPROFEN"), ("ADVIL", "IBUPROFEN", "200 mg"),
                      ("Advil", "Ibuprofen Tablets", "200", "mg")):
            result = self.match(*items)
            self.assertEqual(result.status, Status.UNCERTAIN, items)

    # I
    def test_package_ndc_with_supporting_brand_is_verified(self):
        result = self.match("ADVIL", "IBUPROFEN", "200 mg", "NDC 0573-0150-20")
        self.assertEqual(result.status, Status.VERIFIED)
        self.assertEqual(result.level, Level.PACKAGE)
        self.assertEqual(result.package["package_ndc"], "0573-0150-20")
        self.assertEqual(result.candidates[0].records[0]["product_ndc"], "0573-0150")

    def test_package_ndc_alone_is_not_verified(self):
        result = self.match("NDC 0573-0150-20")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertIsNone(result.package)

    def test_misread_ndc_to_other_advil_product_is_not_verified(self):
        # 0573-0190-20 is Advil Multi-Symptom Cold and Flu, one digit from Advil 0573-0150-20
        result = self.match("ADVIL", "IBUPROFEN", "200 mg", "NDC 0573-0190-20")
        self.assertNotEqual(result.status, Status.VERIFIED)

    def test_ndc_conflicting_with_text_blocks_verification(self):
        motrin_ndc = self.matcher.catalog.find_by_brand("Motrin IB")[0]["packages"][0]["package_ndc"]
        result = self.match("ADVIL PM", "IBUPROFEN", "DIPHENHYDRAMINE CITRATE", "38 mg", "200 mg", "NDC " + motrin_ndc)
        self.assertNotEqual(result.status, Status.VERIFIED)
        self.assertTrue(any(c.evidence.kind == "ndc" for c in result.conflicts))

    # J
    def test_unrelated_text_is_no_match(self):
        result = self.matcher.match([
            {"text": "UNIVERSITY of ALASKA", "confidence": 0.92},
            {"text": "ANCHORAGE", "confidence": 0.82},
            {"text": "1", "confidence": 0.43},
            {"text": "Riday Jana", "confidence": 1.0},
        ])
        self.assertEqual(result.status, Status.NO_MATCH)
        self.assertEqual(result.candidates, [])

    def test_common_words_are_not_brand_evidence(self):
        result = self.match("PAIN RELIEF", "OF")
        self.assertEqual(result.status, Status.NO_MATCH)

    # K
    def test_empty_ocr_is_no_match(self):
        self.assertEqual(self.matcher.match([]).status, Status.NO_MATCH)
        self.assertEqual(self.matcher.match(None).status, Status.NO_MATCH)


class FixtureMatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.matcher = ProductMatcher(ProductCatalog.from_openfda_results(json.load(f)["results"]))

    def test_empty_and_unrelated_text_is_no_match(self):
        self.assertEqual(self.matcher.match([]).status, Status.NO_MATCH)
        self.assertEqual(self.matcher.match(ocr("UNIVERSITY of ALASKA", "ANCHORAGE")).status, Status.NO_MATCH)
        self.assertEqual(self.matcher.match(ocr("200 mg")).status, Status.NO_MATCH)

    def test_strength_split_across_ocr_results(self):
        result = self.matcher.match(ocr("IBUPROFEN", "200", "mg"))
        self.assertIn("200mg", [e.value for e in result.evidence if e.kind == "strength"])

    def test_package_verification_and_ndc_alone(self):
        result = self.matcher.match(ocr("ADVIL", "NDC 0573-0150-20"))
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.PACKAGE))
        self.assertEqual(self.matcher.match(ocr("NDC 0573-0150-20")).status, Status.UNCERTAIN)

    def test_unknown_ndc_is_a_conflict(self):
        result = self.matcher.match(ocr("ADVIL", "NDC 9999-9999-99"))
        self.assertNotEqual(result.status, Status.VERIFIED)
        self.assertTrue(any(c.evidence.kind == "ndc" for c in result.conflicts))

    def test_observations_are_immutable(self):
        result = self.matcher.match(ocr("Advil"))
        observation = result.evidence[0].observations[0]
        self.assertEqual(observation.text, "Advil")
        with self.assertRaises(Exception):
            observation.text = "ADVIL PM"


if __name__ == "__main__":
    unittest.main()
