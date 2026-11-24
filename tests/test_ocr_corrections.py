"""OCR strength correction (O -> 0) and store-brand hints in ProductMatcher."""
import json
import os
import unittest

from vision.products.catalog import ProductCatalog
from vision.products.matcher import Level, ProductMatcher, Status

ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE = os.path.join(ROOT, "vision", "products", "data", "otc_ndc_2026-10-30.json")
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")

strengths = ProductMatcher._strengths


def ocr(*items, confidence=0.95):
    """("2oomg", 0.32) or "ADVIL" -> TextRecognizer-style result dicts."""
    return [{"text": item[0], "confidence": item[1]} if isinstance(item, tuple)
            else {"text": item, "confidence": confidence} for item in items]


def strength_evidence(result):
    return [(e.value, e.corrected_from) for e in result.evidence if e.kind == "strength"]


class StrengthCorrectionTests(unittest.TestCase):
    def test_normal_strengths_are_not_corrected(self):
        self.assertEqual(strengths("TABLETS 200MG"), [("200mg", None)])
        self.assertEqual(strengths("38 MG"), [("38mg", None)])
        self.assertEqual(strengths("100 MG/5 ML"), [("100mg", None), ("5ml", None)])

    def test_letter_o_in_number_is_corrected_and_tracked(self):
        self.assertEqual(strengths("2OOMG"), [("200mg", "2OOMG")])
        self.assertEqual(strengths("TABLETS 20OMG"), [("200mg", "20OMG")])
        self.assertEqual(strengths("1O.5 MG"), [("10.5mg", "1O.5 MG")])

    def test_ambiguous_or_non_numeric_text_is_not_corrected(self):
        for text in ("OOMG", "O MG", "OMG", "COMG", "BO0MG", "MOTRIN", "2OO", "2OO OZ", "NO MG"):
            self.assertEqual(strengths(text), [], text)

    def test_ndc_digits_are_never_corrected(self):
        self.assertEqual(strengths("NDC 0573-O150-20"), [])


class FixtureCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.matcher = ProductMatcher(ProductCatalog.from_openfda_results(json.load(f)["results"]))

    def test_original_observation_is_kept(self):
        result = self.matcher.match(ocr("IBUPROFEN", "2oomg"))
        evidence = [e for e in result.evidence if e.kind == "strength"][0]
        self.assertEqual(evidence.value, "200mg")
        self.assertEqual(evidence.corrected_from, "2OOMG")
        self.assertEqual(evidence.observations[0].text, "2oomg")

    def test_split_number_and_unit_with_letter_o(self):
        self.assertEqual(strength_evidence(self.matcher.match(ocr("IBUPROFEN", "2oo", "mg"))), [("200mg", "2OO MG")])
        self.assertEqual(strength_evidence(self.matcher.match(ocr("IBUPROFEN", "200", "mg"))), [("200mg", None)])

    def test_unknown_ndc_with_letter_is_not_rescued(self):
        result = self.matcher.match(ocr("ADVIL", "NDC 0573-O150-20"))
        self.assertFalse(any(e.kind == "ndc" for e in result.evidence))
        self.assertIsNone(result.package)


@unittest.skipUnless(os.path.exists(CACHE), "full openFDA OTC cache not built")
class RealCatalogCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = ProductMatcher(ProductCatalog.from_file(CACHE))

    def match(self, *items, **kwargs):
        return self.matcher.match(ocr(*items, **kwargs))

    def test_corrected_strength_alone_cannot_verify(self):
        result = self.match("MOTRIN IB", "IBUPROFEN", "2OOmg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertIn("only through OCR correction", result.reason)

    def test_corrected_strength_corroborated_by_plain_reading(self):
        result = self.match("MOTRIN IB", "IBUPROFEN", "2OOmg", "200 mg")
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_corrected_strength_cannot_complete_advil_pm(self):
        result = self.match("ADVIL PM", "IBUPROFEN", "DIPHENHYDRAMINE CITRATE", "38 mg", "2OO mg")
        self.assertEqual(result.status, Status.UNCERTAIN)

    # The actual live Equate observation
    def test_equate_live_observation(self):
        result = self.match(("equate", 1.00), ("Ibuprofen", 1.00), ("2oomg", 0.32), ("mg", 0.84),
                            ("Tablets", 1.00), ("Pain", 1.00), ("Reliever/", 0.67), ("Fever", 1.00), ("(NSAID)", 0.99))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual(len(result.candidates), 7)  # was 18 before the strength correction
        self.assertIn("candidates depend on OCR-corrected text", result.reason)
        self.assertIn("brand hint EQUATE", result.reason)

        ingredients = [{n for n, _ in c.ingredients} for c in result.candidates]
        self.assertEqual(ingredients[0], {"IBUPROFEN"})  # the true formulation remains, ranked first
        self.assertIn({"IBUPROFEN", "DIPHENHYDRAMINE CITRATE"}, ingredients)  # Equate Ibuprofen PM competitor
        self.assertGreater(result.candidates[0].brand_hint_records, 0)

    def test_equate_hint_never_verifies_or_narrows(self):
        with_hint = self.match("equate", "IBUPROFEN", "200 mg")
        without = self.match("IBUPROFEN", "200 mg")
        self.assertEqual(with_hint.status, Status.UNCERTAIN)
        self.assertEqual({c.ingredients for c in with_hint.candidates}, {c.ingredients for c in without.candidates})
        self.assertFalse(any(e.kind == "brand" for e in with_hint.evidence))
        self.assertTrue(any(e.kind == "brand_hint" for e in with_hint.evidence))

    def test_descriptive_words_are_not_brand_hints(self):
        for text in ("ANTACID", "LAXATIVE", "PAIN RELIEF", "SLEEP", "PM"):
            result = self.match(text)
            self.assertFalse(any(e.kind in ("brand", "brand_hint") for e in result.evidence), text)

    def test_compare_to_brand_is_not_the_products_brand(self):
        # Store-brand bottle: before this change it was VERIFIED as Motrin IB
        result = self.match("equate", "Compare to Motrin IB active ingredient", "Ibuprofen", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertFalse(any(e.kind == "brand" for e in result.evidence))
        self.assertIn("comparison text", result.reason)

        # The badge split across OCR results
        result = self.match("equate", "Compare", "to Motrin IB", "active ingredient", "Ibuprofen", "200 mg")
        self.assertEqual(result.status, Status.UNCERTAIN)

    def test_previous_ambiguities_unchanged(self):
        for items in (("ADVIL", "IBUPROFEN", "200 mg"), ("ADVIL PM", "IBUPROFEN", "200 mg"),
                      ("MOTRIN", "IBUPROFEN", "200 mg"), ("IBUPROFEN", "200 mg"),
                      ("ADVIL", "IBUPROFEN", "2oomg"), ("ADVIL", "IBUPROFEN", "200 mg", "NDC 0573-0190-20")):
            self.assertEqual(self.match(*items).status, Status.UNCERTAIN, items)
        self.assertEqual(self.match("MOTRIN IB", "IBUPROFEN", "200 mg", confidence=0.45).status, Status.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
