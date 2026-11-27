"""Concentration regressions: an amount read without its denominator ("15 mg")
must not exclude a liquid listed per volume ("15 mg/5mL").

Found in the Step B catalog comparison (2026-11-19): with only "15 mg" observed,
the strength index matched "15 mg/1" (a tablet) but not "15 mg/5mL", so the
liquid was dropped and a tablet formulation was VERIFIED for a cough syrup.
Six real false VERIFIED results, plus three unresolved formatting cases where
FDA writes a per-unit amount as "1.5 mg/1.5mg".
"""
import os
import unittest

from vision.products.catalog import ProductCatalog, name_key, strength_key
from vision.products.matcher import Level, ProductMatcher, Status
from tests.test_matcher_safety import CACHE, OTHER, catalog, ocr


def formulations(result):
    return {frozenset((n, strength_key(s)) for n, s in c.ingredients) for c in result.candidates}


class SyntheticConcentrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = catalog(
            ("7001-0001", "Coughaway", OTHER, [("DEXTROMETHORPHAN HYDROBROMIDE", "15 mg/1")]),         # softgel
            ("7001-0002", "Coughaway", OTHER, [("DEXTROMETHORPHAN HYDROBROMIDE", "15 mg/5mL")]),       # syrup
            ("7001-0003", "Tabsonly", OTHER, [("DEXTROMETHORPHAN HYDROBROMIDE", "15 mg/1")]),
            ("7001-0004", "Syruponly", OTHER, [("DEXTROMETHORPHAN HYDROBROMIDE", "15 mg/5mL")]),
            ("7001-0005", "Morningpill", OTHER, [("LEVONORGESTREL", "1.5 mg/1.5mg")]),
            ("7001-0006", "Morningpill", OTHER, [("LEVONORGESTREL", "1.5 mg/1")]),
        )

    def test_amount_without_denominator_keeps_the_liquid(self):
        result = self.matcher.match(ocr("COUGHAWAY", "Dextromethorphan HBr 15 mg"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual(formulations(result), {
            frozenset({("DEXTROMETHORPHAN HYDROBROMIDE", "15mg")}),
            frozenset({("DEXTROMETHORPHAN HYDROBROMIDE", "15mg/5ml")}),
        })

    def test_liquid_alone_is_a_candidate_but_not_verified_without_its_concentration(self):
        result = self.matcher.match(ocr("SYRUPONLY", "Dextromethorphan HBr 15 mg"))
        self.assertEqual(result.status, Status.UNCERTAIN)       # was: strength conflict
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.candidates[0].missing_strengths, ["15 mg/5mL"])

    def test_liquid_with_amount_and_volume_read_separately_is_still_not_verified(self):
        result = self.matcher.match(ocr("SYRUPONLY", "Dextromethorphan HBr 15 mg", "per 5 mL"))
        self.assertNotEqual(result.status, Status.VERIFIED)

    def test_tablet_only_product_still_verifies(self):
        result = self.matcher.match(ocr("TABSONLY", "Dextromethorphan HBr 15 mg"))
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_amount_written_per_amount_is_kept_and_not_verified(self):
        result = self.matcher.match(ocr("MORNINGPILL", "Levonorgestrel 1.5 mg"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertIn(frozenset({("LEVONORGESTREL", "1.5mg/1.5mg")}), formulations(result))

    def test_different_amount_still_excludes(self):
        result = self.matcher.match(ocr("COUGHAWAY", "Dextromethorphan HBr 30 mg"))
        self.assertNotEqual(result.status, Status.VERIFIED)
        self.assertTrue(any(c.evidence.kind == "strength" for c in result.conflicts))


# (brand, labeler, [(ingredient, strength)]) as listed in the catalog; the OCR is
# the brand plus each ingredient with the amount only, as in the catalog comparison.
FALSE_VERIFIED = [
    ("Cough Relief Adult", "Dolgencorp", [("DEXTROMETHORPHAN HYDROBROMIDE", "15 mg/5mL")]),
    ("Adult Cough Relief", "QUALITY CHOICE", [("DEXTROMETHORPHAN HYDROBROMIDE", "15 mg/5mL")]),
    ("Vicks DayQuil Severe Cold and Flu", "The Procter & Gamble",
     [("ACETAMINOPHEN", "325 mg/15mL"), ("DEXTROMETHORPHAN HYDROBROMIDE", "10 mg/15mL"),
      ("GUAIFENESIN", "200 mg/15mL"), ("PHENYLEPHRINE HYDROCHLORIDE", "5 mg/15mL")]),
    ("Mejoralito", "OPMX", [("ACETAMINOPHEN", "160 mg/5mL")]),
    ("Pepto-Bismol Ultra", "The Procter & Gamble", [("BISMUTH SUBSALICYLATE", "525 mg/15mL")]),
    ("Pepto-Bismol", "The Procter & Gamble", [("BISMUTH SUBSALICYLATE", "525 mg/30mL")]),
]
UNRESOLVED_FORMAT = [
    ("PostDay One-Step", "Rapha", [("LEVONORGESTREL", "1.5 mg/1.5mg")]),
    ("Morning After", "Rapha", [("LEVONORGESTREL", "1.5 mg/1.5mg")]),
    ("Curist Gas Relief", "Little Pharma", [("DIMETHICONE", "250 mg/250mg")]),
]


@unittest.skipUnless(os.path.exists(CACHE), "full openFDA OTC cache not built")
class RealCatalogConcentrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ProductCatalog.from_file(CACHE)
        cls.matcher = ProductMatcher(cls.catalog)

    def check(self, cases):
        for brand, labeler, ingredients in cases:
            with self.subTest(brand=brand):
                # The listing exists as described (guards against catalog drift).
                self.assertTrue(any(
                    r["brand_name"] == brand and r["labeler_name"].startswith(labeler)
                    and sorted((i["name"], i["strength"]) for i in r["active_ingredients"]) == sorted(ingredients)
                    for r in self.catalog.find_by_brand(brand)), brand)
                text = [brand] + [f"{name} {strength.split('/')[0]}" for name, strength in ingredients]
                result = self.matcher.match(ocr(*text))
                self.assertNotEqual(result.status, Status.VERIFIED, brand)
                truth = frozenset((name_key(n), strength_key(s)) for n, s in ingredients)
                self.assertIn(truth, {frozenset((name_key(n), strength_key(s)) for n, s in c.ingredients)
                                      for c in result.candidates}, brand)

    def test_six_liquid_false_verified_cases(self):
        self.check(FALSE_VERIFIED)

    def test_three_unresolved_formatting_cases(self):
        self.check(UNRESOLVED_FORMAT)


if __name__ == "__main__":
    unittest.main()
