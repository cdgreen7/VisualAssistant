"""Safety regressions for ProductMatcher: brand evidence must not exclude a
combination product that would add unobserved active ingredients.

Two false VERIFIED results reproduced on the real catalog (DailyMed pilot, 2026-11-19):
  BAYER + naproxen sodium 220 mg          -> VERIFIED naproxen only, although
      Aleve PM (naproxen + diphenhydramine) is a Bayer product that prints BAYER.
  ANTIHISTAMINE + diphenhydramine HCl 25  -> VERIFIED diphenhydramine only, because
      "Antihistamine" is one small labeler's exact brand name.

SyntheticSafetyTests use tiny hand-built catalogs that reproduce each mechanism and
always run. RealCatalogSafetyTests use the full dated OTC cache and skip without it.
"""
import os
import unittest
from unittest import mock

from vision.products.catalog import ProductCatalog
from vision.products.matcher import Level, ProductMatcher, Status

ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE = os.path.join(ROOT, "vision", "products", "data", "otc_ndc_2026-10-30.json")

BAYER, LIL, GENDOSE, KENVUE, OTHER = ("Bayer HealthCare LLC.", "Lil' Drug Store Products, Inc.",
                                      "Gendose Pharmaceuticals, LLC", "Kenvue Brands LLC", "Northwind Labs")


def ocr(*texts, confidence=0.95):
    return [{"text": t, "confidence": confidence, "box": (0, 0, 10, 10)} for t in texts]


def catalog(*products):
    """products: (product_ndc, brand, labeler, [(ingredient, strength), ...])."""
    raw = []
    for ndc, brand, labeler, ingredients in products:
        raw.append({
            "product_id": f"{ndc}_test", "product_ndc": ndc, "brand_name": brand, "generic_name": None,
            "labeler_name": labeler, "product_type": "HUMAN OTC DRUG", "dosage_form": "TABLET",
            "active_ingredients": [{"name": n, "strength": s} for n, s in ingredients],
            "packaging": [{"package_ndc": f"{ndc}-01", "description": "24 TABLET in 1 CARTON"}],
            "openfda": {},
        })
    return ProductMatcher(ProductCatalog.from_openfda_results(raw))


NAPROXEN = ("NAPROXEN SODIUM", "220 mg/1")
DPH_HCL = ("DIPHENHYDRAMINE HYDROCHLORIDE", "25 mg/1")
DPH_CITRATE = ("DIPHENHYDRAMINE CITRATE", "38 mg/1")
IBUPROFEN = ("IBUPROFEN", "200 mg/1")
APAP = ("ACETAMINOPHEN", "500 mg/1")
CETIRIZINE = ("CETIRIZINE HYDROCHLORIDE", "10 mg/1")
PSEUDOEPHEDRINE = ("PSEUDOEPHEDRINE HYDROCHLORIDE", "120 mg/1")


def bayer_catalog():
    return catalog(
        ("0280-0001", "Aleve", BAYER, [NAPROXEN]),
        ("0280-0002", "Aleve PM", BAYER, [DPH_HCL, NAPROXEN]),
        ("0280-0003", "Bayer Aspirin", BAYER, [("ASPIRIN", "325 mg/1")]),
        # Repackager listings, as in the real data: an exact "Bayer" brand makes
        # BAYER an extractable brand phrase, and "Bayer Aleve" is the only naproxen record it reaches.
        ("6671-0004", "Bayer Aleve", LIL, [NAPROXEN]),
        ("6671-0005", "Bayer", LIL, [("ASPIRIN", "325 mg/1")]),
    )


def antihistamine_catalog(word="Antihistamine"):
    """As in the real data: a descriptor is an exact brand, and store lines use it
    too; one of those lines also sells a PM combination. Since packaging vocabulary
    (Step B) ANTIHISTAMINE itself is never brand evidence, so the guard path is
    exercised with an invented word that is in no vocabulary list."""
    return catalog(
        ("5555-0001", word, GENDOSE, [DPH_HCL]),
        ("6666-0001", f"ValuMeds Allergy Relief {word}", OTHER, [DPH_HCL]),
        ("6666-0002", "ValuMeds Pain Relief PM", OTHER, [APAP, DPH_HCL]),
        ("5058-0003", "Tylenol PM", KENVUE, [APAP, DPH_HCL]),
    )


def nightcap_catalog():
    """A distinctive word that is not a labeler word narrows away Aleve PM:
    only the superset guard can catch this one."""
    return catalog(
        ("0280-0001", "Aleve", BAYER, [NAPROXEN]),
        ("0280-0002", "Aleve PM", BAYER, [DPH_HCL, NAPROXEN]),
        ("7777-0003", "Nightcap Aleve", OTHER, [NAPROXEN]),
        ("7777-0004", "Nightcap", OTHER, [("ASPIRIN", "325 mg/1")]),   # makes NIGHTCAP an exact brand phrase
    )


def sibling_and_unrelated_catalog():
    return catalog(
        ("5058-0001", "Motrin IB", KENVUE, [IBUPROFEN]),
        ("5058-0002", "Motrin PM", KENVUE, [DPH_CITRATE, IBUPROFEN]),
        ("5058-0003", "Zyrtec", KENVUE, [CETIRIZINE]),
        ("7777-0004", "Allergy Plus D", OTHER, [CETIRIZINE, PSEUDOEPHEDRINE]),
    )


class SyntheticSafetyTests(unittest.TestCase):
    # The two reproduced false VERIFIED mechanisms

    def test_manufacturer_word_does_not_verify_single_ingredient(self):
        matcher = bayer_catalog()
        for texts in (("BAYER", "Naproxen Sodium 220 mg"), ("BAYER", "ALEVE", "Naproxen Sodium 220 mg")):
            result = matcher.match(ocr(*texts))
            self.assertNotEqual(result.status, Status.VERIFIED, texts)
            self.assertIn(frozenset({"NAPROXEN SODIUM", "DIPHENHYDRAMINE HYDROCHLORIDE"}),
                          {frozenset(n for n, _ in c.ingredients) for c in result.candidates}, texts)

    def test_manufacturer_word_is_not_distinctive_brand_evidence(self):
        result = bayer_catalog().match(ocr("BAYER", "Naproxen Sodium 220 mg"))
        self.assertNotIn("BAYER", [e.value for e in result.evidence if e.kind == "brand"])

    def test_descriptor_brand_does_not_verify_when_a_line_using_it_sells_a_superset(self):
        result = antihistamine_catalog().match(ocr("Antihistamine", "Diphenhydramine HCl 25 mg"))
        self.assertEqual((result.status, result.level), (Status.UNCERTAIN, Level.NONE))

        # The same mechanism with a word no vocabulary list knows: the guard catches it.
        result = antihistamine_catalog("Histaclear").match(ocr("Histaclear", "Diphenhydramine HCl 25 mg"))
        self.assertEqual((result.status, result.level), (Status.UNCERTAIN, Level.NONE))
        self.assertIn("ValuMeds Pain Relief PM", result.reason)
        self.assertIn("ACETAMINOPHEN", result.reason)

    def test_distributor_tag_after_comma_does_not_link_products(self):
        # "Motrin IB, CVP HEALTH" and "Advil PM, CVP HEALTH" are a repackager's
        # listings of different brands, not one product line.
        matcher = catalog(
            ("5058-0001", "Motrin IB", KENVUE, [IBUPROFEN]),
            ("6671-0001", "Motrin IB, CVP HEALTH", LIL, [IBUPROFEN]),
            ("6671-0002", "Advil PM, CVP HEALTH", LIL, [DPH_CITRATE, IBUPROFEN]),
        )
        result = matcher.match(ocr("MOTRIN IB", "Ibuprofen 200 mg"))
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_guard_catches_non_labeler_word_through_brand_family(self):
        result = nightcap_catalog().match(ocr("NIGHTCAP", "Naproxen Sodium 220 mg"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertIn("Aleve PM", result.reason)

    # What the guard must leave alone

    def test_observed_sub_line_word_still_excludes_its_sibling(self):
        result = sibling_and_unrelated_catalog().match(ocr("MOTRIN IB", "Ibuprofen 200 mg"))
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_guard_is_not_a_catalog_wide_superset_search(self):
        # Another company's combination is excluded by the brand as before.
        result = sibling_and_unrelated_catalog().match(ocr("ZYRTEC", "Cetirizine HCl 10 mg"))
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_package_verification_is_unchanged(self):
        result = antihistamine_catalog().match(ocr("Antihistamine", "Diphenhydramine HCl 25 mg", "NDC 5555-0001-01"))
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.PACKAGE))
        self.assertEqual(result.package["package_ndc"], "5555-0001-01")

    def test_other_safeguards_still_apply(self):
        matcher = sibling_and_unrelated_catalog()
        low = matcher.match(ocr("MOTRIN IB", "Ibuprofen 200 mg", confidence=0.4))
        self.assertEqual(low.status, Status.UNCERTAIN)
        corrected = matcher.match(ocr("MOTRIN IB", "Ibuprofen 2OOmg"))
        self.assertEqual(corrected.status, Status.UNCERTAIN)
        missing = matcher.match(ocr("MOTRIN", "Ibuprofen 200 mg"))
        self.assertEqual(missing.status, Status.UNCERTAIN)       # Motrin PM still possible

    def test_catalog_ingredient_without_strength_does_not_crash(self):
        # Pre-existing TypeError ("strengths not observed" join over None), which
        # the labeler-word change exposed on a new input in the real catalog.
        matcher = catalog(("8888-0001", "Sunnyday Lotion", OTHER, [("ZINC OXIDE", None)]))
        result = matcher.match(ocr("SUNNYDAY LOTION", "Zinc Oxide"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertIn("not listed in catalog", result.reason)

    def test_guard_only_ever_removes_verified(self):
        """With the guard disabled every other outcome is identical; the guard
        turns some VERIFIED/FORMULATION into UNCERTAIN and nothing else."""
        inputs = [
            (bayer_catalog, ("ALEVE", "Naproxen Sodium 220 mg")),
            (bayer_catalog, ("Naproxen Sodium 220 mg",)),
            (antihistamine_catalog, ("Antihistamine", "Diphenhydramine HCl 25 mg")),
            (antihistamine_catalog, ("Tylenol PM", "Acetaminophen 500 mg", "Diphenhydramine HCl 25 mg")),
            (antihistamine_catalog, ("Antihistamine", "Diphenhydramine HCl 25 mg", "NDC 5555-0001-01")),
            (nightcap_catalog, ("NIGHTCAP", "Naproxen Sodium 220 mg")),
            (sibling_and_unrelated_catalog, ("MOTRIN IB", "Ibuprofen 200 mg")),
            (sibling_and_unrelated_catalog, ("MOTRIN PM", "Ibuprofen 200 mg", "Diphenhydramine Citrate 38 mg")),
            (sibling_and_unrelated_catalog, ("ZYRTEC", "Cetirizine HCl 10 mg")),
            (sibling_and_unrelated_catalog, ("hello",)),
        ]
        for build, texts in inputs:
            matcher = build()
            guarded = matcher.match(ocr(*texts))
            with mock.patch.object(ProductMatcher, "_superset_competitor", return_value=None):
                unguarded = matcher.match(ocr(*texts))
            if guarded.status == Status.VERIFIED:
                self.assertEqual((guarded.status, guarded.level), (unguarded.status, unguarded.level), texts)
            elif unguarded.status == Status.VERIFIED:
                self.assertEqual((unguarded.level, guarded.status), (Level.FORMULATION, Status.UNCERTAIN), texts)
            else:
                self.assertEqual(guarded.status, unguarded.status, texts)
            self.assertEqual([c.ingredients for c in guarded.candidates],
                             [c.ingredients for c in unguarded.candidates], texts)


@unittest.skipUnless(os.path.exists(CACHE), "full openFDA OTC cache not built")
class RealCatalogSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = ProductMatcher(ProductCatalog.from_file(CACHE))

    def match(self, *texts, **kwargs):
        return self.matcher.match(ocr(*texts, **kwargs))

    def test_bayer_naproxen_is_not_verified(self):
        for texts in (("BAYER", "naproxen sodium 220 mg"), ("BAYER", "ALEVE", "naproxen sodium 220 mg")):
            result = self.match(*texts)
            self.assertNotEqual(result.status, Status.VERIFIED, texts)
            self.assertTrue(any("DIPHENHYDRAMINE HYDROCHLORIDE" in {n for n, _ in c.ingredients}
                                for c in result.candidates), texts)   # Aleve PM stays a candidate

    def test_antihistamine_diphenhydramine_is_not_verified(self):
        result = self.match("Antihistamine", "Diphenhydramine HCl 25 mg")
        self.assertNotEqual(result.status, Status.VERIFIED)

    def test_legitimate_formulation_verifications_survive(self):
        cases = [
            (("ADVIL PM", "IBUPROFEN", "DIPHENHYDRAMINE CITRATE", "38 mg", "200 mg"),
             (("DIPHENHYDRAMINE CITRATE", "38 mg/1"), ("IBUPROFEN", "200 mg/1"))),
            (("ZYRTEC", "Cetirizine HCl", "10 mg"), (("CETIRIZINE HYDROCHLORIDE", "10 mg/1"),)),
            (("MOTRIN IB", "IBUPROFEN", "200 mg"), (("IBUPROFEN", "200 mg/1"),)),
            (("Pepcid AC", "Famotidine 20 mg"), (("FAMOTIDINE", "20 mg/1"),)),
        ]
        for texts, formulation in cases:
            result = self.match(*texts)
            self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION), texts)
            self.assertEqual(result.candidates[0].ingredients, formulation, texts)

    def test_package_verifications_survive(self):
        for texts, ndc in ((("ADVIL", "IBUPROFEN 200 mg", "NDC 0573-0150-20"), "0573-0150-20"),
                           (("Benadryl", "Diphenhydramine HCl 25 mg", "NDC 50580-226-12"), "50580-226-12")):
            result = self.match(*texts)
            self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.PACKAGE), texts)
            self.assertEqual(result.package["package_ndc"], ndc)

    def test_major_brands_remain_distinctive(self):
        for brand in ("ADVIL", "ALEVE", "ZYRTEC", "BENADRYL", "MOTRIN", "EXCEDRIN", "CLARITIN", "PEPCID"):
            self.assertTrue(self.matcher._is_distinctive_brand(brand), brand)
        for word in ("BAYER", "KROGER"):
            self.assertFalse(self.matcher._is_distinctive_brand(word), word)


if __name__ == "__main__":
    unittest.main()
