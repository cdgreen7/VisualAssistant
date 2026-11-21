"""Evidence-extraction regressions (Step B): packaging vocabulary, common
substance words, salt abbreviations and base names.

Failures observed in the DailyMed pilot (2026-11-19):
  "Antihistamine", "Outdoor Allergies", "Aspirin (NSAID)" became brand evidence;
  "No water needed" became ingredient WATER; "HBr" was not hydrobromide and
  "Dextromethorphan HBr" became a brand; "Cetirizine HCI" (OCR for HCl) gave no
  ingredient evidence at all.

Synthetic tests use small hand-built catalogs and always run; real-catalog tests
skip without the dated OTC cache.
"""
import os
import unittest

from vision.products.catalog import ProductCatalog
from vision.products.matcher import Level, ProductMatcher, Status, ingredient_key
from tests.test_matcher_safety import CACHE, GENDOSE, KENVUE, OTHER, catalog, ocr

DPH_HCL = ("DIPHENHYDRAMINE HYDROCHLORIDE", "25 mg/1")
CETIRIZINE = ("CETIRIZINE HYDROCHLORIDE", "10 mg/1")
GUAIFENESIN = ("GUAIFENESIN", "600 mg/1")
DXM_HBR = ("DEXTROMETHORPHAN HYDROBROMIDE", "30 mg/1")
PSE = ("PSEUDOEPHEDRINE HYDROCHLORIDE", "30 mg/1")


def label_catalog():
    return catalog(
        ("5058-0010", "Benadryl", KENVUE, [DPH_HCL]),
        ("5555-0001", "Antihistamine", GENDOSE, [DPH_HCL]),           # descriptor used as an exact brand
        ("2222-0001", "Outdoor Allergies", "NaturalCare", [("HISTAMINE DIHYDROCHLORIDE", "12 [hp_X]/1")]),
        ("5058-0011", "Zyrtec", KENVUE, [CETIRIZINE]),
        ("3333-0001", "Cetirizine HCI", "Acme Generics", [CETIRIZINE]),   # FDA brand string with the I/l typo
        ("1234-0001", "Eye Wash", "Clearwater Labs", [("WATER", "98.3 mL/100mL")]),
        ("6382-0001", "Mucinex DM", "Reckitt Benckiser LLC", [DXM_HBR, GUAIFENESIN]),
        ("4444-0001", "Dextromethorphan HBr", "Bionpharma Inc.", [("DEXTROMETHORPHAN HYDROBROMIDE", "15 mg/1")]),
        ("5058-0012", "Sudafed", KENVUE, [PSE]),
    )


def kinds(result, kind):
    return [e.value for e in result.evidence if e.kind == kind]


class VocabularyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = label_catalog()

    def test_purpose_word_is_not_brand_evidence(self):
        result = self.matcher.match(ocr("Benadryl", "Diphenhydramine HCl 25 mg | Antihistamine"))
        self.assertNotIn("ANTIHISTAMINE", kinds(result, "brand") + kinds(result, "brand_hint"))
        self.assertEqual(result.conflicts, [])
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_claim_phrase_is_not_brand_evidence(self):
        result = self.matcher.match(ocr("ZYRTEC", "Cetirizine HCl 10 mg", "Indoor & Outdoor Allergies"))
        self.assertNotIn("OUTDOOR ALLERGIES", kinds(result, "brand") + kinds(result, "brand_hint"))
        self.assertEqual(result.conflicts, [])

    def test_descriptor_alone_is_not_a_brand(self):
        # Step A blocked this through the superset guard; now no brand is observed at all.
        result = self.matcher.match(ocr("Antihistamine", "Diphenhydramine HCl 25 mg"))
        self.assertEqual(kinds(result, "brand"), [])
        self.assertNotEqual(result.status, Status.VERIFIED)


class CommonSubstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = label_catalog()

    def test_water_claim_beside_other_ingredient_is_only_a_hint(self):
        result = self.matcher.match(ocr("ZYRTEC", "Cetirizine HCl 10 mg", "No water needed"))
        self.assertNotIn("WATER", kinds(result, "ingredient"))
        self.assertIn("WATER", kinds(result, "ingredient_hint"))
        self.assertEqual(result.conflicts, [])
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_water_alone_still_proposes_candidates(self):
        result = self.matcher.match(ocr("Eye Wash", "Purified Water 98.3%"))
        self.assertIn("WATER", kinds(result, "ingredient"))
        self.assertTrue(result.candidates)

    def test_real_water_combination_keeps_truth_and_never_verifies_the_water_free_sibling(self):
        """WATER that really is an active ingredient beside another one: Step B treats
        it as a hint, so the true formulation can't be VERIFIED (a known, conservative
        cost), but it stays a candidate, and the sibling without water is not VERIFIED."""
        matcher = catalog(
            ("9999-0001", "Sunveil Tint", OTHER, [("ZINC OXIDE", "8 g/1"), ("WATER", "35 g/1")]),
            ("9999-0002", "Sunveil", OTHER, [("ZINC OXIDE", "8 g/1")]),
        )
        # Water printed with its amount: the amount (still strength evidence) identifies
        # the tinted product, but water is not counted as observed, so it isn't VERIFIED.
        result = matcher.match(ocr("SUNVEIL", "Zinc Oxide 8 g", "Water 35 g"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual([frozenset(n for n, _ in c.ingredients) for c in result.candidates],
                         [frozenset({"ZINC OXIDE", "WATER"})])
        self.assertEqual(result.candidates[0].missing_ingredients, ["WATER"])
        self.assertIn("WATER", kinds(result, "ingredient_hint"))
        self.assertIn("WATER read beside other ingredients", result.reason)

        # Water mentioned without an amount: both products stay candidates, and the
        # water-free one must not be VERIFIED.
        result = matcher.match(ocr("SUNVEIL", "Zinc Oxide 8 g", "Water"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual({frozenset(n for n, _ in c.ingredients) for c in result.candidates},
                         {frozenset({"ZINC OXIDE", "WATER"}), frozenset({"ZINC OXIDE"})})

    def test_hint_never_counts_as_an_observed_active_ingredient(self):
        result = self.matcher.match(ocr("No water needed", "Cetirizine HCl 10 mg"))
        for candidate in result.candidates:
            self.assertNotIn("WATER", {n for n, _ in candidate.ingredients if n not in candidate.missing_ingredients})


class SaltTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = label_catalog()

    def test_hbr_is_hydrobromide(self):
        result = self.matcher.match(ocr("MUCINEX DM", "Dextromethorphan HBr 30 mg", "Guaifenesin 600 mg"))
        self.assertIn("DEXTROMETHORPHAN HYDROBROMIDE", [ingredient_key(v) for v in kinds(result, "ingredient")])
        self.assertNotIn("DEXTROMETHORPHAN HBR", kinds(result, "brand"))
        self.assertEqual(result.candidates[0].base_only_ingredients, [])
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))

    def test_hci_gives_base_only_evidence_that_cannot_verify(self):
        result = self.matcher.match(ocr("ZYRTEC", "Cetirizine HCI 10 mg"))
        self.assertIn("CETIRIZINE", kinds(result, "ingredient"))
        self.assertNotIn("CETIRIZINE HCI", kinds(result, "brand"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual(result.candidates[0].base_only_ingredients, ["CETIRIZINE HYDROCHLORIDE"])

    def test_hci_never_supports_a_package_on_its_own(self):
        # With the full brand, the NDC is supported by the brand (unchanged rule) ...
        with_brand = self.matcher.match(ocr("ZYRTEC", "Cetirizine HCI 10 mg", "NDC 5058-0011-01"))
        self.assertEqual((with_brand.status, with_brand.level), (Status.VERIFIED, Level.PACKAGE))
        # ... but HCI is not accepted as the salt that would support it without one.
        without = self.matcher.match(ocr("Cetirizine HCI 10 mg", "NDC 5058-0011-01"))
        self.assertNotEqual(without.status, Status.VERIFIED)

    def test_salt_base_name_is_extracted_as_base_only(self):
        result = self.matcher.match(ocr("Sudafed", "Pseudoephedrine 30 mg"))
        self.assertIn("PSEUDOEPHEDRINE", kinds(result, "ingredient"))
        self.assertEqual(result.status, Status.UNCERTAIN)
        self.assertEqual(result.candidates[0].base_only_ingredients, ["PSEUDOEPHEDRINE HYDROCHLORIDE"])

    def test_full_salt_still_verifies(self):
        result = self.matcher.match(ocr("Sudafed", "Pseudoephedrine HCl 30 mg"))
        self.assertEqual((result.status, result.level), (Status.VERIFIED, Level.FORMULATION))


@unittest.skipUnless(os.path.exists(CACHE), "full openFDA OTC cache not built")
class RealCatalogEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = ProductMatcher(ProductCatalog.from_file(CACHE))

    def test_packaging_words_are_neither_brands_nor_hints(self):
        for phrase in ("ANTIHISTAMINE", "OUTDOOR ALLERGIES", "ALLERGIES", "HAD", "ALERT", "HEARTBURN", "HIVES",
                       "TOOTHACHE", "NSAID", "HBR", "HCI", "SOFTGEL", "ASPIRIN NSAID", "DEXTROMETHORPHAN HBR"):
            self.assertFalse(self.matcher._is_distinctive_brand(phrase), phrase)
            self.assertFalse(self.matcher._is_brand_hint(phrase), phrase)

    def test_major_brands_remain_distinctive(self):
        for brand in ("ADVIL", "ALEVE", "ZYRTEC", "BENADRYL", "MOTRIN", "EXCEDRIN", "CLARITIN", "PEPCID", "SUDAFED",
                      "DAYQUIL", "NYQUIL", "ROBITUSSIN", "ALLEGRA", "DELSYM", "PRILOSEC", "TUMS", "NEXIUM"):
            self.assertTrue(self.matcher._is_distinctive_brand(brand), brand)

    def test_brand_a_single_company_puts_in_generic_names_stays_distinctive(self):
        # "Dial Gold Antibacterial Bar Soap" is a generic_name; DIAL is still a brand.
        for brand in ("DIAL", "MOTIONSENSE", "ULTRACLEAR"):
            self.assertTrue(self.matcher._is_distinctive_brand(brand), brand)

    def test_pilot_benadryl_front_panel_has_no_antihistamine_conflict(self):
        result = self.matcher.match([{"text": t, "confidence": c} for t, c in (
            ("NDC 50580-226-12", 1.0), ("Benadryl", 0.99), ("ALLERGY", 1.0),
            ("Diphenhydramine HCI 25mg", 1.0), ("Antihistamine", 0.97))])
        self.assertEqual([c for c in result.conflicts if c.evidence.kind == "brand"], [])

    def test_pilot_zyrtec_front_panel_has_no_water_or_allergies_conflict(self):
        result = self.matcher.match([{"text": t, "confidence": c} for t, c in (
            ("NDC 50580-753-24", 1.0), ("ZYRTEC", 0.86), ("Cetirizine HCI", 0.99), ("Indoor &", 1.0),
            ("Outdoor", 1.0), ("Allergies", 1.0), ("No water needed", 1.0))])
        self.assertEqual(result.conflicts, [])
        self.assertIn("CETIRIZINE", kinds(result, "ingredient"))

    def test_real_water_combination_product_is_not_verified_and_stays_a_candidate(self):
        # Catalog record: TINTED SUNSCREEN FOUNDATION, ZINC OXIDE 8 g/1 + WATER 35 g/1
        result = self.matcher.match(ocr("TINTED SUNSCREEN FOUNDATION", "Zinc Oxide 8 g", "Water 35 g"))
        self.assertNotEqual(result.status, Status.VERIFIED)
        self.assertIn(frozenset({"ZINC OXIDE", "WATER"}),
                      [frozenset(n for n, _ in c.ingredients) for c in result.candidates])

    def test_hci_alone_never_verifies_a_formulation(self):
        result = self.matcher.match(ocr("ZYRTEC", "Cetirizine HCI", "10 mg"))
        self.assertEqual(result.status, Status.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
