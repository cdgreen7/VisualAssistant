"""Tests against real openFDA NDC records (export 2026-10-30) in tests/fixtures."""
import json
import os
import unittest

from vision.products.catalog import PACKAGE_FIELDS, RECORD_FIELDS, ProductCatalog, normalize_record

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")

ADVIL = "0573-0150_4cd840a6-9554-de58-e063-6394a90ab543"
ADVIL_SAME_LABEL = "0573-0161_4cd840a6-9554-de58-e063-6394a90ab543"
ADVIL_PM_TABLET = "0573-0164_15971707-84cd-4742-e063-6394a90af906"
ADVIL_PM_LIQUIGEL = "0573-0167_57c5339d-4ab8-ce7f-e063-6394a90a75a5"
MOTRIN_IB = "53209-2001_37e92a7b-971f-c212-e063-6394a90ad710"
FAMOTIDINE_A = "71821-010_0fc43ab9-0e23-258f-e063-6394a90a0c40"
FAMOTIDINE_B = "71821-010_6363f098-8bab-4879-bea8-41b60c4530eb"
INFLUENSEG = "75053-0100_3602f83f-033b-4c57-8769-3e57782429e5"
HYPOTHALTEM = "75053-0010_4e326cd0-c586-4d82-80ef-0ae5dd294dcc"
ELTAMD_PURE = "72043-2582_6f7bbffb-97bc-4a8e-aba3-dbd9bd1ef381"
ELTAMD_CLEAR = "72043-2500_e1ccd02f-6246-493f-8ce5-a39b29b7bde7"
THERACARE = "71101-041_200abefc-e3e4-c65a-e063-6394a90a9a0e"

ADVIL_UPC = "0305730150200"


def ids(records):
    return {record["product_id"] for record in records}


def pair_ids(pairs):
    return {(record["product_id"], package["package_ndc"]) for record, package in pairs}


def ingredient_names(record):
    return {ingredient["name"] for ingredient in record["active_ingredients"]}


class ProductCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.raw = json.load(f)["results"]
        cls.catalog = ProductCatalog.from_openfda_results(cls.raw)

    # Advil / Advil PM / Motrin

    def test_brand_search_keeps_advil_and_advil_pm_separate(self):
        self.assertEqual(ids(self.catalog.find_by_brand("Advil")), {ADVIL, ADVIL_SAME_LABEL})
        self.assertEqual(ids(self.catalog.find_by_brand("advil pm")), {ADVIL_PM_TABLET, ADVIL_PM_LIQUIGEL})
        self.assertEqual(ids(self.catalog.find_by_brand("MOTRIN IB")), {MOTRIN_IB})

    def test_ibuprofen_200mg_returns_separate_products(self):
        found = ids(self.catalog.find_by_ingredient("ibuprofen", "200 mg"))
        self.assertEqual(found, {ADVIL, ADVIL_SAME_LABEL, ADVIL_PM_TABLET, ADVIL_PM_LIQUIGEL, MOTRIN_IB})

    def test_all_active_ingredients_preserved(self):
        advil = self.catalog.get_by_product_id(ADVIL)
        pm_tablet = self.catalog.get_by_product_id(ADVIL_PM_TABLET)
        pm_liquigel = self.catalog.get_by_product_id(ADVIL_PM_LIQUIGEL)
        self.assertEqual(ingredient_names(advil), {"IBUPROFEN"})
        self.assertEqual(ingredient_names(pm_tablet), {"IBUPROFEN", "DIPHENHYDRAMINE CITRATE"})
        self.assertEqual(ingredient_names(pm_liquigel), {"IBUPROFEN", "DIPHENHYDRAMINE HYDROCHLORIDE"})
        self.assertIn({"name": "DIPHENHYDRAMINE CITRATE", "strength": "38 mg/1"}, pm_tablet["active_ingredients"])

    def test_ingredient_and_strength_filters(self):
        self.assertEqual(ids(self.catalog.find_by_ingredient("Diphenhydramine Citrate")), {ADVIL_PM_TABLET})
        self.assertEqual(self.catalog.find_by_ingredient("ibuprofen", "400 mg"), [])

    def test_lookup_by_ndc(self):
        self.assertEqual(ids(self.catalog.find_by_product_ndc("0573-0150")), {ADVIL})
        self.assertEqual(pair_ids(self.catalog.find_by_package_ndc("0573-0150-20")), {(ADVIL, "0573-0150-20")})
        self.assertEqual(pair_ids(self.catalog.find_by_package_ndc("00573015020")), {(ADVIL, "0573-0150-20")})

    def test_package_ndc_belongs_to_one_record(self):
        seen = {}
        for record in self.catalog.records:
            for package in record["packages"]:
                self.assertNotIn(package["package_ndc_norm"], seen)
                seen[package["package_ndc_norm"]] = record["product_id"]

    # Edge case 1: duplicate product NDC across records

    def test_duplicate_product_ndc_returns_every_record(self):
        found = self.catalog.find_by_product_ndc("71821-010")
        self.assertEqual(ids(found), {FAMOTIDINE_A, FAMOTIDINE_B})
        self.assertEqual(len(found), 2)  # not merged

    # Edge case 2: ambiguous undashed 10-digit NDC

    def test_ambiguous_undashed_ndc_returns_every_matching_package(self):
        found = self.catalog.find_by_package_ndc("7505301001")
        self.assertEqual(pair_ids(found), {(INFLUENSEG, "75053-0100-1"), (HYPOTHALTEM, "75053-0010-1")})

    def test_dashed_ndc_is_not_ambiguous(self):
        self.assertEqual(pair_ids(self.catalog.find_by_package_ndc("75053-0100-1")), {(INFLUENSEG, "75053-0100-1")})

    # Edge case 3: UPCs

    def test_upc_is_label_level(self):
        # One UPC on the Advil label covers several products; no Advil PM
        self.assertEqual(ids(self.catalog.find_by_upc(ADVIL_UPC)), {ADVIL, ADVIL_SAME_LABEL})
        self.assertEqual(ids(self.catalog.find_by_upc("305730150200")), {ADVIL, ADVIL_SAME_LABEL})

    def test_upc_ndc_candidate_resolves_to_package(self):
        self.assertEqual(pair_ids(self.catalog.find_by_upc_ndc(ADVIL_UPC)), {(ADVIL, "0573-0150-20")})

    def test_upc_ndc_candidates_that_match_nothing(self):
        eltamd = self.catalog.get_by_product_id(ELTAMD_PURE)
        self.assertEqual(eltamd["upc"], ["0390205025824"])
        self.assertEqual(self.catalog.find_by_upc_ndc("0390205025824"), [])  # number system 3, no NDC inside
        self.assertEqual(ids(self.catalog.find_by_upc("0390205025824")), {ELTAMD_PURE})

    def test_upc_that_cannot_carry_an_ndc(self):
        self.assertEqual(self.catalog.get_by_product_id(THERACARE)["upc"], ["0845717010566"])
        self.assertEqual(self.catalog.find_by_upc_ndc("0845717010566"), [])
        self.assertEqual(self.catalog.find_by_upc_ndc("0305730150201"), [])  # bad check digit
        self.assertEqual(self.catalog.find_by_upc("0305730150201"), [])

    # Edge case 4: duplicate package NDC within one record

    def test_identical_duplicate_package_is_removed(self):
        raw = next(r for r in self.raw if r["product_id"] == ELTAMD_CLEAR)
        raw_ndcs = [p["package_ndc"] for p in raw["packaging"]]
        self.assertEqual(raw_ndcs.count("72043-2500-1"), 2)

        packages = [p["package_ndc"] for p in self.catalog.get_by_product_id(ELTAMD_CLEAR)["packages"]]
        self.assertEqual(packages.count("72043-2500-1"), 1)
        self.assertEqual(len(packages), len(set(raw_ndcs)))
        self.assertEqual(len(self.catalog.find_by_package_ndc("72043-2500-1")), 1)

    def test_conflicting_duplicate_package_raises(self):
        raw = json.loads(json.dumps(next(r for r in self.raw if r["product_id"] == ELTAMD_CLEAR)))
        raw["packaging"].append({"package_ndc": "72043-2500-1", "description": "something else"})
        with self.assertRaises(ValueError):
            normalize_record(raw)

    # Safety and robustness

    def test_records_keep_only_approved_fields(self):
        for record in self.catalog.records:
            self.assertEqual(set(record), set(RECORD_FIELDS))
            for package in record["packages"]:
                self.assertEqual(set(package), set(PACKAGE_FIELDS))

    def test_duplicate_product_id_rejected(self):
        record = self.catalog.get_by_product_id(ADVIL)
        with self.assertRaises(ValueError):
            ProductCatalog([record, record])

    def test_unknown_or_garbled_lookups(self):
        self.assertEqual(self.catalog.find_by_product_ndc("99999-999"), [])
        self.assertEqual(self.catalog.find_by_product_ndc("garbage"), [])
        self.assertEqual(self.catalog.find_by_package_ndc("99999-9999-99"), [])
        self.assertEqual(self.catalog.find_by_package_ndc(None), [])
        self.assertEqual(self.catalog.find_by_upc(""), [])
        self.assertEqual(self.catalog.find_by_brand("Tylenol"), [])
        self.assertEqual(self.catalog.find_by_ingredient("acetaminophen"), [])
        self.assertIsNone(self.catalog.get_by_product_id("nope"))


if __name__ == "__main__":
    unittest.main()
