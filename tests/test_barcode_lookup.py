"""FDA lookup of barcode observations (no image decoding)."""
import json
import os
import unittest

from vision.products.barcode_lookup import LookupStatus, conflicting, lookup_barcode
from vision.products.catalog import ProductCatalog

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")


def upc(value, kind="UPC_A"):
    return {"type": kind, "value": value, "decoded": True}


def raw(product_ndc, packages, upcs=(), brand="TEST", ingredient="IBUPROFEN"):
    return {"product_ndc": product_ndc, "product_id": product_ndc + "_test", "brand_name": brand,
            "generic_name": ingredient.lower(), "labeler_name": "Test Labeler",
            "active_ingredients": [{"name": ingredient, "strength": "200 mg/1"}],
            "packaging": [{"package_ndc": p, "description": p} for p in packages],
            "openfda": {"upc": list(upcs)}}


class FixtureLookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.catalog = ProductCatalog.from_openfda_results(json.load(f)["results"])

    def test_known_package_identifier(self):
        for observation in (upc("305730150200"), upc("0305730150200", "EAN_13")):
            lookup = lookup_barcode(observation, self.catalog)
            self.assertEqual(lookup.status, LookupStatus.RESOLVED)
            self.assertEqual(lookup.record["product_ndc"], "0573-0150")
            self.assertEqual(lookup.package["package_ndc"], "0573-0150-20")
            self.assertIs(lookup.observation, observation)
        self.assertEqual(observation["value"], "0305730150200")  # observation is not rewritten

    def test_unresolved_upc(self):
        lookup = lookup_barcode(upc("012345678905"), self.catalog)
        self.assertEqual(lookup.status, LookupStatus.UNRESOLVED)
        self.assertIsNone(lookup.record)
        self.assertIsNone(lookup.package)

    def test_label_only_upc_is_a_candidate_without_a_package(self):
        lookup = lookup_barcode(upc("0845717010566", "EAN_13"), self.catalog)
        self.assertEqual(lookup.status, LookupStatus.CANDIDATE)
        self.assertIsNotNone(lookup.record)
        self.assertIsNone(lookup.package)

    def test_invalid_values_are_not_repaired(self):
        for value in ("305730150201", "30573015020", "3057301502000000", ""):  # bad check digit, wrong lengths
            self.assertEqual(lookup_barcode(upc(value), self.catalog).status, LookupStatus.UNRESOLVED, value)

    def test_other_formats_are_not_treated_as_upc_or_ndc(self):
        for observation in (upc("305730150200", "CODE_128"), upc("0573-0150-20", "CODE_128"),
                            upc("0100305730150205", "DATA_MATRIX")):
            lookup = lookup_barcode(observation, self.catalog)
            self.assertEqual(lookup.status, LookupStatus.UNRESOLVED)
            self.assertEqual((lookup.label_records, lookup.ndc_matches), ([], []))

    def test_undecoded_observation(self):
        observation = {"type": "QRCODE", "value": None, "decoded": False, "reason": "unsupported format"}
        self.assertEqual(lookup_barcode(observation, self.catalog).status, LookupStatus.UNRESOLVED)

    def test_two_barcodes_naming_different_records_conflict(self):
        advil = lookup_barcode(upc("305730150200"), self.catalog)
        other = lookup_barcode(upc("0845717010566", "EAN_13"), self.catalog)
        unknown = lookup_barcode(upc("012345678905"), self.catalog)
        self.assertTrue(conflicting([advil, other]))
        self.assertFalse(conflicting([advil, advil, unknown]))


class SyntheticLookupTests(unittest.TestCase):
    """Small hand-built catalogs for cases the fixture doesn't contain."""

    UPC = "0305730150200"  # carries 0573-0150-20

    def catalog(self, *records):
        return ProductCatalog.from_openfda_results(list(records))

    def test_label_and_embedded_ndc_disagree(self):
        # Mirrors the real cache: an Advil label lists a UPC whose digits are another record's package
        catalog = self.catalog(raw("0573-0161", ["0573-0161-35"], [self.UPC]),
                               raw("0573-0150", ["0573-0150-20"]))
        lookup = lookup_barcode(upc(self.UPC, "EAN_13"), catalog)
        self.assertEqual(lookup.status, LookupStatus.CONFLICT)
        self.assertIsNone(lookup.record)
        self.assertEqual([r["product_ndc"] for r in lookup.label_records], ["0573-0161"])
        self.assertEqual([p["package_ndc"] for _, p in lookup.ndc_matches], ["0573-0150-20"])

    def test_embedded_ndc_without_label_listing_is_only_a_candidate(self):
        lookup = lookup_barcode(upc(self.UPC, "EAN_13"), self.catalog(raw("0573-0150", ["0573-0150-20"])))
        self.assertEqual(lookup.status, LookupStatus.CANDIDATE)
        self.assertEqual(lookup.package["package_ndc"], "0573-0150-20")

    def test_upc_shared_by_several_records_is_ambiguous(self):
        shared = "0845717010566"
        catalog = self.catalog(raw("11111-111", ["11111-111-01"], [shared]),
                               raw("22222-222", ["22222-222-01"], [shared], ingredient="ACETAMINOPHEN"))
        lookup = lookup_barcode(upc(shared, "EAN_13"), catalog)
        self.assertEqual(lookup.status, LookupStatus.AMBIGUOUS)
        self.assertIsNone(lookup.record)
        self.assertEqual(len(lookup.label_records), 2)

    def test_digits_matching_two_package_ndcs_are_ambiguous(self):
        # 0573015020 reads as 0573-0150-20 (4-4-2) or 05730-150-20 (5-3-2)
        catalog = self.catalog(raw("0573-0150", ["0573-0150-20"]), raw("05730-150", ["05730-150-20"]))
        self.assertEqual(lookup_barcode(upc(self.UPC, "EAN_13"), catalog).status, LookupStatus.AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
