"""Reconciling the FDA barcode lookup with the local UPC registry (no images)."""
import json
import os
import unittest

from tests.test_barcode_lookup import raw, upc
from tests.test_upc_registry import entry, pending
from vision.products.barcode_identify import FinalStatus, identify_barcode
from vision.products.barcode_lookup import LookupStatus
from vision.products.catalog import ProductCatalog
from vision.products.upc_registry import UpcRegistry

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")
ADVIL_UPC = "305730150200"    # FDA: RESOLVED to ADVIL 0573-0150-20
STORE_UPC = "681131699372"    # FDA: UNRESOLVED
STORE = dict(upc=STORE_UPC, product_name="Store Brand Ibuprofen Tablets", brand="Store Brand",
             package_ndc=None, package_description=None)


class IdentifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.catalog = ProductCatalog.from_openfda_results(json.load(f)["results"])

    def identify(self, value, *entries, catalog=None, kind="UPC_A"):
        return identify_barcode(upc(value, kind), catalog or self.catalog, UpcRegistry(entries))

    def test_no_matching_data(self):
        result = self.identify(STORE_UPC)
        self.assertEqual((result.final_status, result.registry_status), (FinalStatus.UNRESOLVED, "NOT_FOUND"))
        self.assertEqual((result.product_name, result.ingredients, result.source, result.package_ndc),
                         (None, [], None, None))

    def test_pending_entry_is_not_an_identification(self):
        result = self.identify(STORE_UPC, pending(STORE_UPC))
        self.assertEqual((result.final_status, result.registry_status), (FinalStatus.UNRESOLVED, "PENDING"))
        self.assertEqual((result.product_name, result.ingredients, result.source), (None, [], None))
        self.assertIsNone(result.registry_entry)

    def test_registry_only_identification(self):
        result = self.identify(STORE_UPC, entry(**STORE))
        self.assertEqual(result.final_status, FinalStatus.IDENTIFIED_FROM_REGISTRY)
        self.assertEqual(result.fda.status, LookupStatus.UNRESOLVED)
        self.assertEqual((result.registry_status, result.source), ("VERIFIED", "Local Verified Registry"))
        self.assertEqual(result.product_name, "Store Brand Ibuprofen Tablets")
        self.assertEqual(result.ingredients, [("IBUPROFEN", "200 mg")])
        self.assertIsNone(result.package_ndc)  # no package is claimed without a confirmed NDC

    def test_fda_only_identification(self):
        result = self.identify(ADVIL_UPC)
        self.assertEqual((result.final_status, result.source), (FinalStatus.IDENTIFIED_FROM_FDA, "FDA"))
        self.assertEqual((result.product_name, result.package_ndc), ("ADVIL", "0573-0150-20"))

    def test_fda_and_registry_agree(self):
        for registry_ndc in ("0573-0150-20", None):
            result = self.identify(ADVIL_UPC, entry(package_ndc=registry_ndc))
            self.assertEqual((result.final_status, result.source), (FinalStatus.IDENTIFIED_FROM_BOTH, "Both"))
            self.assertEqual(result.package_ndc, "0573-0150-20")
        self.assertEqual(result.package_ndc_source, "FDA")

    def test_fda_and_registry_disagree_on_ingredients(self):
        for ingredients in ([{"name": "ACETAMINOPHEN", "strength": "500 mg"}],
                            [{"name": "IBUPROFEN", "strength": "100 mg"}],
                            [{"name": "IBUPROFEN", "strength": "200 mg"},
                             {"name": "DIPHENHYDRAMINE CITRATE", "strength": "38 mg"}]):
            result = self.identify(ADVIL_UPC, entry(ingredients=ingredients, package_ndc=None))
            self.assertEqual(result.final_status, FinalStatus.CONFLICT, ingredients)
            self.assertEqual((result.product_name, result.ingredients, result.source), (None, [], None))

    def test_fda_and_registry_disagree_on_package(self):
        result = self.identify(ADVIL_UPC, entry(package_ndc="0573-0150-30"))
        self.assertEqual(result.final_status, FinalStatus.CONFLICT)
        self.assertIn("different package NDCs", result.reason)

    def test_registry_package_ndc_of_another_formulation_conflicts(self):
        # FDA can't resolve the UPC, but the registry's NDC is a known package of a different drug
        catalog = ProductCatalog.from_openfda_results([raw("11111-111", ["11111-111-01"], ingredient="ACETAMINOPHEN")])
        result = self.identify(STORE_UPC, entry(**dict(STORE, package_ndc="11111-111-01")), catalog=catalog)
        self.assertEqual(result.final_status, FinalStatus.CONFLICT)

    def test_fda_candidate_without_registry_stays_a_candidate(self):
        result = self.identify("0845717010566", kind="EAN_13")
        self.assertEqual(result.final_status, FinalStatus.CANDIDATE)

    def test_multiple_fda_matches_are_ambiguous_and_never_picked(self):
        shared = "0" + STORE_UPC
        catalog = ProductCatalog.from_openfda_results([
            raw("11111-111", ["11111-111-01"], [shared]),
            raw("22222-222", ["22222-222-01"], [shared], ingredient="ACETAMINOPHEN")])
        result = self.identify(STORE_UPC, catalog=catalog)
        self.assertEqual(result.final_status, FinalStatus.AMBIGUOUS)
        self.assertIsNone(result.product_name)

        # A verified entry consistent with one of them identifies via the registry...
        result = self.identify(STORE_UPC, entry(**STORE), catalog=catalog)
        self.assertEqual((result.final_status, result.source),
                         (FinalStatus.IDENTIFIED_FROM_REGISTRY, "Local Verified Registry"))
        # ...and one consistent with none of them is a conflict
        other = entry(**dict(STORE, ingredients=[{"name": "NAPROXEN SODIUM", "strength": "220 mg"}]))
        self.assertEqual(self.identify(STORE_UPC, other, catalog=catalog).final_status, FinalStatus.CONFLICT)

    def test_fda_internal_conflict_is_not_overridden_by_the_registry(self):
        catalog = ProductCatalog.from_openfda_results([raw("0573-0161", ["0573-0161-35"], ["0" + ADVIL_UPC]),
                                                       raw("0573-0150", ["0573-0150-20"])])
        self.assertEqual(self.identify(ADVIL_UPC, entry(), catalog=catalog).final_status, FinalStatus.CONFLICT)

    def test_non_upc_barcodes_never_use_the_registry(self):
        result = self.identify(STORE_UPC, entry(**STORE), kind="CODE_128")
        self.assertEqual((result.final_status, result.registry_status), (FinalStatus.UNRESOLVED, "NOT_FOUND"))

    def test_registry_not_loaded(self):
        result = identify_barcode(upc(ADVIL_UPC), self.catalog, None)
        self.assertEqual((result.final_status, result.registry_status),
                         (FinalStatus.IDENTIFIED_FROM_FDA, "NOT_LOADED"))


if __name__ == "__main__":
    unittest.main()
