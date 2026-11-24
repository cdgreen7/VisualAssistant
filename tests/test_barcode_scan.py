"""The 'b' key helper: clean frame -> barcodes -> FDA + registry lookup -> printed text."""
import json
import os
import unittest

import numpy as np
import zxingcpp

from vision.barcode_scan import NO_BARCODE, format_scan, scan_frame
from tests.test_upc_registry import entry, pending
from vision.products.barcode_identify import FinalStatus
from vision.products.catalog import ProductCatalog
from vision.products.upc_registry import UpcRegistry

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")
F = zxingcpp.BarcodeFormat


def frame_with(*codes):
    """White BGR frame with the given (format, text) barcodes stacked on it."""
    symbols = [np.array(zxingcpp.write_barcode(fmt, text, 300, 120, quiet_zone=20)) for fmt, text in codes]
    frame = np.full((sum(s.shape[0] + 40 for s in symbols) + 40, 480, 3), 255, np.uint8)
    y = 20
    for code in symbols:
        frame[y:y + code.shape[0], 40:40 + code.shape[1]] = code[:, :, None]
        y += code.shape[0] + 40
    return frame


class BarcodeScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.catalog = ProductCatalog.from_openfda_results(json.load(f)["results"])
        cls.registry = UpcRegistry([
            pending("681131699372"),
            entry(upc="012345678905", product_name="Store Brand Ibuprofen Tablets", brand="Store Brand",
                  package_ndc=None, package_description=None)])

    def scan(self, *codes):
        results = scan_frame(frame_with(*codes), self.catalog, self.registry)
        return results, format_scan(results)

    def statuses(self, results):
        return [r.final_status for r in results]

    def test_no_barcode_message(self):
        results, text = self.scan()
        self.assertEqual(results, [])
        self.assertEqual(text, "No barcode detected. Move closer, improve lighting, or rotate the package.")
        self.assertEqual(text, NO_BARCODE)

    def test_fda_resolved_upc_prints_product_and_package(self):
        results, text = self.scan((F.UPCA, "305730150200"))
        self.assertEqual(self.statuses(results), [FinalStatus.IDENTIFIED_FROM_FDA])
        for expected in ("--- BARCODE IDENTIFICATION ---", "Type: UPC-A", "Value: 305730150200",
                         "FDA lookup:\nRESOLVED", "Local registry:\nNOT_FOUND", "Product:\nADVIL",
                         "Ingredients:\nIBUPROFEN 200 mg/1", "product NDC 0573-0150",
                         "Package NDC:\n0573-0150-20 (from FDA)", "Source:\nFDA",
                         "Final status:\nIDENTIFIED_FROM_FDA"):
            self.assertIn(expected, text)

    def test_pending_registry_entry_is_unresolved(self):
        results, text = self.scan((F.UPCA, "681131699372"))
        self.assertEqual(self.statuses(results), [FinalStatus.UNRESOLVED])
        for expected in ("Value: 681131699372", "FDA lookup:\nUNRESOLVED", "Local registry:\nPENDING",
                         "Product:\nNot identified", "Ingredients:\nNot established", "Source:\nNone",
                         "Final status:\nUNRESOLVED"):
            self.assertIn(expected, text)

    def test_registry_only_identification_shows_provenance(self):
        results, text = self.scan((F.UPCA, "012345678905"))
        self.assertEqual(self.statuses(results), [FinalStatus.IDENTIFIED_FROM_REGISTRY])
        for expected in ("FDA lookup:\nUNRESOLVED", "Local registry:\nVERIFIED (confirmed against the package",
                         "Product:\nStore Brand Ibuprofen Tablets", "Ingredients:\nIBUPROFEN 200 mg",
                         "Source:\nLocal Verified Registry - registry entry: Read from the carton (test data), "
                         "verified 2026-11-11", "Final status:\nIDENTIFIED_FROM_REGISTRY"):
            self.assertIn(expected, text)
        self.assertNotIn("Package NDC", text)

    def test_non_upc_and_unsupported_formats_do_not_crash(self):
        results, text = self.scan((F.Code128, "LOT-A123"), (F.QRCode, "hello"))
        self.assertEqual(self.statuses(results), [FinalStatus.UNRESOLVED] * 2)
        self.assertIn("Type: CODE-128\nValue: LOT-A123", text)
        self.assertIn("no established FDA mapping for CODE_128", text)
        self.assertIn("Value: (not decoded: unsupported format)", text)

    def test_final_status_is_never_the_matchers_verified(self):
        for codes in ((), ((F.UPCA, "305730150200"),), ((F.UPCA, "681131699372"),),
                      ((F.UPCA, "012345678905"),), ((F.Code128, "LOT-A123"),)):
            results, text = self.scan(*codes)
            self.assertNotIn("VERIFIED", [r.final_status.value for r in results])
            self.assertNotIn("Final status:\nVERIFIED", text)

    def test_two_barcodes_naming_different_records(self):
        results, text = self.scan((F.UPCA, "305730150200"), (F.EAN13, "0845717010566"))
        self.assertEqual(self.statuses(results), [FinalStatus.IDENTIFIED_FROM_FDA, FinalStatus.CANDIDATE])
        self.assertIn("Final status:\nCANDIDATE", text)
        self.assertIn("name different FDA records: CONFLICT", text)

    def test_scan_without_a_registry(self):
        results, text = scan_frame(frame_with((F.UPCA, "305730150200")), self.catalog), None
        self.assertEqual(self.statuses(results), [FinalStatus.IDENTIFIED_FROM_FDA])
        self.assertIn("Local registry:\nNOT_LOADED", format_scan(results))

    def test_scan_does_not_modify_the_frame(self):
        frame = frame_with((F.UPCA, "305730150200"))
        before = frame.copy()
        scan_frame(frame, self.catalog, self.registry)
        self.assertTrue(np.array_equal(frame, before))


if __name__ == "__main__":
    unittest.main()
