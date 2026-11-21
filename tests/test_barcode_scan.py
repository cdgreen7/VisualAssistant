"""The 'b' key helper: clean frame -> barcodes -> FDA lookup -> printed text."""
import json
import os
import unittest

import numpy as np
import zxingcpp

from vision.barcode_scan import NO_BARCODE, format_scan, scan_frame
from vision.products.barcode_lookup import LookupStatus
from vision.products.catalog import ProductCatalog

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

    def scan(self, *codes):
        lookups = scan_frame(frame_with(*codes), self.catalog)
        return lookups, format_scan(lookups)

    def test_no_barcode_message(self):
        lookups, text = self.scan()
        self.assertEqual(lookups, [])
        self.assertEqual(text, "No barcode detected. Move closer, improve lighting, or rotate the package.")
        self.assertEqual(text, NO_BARCODE)

    def test_resolved_upc_prints_product_and_package(self):
        lookups, text = self.scan((F.UPCA, "305730150200"))
        self.assertEqual([l.status for l in lookups], [LookupStatus.RESOLVED])
        for expected in ("Type: UPC_A", "Value: 305730150200", "ADVIL", "IBUPROFEN 200 mg/1",
                         "product NDC 0573-0150", "NDC 0573-0150-20", "Status: RESOLVED"):
            self.assertIn(expected, text)

    def test_unknown_upc_is_unresolved(self):
        lookups, text = self.scan((F.UPCA, "012345678905"))
        self.assertEqual([l.status for l in lookups], [LookupStatus.UNRESOLVED])
        self.assertIn("Value: 012345678905", text)
        self.assertIn("No reliable product mapping found.", text)
        self.assertIn("Status: UNRESOLVED", text)

    def test_non_upc_and_unsupported_formats_do_not_crash(self):
        lookups, text = self.scan((F.Code128, "LOT-A123"), (F.QRCode, "hello"))
        self.assertEqual([l.status for l in lookups], [LookupStatus.UNRESOLVED] * 2)
        self.assertIn("Value: LOT-A123", text)
        self.assertIn("no established FDA mapping for CODE_128", text)
        self.assertIn("Barcode found but not decoded", text)
        self.assertIn("unsupported format", text)

    def test_never_prints_verified(self):
        for codes in ((), ((F.UPCA, "305730150200"),), ((F.UPCA, "012345678905"),), ((F.Code128, "LOT-A123"),)):
            self.assertNotIn("VERIFIED", self.scan(*codes)[1])

    def test_two_barcodes_naming_different_records(self):
        lookups, text = self.scan((F.UPCA, "305730150200"), (F.EAN13, "0845717010566"))
        self.assertEqual([l.status for l in lookups], [LookupStatus.RESOLVED, LookupStatus.CANDIDATE])
        self.assertIn("Status: CANDIDATE", text)
        self.assertIn("name different FDA records: CONFLICT", text)

    def test_scan_does_not_modify_the_frame(self):
        frame = frame_with((F.UPCA, "305730150200"))
        before = frame.copy()
        scan_frame(frame, self.catalog)
        self.assertTrue(np.array_equal(frame, before))


if __name__ == "__main__":
    unittest.main()
