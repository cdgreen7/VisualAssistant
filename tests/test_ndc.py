import unittest

from vision.products.ndc import (
    ndc_candidates_from_upc,
    normalize_package_ndc,
    normalize_product_ndc,
    normalize_upc,
    package_ndc_candidates,
    product_ndc_candidates,
)


class PackageNdcTests(unittest.TestCase):
    def test_equivalent_forms_normalize_to_same_code(self):
        for value in ("0573-0150-20", "00573-0150-20", "00573015020", "NDC 0573-0150-20", " ndc: 0573-0150-20 "):
            self.assertEqual(normalize_package_ndc(value), "00573015020", value)

    def test_each_dashed_layout(self):
        self.assertEqual(normalize_package_ndc("0573-0150-20"), "00573015020")   # 4-4-2
        self.assertEqual(normalize_package_ndc("71821-010-12"), "71821001012")   # 5-3-2
        self.assertEqual(normalize_package_ndc("75053-0100-1"), "75053010001")   # 5-4-1

    def test_undashed_ten_digits_is_ambiguous(self):
        self.assertIsNone(normalize_package_ndc("0573015020"))
        self.assertEqual(package_ndc_candidates("7505301001"),
                         ["07505301001", "75053001001", "75053010001"])

    def test_invalid_inputs(self):
        for value in (None, "", "abc", "0573-0150", "123-45-6", "05730150201234", "0573-01a0-20"):
            self.assertIsNone(normalize_package_ndc(value), value)
            self.assertEqual(package_ndc_candidates(value), [], value)

    def test_unambiguous_input_has_one_candidate(self):
        self.assertEqual(package_ndc_candidates("0573-0150-20"), ["00573015020"])
        self.assertEqual(package_ndc_candidates("00573015020"), ["00573015020"])


class ProductNdcTests(unittest.TestCase):
    def test_forms(self):
        self.assertEqual(normalize_product_ndc("0573-0150"), "005730150")
        self.assertEqual(normalize_product_ndc("71821-010"), "718210010")
        self.assertEqual(normalize_product_ndc("53209-2001"), "532092001")
        self.assertEqual(normalize_product_ndc("005730150"), "005730150")

    def test_undashed_eight_digits_is_ambiguous(self):
        self.assertIsNone(normalize_product_ndc("05730150"))
        self.assertEqual(product_ndc_candidates("05730150"), ["005730150", "057300150"])


class UpcTests(unittest.TestCase):
    def test_normalize_upc(self):
        self.assertEqual(normalize_upc("305730150200"), "0305730150200")    # UPC-A
        self.assertEqual(normalize_upc("0305730150200"), "0305730150200")   # EAN-13 form
        self.assertIsNone(normalize_upc("305730150201"))                    # bad check digit
        self.assertIsNone(normalize_upc("30573015020"))                     # too short
        self.assertIsNone(normalize_upc(None))

    def test_advil_upc_candidates(self):
        candidates = ndc_candidates_from_upc("0305730150200")
        self.assertIn("00573015020", candidates)  # 0573-0150-20
        self.assertEqual(candidates, package_ndc_candidates("0573015020"))

    def test_upc_without_ndc_gives_no_candidates(self):
        self.assertEqual(ndc_candidates_from_upc("0845717010566"), [])  # number system 8
        self.assertEqual(ndc_candidates_from_upc("305730150201"), [])   # bad check digit
        self.assertEqual(ndc_candidates_from_upc("not a upc"), [])


if __name__ == "__main__":
    unittest.main()
