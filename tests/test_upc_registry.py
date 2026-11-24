"""Local UPC registry: validation and lookup (no FDA catalog, no images)."""
import json
import os
import tempfile
import unittest

from vision.products.upc_registry import DEFAULT_PATH, RegistryError, UpcRegistry, validate_entry

UPC = "305730150200"


def entry(**changes):
    base = {"upc": UPC, "product_name": "Advil Tablets", "brand": "Advil",
            "ingredients": [{"name": "IBUPROFEN", "strength": "200 mg"}],
            "package_description": "24 coated tablets", "package_ndc": "0573-0150-20",
            "source": "Read from the carton (test data)", "verification_status": "VERIFIED",
            "date_verified": "2026-11-11"}
    base.update(changes)
    return base


def pending(upc=UPC):
    return {"upc": upc, "product_name": None, "brand": None, "ingredients": [], "package_description": None,
            "package_ndc": None, "source": None, "verification_status": "PENDING", "date_verified": None}


class ValidationTests(unittest.TestCase):
    def assertInvalid(self, bad, message):
        with self.assertRaises(RegistryError) as raised:
            validate_entry(bad)
        self.assertIn(message, str(raised.exception))

    def test_valid_entries(self):
        validate_entry(entry())
        validate_entry(entry(package_ndc=None, package_description=None))
        validate_entry(pending())

    def test_invalid_check_digit(self):
        self.assertInvalid(entry(upc="305730150201"), "invalid check digit")

    def test_missing_or_malformed_upc(self):
        without = entry()
        del without["upc"]
        for bad in (without, entry(upc=None), entry(upc=""), entry(upc=305730150200),
                    entry(upc="0305730150200"), entry(upc="30573015020"), entry(upc="30573-015020")):
            self.assertInvalid(bad, "exactly 12 digits")

    def test_status_must_be_pending_or_verified(self):
        for status in (None, "verified", "RESOLVED", "CONFIRMED"):
            self.assertInvalid(entry(verification_status=status), "PENDING or VERIFIED")

    def test_verified_entry_must_be_complete(self):
        for name in ("product_name", "brand", "source", "date_verified"):
            self.assertInvalid(entry(**{name: None}), f"missing {name}")
        self.assertInvalid(entry(ingredients=[]), "missing ingredients")
        self.assertInvalid(dict(pending(), verification_status="VERIFIED"), "missing")

    def test_field_formats(self):
        self.assertInvalid(entry(ingredients=[{"name": "IBUPROFEN"}]), "name and a strength")
        self.assertInvalid(entry(ingredients=[{"name": "IBUPROFEN", "strength": ""}]), "name and a strength")
        self.assertInvalid(entry(brand="  "), "brand must be text or null")
        self.assertInvalid(entry(date_verified="11/11/2026"), "YYYY-MM-DD")
        self.assertInvalid(entry(package_ndc="0573015020"), "dashed package NDC")  # undashed is ambiguous
        self.assertInvalid(entry(package_ndc="573-150"), "dashed package NDC")
        self.assertInvalid(entry(notes="x"), "unknown fields")
        self.assertInvalid("305730150200", "not an object")


class RegistryTests(unittest.TestCase):
    def test_verified_entry_is_found_with_exact_upc(self):
        registry = UpcRegistry([entry()])
        self.assertEqual(registry.find(UPC)["upc"], UPC)
        self.assertEqual(registry.find_verified(UPC)["brand"], "Advil")
        self.assertEqual(registry.find("0" + UPC)["upc"], UPC)  # same number read as EAN-13
        self.assertIsNone(registry.find("012345678905"))
        self.assertIsNone(registry.find("305730150201"))  # bad check digit
        self.assertIsNone(registry.find("LOT-A123"))

    def test_pending_entry_is_never_returned_as_verified(self):
        registry = UpcRegistry([pending()])
        self.assertEqual(registry.find(UPC)["verification_status"], "PENDING")
        self.assertIsNone(registry.find_verified(UPC))

    def test_conflicting_duplicates_are_rejected(self):
        with self.assertRaises(RegistryError) as raised:
            UpcRegistry([entry(), entry(brand="Motrin")])
        self.assertIn("conflicting duplicate", str(raised.exception))
        with self.assertRaises(RegistryError):
            UpcRegistry([entry(), pending()])
        self.assertEqual(len(UpcRegistry([entry(), entry()])), 1)  # identical repeat is harmless

    def test_one_invalid_entry_rejects_the_whole_registry(self):
        with self.assertRaises(RegistryError):
            UpcRegistry([entry(), entry(upc="012345678900")])


class RegistryFileTests(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "registry.json")
            with open(path, "w") as f:
                f.write(text)
            return UpcRegistry.from_file(path)

    def test_loads_entries(self):
        self.assertEqual(len(self.load(json.dumps({"entries": [entry()]}))), 1)
        self.assertEqual(len(self.load('{"entries": []}')), 0)

    def test_missing_file_is_an_empty_registry(self):
        self.assertEqual(len(UpcRegistry.from_file("/nonexistent/registry.json")), 0)

    def test_malformed_json(self):
        for text in ("", "{not json", '{"entries": [', "[]", '{"entries": {}}', '{"upcs": []}', '{"entries": ["x"]}'):
            with self.assertRaises(RegistryError, msg=text):
                self.load(text)

    def test_shipped_registry_is_valid_and_has_no_unconfirmed_identity(self):
        registry = UpcRegistry.from_file(DEFAULT_PATH)
        scanned = registry.find("681131699372")
        self.assertEqual(scanned["verification_status"], "PENDING")
        self.assertEqual((scanned["brand"], scanned["product_name"], scanned["ingredients"], scanned["package_ndc"]),
                         (None, None, [], None))
        self.assertIsNone(registry.find_verified("681131699372"))


if __name__ == "__main__":
    unittest.main()
