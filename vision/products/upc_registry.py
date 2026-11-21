"""Local registry of manually confirmed UPC -> product mappings.

The FDA catalog lists a UPC for only a minority of products, so a person can
record what is printed on a package they have checked. Each entry:

    {
      "upc": "681131699372",            exact 12-digit UPC-A, as scanned
      "product_name": null,             as printed on the package
      "brand": null,
      "ingredients": [],                [{"name": "IBUPROFEN", "strength": "200 mg"}]
      "package_description": null,      optional, e.g. "500 coated tablets"
      "package_ndc": null,              optional; only if read from the package itself
      "source": null,                   who/what confirmed it, e.g. "Read from bottle by R.J."
      "verification_status": "PENDING", PENDING or VERIFIED
      "date_verified": null             YYYY-MM-DD, required when VERIFIED
    }

VERIFIED here means "a person confirmed this entry against the package". It
is not ProductMatcher's VERIFIED decision. PENDING entries are placeholders
and are never used to identify a product.

Loading is strict: any invalid entry makes the whole file fail (RegistryError),
so a bad edit can't be half-applied.
"""
import datetime
import json
import os
import re

from vision.products.ndc import normalize_package_ndc, normalize_upc

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "verified_upcs.json")

PENDING, VERIFIED = "PENDING", "VERIFIED"
FIELDS = ("upc", "product_name", "brand", "ingredients", "package_description", "package_ndc",
          "source", "verification_status", "date_verified")
REQUIRED_WHEN_VERIFIED = ("product_name", "brand", "source", "date_verified")


class RegistryError(ValueError):
    """The registry file or one of its entries is invalid."""


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_entry(entry):
    """Raise RegistryError unless entry is a well-formed registry entry."""
    if not isinstance(entry, dict):
        raise RegistryError(f"entry is not an object: {entry!r}")
    upc = entry.get("upc")
    where = f"UPC {upc!r}"
    if not isinstance(upc, str) or not re.fullmatch(r"\d{12}", upc):
        raise RegistryError(f"{where}: a UPC-A must be a string of exactly 12 digits")
    if normalize_upc(upc) is None:
        raise RegistryError(f"{where}: invalid check digit")

    unknown = set(entry) - set(FIELDS)
    if unknown:
        raise RegistryError(f"{where}: unknown fields {sorted(unknown)}")
    status = entry.get("verification_status")
    if status not in (PENDING, VERIFIED):
        raise RegistryError(f"{where}: verification_status must be PENDING or VERIFIED")

    ingredients = entry.get("ingredients", [])
    if not isinstance(ingredients, list):
        raise RegistryError(f"{where}: ingredients must be a list")
    for ingredient in ingredients:
        if not (isinstance(ingredient, dict) and set(ingredient) == {"name", "strength"}
                and _text(ingredient["name"]) and _text(ingredient["strength"])):
            raise RegistryError(f"{where}: each ingredient needs a name and a strength")
    for name in ("product_name", "brand", "package_description", "package_ndc", "source", "date_verified"):
        if entry.get(name) is not None and not _text(entry[name]):
            raise RegistryError(f"{where}: {name} must be text or null")

    package_ndc = entry.get("package_ndc")
    if package_ndc is not None and not ("-" in package_ndc and normalize_package_ndc(package_ndc)):
        raise RegistryError(f"{where}: package_ndc must be a dashed package NDC such as 0573-0150-20")
    if entry.get("date_verified") is not None:
        try:
            datetime.date.fromisoformat(entry["date_verified"])
        except ValueError:
            raise RegistryError(f"{where}: date_verified must be YYYY-MM-DD") from None

    if status == VERIFIED:
        missing = [name for name in REQUIRED_WHEN_VERIFIED if entry.get(name) is None]
        if not ingredients:
            missing.append("ingredients")
        if missing:
            raise RegistryError(f"{where}: VERIFIED entry is missing {', '.join(missing)}")


class UpcRegistry:
    def __init__(self, entries=()):
        self._by_upc = {}
        for entry in entries:
            validate_entry(entry)
            existing = self._by_upc.setdefault(entry["upc"], entry)
            if existing != entry:
                raise RegistryError(f"UPC {entry['upc']!r}: conflicting duplicate entries")

    @classmethod
    def from_file(cls, path=DEFAULT_PATH):
        """Registry from a JSON file; an absent file is an empty registry."""
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError as error:
            raise RegistryError(f"{path} is not valid JSON: {error}") from None
        if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
            raise RegistryError(f'{path} must be an object with an "entries" list')
        return cls(data["entries"])

    def __len__(self):
        return len(self._by_upc)

    def find(self, value):
        """Entry (PENDING or VERIFIED) for a scanned UPC-A / EAN-13 value, or None.

        A 13-digit EAN starting with 0 is the same number as the 12-digit UPC-A.
        """
        key = normalize_upc(value)
        if key is None or not key.startswith("0"):
            return None
        return self._by_upc.get(key[1:])

    def find_verified(self, value):
        """Only entries a person has confirmed; PENDING entries are never returned."""
        entry = self.find(value)
        return entry if entry and entry["verification_status"] == VERIFIED else None
