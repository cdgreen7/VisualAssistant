"""Searchable product knowledge built from FDA NDC Directory (openFDA) records.

This module only represents and searches product data. It has no OCR logic
and never merges records: each openFDA record (keyed by product_id) stays a
separate product, and ambiguous lookups return every match.
"""
import json
import re
from collections import defaultdict

from vision.products.ndc import (
    ndc_candidates_from_upc,
    normalize_package_ndc,
    normalize_product_ndc,
    normalize_upc,
    package_ndc_candidates,
    product_ndc_candidates,
)

# The only fields kept from openFDA. NDC records carry no dosage or usage
# directions, and nothing outside this list is copied.
RECORD_FIELDS = (
    "product_id", "product_ndc", "product_ndc_norm", "brand_name", "generic_name",
    "labeler_name", "product_type", "dosage_form", "active_ingredients",
    "packages", "upc", "spl_set_id",
)
PACKAGE_FIELDS = ("package_ndc", "package_ndc_norm", "description")


def name_key(name):
    """Case/punctuation-insensitive form for exact name comparison."""
    return " ".join(re.sub(r"[^A-Z0-9]+", " ", str(name or "").upper()).split())


def strength_key(strength):
    """'200 mg/1' and '200 mg' compare equal; other units are left as-is."""
    key = re.sub(r"\s+", "", str(strength or "").lower())
    return key[:-2] if key.endswith("/1") else key


def normalize_record(raw):
    """Raw openFDA NDC record -> record containing only RECORD_FIELDS."""
    openfda = raw.get("openfda") or {}

    packages, seen = [], {}
    for package in raw.get("packaging") or []:
        ndc = package.get("package_ndc")
        description = package.get("description")
        if ndc in seen:
            if seen[ndc] != description:
                raise ValueError(f"{raw.get('product_id')}: package {ndc} listed with different descriptions")
            continue  # identical duplicate entry
        seen[ndc] = description
        packages.append({
            "package_ndc": ndc,
            "package_ndc_norm": normalize_package_ndc(ndc),
            "description": description,
        })

    return {
        "product_id": raw.get("product_id"),
        "product_ndc": raw.get("product_ndc"),
        "product_ndc_norm": normalize_product_ndc(raw.get("product_ndc")),
        "brand_name": raw.get("brand_name"),
        "generic_name": raw.get("generic_name"),
        "labeler_name": raw.get("labeler_name"),
        "product_type": raw.get("product_type"),
        "dosage_form": raw.get("dosage_form"),
        "active_ingredients": [
            {"name": ingredient.get("name"), "strength": ingredient.get("strength")}
            for ingredient in raw.get("active_ingredients") or []
        ],
        "packages": packages,
        "upc": list(openfda.get("upc") or []),
        "spl_set_id": list(openfda.get("spl_set_id") or []),
    }


class ProductCatalog:
    def __init__(self, records, meta=None):
        self.records = list(records)
        self.meta = meta or {}

        self._by_product_id = {}
        self._by_product_ndc = defaultdict(list)
        self._by_package_ndc = defaultdict(list)
        self._by_upc = defaultdict(list)
        self._by_brand = defaultdict(list)
        self._by_ingredient = defaultdict(list)

        for record in self.records:
            product_id = record["product_id"]
            if product_id in self._by_product_id:
                raise ValueError(f"duplicate product_id {product_id}")
            self._by_product_id[product_id] = record

            if record["product_ndc_norm"]:
                self._by_product_ndc[record["product_ndc_norm"]].append(record)
            for package in record["packages"]:
                if package["package_ndc_norm"]:
                    self._by_package_ndc[package["package_ndc_norm"]].append((record, package))
            for key in {normalize_upc(upc) for upc in record["upc"]} - {None}:
                self._by_upc[key].append(record)
            if record["brand_name"]:
                self._by_brand[name_key(record["brand_name"])].append(record)
            for key in {name_key(i["name"]) for i in record["active_ingredients"]}:
                self._by_ingredient[key].append(record)

    @classmethod
    def from_openfda_results(cls, results, meta=None):
        return cls([normalize_record(raw) for raw in results], meta)

    @classmethod
    def from_file(cls, path):
        """Load a cache written by build_cache.py."""
        with open(path) as f:
            data = json.load(f)
        return cls(data["records"], data.get("meta"))

    def __len__(self):
        return len(self.records)

    def get_by_product_id(self, product_id):
        return self._by_product_id.get(product_id)

    def find_by_product_ndc(self, ndc):
        """All records for a product NDC (it is not unique across records)."""
        matches = []
        for candidate in product_ndc_candidates(ndc):
            matches.extend(self._by_product_ndc.get(candidate, []))
        return matches

    def find_by_package_ndc(self, ndc):
        """All (record, package) pairs matching any reading of the NDC."""
        matches = []
        for candidate in package_ndc_candidates(ndc):
            matches.extend(self._by_package_ndc.get(candidate, []))
        return matches

    def find_by_upc(self, upc):
        """Records whose openFDA label lists this UPC.

        openFDA attaches UPCs to the whole label (SPL), so several products
        can share one UPC. Treat this as supporting evidence only.
        """
        key = normalize_upc(upc)
        return list(self._by_upc.get(key, [])) if key else []

    def find_by_upc_ndc(self, upc):
        """(record, package) pairs for NDC candidates decoded from a UPC.

        Returns [] when the UPC can't carry an NDC or no candidate matches a
        known package; several pairs when more than one candidate matches.
        """
        matches = []
        for candidate in ndc_candidates_from_upc(upc):
            matches.extend(self._by_package_ndc.get(candidate, []))
        return matches

    def find_by_brand(self, name):
        """Exact brand match ignoring case/punctuation ('Advil' != 'Advil PM')."""
        return list(self._by_brand.get(name_key(name), []))

    def find_by_ingredient(self, name, strength=None):
        """Records containing the ingredient (optionally at that strength)."""
        key = name_key(name)
        matches = self._by_ingredient.get(key, [])
        if strength is None:
            return list(matches)
        wanted = strength_key(strength)
        return [
            record for record in matches
            if any(name_key(i["name"]) == key and strength_key(i["strength"]) == wanted
                   for i in record["active_ingredients"])
        ]
