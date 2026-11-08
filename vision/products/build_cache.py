"""Build a dated local cache of OTC records from the openFDA NDC bulk download.

Usage (from the repo root):
    python -m vision.products.build_cache                 # download and build
    python -m vision.products.build_cache --zip ndc.zip   # use a downloaded zip

Writes vision/products/data/otc_ndc_<last_updated>.json (gitignored).
"""
import argparse
import io
import json
import os
import re
import urllib.request
import zipfile
from collections import Counter

from vision.products.catalog import ProductCatalog, normalize_record
from vision.products.ndc import PACKAGE_FORMATS, PRODUCT_FORMATS

BULK_URL = "https://download.open.fda.gov/drug/ndc/drug-ndc-0001-of-0001.json.zip"
PRODUCT_TYPE = "HUMAN OTC DRUG"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _layout(ndc):
    if not re.fullmatch(r"\d+(-\d+)+", ndc or ""):
        return None
    return tuple(len(part) for part in ndc.split("-"))


def check_assumptions(raw_records):
    """Fail loudly if the data breaks an assumption the catalog relies on."""
    problems = []
    product_ids = Counter(r.get("product_id") for r in raw_records)
    if product_ids.get(None):
        problems.append(f"{product_ids[None]} records have no product_id")
    duplicates = [k for k, v in product_ids.items() if k and v > 1]
    if duplicates:
        problems.append(f"{len(duplicates)} duplicate product_ids, e.g. {duplicates[:3]}")

    bad_products = [r.get("product_ndc") for r in raw_records if _layout(r.get("product_ndc")) not in PRODUCT_FORMATS]
    if bad_products:
        problems.append(f"{len(bad_products)} product NDCs in an unexpected format, e.g. {bad_products[:3]}")

    bad_packages, unprefixed = [], []
    for record in raw_records:
        for package in record.get("packaging") or []:
            ndc = package.get("package_ndc")
            if _layout(ndc) not in PACKAGE_FORMATS:
                bad_packages.append(ndc)
            elif not ndc.startswith(record.get("product_ndc", "") + "-"):
                unprefixed.append(ndc)
    if bad_packages:
        problems.append(f"{len(bad_packages)} package NDCs in an unexpected format, e.g. {bad_packages[:3]}")
    if unprefixed:
        problems.append(f"{len(unprefixed)} package NDCs don't start with their product NDC, e.g. {unprefixed[:3]}")
    return problems


def load_bulk(zip_path=None):
    if zip_path:
        archive = zipfile.ZipFile(zip_path)
    else:
        print(f"Downloading {BULK_URL} ...")
        with urllib.request.urlopen(BULK_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
    return json.loads(archive.read(archive.namelist()[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", help="use an already downloaded openFDA NDC zip")
    parser.add_argument("--out-dir", default=DATA_DIR)
    args = parser.parse_args()

    bulk = load_bulk(args.zip)
    last_updated = bulk["meta"]["last_updated"]
    raw = [r for r in bulk["results"] if r.get("product_type") == PRODUCT_TYPE]
    print(f"openFDA export {last_updated}: {len(bulk['results'])} records, {len(raw)} {PRODUCT_TYPE}")

    problems = check_assumptions(raw)
    if problems:
        raise SystemExit("Data contradicts catalog assumptions:\n  " + "\n  ".join(problems))

    records = [normalize_record(r) for r in raw]
    ProductCatalog(records)  # re-check product_id uniqueness on the normalized records

    duplicate_packages = sum(len(r.get("packaging") or []) for r in raw) - sum(len(r["packages"]) for r in records)
    meta = {
        "source": BULK_URL,
        "last_updated": last_updated,
        "product_type": PRODUCT_TYPE,
        "record_count": len(records),
        "package_count": sum(len(r["packages"]) for r in records),
        "duplicate_package_entries_removed": duplicate_packages,
        "disclaimer": bulk["meta"].get("disclaimer"),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"otc_ndc_{last_updated}.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "records": records}, f)
    print(f"Wrote {len(records)} records ({meta['package_count']} packages, "
          f"{duplicate_packages} duplicate package entries removed) to {path}")


if __name__ == "__main__":
    main()
