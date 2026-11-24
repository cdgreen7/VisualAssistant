"""Decode the barcodes in an image and look them up in the FDA catalog.

Usage (from the repo root):
    python -m vision.barcode_demo path/to/barcode.jpg

Development tool only. A decoded barcode is never reported as VERIFIED.
"""
import argparse
import os
import sys

import cv2

from vision.barcode_scan import NO_BARCODE, format_scan, scan_frame
from vision.products.catalog import ProductCatalog

DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "products", "data", "otc_ndc_2026-10-30.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Couldn't read image: {args.image}")
    if not os.path.exists(args.cache):
        sys.exit(f"FDA cache not found ({args.cache}). Build it with: python -m vision.products.build_cache")
    lookups = scan_frame(image, ProductCatalog.from_file(args.cache))
    print(format_scan(lookups) if lookups else NO_BARCODE + "\n\nStatus: UNRESOLVED")


if __name__ == "__main__":
    main()
