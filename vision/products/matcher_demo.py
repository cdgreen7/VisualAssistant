"""Run a real image through EasyOCR and the ProductMatcher.

Usage (from the repo root):
    python -m vision.products.matcher_demo path/to/medicine.jpg

Development tool only: prints the OCR observations, then the match result.
It does not give dosage or medical advice.
"""
import argparse
import os
import sys

import cv2

from vision.ocr import TextRecognizer
from vision.products.catalog import ProductCatalog
from vision.products.matcher import ProductMatcher, Status

DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "data", "otc_ndc_2026-10-30.json")
MAX_CANDIDATES = 5


def describe_candidate(candidate):
    ingredients = " + ".join(f"{name} {strength}" for name, strength in candidate.ingredients) or "(none listed)"
    labelers = {r["labeler_name"] for r in candidate.records}
    brands = candidate.brand_names
    shown = ", ".join(brands[:4]) + (f", ... ({len(brands)} brands)" if len(brands) > 4 else "")
    return (f"{ingredients}\n"
            f"      {len(candidate.records)} FDA records, {len(labelers)} companies; brands: {shown}")


def print_result(result):
    print("\nMatch result")
    print(f"  Status:     {result.status.value}")
    print(f"  Level:      {result.level.value}")
    print(f"  Reason:     {result.reason}")
    print(f"  Confidence: {result.confidence:.2f} (lowest OCR confidence relied on; not a probability)")

    if result.evidence:
        print("  Evidence:   " + ", ".join(f"{e.kind}={e.value!r} ({e.confidence:.2f})" for e in result.evidence))
    for conflict in result.conflicts:
        print(f"  Conflict:   {conflict.reason}")
    if result.ignored:
        print(f"  Ignored:    {len(result.ignored)} OCR results below the matcher's minimum confidence")

    if result.status == Status.NO_MATCH:
        print("\n  No supported product identity was found in the OCR text.")
        return

    if result.status == Status.VERIFIED:
        print("\n  Identified formulation:")
        print("    " + describe_candidate(result.candidates[0]))
        if result.package:
            print(f"  Identified package: NDC {result.package['package_ndc']} - {result.package['description']}")
        return

    print(f"\n  {len(result.candidates)} formulation candidates (showing up to {MAX_CANDIDATES}):")
    for i, candidate in enumerate(result.candidates[:MAX_CANDIDATES], 1):
        print(f"   {i}. {describe_candidate(candidate)}")
        missing = candidate.missing_ingredients + [f"{n} (salt not read)" for n in candidate.base_only_ingredients]
        if missing:
            print(f"      not observed: {', '.join(missing)}")
        if candidate.missing_strengths:
            print(f"      strengths not observed: {', '.join(candidate.missing_strengths)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="path to an image of medicine packaging")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="FDA OTC cache built by build_cache.py")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"Couldn't read image: {args.image}")
    if not os.path.exists(args.cache):
        sys.exit(f"FDA cache not found: {args.cache}\nBuild it with: python -m vision.products.build_cache")

    ocr_results = TextRecognizer().read(image)
    print(f"OCR observations ({len(ocr_results)}) from {args.image} {image.shape[1]}x{image.shape[0]}:")
    for i, r in enumerate(ocr_results):
        print(f"  [{i}] {r['text']!r}  confidence={r['confidence']:.2f}  box={r['box']}")

    matcher = ProductMatcher(ProductCatalog.from_file(args.cache))
    print_result(matcher.match(ocr_results))


if __name__ == "__main__":
    main()
