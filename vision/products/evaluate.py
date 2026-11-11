"""Evaluate OCR quality and matcher decisions on labeled real images.

Usage (from the repo root):
    python -m vision.products.evaluate
    python -m vision.products.evaluate --crop yolo --yolo-conf 0.1
    python -m vision.products.evaluate --preprocess none,gray,clahe,upscale2x,sharpen

Labels (evaluation/labels.json) describe what is physically on each package,
written by a person looking at the product - never copied from FDA matches.

OCR quality and matcher decisions are scored separately:
  ocr     - did EasyOCR's text contain the label's brand, ingredients, strengths
  matcher - status on the real OCR, whether the true formulation is still a
            candidate, and false VERIFIED results
  oracle  - the matcher run on the label's visible text, i.e. with perfect OCR

Only counts are printed (n is small); no accuracy percentages.
"""
import argparse
import json
import os
import time

import cv2

from vision.products.catalog import name_key, strength_key
from vision.products.matcher import Level, Status, ingredient_key
from vision.products.ndc import normalize_package_ndc

DEFAULT_LABELS = "evaluation/labels.json"
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "data", "otc_ndc_2026-10-30.json")


# Preprocessing variants (evaluation only; not used by the live loop)

def _clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _sharpen(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


PREPROCESSORS = {
    "none": lambda image: image,
    "gray": lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    "clahe": _clahe,
    "upscale2x": lambda image: cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
    "sharpen": _sharpen,
}


def blur_score(image):
    """Variance of the Laplacian; lower means blurrier."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Scoring

def formulation_key(ingredients):
    return frozenset((ingredient_key(name), strength_key(strength)) for name, strength in ingredients)


def score_ocr(label, ocr_results, matcher):
    """Which labeled facts appear in the OCR output (via the matcher's own extraction)."""
    words = set(name_key(" ".join(r["text"] for r in ocr_results)).split())
    evidence = matcher.match(ocr_results).evidence
    ingredients = {ingredient_key(e.value) for e in evidence if e.kind == "ingredient"}
    plain = {e.value for e in evidence if e.kind == "strength" and not e.corrected_from}
    corrected = {e.value for e in evidence if e.kind == "strength" and e.corrected_from}
    strengths = [strength_key(s) for _, s in label["ingredients"]]
    return {
        "brand": all(w in words for w in name_key(" ".join(label["brand_words"])).split()),
        "ingredients": all(ingredient_key(n) in ingredients for n, _ in label["ingredients"]),
        "strengths_plain": all(s in plain for s in strengths),
        "strengths_with_correction": all(s in plain | corrected for s in strengths),
    }


def score_match(label, result):
    truth = formulation_key(label["formulation"])
    keys = [formulation_key(c.ingredients) for c in result.candidates]
    rank = keys.index(truth) + 1 if truth in keys else None

    false_verified = False
    if result.status == Status.VERIFIED:
        false_verified = keys[0] != truth
        if result.level == Level.PACKAGE and label.get("package_ndc"):
            false_verified |= (normalize_package_ndc(result.package["package_ndc"])
                               != normalize_package_ndc(label["package_ndc"]))
    return {
        "status": result.status.value,
        "level": result.level.value,
        "formulations": len(result.candidates),
        "truth_rank": rank,
        "false_verified": false_verified,
        "package_unlabeled": result.level == Level.PACKAGE and not label.get("package_ndc"),
    }


def oracle_result(label, matcher):
    return matcher.match([{"text": line, "confidence": 1.0} for line in label["visible_text"]])


# Running

def region_for(image, crop, detector):
    if crop == "full":
        return image, None
    from vision.identification import crop_box, select_best_detection
    detection = select_best_detection(detector.get_detections(detector.detect(image)))
    if detection is None:
        return None, None
    return crop_box(image, detection["box"]), detection


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--crop", choices=("full", "yolo"), default="full")
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--preprocess", default="none", help="comma-separated: " + ",".join(PREPROCESSORS))
    args = parser.parse_args()

    from vision.ocr import TextRecognizer
    from vision.products.catalog import ProductCatalog
    from vision.products.matcher import ProductMatcher

    with open(args.labels) as f:
        labels = json.load(f)
    variants = list(dict.fromkeys(args.preprocess.split(",")))
    recognizer = TextRecognizer()
    matcher = ProductMatcher(ProductCatalog.from_file(args.cache))
    detector = None
    if args.crop == "yolo":
        from vision.detector import ObjectDetector
        detector = ObjectDetector(conf=args.yolo_conf)

    totals = {v: {"n": 0, "no_region": 0, "brand": 0, "ingredients": 0, "strengths_plain": 0,
                  "strengths_with_correction": 0, "truth_in_candidates": 0, "verified": 0,
                  "uncertain": 0, "no_match": 0, "false_verified": 0, "ocr_seconds": 0.0, "match_ms": 0.0}
              for v in variants}
    oracle = {"n": 0, "truth_in_candidates": 0, "verified": 0, "false_verified": 0}

    for label in labels:
        image = cv2.imread(label["image"])
        if image is None:
            print(f"! can't read {label['image']}")
            continue
        print(f"\n=== {label['image']} - {label['product']}")
        region, detection = region_for(image, args.crop, detector)

        o = score_match(label, oracle_result(label, matcher))
        oracle["n"] += 1
        oracle["truth_in_candidates"] += o["truth_rank"] is not None
        oracle["verified"] += o["status"] == "VERIFIED"
        oracle["false_verified"] += o["false_verified"]
        print(f"  oracle (perfect OCR): {o['status']}/{o['level']}, {o['formulations']} formulations, "
              f"truth rank {o['truth_rank']}")

        for variant in variants:
            t = totals[variant]
            t["n"] += 1
            if region is None:
                t["no_region"] += 1
                print(f"  [{variant}] no {args.crop} region (YOLO found no bottle at conf {args.yolo_conf})")
                continue
            processed = PREPROCESSORS[variant](region)
            start = time.time()
            ocr_results = recognizer.read(processed)
            ocr_seconds = time.time() - start
            start = time.time()
            result = matcher.match(ocr_results)
            match_ms = (time.time() - start) * 1000

            ocr = score_ocr(label, ocr_results, matcher)
            m = score_match(label, result)
            for key in ("brand", "ingredients", "strengths_plain", "strengths_with_correction", "false_verified"):
                t[key] += bool(ocr.get(key, m.get(key)))
            t["truth_in_candidates"] += m["truth_rank"] is not None
            t[{"VERIFIED": "verified", "UNCERTAIN": "uncertain", "NO_MATCH": "no_match"}[m["status"]]] += 1
            t["ocr_seconds"] += ocr_seconds
            t["match_ms"] += match_ms

            size = f"{processed.shape[1]}x{processed.shape[0]}"
            print(f"  [{variant}] {size} blur={blur_score(processed):.0f} ocr={ocr_seconds:.1f}s match={match_ms:.0f}ms")
            print("     OCR: " + ", ".join(f"{r['text']!r} {r['confidence']:.2f}" for r in ocr_results))
            print(f"     ocr scores: {ocr}")
            print(f"     matcher: {m['status']}/{m['level']}, {m['formulations']} formulations, "
                  f"truth rank {m['truth_rank']}, false VERIFIED {m['false_verified']}")
            print(f"     reason: {result.reason}")

    print("\n=== Summary (counts out of n images; not accuracy estimates)")
    print(f"oracle: {oracle}")
    for variant, t in totals.items():
        ran = t["n"] - t["no_region"]
        avg = (f"avg OCR {t['ocr_seconds'] / ran:.1f}s, avg match {t['match_ms'] / ran:.0f}ms" if ran else "")
        print(f"[{variant}] n={t['n']} no_region={t['no_region']} brand={t['brand']} ingredients={t['ingredients']} "
              f"strengths={t['strengths_plain']} (with correction {t['strengths_with_correction']}) "
              f"truth_in_candidates={t['truth_in_candidates']} VERIFIED={t['verified']} "
              f"UNCERTAIN={t['uncertain']} NO_MATCH={t['no_match']} false_VERIFIED={t['false_verified']} {avg}")


if __name__ == "__main__":
    main()
