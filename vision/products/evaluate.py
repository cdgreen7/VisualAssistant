"""Evaluate detection, OCR and matcher decisions on labeled images.

Usage (from the repo root):
    python -m vision.products.evaluate
    python -m vision.products.evaluate --labels research/datasets/photos-v1/labels.json \\
        --inputs full,truth_crop,yolo_crop --save
    python -m vision.products.evaluate --labels research/datasets/dailymed-pilot/labels.json --inputs full,truth_crop
    python -m vision.products.evaluate --labels L --replay research/results/evaluation-<time>.json --save
    python -m vision.products.evaluate --crop yolo --yolo-conf 0.1                       # older flags still work
    python -m vision.products.evaluate --preprocess none,gray,clahe,upscale2x,sharpen

Labels describe what is physically on each package, written by people looking
at the product. They are never copied from FDA matches, model output or an
assistant's transcription.

Review levels (label["review"]["status"]; a label without one is UNREVIEWED):
  UNREVIEWED       descriptive only: matcher statuses, candidate counts, detections
                   and timings. Nothing is scored against the annotation.
  SINGLE_REVIEWED  one person confirmed the annotation: exploratory metrics.
  REVIEWED         two people confirmed it independently and agreed: primary metrics.
The levels are reported separately, never pooled, and nothing here changes a
level. Annotations move between levels only through research/annotation_review.py.

Groups (label["group"]: "photo" for real camera photographs, "artwork" for
official label artwork) are reported separately and never pooled.

Inputs, each scored separately:
  oracle      the matcher on the label's visible_text (perfect OCR)
  full        OCR on the whole image
  truth_crop  OCR on the annotated medicine box (photos) or front-panel roi (artwork)
  yolo_crop   OCR on the box main.py would use (select_best_detection); NO_REGION
              when YOLO finds no bottle

Metrics (scored levels only):
  detection   recall of annotated bottles and of the medicine bottle (a YOLO
              "bottle" box with IoU >= 0.5, matched one-to-one) and which object
              select_best_detection picks when several bottles are present
  ocr         whether the OCR text contains the brand words, every ingredient
              and every strength (plain, and allowing the matcher's O->0 correction)
  matcher     CORRECT_VERIFIED, FALSE_VERIFIED, VERIFIED_PACKAGE_UNSCORED,
              UNCERTAIN, NO_MATCH (and NO_REGION for yolo_crop); candidate recall
              (truth among the candidates), verification coverage (VERIFIED of all
              medicine images) and verification precision (correct of all scored
              VERIFIED). On a non-medicine image (negative) any VERIFIED is false
              and NO_MATCH is a correct rejection.

Every metric is reported per image and per physical product. Views of the same
physical product are not independent, so intervals are computed over products:
a Wilson interval for product-level rates and a cluster bootstrap (resampling
products) for image-level rates, and none below MIN_PRODUCTS_FOR_INTERVAL.
A product's outcome is its worst safety outcome first: any FALSE_VERIFIED view
makes it FALSE_VERIFIED; otherwise any CORRECT_VERIFIED view makes it
CORRECT_VERIFIED, then VERIFIED_PACKAGE_UNSCORED, UNCERTAIN, NO_MATCH, NO_REGION.
Other product-level successes (candidate recall, OCR, detection) mean "in at
least one view".

--save writes research/results/evaluation-<time>.json (never overwritten) with
raw OCR, YOLO detections, timings, image/labels/catalog/model hashes, Git state
and package versions. --replay re-runs only the matcher and the scoring on a
saved run, so matcher or annotation changes are compared on identical OCR. Saved
OCR is reused only for the same image file and, for truth_crop, the same box;
other rows are marked stale and excluded until the evaluation is run again.
Latency percentiles come from research/benchmark.py, not from this single pass.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict

import cv2

from vision.products.catalog import name_key, strength_key
from vision.products.matcher import Level, Status, ingredient_key
from vision.products.ndc import normalize_package_ndc

DEFAULT_LABELS = "evaluation/labels.json"
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "data", "otc_ndc_2026-10-30.json")
DEFAULT_RESULTS = os.path.join("research", "results")
DETECTOR_WEIGHTS = "yolo26n.pt"

REVIEW_LEVELS = ("UNREVIEWED", "SINGLE_REVIEWED", "REVIEWED")
TIERS = {"REVIEWED": "primary", "SINGLE_REVIEWED": "exploratory", "UNREVIEWED": "descriptive"}
# Names that identify software, not a person; never accepted as a reviewer
RESERVED_REVIEWERS = frozenset({"assistant", "claude", "ai", "model", "auto", "automatic", "machine", "yolo",
                                "ocr", "easyocr", "matcher", "qwen", "llm", "gpt"})
BOX_ROLES = ("medicine", "other_bottle")
INPUTS = ("full", "truth_crop", "yolo_crop")
IOU_THRESHOLD = 0.5
MIN_PRODUCTS_FOR_INTERVAL = 5
BOOTSTRAP_RESAMPLES = 2000
VERIFIED_OUTCOMES = ("CORRECT_VERIFIED", "FALSE_VERIFIED", "VERIFIED_PACKAGE_UNSCORED")
# Product outcome precedence: safety failures first
OUTCOME_ORDER = ("FALSE_VERIFIED", "CORRECT_VERIFIED", "VERIFIED_PACKAGE_UNSCORED", "UNCERTAIN", "NO_MATCH", "NO_REGION")


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


# Labels and review levels

def label_id(label):
    return label.get("id") or label["image"]


def review_status(label):
    return (label.get("review") or {}).get("status", "UNREVIEWED")


def tier(label):
    return TIERS[review_status(label)]


def is_negative(label):
    return label.get("negative") is True


def target_boxes(label, role=None):
    return [t["box"] for t in label.get("target_boxes") or [] if role is None or t["role"] == role]


def _valid_box(box):
    return (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) and v >= 0 for v in box)
            and box[0] < box[2] and box[1] < box[3])


def is_person_name(name):
    text = str(name or "").strip()
    return bool(text) and text.lower() not in RESERVED_REVIEWERS


def validate_label(label):
    """Problems (strings) with one label; an empty list means usable."""
    problems = []
    status = review_status(label)
    if status not in REVIEW_LEVELS:
        return [f"unknown review status {status!r}"]
    review = label.get("review") or {}
    reviewers = [str(r).strip() for r in review.get("reviewers") or []]
    if any(not is_person_name(r) for r in reviewers):
        problems.append(f"reviewers must be people, not software or blank: {reviewers}")
    distinct = {r.lower() for r in reviewers}
    if status == "REVIEWED" and (len(distinct) < 2 or not review.get("date")):
        problems.append("REVIEWED needs two different reviewers and a date")
    if status == "SINGLE_REVIEWED" and (len(distinct) != 1 or not review.get("date")):
        problems.append("SINGLE_REVIEWED needs exactly one reviewer and a date")
    if status == "UNREVIEWED" and reviewers:
        problems.append("an UNREVIEWED label must not name reviewers")
    if status != "UNREVIEWED":
        if not isinstance(label.get("negative"), bool):
            problems.append("a reviewed label needs negative true/false")
        elif not label["negative"] and not label.get("formulation"):
            problems.append("a reviewed medicine label needs its complete formulation")
        if not label.get("physical_product_id"):
            problems.append("a reviewed label needs physical_product_id")
    boxes = label.get("target_boxes")
    if boxes is not None:
        if not isinstance(boxes, list) or any(not isinstance(t, dict) or t.get("role") not in BOX_ROLES
                                              or not _valid_box(t.get("box")) for t in boxes):
            problems.append(f"target_boxes must be a list of {{role: {BOX_ROLES}, box: [x1, y1, x2, y2]}}")
    intended = label.get("intended_target")
    if intended is not None and intended != "none" and not (
            isinstance(intended, int) and not isinstance(intended, bool) and isinstance(boxes, list)
            and 0 <= intended < len(boxes)):
        problems.append("intended_target must be null (not annotated), \"none\" or an index into target_boxes")
    if label.get("roi") is not None and not _valid_box(label["roi"]):
        problems.append("roi must be null or [x1, y1, x2, y2]")
    return problems


def load_labels(path):
    """Labels list, validated; raises ValueError listing every problem."""
    with open(path) as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise ValueError(f"{path}: labels must be a list")
    problems, seen = [], set()
    for label in labels:
        key = label_id(label)
        if key in seen:
            problems.append(f"{key}: duplicate id")
        seen.add(key)
        problems += [f"{key}: {problem}" for problem in validate_label(label)]
    if problems:
        raise ValueError("\n".join(problems))
    return labels


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
    truth = formulation_key(label.get("formulation") or [])
    keys = [formulation_key(c.ingredients) for c in result.candidates]
    rank = keys.index(truth) + 1 if truth and truth in keys else None

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


def classify(score, negative=False):
    """Outcome class from score_match output. On a negative any VERIFIED is false."""
    if score["status"] != "VERIFIED":
        return score["status"]
    if negative or score["false_verified"]:
        return "FALSE_VERIFIED"
    if score["package_unlabeled"]:
        return "VERIFIED_PACKAGE_UNSCORED"
    return "CORRECT_VERIFIED"


def oracle_result(label, matcher):
    return matcher.match([{"text": line, "confidence": 1.0} for line in label["visible_text"]])


# Detection

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(truth, predicted, threshold=IOU_THRESHOLD):
    """One-to-one (truth index, predicted index, IoU) pairs, greedy by IoU, each >= threshold."""
    pairs = sorted(((iou(t, p), ti, pi) for ti, t in enumerate(truth) for pi, p in enumerate(predicted)), reverse=True)
    used_t, used_p, matched = set(), set(), []
    for overlap, ti, pi in pairs:
        if overlap < threshold or ti in used_t or pi in used_p:
            continue
        used_t.add(ti)
        used_p.add(pi)
        matched.append((ti, pi, round(overlap, 4)))
    return sorted(matched)


def selection_outcome(label, selected, threshold=IOU_THRESHOLD):
    """Outcome of reading `selected` (a detection or None) for this annotated label."""
    medicine = target_boxes(label, "medicine")
    if not medicine:
        return "NOTHING_SELECTED" if selected is None else "NON_MEDICINE_SELECTED"
    if selected is None:
        return "NO_BOTTLE_DETECTED"
    if max(iou(selected["box"], box) for box in medicine) >= threshold:
        return "MEDICINE_SELECTED"
    return "WRONG_OBJECT_SELECTED"


def intent_outcome(label, selected, threshold=IOU_THRESHOLD):
    """Outcome against the annotated INTENDED target, or None when intent wasn't annotated.

    The intended target is what the person meant the system to read, recorded by the
    reviewers (label["intended_target"]: an index into target_boxes, or "none" when
    nothing should be read). It is separate from the medicine boxes: a photo can show
    two medicines, or the user can mean another object."""
    intended = label.get("intended_target")
    if intended is None:
        return None
    if intended == "none":
        return "CORRECTLY_NOTHING" if selected is None else "UNINTENDED_SELECTED"
    if selected is None:
        return "NOTHING_SELECTED"
    box = label["target_boxes"][intended]["box"]
    return "INTENDED_SELECTED" if iou(selected["box"], box) >= threshold else "UNINTENDED_SELECTED"


def target_selection(label, detections, threshold=IOU_THRESHOLD, class_name="bottle"):
    """Which object main.py's default select_best_detection would read."""
    from vision.identification import select_best_detection
    return selection_outcome(label, select_best_detection(detections, class_name), threshold)


ORIGINAL_BASELINE = "confidence_original"      # identification.select_best_detection, as main.py by default


def selection_choices(detections, frame_size, methods=None, class_name="bottle"):
    """{method: {"box", "index", "score"} or None}: the original baseline
    (select_best_detection, no score) and every target-selection method, all on the same
    detections. Descriptive: it says which box each method picks, not whether it's right."""
    from vision.identification import select_best_detection
    from vision.target_selection import METHODS, select_target
    original = select_best_detection(detections, class_name)
    out = {ORIGINAL_BASELINE: None if original is None else {
        "box": list(original["box"]), "index": next(i for i, d in enumerate(detections) if d is original),
        "score": None}}
    for name, weights in (methods or METHODS).items():
        selection = select_target(detections, frame_size, weights=weights, class_name=class_name)
        chosen = selection.candidates[0] if selection.detection is not None else None
        out[name] = None if chosen is None else {"box": list(chosen.detection["box"]), "index": chosen.index,
                                                 "score": round(chosen.score, 6)}
    return out


def score_detection(label, detections, threshold=IOU_THRESHOLD, class_name="bottle", frame_size=None, methods=None):
    """Detection scores, or None when the label has no annotated boxes. With frame_size,
    also the outcome of every target-selection method on the same detections."""
    if label.get("target_boxes") is None:
        return None
    truth = label["target_boxes"]
    bottles = [d for d in detections if d["class_name"] == class_name]
    matched = {ti for ti, _, _ in match_boxes([t["box"] for t in truth], [list(d["box"]) for d in bottles], threshold)}
    medicine = [i for i, t in enumerate(truth) if t["role"] == "medicine"]
    choices = None if frame_size is None else selection_choices(detections, frame_size, methods, class_name)
    return {
        "bottle_targets": len(truth),
        "bottles_found": len(matched),
        "medicine_targets": len(medicine),
        "medicine_found": sum(i in matched for i in medicine),
        "bottle_detections": len(bottles),
        "unmatched_bottle_detections": len(bottles) - len(matched),
        "multiple_bottles": len(truth) > 1,
        "selection": target_selection(label, detections, threshold, class_name),
        "intent_annotated": label.get("intended_target") is not None,
        "selection_by_method": None if choices is None else {
            name: selection_outcome(label, None if choice is None else {"box": choice["box"]}, threshold)
            for name, choice in choices.items()},
        "intent_by_method": None if choices is None or label.get("intended_target") is None else {
            name: intent_outcome(label, None if choice is None else {"box": choice["box"]}, threshold)
            for name, choice in choices.items()},
    }


def truth_crop_box(label):
    """The single annotated medicine box, else the artwork front-panel roi, else None."""
    medicine = target_boxes(label, "medicine")
    if medicine:
        return medicine[0] if len(medicine) == 1 else None
    return label.get("roi")


# Running

def region_for(image, crop, detector):
    if crop == "full":
        return image, None
    from vision.identification import crop_box, select_best_detection
    detection = select_best_detection(detector.get_detections(detector.detect(image)))
    if detection is None:
        return None, None
    return crop_box(image, detection["box"]), detection


def _crop(image, box):
    from vision.identification import crop_box
    return crop_box(image, box)


def _ocr_list(ocr_results):
    return [[r["text"], round(float(r["confidence"]), 4)] for r in ocr_results]


def make_row(label, input_name, variant, result, matcher, ocr=None, extra=None):
    """One result row. Truth-based fields only for reviewed (non-descriptive) labels."""
    status = review_status(label)
    row = {"id": label_id(label), "group": label.get("group", "unspecified"), "review_status": status,
           "tier": TIERS[status], "physical_product_id": label.get("physical_product_id"),
           "negative": label.get("negative"), "input": input_name, "variant": variant}
    if result is None:                                    # yolo_crop without a detected bottle
        row.update(status="NO_REGION", level=None, formulations=0, reason="YOLO found no bottle", ocr=None)
    else:
        row.update(status=result.status.value, level=result.level.value, formulations=len(result.candidates),
                   reason=result.reason, package=result.package, ocr=None if ocr is None else _ocr_list(ocr))
    row.update(extra or {})
    if row["tier"] == "descriptive":
        return row
    if result is None:
        row.update(outcome="NO_REGION", truth_rank=None, false_verified=False)
        return row
    score = score_match(label, result)
    row.update(outcome=classify(score, is_negative(label)), truth_rank=score["truth_rank"],
               false_verified=score["false_verified"] or (is_negative(label) and score["status"] == "VERIFIED"),
               package_unlabeled=score["package_unlabeled"])
    if ocr is not None and not is_negative(label) and label.get("ingredients"):
        row["ocr_scores"] = score_ocr(label, ocr, matcher)
    return row


def evaluate_label(label, image, recognizer, matcher, detector=None, inputs=("full",), variants=("none",),
                   threshold=IOU_THRESHOLD, methods=None):
    """(rows, image_record) for one label. rows: oracle plus one per input and variant."""
    record = {"id": label_id(label), "image": label["image"], "size": f"{image.shape[1]}x{image.shape[0]}",
              "review_status": review_status(label), "group": label.get("group", "unspecified"),
              "physical_product_id": label.get("physical_product_id"), "skipped": []}
    detections = None
    if detector is not None and ("yolo_crop" in inputs or label.get("target_boxes") is not None):
        started = time.perf_counter()
        detections = detector.get_detections(detector.detect(image))
        record["yolo_ms"] = round((time.perf_counter() - started) * 1000, 2)
        record["detections"] = [{"class_name": d["class_name"], "confidence": round(d["confidence"], 4),
                                 "box": list(d["box"])} for d in detections]
        frame_size = (image.shape[1], image.shape[0])
        record["selection_choices"] = selection_choices(detections, frame_size, methods)
        record["detection"] = (score_detection(label, detections, threshold, frame_size=frame_size, methods=methods)
                               if tier(label) != "descriptive" else None)

    rows = []
    if label.get("visible_text"):
        started = time.perf_counter()
        result = oracle_result(label, matcher)
        rows.append(make_row(label, "oracle", None, result, matcher,
                             extra={"match_ms": round((time.perf_counter() - started) * 1000, 2)}))
    for input_name in inputs:
        if input_name == "full":
            region, box = image, None
        elif input_name == "truth_crop":
            box = truth_crop_box(label)
            region = _crop(image, box) if box else None
            if region is None:
                record["skipped"].append({"input": "truth_crop", "reason": "no single annotated medicine box or roi"})
                continue
        else:
            from vision.identification import select_best_detection
            chosen = select_best_detection(detections) if detections is not None else None
            box = list(chosen["box"]) if chosen else None
            region = _crop(image, box) if box else None
            if region is None:
                for variant in variants:
                    rows.append(make_row(label, "yolo_crop", variant, None, matcher, extra={"crop_box": None}))
                continue
        for variant in variants:
            processed = PREPROCESSORS[variant](region)
            started = time.perf_counter()
            ocr = recognizer.read(processed)
            ocr_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            result = matcher.match(ocr)
            match_ms = (time.perf_counter() - started) * 1000
            rows.append(make_row(label, input_name, variant, result, matcher, ocr, extra={
                "crop_box": box, "region_size": f"{processed.shape[1]}x{processed.shape[0]}",
                "blur": round(blur_score(processed), 1), "ocr_ms": round(ocr_ms, 2), "match_ms": round(match_ms, 2)}))
    return rows, record


def stale_reason(old_row, label, saved_image_sha):
    """Why a saved row's OCR must not be reused for this label, or None.

    OCR is only valid for the exact pixels it read: the same image file (sha256)
    and, for truth_crop, the same annotated box. Full-frame and YOLO-crop OCR depend
    only on the file, since the saved detections are from that file."""
    if old_row["input"] == "oracle":
        return None                                       # recomputed from the current visible_text
    if label.get("sha256") and saved_image_sha and label["sha256"] != saved_image_sha:
        return "image file changed since the saved run"
    if old_row["input"] == "truth_crop" and old_row.get("crop_box") != truth_crop_box(label):
        return "annotated crop changed since the saved run"
    return None


def _parse_size(size):
    """'WxH' -> (W, H), or None."""
    try:
        width, height = (int(v) for v in str(size).split("x"))
        return width, height
    except ValueError:
        return None


def replay(previous, labels, matcher, threshold=IOU_THRESHOLD, methods=None):
    """Re-score a saved run with the current matcher and annotations, on its saved OCR
    and detections. OCR and YOLO are not repeated.

    Saved OCR is reused only for the same pixels (see stale_reason). A stale row keeps
    no OCR, is not matched or scored, and is excluded from the summary; run the
    evaluation again for it. A label that now has a truth crop the saved run never read
    is listed under the image's "needs_ocr"."""
    by_id = {label_id(label): label for label in labels}
    missing = sorted({row["id"] for row in previous["rows"]} - set(by_id))
    if missing:
        raise ValueError(f"saved rows have no label: {missing}")
    saved_sha = {img["id"]: img.get("sha256") for img in previous.get("images", [])}
    rows = []
    for old in previous["rows"]:
        label = by_id[old["id"]]
        previous_fields = {"status": old.get("status"), "outcome": old.get("outcome"),
                           "review_status": old.get("review_status")}
        reason = stale_reason(old, label, saved_sha.get(old["id"]))
        if reason:
            status = review_status(label)
            rows.append({"id": old["id"], "group": label.get("group", "unspecified"), "review_status": status,
                         "tier": TIERS[status], "physical_product_id": label.get("physical_product_id"),
                         "negative": label.get("negative"), "input": old["input"], "variant": old.get("variant"),
                         "status": "STALE", "stale": reason, "crop_box": old.get("crop_box"),
                         "current_crop_box": truth_crop_box(label) if old["input"] == "truth_crop" else None,
                         "ocr": None, "previous": previous_fields})
            continue
        keep = {k: old[k] for k in ("crop_box", "region_size", "blur", "ocr_ms") if k in old}
        if old["input"] == "oracle":
            if not label.get("visible_text"):
                continue
            row = make_row(label, "oracle", None, oracle_result(label, matcher), matcher)
        elif old.get("ocr") is None:
            row = make_row(label, old["input"], old.get("variant"), None, matcher, extra=keep)
        else:
            ocr = [{"text": text, "confidence": confidence} for text, confidence in old["ocr"]]
            row = make_row(label, old["input"], old.get("variant"), matcher.match(ocr), matcher, ocr, extra=keep)
        row["previous"] = previous_fields
        rows.append(row)
    read_truth_crop = {r["id"] for r in previous["rows"] if r["input"] == "truth_crop"}
    images = []
    for old in previous.get("images", []):
        label = by_id[old["id"]]
        record = dict(old, review_status=review_status(label), physical_product_id=label.get("physical_product_id"),
                      needs_ocr=[])
        changed = bool(label.get("sha256") and old.get("sha256") and label["sha256"] != old["sha256"])
        if changed:
            record["stale"] = "image file changed since the saved run"
            record["detection"] = None                    # detections were made on other pixels
        elif old.get("detections") is not None:
            frame_size = _parse_size(old.get("size"))
            record["selection_choices"] = selection_choices(old["detections"], frame_size, methods) if frame_size else None
            record["detection"] = (score_detection(label, old["detections"], threshold, frame_size=frame_size,
                                                   methods=methods) if tier(label) != "descriptive" else None)
        else:
            record["detection"] = None
        if truth_crop_box(label) and old["id"] not in read_truth_crop:
            record["needs_ocr"].append("truth_crop")
        images.append(record)
    return rows, images


# Statistics

def wilson(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def cluster_bootstrap(pairs_by_cluster, seed=0, resamples=BOOTSTRAP_RESAMPLES):
    """95% percentile interval of sum(numerators)/sum(denominators), resampling clusters
    (physical products) with replacement. pairs_by_cluster: {cluster: (num, den)}."""
    clusters = list(pairs_by_cluster.values())
    if not clusters:
        return None
    rng, ratios = random.Random(seed), []
    for _ in range(resamples):
        sample = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        den = sum(d for _, d in sample)
        if den:
            ratios.append(sum(n for n, _ in sample) / den)
    if not ratios:
        return None
    ratios.sort()
    return [round(ratios[int(0.025 * (len(ratios) - 1))], 4), round(ratios[int(0.975 * (len(ratios) - 1))], 4)]


def _product(item):
    return item.get("physical_product_id") or f"unassigned:{item['id']}"


def rate(items, success, applies=lambda item: True, seed=0):
    """Image-level and product-level rate of success among items where applies.
    Product level: a product succeeds when any of its applicable views succeeds."""
    used = [item for item in items if applies(item)]
    by_product = defaultdict(list)
    for item in used:
        by_product[_product(item)].append(bool(success(item)))
    n_products = len(by_product)
    hits = sum(any(v) for v in by_product.values())
    image_hits = sum(sum(v) for v in by_product.values())
    enough = n_products >= MIN_PRODUCTS_FOR_INTERVAL
    return {
        "images": [image_hits, len(used)],
        "image_rate": round(image_hits / len(used), 4) if used else None,
        "image_ci95": cluster_bootstrap({p: (sum(v), len(v)) for p, v in by_product.items()}, seed) if enough else None,
        "products": [hits, n_products],
        "product_rate": round(hits / n_products, 4) if n_products else None,
        "product_ci95": wilson(hits, n_products) if enough else None,
    }


def product_outcomes(rows):
    """{product: outcome}, safety failures first (OUTCOME_ORDER)."""
    grouped = defaultdict(set)
    for row in rows:
        grouped[_product(row)].add(row["outcome"])
    return {p: next(o for o in OUTCOME_ORDER if o in outcomes) for p, outcomes in grouped.items()}


def _counts(values, keys):
    return {k: sum(v == k for v in values) for k in keys}


def summarize_matcher(rows, seed=0):
    """Metrics for rows of one group, review level, input and variant."""
    products = {_product(r) for r in rows}
    summary = {"images": len(rows), "products": len(products),
               "unassigned_product_ids": sum(not r.get("physical_product_id") for r in rows)}
    if rows and rows[0]["tier"] == "descriptive":
        summary["note"] = "UNREVIEWED annotations: matcher statuses only, nothing is scored"
        summary["status_counts"] = _counts([r["status"] for r in rows], ("VERIFIED", "UNCERTAIN", "NO_MATCH", "NO_REGION"))
        summary["median_candidates"] = statistics.median(r["formulations"] for r in rows) if rows else None
        return summary
    positives = [r for r in rows if r.get("negative") is False]
    negatives = [r for r in rows if r.get("negative") is True]
    keys = OUTCOME_ORDER
    if positives:
        by_product = product_outcomes(positives)
        verified = lambda r: r["outcome"] in VERIFIED_OUTCOMES
        scored = lambda r: r["outcome"] in ("CORRECT_VERIFIED", "FALSE_VERIFIED")
        product_rows = [{"id": p, "physical_product_id": p, "outcome": o} for p, o in by_product.items()]
        summary["medicine"] = {
            "outcomes_images": _counts([r["outcome"] for r in positives], keys),
            "outcomes_products": _counts(list(by_product.values()), keys),
            "candidate_recall": rate(positives, lambda r: r["truth_rank"] is not None, seed=seed),
            "verification_coverage": rate(positives, verified, seed=seed),
            "false_verified": rate(positives, lambda r: r["outcome"] == "FALSE_VERIFIED", seed=seed),
            "verification_precision_images": rate(positives, lambda r: r["outcome"] == "CORRECT_VERIFIED",
                                                  applies=scored, seed=seed),
            "verification_precision_products": rate(product_rows, lambda r: r["outcome"] == "CORRECT_VERIFIED",
                                                    applies=scored, seed=seed)["products"],
        }
        with_ocr = [r for r in positives if "ocr_scores" in r]
        if with_ocr:
            summary["ocr"] = {field: rate(with_ocr, lambda r, f=field: r["ocr_scores"][f], seed=seed)
                              for field in ("brand", "ingredients", "strengths_plain", "strengths_with_correction")}
    if negatives:
        summary["non_medicine"] = {
            "outcomes_images": _counts([r["outcome"] for r in negatives], keys),
            "false_verified": rate(negatives, lambda r: r["outcome"] == "FALSE_VERIFIED", seed=seed),
            "correct_rejection": rate(negatives, lambda r: r["outcome"] == "NO_MATCH", seed=seed),
        }
    return summary


def summarize_detection(images, seed=0):
    scored = [i for i in images if i.get("detection")]
    if not scored:
        return None
    flat = lambda key_found, key_total: [  # one item per annotated box, clustered by product
        {"id": i["id"], "physical_product_id": i.get("physical_product_id"), "found": k < i["detection"][key_found]}
        for i in scored for k in range(i["detection"][key_total])]
    with_medicine = [i for i in scored if i["detection"]["medicine_targets"]]
    selected = lambda i: i["detection"]["selection"] == "MEDICINE_SELECTED"
    return {
        "images": len(scored),
        "bottle_recall": rate(flat("bottles_found", "bottle_targets"), lambda b: b["found"], seed=seed),
        "medicine_recall": rate(flat("medicine_found", "medicine_targets"), lambda b: b["found"], seed=seed),
        "target_selection": rate(with_medicine, selected, seed=seed),
        "target_selection_multiple_bottles": rate(with_medicine, selected,
                                                  applies=lambda i: i["detection"]["multiple_bottles"], seed=seed),
        "selection_counts": _counts([i["detection"]["selection"] for i in scored],
                                    ("MEDICINE_SELECTED", "WRONG_OBJECT_SELECTED", "NO_BOTTLE_DETECTED",
                                     "NON_MEDICINE_SELECTED", "NOTHING_SELECTED")),
        "unmatched_bottle_detections": sum(i["detection"]["unmatched_bottle_detections"] for i in scored),
        "selection_methods": summarize_selection_methods(scored, seed),
    }


SELECTION_OUTCOMES = ("MEDICINE_SELECTED", "WRONG_OBJECT_SELECTED", "NO_BOTTLE_DETECTED",
                      "NON_MEDICINE_SELECTED", "NOTHING_SELECTED")


INTENT_OUTCOMES = ("INTENDED_SELECTED", "UNINTENDED_SELECTED", "NOTHING_SELECTED", "CORRECTLY_NOTHING")


def summarize_selection_methods(scored, seed=0):
    """Per method, on identical detections, two separate questions:

    medicine_*   did it read an annotated MEDICINE box (any medicine, on images with one),
                 split into several-bottle and single-bottle images, and what it read on
                 images without a medicine;
    intended_*   did it read the object the person INTENDED (only images whose reviewers
                 annotated intent; otherwise not computed).
    """
    by_method, intent_by_method = defaultdict(list), defaultdict(list)
    for image in scored:
        for name, outcome in (image["detection"].get("selection_by_method") or {}).items():
            by_method[name].append(dict(image, outcome=outcome))
        for name, outcome in (image["detection"].get("intent_by_method") or {}).items():
            intent_by_method[name].append(dict(image, outcome=outcome))
    out = {}
    for name, items in sorted(by_method.items()):
        medicine = lambda i: i["detection"]["medicine_targets"] > 0
        several = lambda i: medicine(i) and i["detection"]["bottle_targets"] > 1
        single = lambda i: medicine(i) and i["detection"]["bottle_targets"] == 1
        right = lambda i: i["outcome"] == "MEDICINE_SELECTED"
        wrong = lambda i: i["outcome"] == "WRONG_OBJECT_SELECTED"
        intent = intent_by_method.get(name, [])
        out[name] = {
            "medicine_selected": rate(items, right, applies=medicine, seed=seed),
            "wrong_object_selected": rate(items, wrong, applies=medicine, seed=seed),
            "medicine_selected_several_bottles": rate(items, right, applies=several, seed=seed),
            "medicine_selected_single_bottle": rate(items, right, applies=single, seed=seed),
            "no_medicine_images": _counts([i["outcome"] for i in items if not medicine(i)],
                                          ("NOTHING_SELECTED", "NON_MEDICINE_SELECTED")),
            "counts": _counts([i["outcome"] for i in items], SELECTION_OUTCOMES),
            "intent_annotated_images": len(intent),
            "intended_target": rate(intent, lambda i: i["outcome"] in ("INTENDED_SELECTED", "CORRECTLY_NOTHING"),
                                    seed=seed) if intent else None,
            "intent_counts": _counts([i["outcome"] for i in intent], INTENT_OUTCOMES) if intent else None,
        }
    return out


def summarize(rows, images, seed=0):
    """{group: {review status: {"detection": ..., "<input>/<variant>": ...}}}; never pooled."""
    report = defaultdict(dict)
    stale = [r for r in rows if r.get("stale")]
    rows = [r for r in rows if not r.get("stale")]           # never scored, never counted
    images = [i for i in images if not i.get("stale")]
    keys = sorted({(r["group"], r["review_status"]) for r in rows + stale}
                  | {(i["group"], i["review_status"]) for i in images})
    for group, status in keys:
        section = {"tier": TIERS[status],
                   "stale_rows_excluded": sum((r["group"], r["review_status"]) == (group, status) for r in stale)}
        if TIERS[status] != "descriptive":
            section["detection"] = summarize_detection(
                [i for i in images if i["group"] == group and i["review_status"] == status], seed)
        for input_name in ("oracle",) + INPUTS:
            subset = [r for r in rows if (r["group"], r["review_status"], r["input"]) == (group, status, input_name)]
            for variant in sorted({str(r["variant"]) for r in subset}):
                chosen = [r for r in subset if str(r["variant"]) == variant]
                name = input_name if input_name == "oracle" else f"{input_name}/{variant}"
                section[name] = summarize_matcher(chosen, seed)
        report[group][status] = section
    return dict(report)


# Reproducibility

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _version(distribution):
    from importlib import metadata
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def environment(labels_path, cache_path, catalog=None, detector_used=False, ocr_used=False):
    """What is needed to reproduce a run: code, data and model identities, versions, hardware."""
    easyocr_dir = os.path.join(os.environ.get("EASYOCR_MODULE_PATH", os.path.expanduser("~/.EasyOCR")), "model")
    ocr_models = {}
    if ocr_used and os.path.isdir(easyocr_dir):
        ocr_models = {name: file_sha256(os.path.join(easyocr_dir, name))
                      for name in sorted(os.listdir(easyocr_dir)) if name.endswith(".pth")}
    try:
        import torch
        threads = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    except Exception:
        threads = None
    return {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git": {"commit": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "tracked_changes": bool(_git("status", "--porcelain", "--untracked-files=no"))},
        "python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
        "versions": {name: _version(name) for name in ("torch", "torchvision", "ultralytics", "easyocr",
                                                       "opencv-python", "numpy", "zxing-cpp", "Pillow")},
        "torch_threads": threads,
        "labels": {"path": labels_path, "sha256": file_sha256(labels_path)},
        "catalog": {"file": os.path.basename(cache_path), "sha256": file_sha256(cache_path),
                    "last_updated": (catalog.meta.get("last_updated") if catalog is not None else None)},
        "detector": {"weights": DETECTOR_WEIGHTS, "sha256": file_sha256(DETECTOR_WEIGHTS)}
        if detector_used and os.path.exists(DETECTOR_WEIGHTS) else None,
        "ocr_models": ocr_models or None,
    }


def save_results(data, out_dir=DEFAULT_RESULTS, prefix="evaluation", now=None):
    os.makedirs(out_dir, exist_ok=True)
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(out_dir, f"{prefix}-{stamp}.json")
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; results are never overwritten")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp, path)
    return path


def read_image(label):
    """The image as cv2 loads it (EXIF orientation applied); checks the recorded sha256."""
    path = label["image"]
    if not os.path.exists(path):
        return None, None
    digest = file_sha256(path)
    if label.get("sha256") and label["sha256"] != digest:
        raise ValueError(f"{label_id(label)}: {path} does not match its recorded sha256")
    return cv2.imread(path), digest


# Printing

def _fmt(metric):
    if not metric:
        return "-"
    (hits, n), (phits, pn) = metric["images"], metric["products"]
    ci = metric.get("product_ci95")
    tail = f", products 95% CI {ci[0]:.0%}-{ci[1]:.0%}" if ci else ""
    return f"{hits}/{n} images, {phits}/{pn} products{tail}"


def print_report(report):
    for group, levels in report.items():
        for status, section in levels.items():
            print(f"\n=== group {group} | {status} ({section['tier']})")
            if section.get("detection"):
                d = section["detection"]
                print(f"  YOLO bottle recall: {_fmt(d['bottle_recall'])}; medicine recall: {_fmt(d['medicine_recall'])}")
                print(f"  target selection: {_fmt(d['target_selection'])}; with several bottles: "
                      f"{_fmt(d['target_selection_multiple_bottles'])}; {d['selection_counts']}")
                for name, m in (d.get("selection_methods") or {}).items():
                    print(f"    [{name}] medicine selected {_fmt(m['medicine_selected'])}; wrong object "
                          f"{_fmt(m['wrong_object_selected'])}; several bottles "
                          f"{_fmt(m['medicine_selected_several_bottles'])}; single bottle "
                          f"{_fmt(m['medicine_selected_single_bottle'])}; no medicine {m['no_medicine_images']}")
                    print(f"      intended target: " + (_fmt(m["intended_target"]) if m["intended_target"]
                                                         else "not annotated (intent is not inferred from medicine boxes)"))
            if section.get("stale_rows_excluded"):
                print(f"  {section['stale_rows_excluded']} stale rows excluded (OCR from other pixels; re-run them)")
            for name, s in section.items():
                if name in ("tier", "detection", "stale_rows_excluded") or not isinstance(s, dict):
                    continue
                print(f"  [{name}] images={s['images']} products={s['products']}")
                if "status_counts" in s:
                    print(f"     {s['note']}: {s['status_counts']}, median candidates {s['median_candidates']}")
                if "medicine" in s:
                    m = s["medicine"]
                    print(f"     outcomes (images): {m['outcomes_images']}")
                    print(f"     outcomes (products): {m['outcomes_products']}")
                    print(f"     candidate recall: {_fmt(m['candidate_recall'])}")
                    print(f"     verification coverage: {_fmt(m['verification_coverage'])}")
                    print(f"     false VERIFIED: {_fmt(m['false_verified'])}")
                    print(f"     verification precision (images): {_fmt(m['verification_precision_images'])}; "
                          f"products {m['verification_precision_products']}")
                if "ocr" in s:
                    print("     OCR: " + "; ".join(f"{k} {_fmt(v)}" for k, v in s["ocr"].items()))
                if "non_medicine" in s:
                    n = s["non_medicine"]
                    print(f"     non-medicine: false VERIFIED {_fmt(n['false_verified'])}; "
                          f"correct rejection {_fmt(n['correct_rejection'])}")
    print(f"\nIntervals need at least {MIN_PRODUCTS_FOR_INTERVAL} physical products. "
          "UNREVIEWED results are descriptive only; nothing is pooled across groups or review levels.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--inputs", help="comma-separated: " + ",".join(INPUTS) + " (default full)")
    parser.add_argument("--crop", choices=("full", "yolo"), help="older form of --inputs full / --inputs yolo_crop")
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--preprocess", default="none", help="comma-separated: " + ",".join(PREPROCESSORS))
    parser.add_argument("--replay", help="a saved evaluation JSON: re-score its OCR and detections (no OCR or YOLO)")
    parser.add_argument("--save", action="store_true", help=f"write the results JSON under {DEFAULT_RESULTS}/")
    parser.add_argument("--out", default=DEFAULT_RESULTS)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    parser.add_argument("--selection-weights", help="also compare custom target-selection weights wc,wa,wp "
                                                    "(confidence, area, centre; non-negative, sum 1)")
    args = parser.parse_args(argv)

    inputs = [i for i in (args.inputs or ("yolo_crop" if args.crop == "yolo" else "full")).split(",") if i]
    variants = list(dict.fromkeys(args.preprocess.split(",")))
    unknown = [i for i in inputs if i not in INPUTS] + [v for v in variants if v not in PREPROCESSORS]
    if unknown:
        parser.error(f"unknown inputs or preprocessors: {unknown}")

    from vision.products.catalog import ProductCatalog
    from vision.products.matcher import ProductMatcher
    from vision.target_selection import METHODS, Weights

    methods = dict(METHODS)
    if args.selection_weights:
        try:
            methods["custom"] = Weights(*(float(v) for v in args.selection_weights.split(",")))
        except (TypeError, ValueError) as error:
            parser.error(f"--selection-weights: {error}")

    labels = load_labels(args.labels)
    catalog = ProductCatalog.from_file(args.cache)
    matcher = ProductMatcher(catalog)
    settings = {"inputs": inputs, "variants": variants, "yolo_conf": args.yolo_conf, "iou_threshold": args.iou,
                "bootstrap_seed": args.seed, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "min_products_for_interval": MIN_PRODUCTS_FOR_INTERVAL,
                "selection_methods": {name: [w.confidence, w.area, w.center] for name, w in methods.items()}}

    if args.replay:
        with open(args.replay) as f:
            previous = json.load(f)
        rows, images = replay(previous, labels, matcher, args.iou, methods)
        for row in rows:
            if row.get("stale"):
                print(f"stale, not scored: {row['id']} {row['input']} ({row['stale']}); run the evaluation again")
        for image in images:
            if image.get("needs_ocr"):
                print(f"not in the saved run: {image['id']} {', '.join(image['needs_ocr'])}; run the evaluation again")
        env = environment(args.labels, args.cache, catalog)
        if previous.get("environment", {}).get("catalog", {}).get("sha256") != env["catalog"]["sha256"]:
            print("note: the catalog differs from the replayed run's catalog")
        data = {"environment": env, "settings": dict(previous.get("settings", {}), replay_of=args.replay),
                "previous_environment": previous.get("environment"), "rows": rows, "images": images}
    else:
        detector = None
        if "yolo_crop" in inputs or any(label.get("target_boxes") is not None for label in labels):
            from vision.detector import ObjectDetector
            detector = ObjectDetector(conf=args.yolo_conf)
        recognizer = None
        if inputs:
            from vision.ocr import TextRecognizer
            recognizer = TextRecognizer()
        rows, images = [], []
        for label in labels:
            image, digest = read_image(label)
            if image is None:
                print(f"! can't read {label['image']}; skipped")
                images.append({"id": label_id(label), "image": label["image"], "missing": True,
                               "group": label.get("group", "unspecified"), "review_status": review_status(label)})
                continue
            print(f"=== {label_id(label)} [{label.get('group', 'unspecified')}, {review_status(label)}]")
            label_rows, record = evaluate_label(label, image, recognizer, matcher, detector, inputs, variants, args.iou,
                                                methods)
            record["sha256"] = digest
            rows += label_rows
            images.append(record)
            for row in label_rows:
                outcome = row.get("outcome", "(not scored)")
                print(f"  {row['input']:10} {str(row['variant'] or ''):9} {row['status']:9} {outcome:26} "
                      f"candidates={row['formulations']}")
        env = environment(args.labels, args.cache, catalog, detector_used=detector is not None,
                          ocr_used=recognizer is not None)
        data = {"environment": env, "settings": settings, "rows": rows, "images": images}

    for image in images:
        choices = image.get("selection_choices")
        if choices:
            picked = ", ".join(f"{name} #{c['index']}" if c else f"{name} none" for name, c in choices.items())
            note = "" if len({c and c["index"] for c in choices.values()}) == 1 else "  (methods disagree)"
            print(f"target selection {image['id']}: {picked}{note}")
    images_present = [i for i in images if not i.get("missing")]
    data["report"] = summarize(rows, images_present, args.seed)
    print_report(data["report"])
    if args.save:
        print("saved:", save_results(data, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
