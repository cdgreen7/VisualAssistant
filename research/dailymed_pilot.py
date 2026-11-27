"""DailyMed pilot: a small, reproducible label-image set for OCR + ProductMatcher.

From the repository root:

    .venv/bin/python -m research.dailymed_pilot fetch    # download only the listed images
    .venv/bin/python -m research.dailymed_pilot verify   # labels vs. local openFDA catalog + files
    .venv/bin/python -m research.dailymed_pilot run      # OCR + matcher, full image and front panel

Labels (research/datasets/dailymed-pilot/labels.json) use the format of
vision/products/evaluate.py, plus optional keys that it ignores (id, category,
roi, barcodes, provenance). So the existing evaluator also runs on them:

    .venv/bin/python -m vision.products.evaluate --labels research/datasets/dailymed-pilot/labels.json

Scoring is evaluate.py's own (score_ocr, score_match, oracle_result); this
module adds the front-panel ROI input, per-image outcome classes and a JSON
result file. Nothing here changes the matcher or its decision rules.

The pilot labels are UNREVIEWED: run/replay counts compare matcher versions on
fixed inputs (a regression check), not accuracy. Review-gated metrics come from
vision/products/evaluate.py.

Outcomes (UNCERTAIN is never counted as a correct identification):
  CORRECT_VERIFIED           VERIFIED and the formulation (and package, if labeled) is right
  FALSE_VERIFIED             VERIFIED with a wrong formulation or a wrong labeled package
  VERIFIED_PACKAGE_UNSCORED  VERIFIED/PACKAGE, right formulation, package truth unknown
  UNCERTAIN / NO_MATCH       the matcher's own status
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("research/datasets/dailymed-pilot")
LABELS = DATASET / "labels.json"
RETRIEVAL = DATASET / "retrieval.json"
RESULTS = Path("research/results")
SOURCE_PREFIX = "https://dailymed.nlm.nih.gov/dailymed/image.cfm?setid="
MAX_BYTES = 15 * 1024 * 1024
USER_AGENT = "visual-assistant-research"
INPUTS = ("full", "front_panel")
UNREVIEWED_NOTE = ("These labels are UNREVIEWED (transcribed by an assistant): the counts are a regression "
                   "comparison of matcher versions on fixed inputs, not accuracy. For review-gated metrics run "
                   "python -m vision.products.evaluate --labels research/datasets/dailymed-pilot/labels.json")
REQUIRED = ("id", "image", "product", "brand_words", "ingredients", "formulation",
            "visible_text", "provenance")


def _abs(path):
    return path if Path(path).is_absolute() else ROOT / path


def load_labels(path=LABELS):
    """Labels list, validated. Raises ValueError on the first problem."""
    labels = json.loads(_abs(path).read_text())
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels must be a non-empty list")
    seen = set()
    for label in labels:
        missing = [key for key in REQUIRED if key not in label]
        if missing:
            raise ValueError(f"{label.get('id')}: missing {missing}")
        if label["id"] in seen:
            raise ValueError(f"duplicate id {label['id']}")
        seen.add(label["id"])
        image = Path(label["image"])
        if image.is_absolute() or image.parent != DATASET / "images" or image.suffix != ".jpg":
            raise ValueError(f"{label['id']}: image must be {DATASET}/images/<name>.jpg")
        source = label["provenance"]
        if not source.get("spl_set_id") or not source.get("source_url", "").startswith(
                SOURCE_PREFIX + source["spl_set_id"] + "&"):
            raise ValueError(f"{label['id']}: source_url must be a DailyMed image of its spl_set_id")
        roi = label.get("roi")
        if roi is not None and not (isinstance(roi, list) and len(roi) == 4 and all(isinstance(v, int) for v in roi)
                                    and 0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3]):
            raise ValueError(f"{label['id']}: roi must be null or [x1, y1, x2, y2] in image pixels")
    return labels


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _download(url, opener):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(request, timeout=60) as response:
        data = response.read(MAX_BYTES + 1)
        content_type = response.headers.get("Content-Type", "")
    if len(data) > MAX_BYTES:
        raise ValueError(f"{url}: larger than {MAX_BYTES} bytes")
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"{url}: not a JPEG")
    return data, content_type


def fetch(labels, retrieval_path=RETRIEVAL, opener=urllib.request.urlopen, pause=0.5, now=None):
    """Download each listed image once, recording where and when it came from.

    An image already on disk is kept only if it matches its recorded hash; a
    file without a record, or a changed file, is an error (never overwritten).
    Returns the ids downloaded in this call.
    """
    retrieval_path = _abs(retrieval_path)
    records = json.loads(retrieval_path.read_text()) if retrieval_path.exists() else {}
    downloaded = []
    for label in labels:
        dest, record = _abs(label["image"]), records.get(label["id"])
        if dest.exists():
            if record is None:
                raise ValueError(f"{dest} exists without a retrieval record; move it away and fetch again")
            if _sha256(dest.read_bytes()) != record["sha256"]:
                raise ValueError(f"{dest} does not match its recorded sha256")
            continue
        if downloaded and pause:
            time.sleep(pause)
        url = label["provenance"]["source_url"]
        data, content_type = _download(url, opener)
        if record is not None and _sha256(data) != record["sha256"]:
            raise ValueError(f"{url}: DailyMed now serves different bytes than recorded; review before reusing labels")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(dest) + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
        records[label["id"]] = record or {
            "source_url": url,
            "spl_set_id": label["provenance"]["spl_set_id"],
            "media_name": label["provenance"].get("media_name"),
            "retrieved_at": (now or datetime.datetime.now(datetime.timezone.utc)).isoformat(timespec="seconds"),
            "sha256": _sha256(data),
            "bytes": len(data),
            "content_type": content_type,
        }
        _write_json(retrieval_path, records)
        downloaded.append(label["id"])
    return downloaded


def verify(labels, catalog, retrieval_path=RETRIEVAL, check_files=True):
    """Problems (list of strings) with labels vs. the local catalog and files.

    Checks: the SPL set ID has catalog records; the labeled formulation equals
    one of them; a labeled package NDC is a package of that set; each image
    matches its retrieval hash and the ROI lies inside it. A printed UPC that
    FDA does not list for the set is reported as a note, not a problem.
    """
    from vision.products.evaluate import formulation_key

    problems, notes = [], []
    records = json.loads(_abs(retrieval_path).read_text()) if check_files and _abs(retrieval_path).exists() else {}
    for label in labels:
        tag, set_id = label["id"], label["provenance"]["spl_set_id"]
        in_set = [r for r in catalog.records if set_id in r["spl_set_id"]]
        if not in_set:
            problems.append(f"{tag}: SPL set {set_id} has no records in the catalog")
            continue
        truth = formulation_key(label["formulation"])
        if not any(formulation_key([(i["name"], i["strength"]) for i in r["active_ingredients"]]) == truth
                   for r in in_set):
            problems.append(f"{tag}: labeled formulation is not a formulation of SPL set {set_id}")
        if label.get("package_ndc"):
            ids = {r["product_id"] for r in in_set}
            if not any(r["product_id"] in ids for r, _ in catalog.find_by_package_ndc(label["package_ndc"])):
                problems.append(f"{tag}: package NDC {label['package_ndc']} is not a package of SPL set {set_id}")
        for barcode in label.get("barcodes", []):
            if not any(set_id in r["spl_set_id"] for r in catalog.find_by_upc(barcode["digits"])):
                notes.append(f"{tag}: UPC {barcode['digits']} is not listed on this SPL set's FDA label")
        if check_files:
            problems += _check_file(label, records.get(tag))
    return problems, notes


def _check_file(label, record):
    import cv2

    path = _abs(label["image"])
    if not path.exists():
        return [f"{label['id']}: image missing (run fetch)"]
    if record is None or _sha256(path.read_bytes()) != record["sha256"]:
        return [f"{label['id']}: image does not match its retrieval record"]
    image = cv2.imread(str(path))
    if image is None:
        return [f"{label['id']}: image is unreadable"]
    roi = label.get("roi")
    if roi and (roi[2] > image.shape[1] or roi[3] > image.shape[0]):
        return [f"{label['id']}: roi {roi} exceeds image size {image.shape[1]}x{image.shape[0]}"]
    return []


def region(image, label, which):
    """The full image, or its labeled front panel (the whole image when roi is null)."""
    if which == "full" or label.get("roi") is None:
        return image
    x1, y1, x2, y2 = label["roi"]
    return image[y1:y2, x1:x2]


def classify(score):
    """Outcome class from evaluate.score_match output (evaluate.classify, one definition)."""
    from vision.products.evaluate import classify as evaluate_classify
    return evaluate_classify(score)


def evaluate_labels(labels, recognizer, matcher, inputs=INPUTS):
    """One row per (label, input) plus one oracle row per label."""
    import cv2
    from vision.products import evaluate

    rows = []
    for label in labels:
        image = cv2.imread(str(_abs(label["image"])))
        if image is None:
            raise ValueError(f"{label['id']}: can't read {label['image']} (run fetch)")
        oracle = evaluate.score_match(label, evaluate.oracle_result(label, matcher))
        rows.append(dict(id=label["id"], product=label["product"], category=label.get("category"),
                         input="oracle", outcome=classify(oracle), **oracle))
        for which in inputs:
            crop = region(image, label, which)
            started = time.perf_counter()
            ocr = recognizer.read(crop)
            ocr_s = time.perf_counter() - started
            started = time.perf_counter()
            result = matcher.match(ocr)
            match_ms = (time.perf_counter() - started) * 1000
            score = evaluate.score_match(label, result)
            rows.append(dict(
                id=label["id"], product=label["product"], category=label.get("category"), input=which,
                size=f"{crop.shape[1]}x{crop.shape[0]}", outcome=classify(score), **score,
                ocr_scores=evaluate.score_ocr(label, ocr, matcher),
                reason=result.reason, package=result.package,
                ocr=[[r["text"], round(float(r["confidence"]), 3)] for r in ocr],
                ocr_seconds=round(ocr_s, 2), match_ms=round(match_ms, 1)))
    return rows


def replay(previous, labels, matcher):
    """Re-run only the matcher on the OCR text saved in a previous `run` result.

    OCR is not repeated, so OCR variation can't affect the comparison. Oracle rows
    use the labels' visible_text, as in `run`. Each row keeps the previous outcome
    and reason next to the new ones.
    """
    from vision.products import evaluate

    by_id = {label["id"]: label for label in labels}
    rows = []
    for old in previous["rows"]:
        label = by_id[old["id"]]
        if old["input"] == "oracle":
            result = evaluate.oracle_result(label, matcher)
        else:
            result = matcher.match([{"text": text, "confidence": confidence} for text, confidence in old["ocr"]])
        score = evaluate.score_match(label, result)
        rows.append(dict(
            id=old["id"], product=old["product"], category=old.get("category"), input=old["input"],
            outcome=classify(score), **score, reason=result.reason, package=result.package,
            previous_outcome=old["outcome"], previous_level=old["level"], previous_formulations=old["formulations"],
            previous_reason=old.get("reason"), ocr=old.get("ocr")))
    return rows


def summarize(rows):
    summary = {}
    for row in rows:
        counts = summary.setdefault(row["input"], {"n": 0, "CORRECT_VERIFIED": 0, "FALSE_VERIFIED": 0,
                                                   "VERIFIED_PACKAGE_UNSCORED": 0, "UNCERTAIN": 0,
                                                   "NO_MATCH": 0, "truth_in_candidates": 0})
        counts["n"] += 1
        counts[row["outcome"]] += 1
        counts["truth_in_candidates"] += row["truth_rank"] is not None
    return summary


def _git_state():
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "verify", "run", "replay"))
    parser.add_argument("--labels", default=str(LABELS))
    parser.add_argument("--inputs", default=",".join(INPUTS), help="comma-separated: " + ",".join(INPUTS))
    parser.add_argument("--results", help="replay: a previous dailymed-pilot results JSON")
    args = parser.parse_args(argv)
    labels = load_labels(args.labels)

    if args.command == "fetch":
        print("downloaded:", fetch(labels) or "nothing (all present and unchanged)")
        return 0

    if args.command == "replay":
        return _replay_main(args, labels)

    from vision.products import evaluate
    from vision.products.catalog import ProductCatalog
    catalog = ProductCatalog.from_file(evaluate.DEFAULT_CACHE)
    problems, notes = verify(labels, catalog)
    for line in notes:
        print("note:", line)
    for line in problems:
        print("PROBLEM:", line)
    if args.command == "verify" or problems:
        print(f"{len(labels)} labels, {len(problems)} problems")
        return 1 if problems else 0

    inputs = [i for i in args.inputs.split(",") if i]
    if not set(inputs) <= set(INPUTS):
        parser.error(f"--inputs must be from {INPUTS}")
    from vision.ocr import TextRecognizer
    from vision.products.matcher import ProductMatcher
    recognizer, matcher = TextRecognizer(), ProductMatcher(catalog)
    rows = evaluate_labels(labels, recognizer, matcher, inputs)

    for row in rows:
        if row["input"] == "oracle":
            continue
        print(f"{row['id']:18} {row['input']:12} {row['size']:>10} {row['outcome']:26} "
              f"cands={row['formulations']:<3} rank={row['truth_rank']} ocr={row['ocr_seconds']}s "
              f"match={row['match_ms']}ms")
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print(UNREVIEWED_NOTE)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = _abs(RESULTS / f"dailymed-pilot-{stamp}.json")
    if out.exists():
        raise SystemExit(f"{out} already exists")
    _write_json(out, {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "labels": args.labels, "labels_sha256": _sha256(_abs(args.labels).read_bytes()),
        "catalog": os.path.basename(evaluate.DEFAULT_CACHE), "catalog_meta": catalog.meta.get("last_updated"),
        "git": _git_state(), "python": platform.python_version(), "machine": platform.machine(),
        "summary": summary, "rows": rows,
    })
    print("results:", out.relative_to(ROOT))
    return 0


def _replay_main(args, labels):
    from vision.products import evaluate
    from vision.products.catalog import ProductCatalog
    from vision.products.matcher import ProductMatcher

    if not args.results:
        raise SystemExit("replay needs --results PATH")
    source = _abs(Path(args.results))
    previous = json.loads(source.read_text())
    labels_sha = _sha256(_abs(args.labels).read_bytes())
    if previous.get("labels_sha256") != labels_sha:
        raise SystemExit("labels changed since that run; a replay would not compare like with like")
    catalog = ProductCatalog.from_file(evaluate.DEFAULT_CACHE)
    rows = replay(previous, labels, ProductMatcher(catalog))

    for row in rows:
        changed = "  <-- changed" if row["outcome"] != row["previous_outcome"] else ""
        print(f"{row['id']:18} {row['input']:12} {row['previous_outcome']:18} -> {row['outcome']:18}{changed}")
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print(UNREVIEWED_NOTE)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = _abs(RESULTS / f"dailymed-pilot-replay-{stamp}.json")
    if out.exists():
        raise SystemExit(f"{out} already exists")
    _write_json(out, {
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "replay_of": str(source.relative_to(ROOT)), "labels": args.labels, "labels_sha256": labels_sha,
        "catalog": os.path.basename(evaluate.DEFAULT_CACHE), "git": _git_state(),
        "summary": summary, "previous_summary": previous.get("summary"), "rows": rows,
    })
    print("results:", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
