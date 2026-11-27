"""Reproducible latency benchmark for the Visual Assistant's local components.

Each component runs in its own Python process, so that
  * "load" and the first ("cold") call are measured from a fresh process, and
  * peak memory is the peak of that component alone.
No camera, network, Flask app or vision-language model is used, and nothing is
downloaded.

Usage (from the repo root):
    python -m research.benchmark --images DIR
    python -m research.benchmark --images DIR --components yolo,ocr --trials 20
    python -m research.benchmark --images DIR --max-sides 0,1280,640     # resolution sweep

These are component times on this machine, not end-to-end latency of main.py
(camera capture, display and speech are not included). Percentile gates: the
median is reported for n >= 3 and p95 for n >= 20 warm calls; smaller samples
report only n, min and max. Accuracy is measured by vision/products/evaluate.py.
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time

try:
    import resource                     # Unix only; peak memory is unavailable without it (e.g. Windows)
except ImportError:
    resource = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_TYPES = (".jpg", ".jpeg", ".png")

# One real OCR reading (the live Equate observation recorded on 2026-11-08), used so the
# matcher can be timed without running OCR first. It is an input, not a ground-truth label.
FIXED_OCR = [("equate", 1.00), ("Ibuprofen", 1.00), ("2oomg", 0.32), ("mg", 0.84), ("Tablets", 1.00),
             ("Pain", 1.00), ("Reliever/", 0.67), ("Fever", 1.00), ("(NSAID)", 0.99)]
FIXED_UPCS = ["305730150200", "681131699372"]            # one FDA-resolvable, one unresolved
MIN_N_MEDIAN, MIN_N_P90, MIN_N_P95 = 3, 10, 20             # percentile gates (research/methodology)


# ---------------------------------------------------------------- helpers

def now_ms():
    return time.perf_counter() * 1000


def peak_rss_mb():
    """Peak resident memory of this process in MiB, or None where the platform can't report it."""
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)   # bytes on macOS, KB on Linux


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = (len(ordered) - 1) * p
    low, high = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (rank - low), 2)


def summarize(values):
    """n, min, max always; median, p90 and p95 only once n reaches their gate."""
    if not values:
        return {"n": 0}
    out = {"n": len(values)}
    if len(values) >= MIN_N_MEDIAN:
        out["median"] = round(statistics.median(values), 2)
    if len(values) >= MIN_N_P90:
        out["p90"] = percentile(values, 0.90)
    if len(values) >= MIN_N_P95:
        out["p95"] = percentile(values, 0.95)
    out.update(min=round(min(values), 2), max=round(max(values), 2))
    return out


def list_images(directory):
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(IMAGE_TYPES))
    return [os.path.join(directory, n) for n in names]


def resized(image, max_side):
    import cv2
    if not max_side or max(image.shape[:2]) <= max_side:
        return image
    scale = max_side / max(image.shape[:2])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- components
# Each returns (config, call, extra) where call(image) runs the component once
# and may return a small dict of facts to keep (never image data).

def c_decode(args):
    import cv2
    import numpy as np
    files = {}

    def call(image, path=None):
        data = files.get(path)
        if data is None:
            with open(path, "rb") as f:
                data = files[path] = np.frombuffer(f.read(), np.uint8)
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return {"pixels": int(decoded.shape[0] * decoded.shape[1])}
    return {"what": "cv2.imdecode of the image file bytes"}, call


def c_yolo(args):
    from vision.detector import ObjectDetector
    detector = ObjectDetector()
    config = {"model": "yolo26n.pt", "confidence": detector.conf, "device": str(detector.model.device)}

    def call(image, path=None):
        found = detector.get_detections(detector.detect(image))
        return {"objects": len(found), "bottles": sum(d["class_name"] == "bottle" for d in found)}
    return config, call


def c_crop(args):
    from vision.identification import crop_box

    def call(image, path=None):
        height, width = image.shape[:2]
        crop = crop_box(image, (width // 4, height // 4, 3 * width // 4, 3 * height // 4))
        return {"pixels": int(crop.shape[0] * crop.shape[1])}
    return {"what": "crop_box of the centre half of the frame (a NumPy view)"}, call


def c_ocr(args):
    from vision.ocr import TextRecognizer
    recognizer = TextRecognizer()

    def call(image, path=None):
        results = recognizer.read(image)
        return {"text_items": len(results)}
    return {"model": "EasyOCR en", "gpu": False, "input": "whole frame (not a YOLO crop)"}, call


def c_matcher(args):
    from vision.products.catalog import ProductCatalog
    from vision.products.matcher import ProductMatcher
    from vision.products.evaluate import DEFAULT_CACHE as FDA_CACHE
    matcher = ProductMatcher(ProductCatalog.from_file(FDA_CACHE))
    observations = [{"text": text, "confidence": confidence} for text, confidence in FIXED_OCR]

    def call(image, path=None):
        result = matcher.match(observations)
        return {"status": result.status.value, "candidates": len(result.candidates)}
    return {"catalog": os.path.basename(FDA_CACHE), "input": "one fixed real OCR reading (no image used)",
            "synthetic_input": True}, call


def c_barcode_decode(args):
    from vision.barcode import decode_barcodes

    def call(image, path=None):
        found = decode_barcodes(image)
        return {"barcodes": len(found), "decoded": sum(o["decoded"] for o in found)}
    return {"library": "zxing-cpp"}, call


def c_barcode_lookup(args):
    from vision.products.barcode_lookup import lookup_barcode
    from vision.products.catalog import ProductCatalog
    from vision.products.evaluate import DEFAULT_CACHE as FDA_CACHE
    catalog = ProductCatalog.from_file(FDA_CACHE)

    def call(image, path=None):
        statuses = [lookup_barcode({"type": "UPC_A", "value": upc, "decoded": True}, catalog).status.value
                    for upc in FIXED_UPCS]
        return {"statuses": ",".join(statuses)}
    return {"input": f"{len(FIXED_UPCS)} fixed UPC values per call (no image used)", "synthetic_input": True}, call


COMPONENTS = {
    "decode": c_decode, "yolo": c_yolo, "crop": c_crop, "ocr": c_ocr, "matcher": c_matcher,
    "barcode_decode": c_barcode_decode, "barcode_lookup": c_barcode_lookup,
}
NO_IMAGE = ("matcher", "barcode_lookup")
RESIZABLE = ("yolo", "ocr", "barcode_decode")
DEFAULT = list(COMPONENTS)


# ---------------------------------------------------------------- child process: one component

def run_child(args):
    import cv2
    name, max_side = args.child, args.child_max_side
    out = {"component": name, "max_side": max_side or None, "status": "ok", "failures": []}
    try:
        started = now_ms()
        config, call = COMPONENTS[name](args)
        out["config"], out["load_ms"] = config, round(now_ms() - started, 2)
        paths = list_images(args.images)
        images = [(p, resized(cv2.imread(p), max_side)) for p in paths]
        images = [(p, image) for p, image in images if image is not None]
        if name in NO_IMAGE:
            images = images[:1] or [(None, None)]
        out["image_sizes"] = sorted({f"{i.shape[1]}x{i.shape[0]}" for _, i in images if i is not None})

        def one(path, image):
            begun = now_ms()
            facts = call(image, path=path) or {}
            elapsed = now_ms() - begun
            if facts.pop("_fail", False):
                raise RuntimeError(f"HTTP {facts.get('http_status')}")
            return elapsed, facts

        warm, stages, facts_seen = [], {}, []
        for trial in range(args.trials + 1):                          # trial 0 is the cold call
            for index, (path, image) in enumerate(images):
                try:
                    elapsed, facts = one(path, image)
                except Exception as error:
                    out["failures"].append({"trial": trial, "image": os.path.basename(path or ""),
                                            "error": f"{type(error).__name__}: {str(error)[:160]}"})
                    continue
                stage_times = facts.pop("_stages", {})
                if trial == 0 and index == 0:
                    out["cold_ms"], out["cold_stages"] = round(elapsed, 2), stage_times
                    out["first_result"] = facts
                    continue
                if trial == 0:
                    continue                                     # other images on the cold pass: warm-up only
                warm.append(elapsed)
                for stage, value in stage_times.items():
                    stages.setdefault(stage, []).append(value)
                facts_seen.append(facts)
        out["warm_ms"] = summarize(warm)
        out["warm_stages_ms"] = {stage: summarize(values) for stage, values in stages.items()}
        outcomes = {}
        for facts in facts_seen:
            key = json.dumps(facts, sort_keys=True)
            outcomes[key] = outcomes.get(key, 0) + 1
        out["warm_outcomes"] = [{"result": json.loads(k), "count": v} for k, v in outcomes.items()][:8]
        if out["failures"] and not warm and "cold_ms" not in out:
            out["status"] = "failed"
    except Exception as error:
        out["status"], out["error"] = "failed", f"{type(error).__name__}: {str(error)[:300]}"
    out["peak_rss_mb"] = peak_rss_mb()
    if out["peak_rss_mb"] is None:
        out["peak_rss_note"] = "peak memory unavailable on this platform (no resource module)"
    print("##RESULT##" + json.dumps(out))


# ---------------------------------------------------------------- parent: run every component

def environment(args):
    def version(module):
        from importlib import metadata
        for distribution in (module, {"cv2": "opencv-python", "zxingcpp": "zxing-cpp"}.get(module, module)):
            try:
                return metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
        return None

    def git(*command):
        try:
            return subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    cpu = None
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    import cv2
    images = []
    for path in list_images(args.images):
        image = cv2.imread(path)
        images.append({"file": os.path.basename(path), "bytes": os.path.getsize(path),
                       "size": None if image is None else f"{image.shape[1]}x{image.shape[0]}"})
    return {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "platform": platform.platform(), "cpu": cpu or platform.processor(),
        "cpu_count": os.cpu_count(), "python": platform.python_version(),
        "versions": {m: version(m) for m in ("torch", "ultralytics", "easyocr", "cv2", "numpy", "zxingcpp")},
        "git_commit": git("rev-parse", "--short", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "images": images, "trials": args.trials, "timeout_s": args.timeout,
    }


def run_component(args, name, max_side):
    command = [sys.executable, "-m", "research.benchmark", "--images", args.images, "--child", name,
               "--child-max-side", str(max_side), "--trials", str(args.trials)]
    started = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return {"component": name, "max_side": max_side or None, "status": "timeout", "timeout_s": args.timeout}
    for line in done.stdout.splitlines():
        if line.startswith("##RESULT##"):
            result = json.loads(line[len("##RESULT##"):])
            result["process_seconds"] = round(time.perf_counter() - started, 1)
            return result
    return {"component": name, "max_side": max_side or None, "status": "failed",
            "error": (done.stderr or done.stdout)[-400:]}


def write_csv(path, results):
    fields = ["component", "max_side", "status", "load_ms", "cold_ms", "warm_n", "warm_median_ms", "warm_p90_ms",
              "warm_p95_ms", "warm_min_ms", "warm_max_ms", "failures", "peak_rss_mb"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            warm = r.get("warm_ms", {})
            writer.writerow({"component": r["component"], "max_side": r.get("max_side") or "", "status": r["status"],
                             "load_ms": r.get("load_ms", ""), "cold_ms": r.get("cold_ms", ""), "warm_n": warm.get("n", 0),
                             "warm_median_ms": warm.get("median", ""), "warm_p90_ms": warm.get("p90", ""),
                             "warm_p95_ms": warm.get("p95", ""), "warm_min_ms": warm.get("min", ""),
                             "warm_max_ms": warm.get("max", ""), "failures": len(r.get("failures", [])),
                             "peak_rss_mb": r.get("peak_rss_mb", "")})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="directory of .jpg/.png images")
    parser.add_argument("--components", default=",".join(DEFAULT), help="comma-separated: " + ",".join(COMPONENTS))
    parser.add_argument("--trials", type=int, default=10, help="warm passes over the images (default 10)")
    parser.add_argument("--max-sides", default="0", help="comma-separated longest-side limits; 0 = original size")
    parser.add_argument("--timeout", type=int, default=600, help="seconds allowed per component process")
    parser.add_argument("--out", default=os.path.join(ROOT, "research", "results"))
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-max-side", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return run_child(args)

    if not os.path.isdir(args.images) or not list_images(args.images):
        sys.exit(f"No .jpg/.png images in {args.images}")
    names = [n.strip() for n in args.components.split(",") if n.strip()]
    unknown = [n for n in names if n not in COMPONENTS]
    if unknown:
        sys.exit(f"Unknown components: {unknown}. Choose from: {', '.join(COMPONENTS)}")
    sides = [int(s) for s in args.max_sides.split(",")]
    env = environment(args)

    results = []
    for name in names:
        for side in (sides if name in RESIZABLE else [0]):
            label = f"{name}" + (f" @{side}px" if side else "")
            print(f"running {label} ...", flush=True)
            result = run_component(args, name, side)
            results.append(result)
            warm = result.get("warm_ms", {})
            print(f"  {result['status']}: load {result.get('load_ms', '-')} ms, cold {result.get('cold_ms', '-')} ms, "
                  f"warm median {warm.get('median', '-')} ms (p95 {warm.get('p95', '-')}, n={warm.get('n', 0)}), "
                  f"failures {len(result.get('failures', []))}, peak {result.get('peak_rss_mb') or 'unavailable'} MB", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path, csv_path = os.path.join(args.out, f"benchmark-{stamp}.json"), os.path.join(args.out, f"benchmark-{stamp}.csv")
    with open(json_path, "w") as f:
        json.dump({"environment": env, "results": results}, f, indent=2)
    write_csv(csv_path, [r for r in results if "component" in r])
    print(f"\nSaved {os.path.relpath(json_path, ROOT)} and {os.path.relpath(csv_path, ROOT)}")


if __name__ == "__main__":
    main()
