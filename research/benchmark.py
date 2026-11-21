"""Reproducible latency benchmark for the existing Visual Assistant components.

Each component runs in its own Python process, so that
  * "load" and the first ("cold") call are measured from a fresh process, and
  * peak memory is the peak of that component alone.
No phone, camera or browser is needed, and nothing is downloaded.

Usage (from the repo root):
    python -m research.benchmark --images vision
    python -m research.benchmark --images DIR --components yolo,ocr --trials 20
    python -m research.benchmark --images DIR --max-sides 0,1280,640     # resolution sweep
    python -m research.benchmark --images DIR --online                   # also openFDA research

What is and is not measured: see research/methodology.md. In short, these are
component and server-side times on this machine. They are not end-to-end
phone latencies; those come from the browser instrumentation (web/static/timing.js).
"""
import argparse
import csv
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_TYPES = (".jpg", ".jpeg", ".png")
OLLAMA = "http://127.0.0.1:11434/api/"

# One real OCR reading (the live Equate observation recorded on 2026-11-08), used so the
# matcher can be timed without running OCR first. It is an input, not a ground-truth label.
FIXED_OCR = [("equate", 1.00), ("Ibuprofen", 1.00), ("2oomg", 0.32), ("mg", 0.84), ("Tablets", 1.00),
             ("Pain", 1.00), ("Reliever/", 0.67), ("Fever", 1.00), ("(NSAID)", 0.99)]
FIXED_UPCS = ["305730150200", "681131699372"]            # one FDA-resolvable, one unresolved
FIXED_RESEARCH = {"evidence": [{"kind": "ingredient", "value": "IBUPROFEN"}, {"kind": "strength", "value": "200mg"}],
                  "ocr": [{"text": "Ibuprofen Tablets 200 mg", "confidence": 0.95}]}


# ---------------------------------------------------------------- helpers

def now_ms():
    return time.perf_counter() * 1000


def peak_rss_mb():
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
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": round(statistics.median(values), 2), "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95), "min": round(min(values), 2), "max": round(max(values), 2)}


def list_images(directory):
    names = sorted(n for n in os.listdir(directory) if n.lower().endswith(IMAGE_TYPES))
    return [os.path.join(directory, n) for n in names]


def resized(image, max_side):
    import cv2
    if not max_side or max(image.shape[:2]) <= max_side:
        return image
    scale = max_side / max(image.shape[:2])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def ollama(path, payload=None, timeout=5):
    request = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode() if payload else None,
                                     headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def qwen_model():
    return os.environ.get("QWEN_VISION_MODEL", "qwen2.5vl:3b")


def qwen_unavailable():
    """None when the configured local model is installed in a running Ollama; else why not."""
    try:
        installed = [m.get("name") for m in ollama("tags").get("models", [])]
    except Exception as error:
        return f"Ollama is not reachable on 127.0.0.1:11434 ({type(error).__name__})"
    if qwen_model() not in installed:
        return f"{qwen_model()} is not installed in Ollama (not downloading it)"
    return None


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
    return {"what": "cv2.imdecode of the file bytes (as the server does for an upload)"}, call


def c_yolo(args):
    from vision.detector import ObjectDetector
    detector = ObjectDetector()
    config = {"model": "yolo26n.pt", "confidence": detector.conf, "device": str(detector.model.device)}

    def call(image, path=None):
        found = detector.get_detections(detector.detect(image))
        return {"objects": len(found), "bottles": sum(d["class_name"] == "bottle" for d in found)}
    return config, call


def c_tracking(args):
    from vision.tracking import ObjectTracker
    tracker, state = ObjectTracker(), {"t": 0.0}
    boxes = [(40 + 55 * i, 60, 40 + 55 * i + 50, 300) for i in range(10)]

    def call(image, path=None):
        state["t"] += 1.5
        shift = int(state["t"]) % 7
        detections = [{"class_name": "person" if i % 2 else "chair", "box": (x1 + shift, y1, x2 + shift, y2)}
                      for i, (x1, y1, x2, y2) in enumerate(boxes)]
        ids = tracker.update(detections, state["t"])
        return {"objects": len(ids)}
    return {"what": "ObjectTracker.update on 10 synthetic boxes (no image used)", "synthetic_input": True}, call


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
    from web.services import FDA_CACHE
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
    from vision.products.barcode_identify import identify_barcode
    from vision.products.catalog import ProductCatalog
    from vision.products.upc_registry import UpcRegistry
    from web.services import FDA_CACHE
    catalog, registry = ProductCatalog.from_file(FDA_CACHE), UpcRegistry.from_file()

    def call(image, path=None):
        statuses = [identify_barcode({"type": "UPC_A", "value": upc, "decoded": True}, catalog, registry).final_status.value
                    for upc in FIXED_UPCS]
        return {"statuses": ",".join(statuses)}
    return {"input": f"{len(FIXED_UPCS)} fixed UPC values per call (no image used)", "synthetic_input": True}, call


def _api(endpoint, field_builder=None):
    """A real request through the Flask app in-process: server-side total plus Server-Timing stages."""
    def build(args):
        import cv2
        from web.server import create_app
        from web.services import VisionServices
        services = VisionServices()
        services.preload()
        client = create_app(services).test_client()
        encoded = {}

        def call(image, path=None):
            import io
            key = (path, image.shape)
            if key not in encoded:
                encoded[key] = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
            data = {"image": (io.BytesIO(encoded[key]), "frame.jpg")}
            data.update(field_builder(call) if field_builder else {})
            response = client.post(endpoint, data=data, content_type="multipart/form-data")
            stages = {}
            for part in response.headers.get("Server-Timing", "").split(","):
                name, _, rest = part.strip().partition(";dur=")
                if rest:
                    stages[name] = float(rest)
            body = response.get_json(silent=True) or {}
            return {"http_status": response.status_code, "outcome": body.get("status") or body.get("tracking"),
                    "_stages": stages, "_fail": response.status_code >= 400}
        call.frame = 0
        return {"what": f"POST {endpoint} through the Flask app in-process (no network, no TLS)",
                "upload": "JPEG quality 90 of the benchmark image"}, call
    return build


def _tracking_fields(call):
    call.frame += 1
    return {"session_id": "benchmark-session", "frame": str(call.frame)}


def _qwen(reader, keep_alive):
    """keep_alive None = exactly as deployed (model unloaded after every call)."""
    def build(args):
        import cv2
        import numpy as np
        from werkzeug.datastructures import FileStorage
        import io
        from web.qwen_label import QwenLabelReader
        from web.qwen_scene import QwenSceneService, prepare_image
        service = QwenLabelReader() if reader else QwenSceneService()
        original, seen = service._post, {}

        def post(path, payload, deadline):
            if path == "chat" and keep_alive is not None:
                payload = dict(payload, keep_alive=keep_alive)        # benchmark-only variant
            body = original(path, payload, deadline)
            if path == "chat":
                seen.update({k: round(body.get(k, 0) / 1e6, 1) for k in
                             ("load_duration", "prompt_eval_duration", "eval_duration", "total_duration")})
                seen["output_tokens"] = body.get("eval_count")
            return body
        service._post = post

        def call(image, path=None):
            seen.clear()
            if keep_alive is not None:
                # Ollama reuses its work for an image it has just seen. Shift the picture a
                # little each call so a loaded model is still timed on an unseen image.
                call.shift += 8
                image = np.roll(image, call.shift, axis=1)
            if reader:
                out = service.read(image, [])
                facts = {"lines": len(out["lines"])}
            else:
                jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                out = service.describe(prepare_image(FileStorage(io.BytesIO(jpeg), "f.jpg", content_type="image/jpeg")))
                facts = {"description_chars": len(out["description"])}
            facts["_stages"] = {f"ollama_{k}": v for k, v in seen.items() if k != "output_tokens"}
            facts["output_tokens"] = seen.get("output_tokens")
            return facts
        call.shift = 0
        return {"model": service.model, "keep_alive": "0 (as deployed: unloaded after each call)" if keep_alive is None
                else f"{keep_alive} (benchmark-only variant: model stays loaded; image shifted each call so it is unseen)",
                "image": "resized to at most 1024 px by the existing code", "memory_note":
                "the model runs inside Ollama; this process's peak memory does not include it"}, call
    return build


def c_research(args):
    from web.product_research import ProductResearch
    service = ProductResearch()

    def call(image, path=None):
        service.cache.clear()                                    # time the network call, not the cache
        out = service.research(json.loads(json.dumps(FIXED_RESEARCH)))
        return {"status": out["status"], "records": len(out["records"])}
    return {"service": "api.fda.gov/drug/label.json", "input": "fixed ingredient + strength evidence",
            "network": True, "synthetic_input": True}, call


COMPONENTS = {
    "decode": c_decode, "yolo": c_yolo, "tracking": c_tracking, "crop": c_crop, "ocr": c_ocr, "matcher": c_matcher,
    "barcode_decode": c_barcode_decode, "barcode_lookup": c_barcode_lookup,
    "api_detect": _api("/api/detect", _tracking_fields), "api_identify": _api("/api/identify"),
    "api_barcode": _api("/api/barcode"),
    "qwen_scene": _qwen(False, None), "qwen_scene_loaded": _qwen(False, "5m"), "qwen_label": _qwen(True, None),
    "research": c_research,
}
VLM = ("qwen_scene", "qwen_scene_loaded", "qwen_label")
NO_IMAGE = ("tracking", "matcher", "barcode_lookup", "research")
RESIZABLE = ("yolo", "ocr", "barcode_decode", "api_detect", "api_identify")
DEFAULT = [c for c in COMPONENTS if c != "research"]


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
        trials = args.vlm_trials if name in VLM else args.trials
        for trial in range(trials + 1):                          # trial 0 is the cold call
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
        "versions": {m: version(m) for m in ("torch", "ultralytics", "easyocr", "cv2", "flask", "zxingcpp")},
        "git_commit": git("rev-parse", "--short", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
        "qwen_model": qwen_model(), "qwen_unavailable": qwen_unavailable(),
        "images": images, "trials": args.trials, "vlm_trials": args.vlm_trials, "timeout_s": args.timeout,
    }


def run_component(args, name, max_side):
    command = [sys.executable, "-m", "research.benchmark", "--images", args.images, "--child", name,
               "--child-max-side", str(max_side), "--trials", str(args.trials), "--vlm-trials", str(args.vlm_trials)]
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
    parser.add_argument("--vlm-trials", type=int, default=3, help="warm passes for the Qwen components (default 3)")
    parser.add_argument("--max-sides", default="0", help="comma-separated longest-side limits; 0 = original size")
    parser.add_argument("--timeout", type=int, default=600, help="seconds allowed per component process")
    parser.add_argument("--online", action="store_true", help="allow the openFDA product research component")
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
        if name == "research" and not args.online:
            results.append({"component": name, "status": "skipped", "reason": "needs internet; pass --online"})
            continue
        if name in VLM and env["qwen_unavailable"]:
            results.append({"component": name, "status": "unavailable", "reason": env["qwen_unavailable"]})
            continue
        for side in (sides if name in RESIZABLE else [0]):
            label = f"{name}" + (f" @{side}px" if side else "")
            print(f"running {label} ...", flush=True)
            result = run_component(args, name, side)
            results.append(result)
            warm = result.get("warm_ms", {})
            print(f"  {result['status']}: load {result.get('load_ms', '-')} ms, cold {result.get('cold_ms', '-')} ms, "
                  f"warm median {warm.get('median', '-')} ms (p95 {warm.get('p95', '-')}, n={warm.get('n', 0)}), "
                  f"failures {len(result.get('failures', []))}, peak {result.get('peak_rss_mb', '-')} MB", flush=True)

    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path, csv_path = os.path.join(args.out, f"benchmark-{stamp}.json"), os.path.join(args.out, f"benchmark-{stamp}.csv")
    with open(json_path, "w") as f:
        json.dump({"environment": env, "results": results}, f, indent=2)
    write_csv(csv_path, [r for r in results if "component" in r])
    print(f"\nSaved {os.path.relpath(json_path, ROOT)} and {os.path.relpath(csv_path, ROOT)}")


if __name__ == "__main__":
    main()
