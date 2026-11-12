"""The vision pipeline as a long-lived service for the web app.

Models and data are loaded once (at startup via preload(), or on first use)
and reused for every request. All business logic stays in vision/: this
module only wires the existing pieces together, the same way main.py does.
"""
import os
import threading
import time

FDA_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "vision", "products", "data", "otc_ndc_2026-10-30.json")


class ServiceError(Exception):
    """A failure to report to the client as JSON: (http status, code, message)."""

    def __init__(self, http_status, code, message):
        super().__init__(message)
        self.http_status, self.code, self.message = http_status, code, message


def _load_catalog():
    from vision.products.catalog import ProductCatalog
    if not os.path.exists(FDA_CACHE):
        raise FileNotFoundError("FDA cache not found. Build it with: python -m vision.products.build_cache")
    return ProductCatalog.from_file(FDA_CACHE)


def _load_registry():
    from vision.products.upc_registry import UpcRegistry
    return UpcRegistry.from_file()


def _load_detector():
    from vision.detector import ObjectDetector
    return ObjectDetector()


def _load_recognizer():
    from vision.ocr import TextRecognizer
    return TextRecognizer()


DEFAULT_LOADERS = {"catalog": _load_catalog, "registry": _load_registry,
                   "detector": _load_detector, "recognizer": _load_recognizer}


class VisionServices:
    def __init__(self, loaders=None, busy_timeout=30, detect_busy_timeout=2):
        self._loaders = dict(DEFAULT_LOADERS, **(loaders or {}))
        self._loaded = {}
        self._load_lock = threading.Lock()
        self._ocr_lock = threading.Lock()
        self._work_lock = threading.Lock()  # YOLO/EasyOCR run one request at a time
        self._busy_timeout = busy_timeout
        self._detect_busy_timeout = detect_busy_timeout  # awareness frames give up quickly instead of queueing
        self.registry_error = None

    def _get(self, name):
        """The named component, loading it once. A failed load is retried next time."""
        if name in self._loaded:
            return self._loaded[name]
        with self._load_lock:
            if name not in self._loaded:
                try:
                    if name == "matcher":
                        from vision.products.matcher import ProductMatcher
                        self._loaded[name] = ProductMatcher(self._get_unlocked("catalog"))
                    else:
                        self._loaded[name] = self._loaders[name]()
                except Exception as error:
                    if name == "registry":  # like main.py: scans still run, FDA lookup only
                        self.registry_error = str(error)
                        self._loaded[name] = None
                    else:
                        raise ServiceError(503, "initialization_failed",
                                           f"Could not load the {name}: {error}") from error
            return self._loaded[name]

    def _get_unlocked(self, name):
        if name not in self._loaded:
            self._loaded[name] = self._loaders[name]()
        return self._loaded[name]

    def preload(self, names=("catalog", "registry", "matcher", "detector", "recognizer")):
        for name in names:
            self._get(name)

    def loaded(self):
        return sorted(name for name, value in self._loaded.items() if value is not None)

    def _working(self, timeout=None):
        if not self._work_lock.acquire(timeout=self._busy_timeout if timeout is None else timeout):
            raise ServiceError(503, "busy", "The server is still processing another image. Try again.")
        return self._work_lock

    def detect(self, image):
        """Every YOLO detection at the detector's configured threshold.

        Returns (detections, inference_ms). Uses the same detector instance as identify().
        """
        detector = self._get("detector")
        lock = self._working(self._detect_busy_timeout)
        try:
            started = time.perf_counter()
            detections = detector.get_detections(detector.detect(image))
            return detections, (time.perf_counter() - started) * 1000
        finally:
            lock.release()

    def scan_barcodes(self, image):
        """BarcodeIdentification list for the image (FDA lookup + local registry)."""
        from vision.barcode_scan import scan_frame
        catalog, registry = self._get("catalog"), self._get("registry")
        lock = self._working()
        try:
            return scan_frame(image, catalog, registry)
        finally:
            lock.release()

    def identify(self, image):
        """YOLO -> best bottle -> crop -> OCR -> ProductMatcher.

        Returns (all detections, identified) where identified is
        (detection, ocr_results, match_result) or None when there is no bottle.
        """
        from vision.identification import identify_detection
        detector, recognizer, matcher = self._get("detector"), self._get("recognizer"), self._get("matcher")
        detections, _ = self.detect(image)
        if not self._ocr_lock.acquire(timeout=self._busy_timeout):
            raise ServiceError(503, "busy", "Another label scan is running.")
        try:
            return detections, identify_detection(image, detections, recognizer, matcher)
        finally:
            self._ocr_lock.release()

    def identify_crop(self, image):
        """A Phase 4A crop is already localized. Reuse the existing helper with
        its whole-image ROI; never infer a second bottle inside the crop.
        """
        from vision.identification import identify_detection
        from web.medicine_result import explain
        if not self._ocr_lock.acquire(blocking=False):
            raise ServiceError(409, "identification_busy", "Another label scan is running. Rescan when it finishes.")
        started = time.perf_counter()
        times = {"ocr_ms": 0, "matcher_ms": 0}
        class Timed:
            def __init__(self, component, method, key):
                self.component, self.method, self.key = component, method, key
            def __getattr__(self, method):
                def call(*args):
                    before = time.perf_counter()
                    try:
                        return getattr(self.component, self.method)(*args)
                    finally:
                        times[self.key] = round((time.perf_counter()-before)*1000, 2)
                return call
        try:
            recognizer, matcher = self._get("recognizer"), self._get("matcher")
            height, width = image.shape[:2]
            roi = {"class_name": "bottle", "confidence": 1.0, "box": (0, 0, width, height)}
            _, ocr, matched = identify_detection(image, [roi],
                Timed(recognizer, "read", "ocr_ms"), Timed(matcher, "match", "matcher_ms"))
            body = explain(ocr, matched)
            body["ocr"] = [dict(text=r["text"], confidence=float(r["confidence"])) for r in ocr]
            times["total_ms"] = round((time.perf_counter()-started)*1000, 2)
            body["timing"] = times
            return body
        finally:
            self._ocr_lock.release()
