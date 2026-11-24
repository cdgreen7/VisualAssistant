"""Web API tests. YOLO and EasyOCR are replaced by fakes; barcode decoding,
the FDA lookup (fixture catalog), the registry and the matcher are real."""
import io
import json
import os
import unittest

import cv2
import numpy as np
import zxingcpp

from tests.test_barcode_lookup import raw
from tests.test_upc_registry import entry, pending
from vision.products.catalog import ProductCatalog
from vision.products.upc_registry import UpcRegistry
from web.server import create_app
from web.services import VisionServices

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "openfda_ndc_sample.json")
F = zxingcpp.BarcodeFormat


def png(image):
    return cv2.imencode(".png", image)[1].tobytes()


def barcode_image(fmt, text):
    return png(np.array(zxingcpp.write_barcode(fmt, text, 300, 120, quiet_zone=20)))


BLANK = png(np.full((240, 320, 3), 255, np.uint8))


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, frame):
        return "raw result"

    def get_detections(self, result):
        return self.detections


class FakeRecognizer:
    def __init__(self, results):
        self.results, self.seen_shapes = results, []

    def read(self, image):
        self.seen_shapes.append(image.shape)
        return self.results


def bottle(confidence=np.float32(0.8), box=(np.int64(10), np.int64(20), np.int64(110), np.int64(220))):
    return {"class_id": 39, "class_name": "bottle", "confidence": confidence, "box": box}


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE) as f:
            cls.catalog = ProductCatalog.from_openfda_results(json.load(f)["results"])
        cls.registry = UpcRegistry([
            pending("681131699372"),
            entry(upc="012345678905", product_name="Store Brand Ibuprofen Tablets", brand="Store Brand",
                  package_ndc=None, package_description=None)])

    def client(self, detections=(), ocr=(), **loaders):
        self.recognizer = FakeRecognizer(list(ocr))
        self.load_counts = {}

        def counted(name, value):
            def load():
                self.load_counts[name] = self.load_counts.get(name, 0) + 1
                return value
            return load

        defaults = {"catalog": counted("catalog", self.catalog), "registry": counted("registry", self.registry),
                    "detector": counted("detector", FakeDetector(list(detections))),
                    "recognizer": counted("recognizer", self.recognizer)}
        self.services = VisionServices(dict(defaults, **loaders), busy_timeout=1, detect_busy_timeout=0.05)
        return create_app(self.services, max_upload_bytes=200_000).test_client()

    def post(self, client, path, data=BLANK, field="image"):
        files = {} if data is None else {field: (io.BytesIO(data), "capture.jpg")}
        return client.post(path, data=files, content_type="multipart/form-data")


class GeneralTests(ApiTestCase):
    def test_health(self):
        response = self.client().get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_index_page(self):
        response = self.client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Capture barcode", response.data)
        self.assertIn(b"Identify medicine", response.data)

    def test_page_has_voice_controls_off_by_default(self):
        client = self.client()
        page = client.get("/").data.decode()
        for expected in ('id="voice-toggle" type="button" class="tile" aria-pressed="false"', 'id="voice-volume"',
                         "static/announcer.js", "static/view.js", "static/ui.js", 'id="scene"',
                         'id="awareness-toggle"', 'id="ask-ai"', 'id="more-tools"', 'id="welcome-start"',
                         "The camera is off.", 'id="flip-camera"', 'id="scan-barcode"', 'id="identify"'):
            self.assertIn(expected, page)
        self.assertLess(page.index("announcer.js"), page.index("app.js"))  # app.js uses it
        self.assertEqual(client.get("/static/announcer.js").status_code, 200)

    def test_unknown_route_and_method_are_json_errors(self):
        client = self.client()
        self.assertEqual(client.get("/api/nope").get_json()["error"]["code"], "http_error")
        self.assertEqual(client.get("/api/barcode").status_code, 405)


class UploadValidationTests(ApiTestCase):
    def test_missing_image(self):
        client = self.client()
        for path in ("/api/barcode", "/api/identify"):
            for response in (self.post(client, path, None), self.post(client, path, BLANK, field="file"),
                             self.post(client, path, b"")):
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"]["code"], "missing_image")

    def test_invalid_image(self):
        client = self.client()
        for path in ("/api/barcode", "/api/identify"):
            for data in (b"this is not an image", BLANK[:40], b"/etc/passwd"):
                response = self.post(client, path, data)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"]["code"], "invalid_image")

    def test_oversized_image(self):
        client = self.client()
        for path in ("/api/barcode", "/api/identify"):
            response = self.post(client, path, b"x" * 300_000)
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.get_json()["error"]["code"], "image_too_large")

    def test_a_path_in_the_form_is_never_read(self):
        response = self.client().post("/api/barcode", data={"image": "vision/advil_front.jpg", "path": "/etc/passwd"})
        self.assertEqual(response.status_code, 400)


class BarcodeApiTests(ApiTestCase):
    def scan(self, data):
        response = self.post(self.client(), "/api/barcode", data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        return response.get_json()

    def test_valid_barcode_resolved_by_fda(self):
        body = self.scan(barcode_image(F.UPCA, "305730150200"))
        self.assertEqual(body["status"], "OK")
        (found,) = body["barcodes"]
        self.assertEqual(found["barcode"], {"type": "UPC_A", "value": "305730150200", "decoded": True,
                                            "reason": None, "box": found["barcode"]["box"]})
        self.assertEqual((found["final_status"], found["source"]), ("IDENTIFIED_FROM_FDA", "FDA"))
        self.assertEqual(found["fda"]["status"], "RESOLVED")
        self.assertEqual(found["product_name"], "ADVIL")
        self.assertEqual(found["ingredients"], [{"name": "IBUPROFEN", "strength": "200 mg/1"}])
        self.assertEqual(found["package_ndc"], "0573-0150-20")

    def test_no_barcode(self):
        body = self.scan(BLANK)
        self.assertEqual((body["status"], body["barcodes"]), ("NO_BARCODE", []))
        self.assertEqual(body["message"], "No barcode detected. Move closer, improve lighting, or rotate the package.")

    def test_unresolved_upc_with_pending_registry_entry(self):
        (found,) = self.scan(barcode_image(F.UPCA, "681131699372"))["barcodes"]
        self.assertEqual(found["final_status"], "UNRESOLVED")
        self.assertEqual((found["fda"]["status"], found["registry"]["status"]), ("UNRESOLVED", "PENDING"))
        self.assertEqual((found["product_name"], found["ingredients"], found["source"]), (None, [], None))

    def test_registry_identified_upc(self):
        (found,) = self.scan(barcode_image(F.UPCA, "012345678905"))["barcodes"]
        self.assertEqual(found["final_status"], "IDENTIFIED_FROM_REGISTRY")
        self.assertEqual(found["source"], "Local Verified Registry")
        self.assertEqual(found["registry"]["status"], "VERIFIED")
        self.assertEqual(found["registry"]["date_verified"], "2026-11-11")
        self.assertEqual(found["product_name"], "Store Brand Ibuprofen Tablets")
        self.assertIsNone(found["package_ndc"])

    def test_unsupported_format_is_reported_not_crashed(self):
        (found,) = self.scan(barcode_image(F.QRCode, "hello"))["barcodes"]
        self.assertEqual((found["barcode"]["decoded"], found["barcode"]["value"]), (False, None))
        self.assertEqual(found["final_status"], "UNRESOLVED")

    def test_barcode_scan_does_not_load_yolo_or_ocr(self):
        client = self.client()
        self.post(client, "/api/barcode", barcode_image(F.UPCA, "305730150200"))
        self.post(client, "/api/barcode", BLANK)
        self.assertEqual(self.load_counts, {"catalog": 1, "registry": 1})

    def test_invalid_registry_falls_back_to_fda_only(self):
        def broken():
            raise ValueError("bad registry file")
        client = self.client(registry=broken)
        (found,) = self.post(client, "/api/barcode", barcode_image(F.UPCA, "305730150200")).get_json()["barcodes"]
        self.assertEqual((found["final_status"], found["registry"]["status"]), ("IDENTIFIED_FROM_FDA", "NOT_LOADED"))
        self.assertEqual(client.get("/health").get_json()["registry_error"], "bad registry file")


class FrontLabelFallbackTests(ApiTestCase):
    """An unresolved barcode suggests scanning the front label; nothing else changes."""

    MESSAGE = ("Barcode scanned successfully. We couldn\u2019t identify this product in our database. "
               "Try scanning the front label instead.")

    def scan(self, data, **loaders):
        return self.post(self.client(**loaders), "/api/barcode", data).get_json()

    def test_unresolved_barcode_suggests_the_front_label(self):
        (found,) = self.scan(barcode_image(F.UPCA, "681131699372"))["barcodes"]
        self.assertEqual((found["next_step"], found["message"]), ("scan_front_label", self.MESSAGE))
        self.assertEqual(found["final_status"], "UNRESOLVED")           # status is unchanged
        self.assertEqual(found["barcode"]["value"], "681131699372")     # digits stay available to show
        self.assertEqual((found["product_name"], found["ingredients"], found["source"]), (None, [], None))

    def test_unknown_upc_and_non_upc_formats_also_suggest_it(self):
        (found,) = self.scan(barcode_image(F.UPCA, "036000291452"))["barcodes"]
        self.assertEqual((found["final_status"], found["next_step"]), ("UNRESOLVED", "scan_front_label"))
        (found,) = self.scan(barcode_image(F.Code128, "LOT-A123"))["barcodes"]
        self.assertEqual((found["final_status"], found["next_step"]), ("UNRESOLVED", "scan_front_label"))

    def test_other_statuses_get_no_fallback_and_keep_their_status(self):
        shared = "0681131699372"
        catalogs = {
            "IDENTIFIED_FROM_FDA": (self.catalog, "305730150200"),
            "IDENTIFIED_FROM_REGISTRY": (self.catalog, "012345678905"),
            "CANDIDATE": (self.catalog, "845717010566"),
            "AMBIGUOUS": (ProductCatalog.from_openfda_results([
                raw("11111-111", ["11111-111-01"], [shared]),
                raw("22222-222", ["22222-222-01"], [shared], ingredient="ACETAMINOPHEN")]), "681131699372"),
            "CONFLICT": (ProductCatalog.from_openfda_results([
                raw("0573-0161", ["0573-0161-35"], ["0305730150200"]),
                raw("0573-0150", ["0573-0150-20"])]), "305730150200"),
        }
        for status, (catalog, value) in catalogs.items():
            (found,) = self.scan(barcode_image(F.UPCA, value), catalog=lambda c=catalog: c)["barcodes"]
            self.assertEqual(found["final_status"], status)
            self.assertEqual((found["next_step"], found["message"]), (None, None), status)

    def test_undecoded_barcode_is_not_called_a_successful_scan(self):
        (found,) = self.scan(barcode_image(F.QRCode, "hello"))["barcodes"]
        self.assertEqual((found["final_status"], found["next_step"], found["message"]), ("UNRESOLVED", None, None))

    def test_no_barcode_has_no_fallback(self):
        body = self.scan(BLANK)
        self.assertEqual((body["status"], body["barcodes"]), ("NO_BARCODE", []))

    def test_front_label_capture_uses_the_existing_identify_endpoint(self):
        # The page's fallback posts a fresh image to /api/identify; its result is the ordinary one
        ocr = [{"text": "Ibuprofen", "confidence": 0.99, "box": (1, 2, 3, 4)},
               {"text": "200 mg", "confidence": 0.95, "box": (1, 2, 3, 4)}]
        client = self.client(detections=[bottle()], ocr=ocr)
        self.assertEqual(self.post(client, "/api/barcode", barcode_image(F.UPCA, "681131699372"))
                         .get_json()["barcodes"][0]["next_step"], "scan_front_label")
        match = self.post(client, "/api/identify").get_json()["match"]
        self.assertEqual(match["status"], "UNCERTAIN")  # possibilities, not an identification
        self.assertGreater(len(match["candidates"]), 0)
        self.assertEqual(self.post(self.client(), "/api/identify").get_json()["status"], "NO_BOTTLE")

    def test_page_script_wires_the_fallback_to_identify(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "web", "static", "app.js")) as f:
            script = f.read()
        for expected in ('"scan_front_label"', "Scan front label", "Capture front label",
                         'send("/api/identify", "Identification", renderIdentify)',
                         "Move the camera back and keep the entire bottle visible"):
            self.assertIn(expected, script)


class DetectApiTests(ApiTestCase):
    """Environmental awareness: every YOLO detection with its position (YOLO itself is faked)."""

    WIDE = png(np.full((300, 900, 3), 255, np.uint8))  # thirds at x = 300 and 600

    def found(self, name, confidence, box):
        return {"class_id": 0, "class_name": name, "confidence": confidence, "box": box}

    def detect(self, detections, data=None):
        response = self.post(self.client(detections=detections), "/api/detect", data or self.WIDE)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        return json.loads(response.data)

    def test_valid_detection_response(self):
        body = self.detect([self.found("person", np.float32(0.94), (np.int64(320), np.int64(20), np.int64(580), np.int64(290)))])
        # every field existing clients rely on is still there (tracking only adds fields)
        self.assertLessEqual({"detections", "image_width", "image_height", "inference_time_ms", "detection_count"},
                             set(body))
        self.assertEqual((body["image_width"], body["image_height"], body["detection_count"]), (900, 300, 1))
        (found,) = body["detections"]
        self.assertEqual({k: found[k] for k in ("class_name", "position", "bounding_box")},
                         {"class_name": "person", "position": "CENTER", "bounding_box": [320, 20, 580, 290]})
        self.assertAlmostEqual(found["confidence"], 0.94, places=5)
        # a request without a session is not tracked: the field is present but empty
        self.assertEqual((found["track_id"], body["tracking"]), (None, "off"))
        self.assertIsInstance(body["inference_time_ms"], int)
        self.assertGreaterEqual(body["inference_time_ms"], 0)

    def test_multiple_objects_and_position_classification(self):
        body = self.detect([self.found("chair", 0.71, (10, 50, 200, 280)),
                            self.found("person", 0.94, (320, 20, 580, 290)),
                            self.found("tv", 0.66, (650, 10, 890, 150)),
                            self.found("bottle", 0.52, (250, 100, 340, 200))])  # centre x = 295 -> LEFT
        self.assertEqual(body["detection_count"], 4)
        self.assertEqual([(d["class_name"], d["position"]) for d in body["detections"]],
                         [("chair", "LEFT"), ("person", "CENTER"), ("tv", "RIGHT"), ("bottle", "LEFT")])

    def test_no_detections(self):
        body = self.detect([])
        self.assertEqual((body["detections"], body["detection_count"]), ([], 0))
        self.assertEqual((body["image_width"], body["image_height"]), (900, 300))

    def test_confidence_and_box_are_plain_json_numbers(self):
        (found,) = self.detect([self.found("cup", np.float32(0.5), (np.int64(1), np.float32(2.0), 30, 40))])["detections"]
        self.assertIsInstance(found["confidence"], float)
        self.assertAlmostEqual(found["confidence"], 0.5)
        self.assertEqual(found["bounding_box"], [1, 2, 30, 40])
        self.assertTrue(all(type(v) is int for v in found["bounding_box"]))

    def test_boxes_use_the_submitted_images_coordinates(self):
        small = png(np.full((120, 160, 3), 255, np.uint8))
        body = self.detect([self.found("cat", 0.9, (120, 10, 158, 110))], small)
        self.assertEqual((body["image_width"], body["image_height"]), (160, 120))
        self.assertEqual((body["detections"][0]["bounding_box"], body["detections"][0]["position"]),
                         ([120, 10, 158, 110], "RIGHT"))

    def test_detect_runs_no_ocr_matcher_or_barcode_lookup(self):
        client = self.client(detections=[self.found("person", 0.9, (1, 2, 3, 4))])
        self.post(client, "/api/detect", barcode_image(F.UPCA, "305730150200"))
        self.assertEqual(self.load_counts, {"detector": 1})
        self.assertEqual(self.recognizer.seen_shapes, [])

    def test_invalid_uploads(self):
        client = self.client()
        cases = ((None, 400, "missing_image"), (b"", 400, "missing_image"), (b"not an image", 400, "invalid_image"),
                 (b"x" * 300_000, 413, "image_too_large"))
        for data, status, code in cases:
            response = self.post(client, "/api/detect", data)
            self.assertEqual((response.status_code, response.get_json()["error"]["code"]), (status, code))
        self.assertEqual(self.load_counts, {})  # rejected before any model is touched

    def test_yolo_model_is_reused_across_detect_and_identify(self):
        client = self.client(detections=[bottle()], ocr=[{"text": "Ibuprofen", "confidence": 0.9, "box": (1, 2, 3, 4)}])
        for path in ("/api/detect", "/api/detect", "/api/identify", "/api/detect"):
            self.assertEqual(self.post(client, path).status_code, 200)
        self.assertEqual(self.load_counts["detector"], 1)

    def test_busy_server_answers_quickly_instead_of_queueing(self):
        client = self.client(detections=[self.found("person", 0.9, (1, 2, 3, 4))])
        self.assertEqual(self.post(client, "/api/detect").status_code, 200)
        self.services._work_lock.acquire()  # another request is being processed
        try:
            response = self.post(client, "/api/detect")
        finally:
            self.services._work_lock.release()
        self.assertEqual((response.status_code, response.get_json()["error"]["code"]), (503, "busy"))
        self.assertEqual(self.post(client, "/api/detect").status_code, 200)  # works again afterwards

    def test_model_initialization_failure(self):
        def broken():
            raise RuntimeError("weights file missing")
        response = self.post(self.client(detector=broken), "/api/detect")
        self.assertEqual((response.status_code, response.get_json()["error"]["code"]), (503, "initialization_failed"))


class TrackingApiTests(ApiTestCase):
    """track_id on /api/detect: state lives per session id, between separate requests."""

    def setUp(self):
        self.detector = FakeDetector([])
        self.http = self.client(detector=lambda: self.detector)

    def frame(self, session, number, detections, **extra):
        self.detector.detections = detections
        data = {"image": (io.BytesIO(DetectApiTests.WIDE), "frame.jpg"), **extra}
        if session is not None:
            data.update(session_id=session, frame=str(number))
        return self.http.post("/api/detect", data=data, content_type="multipart/form-data")

    def ids(self, session, number, detections):
        body = self.frame(session, number, detections).get_json()
        return [d["track_id"] for d in body["detections"]], body["tracking"]

    @staticmethod
    def person(x, confidence=0.9):
        return {"class_id": 0, "class_name": "person", "confidence": confidence, "box": (x, 20, x + 140, 290)}

    def test_ids_persist_across_separate_requests(self):
        self.assertEqual(self.ids("session-one", 1, [self.person(50), self.person(500)]), ([1, 2], "ok"))
        body = self.frame("session-one", 2, [self.person(560), self.person(120)]).get_json()
        self.assertEqual(body["tracking"], "ok")
        self.assertEqual([(d["track_id"], d["position"]) for d in body["detections"]], [(2, "RIGHT"), (1, "LEFT")])
        self.assertEqual(self.ids("session-one", 3, [self.person(300)]), ([1], "ok"))   # moved to the centre

    def test_without_a_session_nothing_is_tracked(self):
        for _ in range(2):
            self.assertEqual(self.ids(None, None, [self.person(50)]), ([None], "off"))

    def test_concurrent_browser_sessions_are_independent(self):
        self.assertEqual(self.ids("iphone-session", 1, [self.person(50)]), ([1], "ok"))
        self.assertEqual(self.ids("laptop-session", 1, [self.person(60), self.person(500)]), ([1, 2], "ok"))
        self.assertEqual(self.ids("iphone-session", 2, [self.person(500), self.person(55)]), ([2, 1], "ok"))
        self.assertEqual(self.ids("laptop-session", 2, [self.person(505)]), ([2], "ok"))

    def test_stale_and_out_of_order_frames(self):
        self.ids("session-one", 1, [self.person(50)])
        self.ids("session-one", 3, [self.person(60)])
        self.assertEqual(self.ids("session-one", 2, [self.person(700)]), ([None], "stale"))
        self.assertEqual(self.ids("session-one", 3, [self.person(700)]), ([None], "stale"))
        self.assertEqual(self.ids("session-one", 4, [self.person(64)]), ([1], "ok"))

    def test_reset_endpoint_and_new_session_start_from_one(self):
        self.ids("session-one", 1, [self.person(50), self.person(500)])
        response = self.http.post("/api/detect/reset", data={"session_id": "session-one"})
        self.assertEqual((response.status_code, response.get_json()), (200, {"reset": True}))
        self.assertEqual(self.http.post("/api/detect/reset", data={"session_id": "session-one"}).get_json(),
                         {"reset": False})
        self.assertEqual(self.ids("session-one", 1, [self.person(500)]), ([1], "ok"))
        self.assertEqual(self.ids("session-two", 1, [self.person(500)]), ([1], "ok"))   # awareness restarted

    def test_invalid_session_or_frame_is_rejected_before_any_work(self):
        for session, number in (("short", 1), ("has spaces in it", 1), ("../../etc/passwd", 1), ("x" * 65, 1),
                                ("session-one", "abc"), ("session-one", "-1"), ("session-one", "")):
            response = self.frame(session, number, [self.person(50)])
            self.assertEqual((response.status_code, response.get_json()["error"]["code"]), (400, "invalid_tracking"))
        self.assertEqual(self.http.post("/api/detect/reset", data={}).status_code, 400)
        self.assertEqual(self.load_counts, {})

    def test_tracking_does_not_load_or_run_anything_else(self):
        self.ids("session-one", 1, [self.person(50)])
        self.ids("session-one", 2, [self.person(52)])
        self.assertEqual(self.load_counts, {"detector": 1} if "detector" in self.load_counts else {})
        self.assertEqual(self.recognizer.seen_shapes, [])


class IdentifyApiTests(ApiTestCase):
    OCR = [{"text": "MOTRIN IB", "confidence": np.float64(0.97), "box": (np.int64(1), np.int64(2), 30, 12)},
           {"text": "Ibuprofen Tablets 2OOmg", "confidence": np.float32(0.9), "box": (1, 20, 60, 30)},
           {"text": "200 mg", "confidence": 0.95, "box": (1, 40, 30, 50)},
           {"text": "blurry", "confidence": 0.1, "box": (1, 60, 30, 70)}]

    def test_no_bottle(self):
        chair = dict(bottle(), class_name="chair")
        response = self.post(self.client(detections=[chair]), "/api/identify")
        body = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual((body["status"], body["message"]), ("NO_BOTTLE", "No bottle detected."))
        self.assertEqual((body["detection"], body["ocr"], body["match"]), (None, [], None))
        self.assertEqual(body["detections_seen"][0]["class_name"], "chair")
        self.assertEqual(self.recognizer.seen_shapes, [])  # OCR never ran

    def test_identification_is_serialized_safely(self):
        response = self.post(self.client(detections=[bottle(np.float32(0.6)), bottle()], ocr=self.OCR), "/api/identify")
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.data)  # plain JSON: NumPy numbers, tuples and enums all converted
        self.assertEqual(body["status"], "OK")
        self.assertEqual(body["detection"], {"class_name": "bottle", "confidence": body["detection"]["confidence"],
                                             "box": [10, 20, 110, 220], "position": "LEFT"})
        self.assertAlmostEqual(body["detection"]["confidence"], 0.8, places=5)
        self.assertEqual(self.recognizer.seen_shapes, [(200, 100, 3)])  # OCR saw only the bottle crop
        self.assertEqual(body["ocr"][0], {"text": "MOTRIN IB", "confidence": 0.97, "box": [1, 2, 30, 12]})

        match = body["match"]
        self.assertEqual((match["status"], match["level"]), ("VERIFIED", "FORMULATION"))
        self.assertEqual(match["candidates"][0]["ingredients"], [{"name": "IBUPROFEN", "strength": "200 mg/1"}])
        self.assertIn("Motrin IB", match["candidates"][0]["brand_names"])
        self.assertIsInstance(match["candidates"][0]["record_count"], int)
        self.assertNotIn("records", match["candidates"][0])  # no raw FDA record dumps
        corrected = [e for e in match["evidence"] if e["corrected_from"]]
        self.assertEqual([(e["kind"], e["value"]) for e in corrected], [("strength", "200mg")])
        self.assertEqual(match["ignored"], ["blurry"])

    def test_uncertain_result_keeps_its_status(self):
        ocr = [{"text": "Ibuprofen", "confidence": 0.99, "box": (1, 2, 3, 4)},
               {"text": "200 mg", "confidence": 0.95, "box": (1, 2, 3, 4)}]
        match = self.post(self.client(detections=[bottle()], ocr=ocr), "/api/identify").get_json()["match"]
        self.assertEqual(match["status"], "UNCERTAIN")
        self.assertGreater(match["candidate_count"], 0)
        self.assertLessEqual(len(match["candidates"]), 10)

    def test_no_text_read(self):
        body = self.post(self.client(detections=[bottle()]), "/api/identify").get_json()
        self.assertEqual((body["ocr"], body["match"]["status"]), ([], "NO_MATCH"))

    def test_models_are_loaded_once_across_requests(self):
        client = self.client(detections=[bottle()], ocr=self.OCR)
        for _ in range(3):
            self.assertEqual(self.post(client, "/api/identify").status_code, 200)
        self.assertEqual(self.load_counts, {"catalog": 1, "detector": 1, "recognizer": 1})


class InitializationErrorTests(ApiTestCase):
    def test_model_load_failure_is_a_clear_503_and_is_retried(self):
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("weights file missing")
            return FakeDetector([])

        client = self.client(detector=flaky)
        response = self.post(client, "/api/identify")
        self.assertEqual(response.status_code, 503)
        error = response.get_json()["error"]
        self.assertEqual(error["code"], "initialization_failed")
        self.assertIn("detector", error["message"])
        self.assertIn("weights file missing", error["message"])
        self.assertEqual(self.post(client, "/api/identify").status_code, 200)  # loads on the next request

    def test_missing_fda_catalog(self):
        def missing():
            raise FileNotFoundError("FDA cache not found.")
        client = self.client(catalog=missing)
        for path in ("/api/barcode", "/api/identify"):
            response = self.post(client, path, barcode_image(F.UPCA, "305730150200"))
            self.assertEqual((response.status_code, response.get_json()["error"]["code"]),
                             (503, "initialization_failed"))

    def test_unexpected_processing_error_is_json_without_details(self):
        class Exploding(FakeDetector):
            def detect(self, frame):
                raise RuntimeError("secret internal path /Users/x")
        with self.assertLogs("web.server", level="ERROR"):
            response = self.post(self.client(detector=lambda: Exploding([])), "/api/identify")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"],
                         {"code": "internal_error", "message": "Processing failed: RuntimeError"})


if __name__ == "__main__":
    unittest.main()
