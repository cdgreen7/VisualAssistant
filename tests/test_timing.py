"""Server-side stage timing (web/timing.py): the Server-Timing header and the
optional browser-timing log. YOLO and OCR are fakes, as in test_web_api."""
import json
import os
import tempfile
import unittest

from tests.test_web_api import ApiTestCase, BLANK, bottle
from web import timing
from web.server import create_app


def stages(response):
    out = {}
    for part in response.headers["Server-Timing"].split(","):
        name, _, value = part.strip().partition(";dur=")
        out[name] = float(value)
    return out


class ServerTimingTests(ApiTestCase):
    OCR = [{"text": "MOTRIN IB", "confidence": 0.97, "box": (1, 2, 30, 12)},
           {"text": "Ibuprofen 200 mg", "confidence": 0.95, "box": (1, 20, 60, 30)}]

    def test_every_response_has_a_total(self):
        client = self.client()
        for response in (client.get("/health"), client.get("/"), client.get("/api/nope"),
                         self.post(client, "/api/detect", b"not an image")):
            self.assertGreaterEqual(stages(response)["total"], 0)

    def test_detect_reports_decode_yolo_tracking_and_json(self):
        client = self.client(detections=[bottle()])
        data = {"image": (__import__("io").BytesIO(BLANK), "f.jpg"), "session_id": "timing-session", "frame": "1"}
        response = client.post("/api/detect", data=data, content_type="multipart/form-data")
        found = stages(response)
        self.assertEqual(set(found), {"decode", "yolo", "tracking", "json", "total"})
        self.assertTrue(all(value >= 0 for value in found.values()))
        self.assertGreaterEqual(found["total"] + 0.11, found["decode"] + found["yolo"] + found["tracking"] + found["json"])
        # the response body is unchanged by the instrumentation
        self.assertEqual(set(response.get_json()), {"detections", "image_width", "image_height", "inference_time_ms",
                                                    "detection_count", "tracking"})

    def test_untracked_detect_has_no_tracking_stage(self):
        self.assertNotIn("tracking", stages(self.post(self.client(), "/api/detect")))

    def test_identify_separates_ocr_and_matching(self):
        response = self.post(self.client(detections=[bottle()], ocr=self.OCR), "/api/identify")
        found = stages(response)
        self.assertLessEqual({"decode", "yolo", "identify", "ocr", "match", "json", "total"}, set(found))
        self.assertGreaterEqual(found["identify"] + 0.11, found["ocr"] + found["match"])   # crop/select is the rest
        self.assertEqual(response.get_json()["match"]["status"], "VERIFIED")               # behaviour unchanged
        self.assertEqual(self.recognizer.seen_shapes, [(200, 100, 3)])

    def test_identify_without_a_bottle_runs_no_ocr_stage(self):
        found = stages(self.post(self.client(), "/api/identify"))
        self.assertNotIn("ocr", found)
        self.assertNotIn("match", found)

    def test_barcode_stage(self):
        self.assertIn("barcode", stages(self.post(self.client(), "/api/barcode")))

    def test_requests_do_not_share_timings(self):
        client = self.client(detections=[bottle()], ocr=self.OCR)
        self.post(client, "/api/identify")
        self.assertNotIn("ocr", stages(client.get("/health")))


class TimingHelperTests(unittest.TestCase):
    def test_stage_and_record_outside_a_request_do_nothing(self):
        with timing.stage("yolo"):
            pass
        timing.record("ocr", 5)

    def test_timed_proxy_only_times_the_named_method(self):
        class Component:
            name = "reader"

            def read(self, value, scale=1):
                return value * scale

            def other(self):
                return "untouched"
        proxy = timing.Timed(Component(), "read", "ocr")
        self.assertEqual((proxy.read(3, scale=2), proxy.other(), proxy.name), (6, "untouched", "reader"))

    def test_client_label(self):
        cases = {
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1": "iphone-safari",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 CriOS/130.0 Mobile/15E148 Safari/604.1": "iphone-chrome",
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/130.0.0.0 Mobile Safari/537.36": "android-chrome",
            "Mozilla/5.0 (Linux; Android 14; SM-S921B) AppleWebKit/537.36 SamsungBrowser/26.0 Chrome/122.0 Mobile Safari/537.36": "android-samsung",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15": "desktop-safari",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36": "desktop-chrome",
            "": "desktop-other", None: "desktop-other",
        }
        for user_agent, expected in cases.items():
            self.assertEqual(timing.client_label(user_agent), expected, user_agent)


class TimingLogTests(ApiTestCase):
    IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1"
    RECORD = {"kind": "detect", "prepare_ms": 12.5, "roundtrip_ms": 180.2, "total_ms": 201.0,
              "server": {"yolo": 24.1, "total": 29.0}, "transfer_ms": 151.2}

    def app(self, path):
        self.client()                       # builds self.services with the fakes
        return create_app(self.services, timing_log=path).test_client()

    def send(self, client, payload, **headers):
        return client.post("/api/research/timings", data=json.dumps(payload), content_type="application/json", headers=headers)

    def test_collection_is_off_unless_the_server_was_started_with_a_log(self):
        response = self.send(self.client(), {"records": [self.RECORD]})
        self.assertEqual((response.status_code, response.get_json()["error"]["code"]), (404, "timing_log_off"))

    def test_records_are_stored_with_the_client_family_and_nothing_else(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nested", "timings.jsonl")
            client = self.app(path)
            payload = {"meta": {"client": "iphone-safari", "viewport": "393x852", "extra": {"deep": {"deeper": {"x": 1}}}},
                       "records": [self.RECORD, {"kind": "barcode", "note": "x" * 500, "image": [1, 2, 3]}, "not a record"]}
            response = self.send(client, payload, **{"User-Agent": self.IPHONE})
            self.assertEqual(response.get_json(), {"stored": 2})
            with open(path) as f:
                lines = [json.loads(line) for line in f]
            self.assertEqual(lines[0]["record"], self.RECORD)
            self.assertEqual(lines[0]["meta"]["client_from_server"], "iphone-safari")
            self.assertEqual(len(lines[1]["record"]["note"]), 120)            # long strings are cut
            self.assertIsNone(lines[1]["record"]["image"])                    # lists (e.g. pixel data) are dropped
            self.assertNotIn("remote_addr", json.dumps(lines))                # no IP address is stored
            self.assertEqual(self.send(client, {"records": [self.RECORD]}).get_json(), {"stored": 1})
            with open(path) as f:
                self.assertEqual(len(f.readlines()), 3)

    def test_bad_batches_store_nothing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "timings.jsonl")
            client = self.app(path)
            for payload in ({}, {"records": []}, {"records": "x"}, [1, 2], {"records": [self.RECORD] * 101}):
                self.assertEqual(self.send(client, payload).get_json(), {"stored": 0})
            big = self.send(client, {"records": [dict(self.RECORD, pad="y" * 70000)]})
            self.assertEqual(big.status_code, 413)
            self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
