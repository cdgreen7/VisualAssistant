"""Additions to the batch annotation assistant: HEIC decoding without extra
installs, reading every detected bottle from the original pixels, the
single-amount suggestion, refreshing unreviewed drafts, and the optional Qwen
observation. No model is loaded: YOLO, OCR and Qwen are fakes."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from research import annotation_batch, dataset_images
from research.annotate_medicine import load_manifest
from research.annotation_batch import LocalReader, batch, bottle_crops, make_row, suggest
from research.dataset_images import read


def evidence(*names, brands=()):
    return SimpleNamespace(evidence=[SimpleNamespace(kind="ingredient", value=n) for n in names] +
                                    [SimpleNamespace(kind="brand", value=b) for b in brands],
                           status=SimpleNamespace(value="UNCERTAIN"))


def rows(*items):
    return [{"text": text, "confidence": confidence, "region": region} for text, confidence, region in items]


def bottle(confidence, box):
    return {"class_name": "bottle", "confidence": confidence, "box": box}


class HeicTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.heic = Path(self.tmp.name) / "IMG_0001.HEIC"
        self.heic.write_bytes(b"original heic bytes, never rewritten")

    def fake_sips(self, command, **_):
        """Stands in for /usr/bin/sips: writes a JPEG to the requested --out path."""
        Image.new("RGB", (5712, 4284), "navy").save(command[command.index("--out") + 1], "JPEG")
        return SimpleNamespace(returncode=0)

    def test_heic_is_decoded_with_sips_and_hashed_from_the_original(self):
        import hashlib
        with patch.object(dataset_images.subprocess, "run", side_effect=self.fake_sips) as run, \
             patch.object(dataset_images.shutil, "which", return_value="/usr/bin/sips"), \
             patch.dict("sys.modules", {"pillow_heif": None}):
            data, info, image = read(self.heic)
        self.assertEqual(data, b"original heic bytes, never rewritten")
        self.assertEqual(info["sha256"], hashlib.sha256(data).hexdigest())      # hash of the original, not the copy
        self.assertEqual((info["extension"], info["width"], info["height"]), (".heic", 5712, 4284))
        self.assertIn("sips", info["decoder"])
        self.assertEqual(image.size, (5712, 4284))
        self.assertEqual(self.heic.read_bytes(), data)                          # original untouched
        self.assertEqual(Path(run.call_args[0][0][-3]), self.heic)              # sips read it, wrote elsewhere
        self.assertNotEqual(Path(run.call_args[0][0][-1]).parent, self.heic.parent)

    def test_no_decoder_and_failed_decoding_are_clear_errors(self):
        with patch.dict("sys.modules", {"pillow_heif": None}):
            with patch.object(dataset_images.shutil, "which", return_value=None):
                with self.assertRaisesRegex(ValueError, "HEIC decoder unavailable"):
                    read(self.heic)
            with patch.object(dataset_images.shutil, "which", return_value="/usr/bin/sips"), \
                 patch.object(dataset_images.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
                with self.assertRaisesRegex(ValueError, "HEIC decoding failed"):
                    read(self.heic)

    def test_jpeg_records_its_decoder_too(self):
        path = Path(self.tmp.name) / "a.jpg"
        Image.new("RGB", (40, 60)).save(path)
        self.assertEqual(read(path)[1]["decoder"], "Pillow")


class BottleCropTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (4000, 6000), "white")       # the original; OCR's working copy is 1000x1500
        self.working = (1000, 1500)

    def test_every_bottle_is_cut_from_the_original_pixels_most_confident_first(self):
        found = [bottle(0.70, (100, 500, 300, 1000)), {"class_name": "tv", "confidence": 0.99, "box": (0, 0, 500, 300)},
                 bottle(0.75, (600, 0, 900, 800))]
        crops = bottle_crops(self.image, self.working, found)
        self.assertEqual([round(c[2], 2) for c in crops], [0.75, 0.70])
        _, box, _ = crops[1]
        self.assertEqual(box, [368, 1920, 1232, 4080])              # 4x the working box, plus the 4% margin
        self.assertEqual(crops[1][0].size, (864, 2160))

    def test_boxes_are_clamped_and_useless_crops_skipped(self):
        edge = bottle_crops(self.image, self.working, [bottle(0.9, (900, 0, 1000, 400))])
        self.assertEqual(edge[0][1][2], 4000)                       # never past the image edge
        self.assertEqual(edge[0][1][1], 0)
        self.assertEqual(bottle_crops(self.image, self.working, [bottle(0.9, (0, 0, 990, 1490))]), [])   # nearly the whole frame
        self.assertEqual(bottle_crops(self.image, self.working, [bottle(0.9, (10, 10, 14, 14))]), [])    # too small to read
        self.assertEqual(bottle_crops(self.image, self.working, []), [])

    def test_at_most_three_bottles_are_read(self):
        found = [bottle(0.5 + i / 100, (i * 150, 100, i * 150 + 100, 500)) for i in range(6)]
        self.assertEqual(len(bottle_crops(self.image, self.working, found)), annotation_batch.MAX_CROPS)

    def test_reader_reads_frame_and_each_crop_and_records_where_text_came_from(self):
        class Recognizer:
            def __init__(self):
                self.shapes = []

            def read(self, frame):
                self.shapes.append(frame.shape[:2])
                first = len(self.shapes) == 1
                return [{"text": "blur" if first else "Ibuprofen", "confidence": 0.2 if first else 0.95, "box": (0, 0, 1, 1)}]

        recognizer, seen = Recognizer(), {}
        detector = SimpleNamespace(detect=lambda frame: None,
                                   get_detections=lambda result: [bottle(0.8, (100, 200, 300, 700)), bottle(0.6, (500, 200, 700, 700))])

        def match(ocr):
            seen["ocr"] = ocr
            return evidence("IBUPROFEN")
        parts = {"detector": detector, "recognizer": recognizer, "matcher": SimpleNamespace(match=match)}
        reader = LocalReader(side=1000)
        reader.services = SimpleNamespace(_get=parts.__getitem__)
        out = reader(Image.new("RGB", (2000, 3000), "white"))
        self.assertEqual(recognizer.shapes[0], (1000, 667))         # the reduced whole frame
        self.assertEqual(len(recognizer.shapes), 3)                 # plus two bottle crops
        self.assertEqual([r["region"] for r in seen["ocr"]], ["frame", "bottle_crop_1", "bottle_crop_2"])
        observed = out["observations"]
        self.assertEqual([c["region"] for c in observed["bottle_crops"]], ["bottle_crop_1", "bottle_crop_2"])
        self.assertEqual(observed["bottle_crops"][0]["box_in_original"], [275, 540, 923, 2160])   # working copy is 667 px wide, so the scale is just under 3
        self.assertEqual(out["fields"]["bottles_detected"], 2)
        self.assertIn("more than one bottle", out["field_sources"]["bottles_detected"])
        self.assertFalse(observed["models"]["qwen"])


class SuggestionTests(unittest.TestCase):
    def test_single_ingredient_with_its_amount_on_another_line(self):
        observed = {"ocr": rows(("Ibuprofen", 0.95, "bottle_crop_1"), ("USP 200 mg", 0.96, "bottle_crop_1"),
                                ("Tablets", 0.9, "bottle_crop_1")), "detections": [bottle(0.7, (0, 0, 9, 9))]}
        out = suggest(observed, evidence("IBUPROFEN"))
        self.assertEqual(out["fields"]["ingredients"], [{"name": "IBUPROFEN", "strength": "200", "unit": "mg"}])
        self.assertIn("only amount read", out["field_sources"]["ingredients"])
        self.assertEqual(out["fields"]["category"], "single_ingredient")

    def test_two_different_amounts_or_low_confidence_leave_strength_unknown(self):
        unknown = [{"name": "IBUPROFEN", "strength": None, "unit": None}]
        two = {"ocr": rows(("Ibuprofen", 0.95, "frame"), ("200 mg", 0.9, "frame"), ("500 mg", 0.9, "frame")), "detections": []}
        self.assertEqual(suggest(two, evidence("IBUPROFEN"))["fields"]["ingredients"], unknown)
        faint = {"ocr": rows(("Ibuprofen", 0.95, "frame"), ("200 mg", 0.4, "frame")), "detections": []}
        self.assertEqual(suggest(faint, evidence("IBUPROFEN"))["fields"]["ingredients"], unknown)

    def test_lone_amount_is_never_shared_between_two_ingredients(self):
        observed = {"ocr": rows(("Ibuprofen", 0.95, "frame"), ("Diphenhydramine citrate", 0.95, "frame"), ("200 mg", 0.9, "frame")),
                    "detections": []}
        fields = suggest(observed, evidence("IBUPROFEN", "DIPHENHYDRAMINE CITRATE"))["fields"]
        self.assertEqual({i["strength"] for i in fields["ingredients"]}, {None})
        self.assertEqual((fields["category"], fields["multiple_active_ingredients"]), ("combination", True))

    def test_product_name_uses_confident_wordy_lines_bottle_crops_first(self):
        observed = {"ocr": rows(("0us", 0.0, "frame"), ("g", 0.67, "frame"), ("Pain Reliever/", 0.77, "frame"),
                                ("equate", 0.99, "bottle_crop_2"), ("Ibuprofen", 0.95, "bottle_crop_2"), ("4", 0.98, "bottle_crop_2"),
                                ("Ibuprofen", 0.9, "frame")), "detections": []}
        self.assertEqual(suggest(observed, evidence("IBUPROFEN"))["fields"]["product_name"], "equate · Ibuprofen · Pain Reliever/")
        nothing = {"ocr": rows(("9", 0.9, "frame"), ("xx", 0.1, "frame")), "detections": []}
        self.assertIsNone(suggest(nothing, evidence())["fields"]["product_name"])

    def test_a_bottle_without_ingredient_text_is_not_called_medicine_or_non_medicine(self):
        observed = {"ocr": rows(("UIC", 0.9, "frame")), "detections": [bottle(0.8, (0, 0, 9, 9))]}
        fields = suggest(observed, evidence())["fields"]
        self.assertEqual((fields["category"], fields["possible_medicine"], fields["negative"]), ("unclear_label", None, None))
        self.assertEqual(suggest({"ocr": [], "detections": []}, evidence())["fields"]["category"], "no_bottle")

    def test_matcher_output_never_becomes_confirmed_truth(self):
        observed = {"ocr": rows(("Ibuprofen 200 mg", 0.99, "frame")), "detections": []}
        match = evidence("IBUPROFEN", brands=("MOTRIN IB",))
        match.status = SimpleNamespace(value="VERIFIED")
        out = suggest(dict(observed, matcher_status="VERIFIED"), match)
        item = {"id": "photo-1", "sha256": "0" * 64, "suggestions": out, "info": {}}
        row = make_row(item, out["fields"], independent=False)
        self.assertFalse(row["truth"]["confirmed"])
        self.assertEqual((row["truth"]["ingredients"], row["truth"]["brand"]), ([], None))
        self.assertEqual(row["reviewed_suggestions"]["brand"], "MOTRIN IB")     # kept, but apart from truth


class QwenObservationTests(unittest.TestCase):
    def reader(self, label_reader):
        reader = LocalReader(qwen=True)
        reader.label_reader = label_reader
        return reader

    def test_qwen_lines_are_stored_but_never_fill_a_field(self):
        qwen = SimpleNamespace(read=lambda frame, ocr: {"model": "qwen2.5vl:3b", "lines": ["ADVIL PM", "Naproxen 220 mg"],
                                                        "agrees_with_ocr": []})
        observed = {"ocr": rows(("Ibuprofen 200 mg", 0.95, "frame")), "detections": [], "models": {"qwen": False}}
        self.reader(qwen).second_reader(None, observed["ocr"], observed)
        out = suggest(observed, evidence("IBUPROFEN"))
        self.assertEqual(out["observations"]["qwen"]["lines"], ["ADVIL PM", "Naproxen 220 mg"])
        self.assertFalse(out["observations"]["qwen"]["used_for_suggestions"])
        self.assertEqual(out["fields"]["ingredients"], [{"name": "IBUPROFEN", "strength": "200", "unit": "mg"}])
        self.assertIsNone(out["fields"]["brand"])
        self.assertNotIn("Naproxen", json.dumps(out["fields"]))

    def test_qwen_failure_is_recorded_and_does_not_stop_the_batch(self):
        from web.services import ServiceError

        def unavailable(frame, ocr):
            raise ServiceError(503, "model_unavailable", "Cannot reach local Ollama.")
        observed = {"models": {"qwen": False}}
        self.reader(SimpleNamespace(read=unavailable)).second_reader(None, [], observed)
        self.assertEqual(observed["qwen"], {"error": "Cannot reach local Ollama.", "used_for_suggestions": False})

    def test_qwen_is_off_unless_asked_for(self):
        self.assertFalse(LocalReader().qwen)


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        base = Path(self.tmp.name)
        self.photos, self.manifest = base / "photos", base / "manifest.json"
        self.photos.mkdir()
        self.manifest.write_text(json.dumps({"schema_version": 1, "dataset_id": "test", "images": []}))
        for name, colour in (("a.jpg", "red"), ("b.jpg", "blue")):
            Image.new("RGB", (40, 60), colour).save(self.photos / name)
        self.calls = []

    def reader(self, version):
        def read_image(image):
            self.calls.append(version)
            fields = {"category": "unclear_label", "object_category": "bottle", "brand": version, "product_name": None,
                      "ingredients": [], "negative": None, "readable": None, "dosage_form": None, "exact_product": None}
            return {"fields": fields, "field_sources": {k: "test" for k in fields}, "observations": {}}
        return read_image

    def run_batch(self, version, **options):
        with patch("builtins.print"):
            batch(self.manifest, self.photos, reader=self.reader(version), **options)
        queue = json.loads((self.manifest.parent / "drafts.json").read_text())
        return {Path(i["source"]).name: i for i in queue["items"]}

    def test_rerunning_without_refresh_does_no_work(self):
        self.run_batch("v1")
        items = self.run_batch("v2")
        self.assertEqual(self.calls, ["v1", "v1"])
        self.assertEqual(items["a.jpg"]["suggestions"]["fields"]["brand"], "v1")

    def test_refresh_reinspects_only_drafts_nobody_has_touched(self):
        self.run_batch("v1")
        path = self.manifest.parent / "drafts.json"
        queue = json.loads(path.read_text())
        edited = next(i for i in queue["items"] if i["source"].endswith("b.jpg"))
        edited["reviewed_fields"] = dict(edited["suggestions"]["fields"], brand="Read by a person")
        path.write_text(json.dumps(queue))

        items = self.run_batch("v2", refresh=True)
        self.assertEqual(self.calls, ["v1", "v1", "v2"])                        # only a.jpg was re-read
        self.assertEqual(items["a.jpg"]["suggestions"]["fields"]["brand"], "v2")
        self.assertEqual(items["b.jpg"]["suggestions"]["fields"]["brand"], "v1")
        self.assertEqual(items["b.jpg"]["reviewed_fields"]["brand"], "Read by a person")   # human edit kept
        self.assertTrue(all(i["truth"] == {"confirmed": False} for i in items.values()))
        self.assertEqual(load_manifest(self.manifest)["images"], [])            # batch never writes the manifest


if __name__ == "__main__":
    unittest.main()
