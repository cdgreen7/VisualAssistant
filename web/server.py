"""Web interface for the Visual Assistant: a phone (or this Mac) captures a
still image in the browser, this server runs the existing vision pipeline.

Usage (from the repo root):
    python -m web.server                                  # this Mac only: http://localhost:5443
    python -m web.server --host 0.0.0.0 --cert CERT.pem --key KEY.pem   # phones on the same network

See web/README.md for the HTTPS certificate setup an iPhone needs.
The server never runs in debug mode and only ever reads the uploaded image.
"""
import hashlib
from pathlib import Path
import argparse
import io
import sys

import cv2
import numpy as np
from flask import Flask, Request, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from vision.barcode_scan import NO_BARCODE
from vision.products.barcode_lookup import conflicting
from web import serialize
from web.qwen_scene import QwenSceneService, prepare_image
from web.services import ServiceError, VisionServices
from web.track_sessions import OFF, SESSION_ID, TrackSessions

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
NO_BOTTLE = "No bottle detected."
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def read_image(files):
    """The uploaded "image" as a BGR array. Only the uploaded bytes are read."""
    upload = files.get("image")
    if upload is None:
        raise ServiceError(400, "missing_image", 'Upload an image in the "image" form field.')
    data = upload.read()
    if not data:
        raise ServiceError(400, "missing_image", "The uploaded image is empty.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ServiceError(400, "invalid_image", "The upload is not a readable image (use JPEG or PNG).")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ServiceError(413, "image_too_large", "The image has too many pixels.")
    return image


def read_tracking(form):
    """(session id, frame number) when the client asks for tracking, else (None, None)."""
    session_id = form.get("session_id")
    if session_id is None:
        return None, None
    frame = form.get("frame", "")
    if not SESSION_ID.fullmatch(session_id) or not frame.isdigit() or len(frame) > 12:
        raise ServiceError(400, "invalid_tracking", "session_id or frame is not valid.")
    return session_id, int(frame)


class CameraRequest(Request):
    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        # Werkzeug normally spools larger uploads to disk. Ask AI stays in RAM.
        if self.path in ("/api/scene/describe", "/api/medicine/identify-crop"):
            return io.BytesIO()
        return super()._get_file_stream(total_content_length, content_type, filename, content_length)


def create_app(services=None, max_upload_bytes=MAX_UPLOAD_BYTES, track_sessions=None, scene_service=None, label_reader=None, research_service=None):
    app = Flask(__name__)
    root = Path(__file__).parent
    # One startup fingerprint for backend, template and every frontend module.
    sources = sorted(p for p in root.rglob('*') if p.suffix in {'.py', '.js', '.css', '.html'})
    build = '4b-debug-' + hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:10]
    app.config['APP_BUILD'] = build
    app.context_processor(lambda: {'app_build': build})
    app.request_class = CameraRequest
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes
    services = services or VisionServices()
    track_sessions = track_sessions or TrackSessions()

    scene_service = scene_service or QwenSceneService()
    from web.qwen_label import QwenLabelReader
    label_reader = label_reader or QwenLabelReader()

    from web.product_research import ProductResearch
    research_service = research_service or ProductResearch()

    @app.post("/api/products/research")
    def product_research():
        if request.content_length and request.content_length > 32768:
            raise ServiceError(413, "evidence_too_large", "Research evidence must be under 32 KB.")
        return jsonify(research_service.research(request.get_json(silent=True)))

    def error(http_status, code, message):
        return jsonify({"error": {"code": code, "message": message}}), http_status

    @app.errorhandler(ServiceError)
    def service_error(failure):
        return error(failure.http_status, failure.code, failure.message)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return error(413, "image_too_large", f"The upload is larger than {max_upload_bytes // (1024 * 1024)} MB.")

    @app.errorhandler(HTTPException)
    def http_error(failure):
        return error(failure.code, "http_error", failure.name)

    @app.errorhandler(Exception)
    def unexpected(failure):
        app.logger.exception("request failed")
        return error(500, "internal_error", f"Processing failed: {type(failure).__name__}")

    @app.after_request
    def no_store(response):
        response.headers["X-Visual-Assistant-Build"] = build
        if request.path == "/" or request.path == "/health" or request.path.startswith(("/api/", "/static/")):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "build": build, "loaded": services.loaded(), "registry_error": services.registry_error})

    @app.post("/api/scene/describe")
    def describe_scene():
        image = prepare_image(request.files.get("image"))
        return jsonify(scene_service.describe(image))

    @app.post("/api/detect")
    def detect():
        session_id, frame = read_tracking(request.form)
        image = read_image(request.files)
        height, width = image.shape[:2]
        detections, inference_ms = services.detect(image)
        track_ids, tracking = [None] * len(detections), OFF
        if session_id is not None:  # YOLO alone keeps no identity between requests
            track_ids, tracking = track_sessions.track(session_id, frame, detections)
        return jsonify({
            "detections": [serialize.environment_detection(d, width, track_id)
                           for d, track_id in zip(detections, track_ids)],
            "image_width": width,
            "image_height": height,
            "inference_time_ms": round(inference_ms),
            "detection_count": len(detections),
            "tracking": tracking,
        })

    @app.post("/api/detect/reset")
    def reset_tracking():
        session_id = request.form.get("session_id", "")
        if not SESSION_ID.fullmatch(session_id):
            raise ServiceError(400, "invalid_tracking", "session_id is not valid.")
        return jsonify({"reset": track_sessions.reset(session_id)})

    @app.post("/api/barcode")
    def barcode():
        image = read_image(request.files)
        found = services.scan_barcodes(image)
        return jsonify({
            "status": "OK" if found else "NO_BARCODE",
            "message": None if found else NO_BARCODE,
            "barcodes": [serialize.barcode_identification(i) for i in found],
            "barcodes_conflict": conflicting([i.fda for i in found]),
        })

    @app.post("/api/medicine/identify-crop")
    def identify_medicine_crop():
        session_id = request.form.get("session_id", "")
        capture_id = request.form.get("capture_id", "")
        if not SESSION_ID.fullmatch(session_id) or not SESSION_ID.fullmatch(capture_id):
            raise ServiceError(400, "invalid_capture", "A valid session_id and capture_id are required.")
        # Validate JPEG/PNG with the same in-memory validator, without its scene
        # resizing: OCR receives every uploaded pixel.
        from web.medicine_upload import read_crop
        image = read_crop(request.files.get("image"))
        app.logger.warning("medicine crop accepted: %dx%d", image.shape[1], image.shape[0])
        body = services.identify_crop(image)
        app.logger.warning("medicine OCR/matcher complete: %s; timing=%s", body["match"]["status"], body["timing"])
        body["qwen"] = {"enabled": False, "used_for_matching": False}
        if request.form.get("assist") == "true":
            import time
            started = time.perf_counter()
            try:
                body["qwen"] = dict(label_reader.read(image, body["ocr"]), enabled=True)
            except ServiceError as failure:
                body["qwen"] = {"enabled": True, "error": failure.message, "used_for_matching": False}
            body["timing"]["qwen_ms"] = round((time.perf_counter()-started)*1000, 2)
        else:
            body["timing"]["qwen_ms"] = 0
        body["timing"]["total_ms"] += body["timing"]["qwen_ms"]
        body.update(session_id=session_id, capture_id=capture_id)
        return jsonify(body)

    @app.post("/api/identify")
    def identify():
        image = read_image(request.files)
        width = image.shape[1]
        detections, identified = services.identify(image)
        body = {"status": "NO_BOTTLE", "message": NO_BOTTLE, "detection": None, "ocr": [], "match": None,
                "detections_seen": [serialize.detection(d, width) for d in detections]}
        if identified is not None:
            found, ocr_results, result = identified
            body.update(status="OK", message=None, detection=serialize.detection(found, width),
                        ocr=serialize.ocr_results(ocr_results), match=serialize.match_result(result))
        return jsonify(body)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept phones on the local network")
    parser.add_argument("--port", type=int, default=5443)
    parser.add_argument("--cert", help="TLS certificate (PEM)")
    parser.add_argument("--key", help="TLS private key (PEM)")
    parser.add_argument("--lazy", action="store_true", help="load models on first use instead of at startup")
    args = parser.parse_args()

    if bool(args.cert) != bool(args.key):
        sys.exit("Give both --cert and --key, or neither.")
    if args.host not in LOOPBACK and not args.cert:
        sys.exit("Refusing to serve other devices over plain HTTP: phone browsers only allow the camera "
                 "over HTTPS. Pass --cert and --key (see web/README.md).")

    services = VisionServices()
    if not args.lazy:
        print("Loading FDA catalog, UPC registry, YOLO and EasyOCR...")
        services.preload()
        if services.registry_error:
            print(f"Local UPC registry not loaded: {services.registry_error}")
    scheme = "https" if args.cert else "http"
    print(f"Visual Assistant web app: {scheme}://{'localhost' if args.host in LOOPBACK else args.host}:{args.port}")
    create_app(services).run(host=args.host, port=args.port, debug=False, threaded=True,
                             ssl_context=(args.cert, args.key) if args.cert else None)


if __name__ == "__main__":
    main()
