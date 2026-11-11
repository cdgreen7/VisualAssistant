"""Web interface for the Visual Assistant: a phone (or this Mac) captures a
still image in the browser, this server runs the existing vision pipeline.

Usage (from the repo root):
    python -m web.server                                  # this Mac only: http://localhost:5443
    python -m web.server --host 0.0.0.0 --cert CERT.pem --key KEY.pem   # phones on the same network

See web/README.md for the HTTPS certificate setup an iPhone needs.
The server never runs in debug mode and only ever reads the uploaded image.
"""
import argparse
import sys

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from vision.barcode_scan import NO_BARCODE
from vision.products.barcode_lookup import conflicting
from web import serialize
from web.services import ServiceError, VisionServices

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


def create_app(services=None, max_upload_bytes=MAX_UPLOAD_BYTES):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = max_upload_bytes
    services = services or VisionServices()

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
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "loaded": services.loaded(), "registry_error": services.registry_error})

    @app.post("/api/detect")
    def detect():
        image = read_image(request.files)
        height, width = image.shape[:2]
        detections, inference_ms = services.detect(image)
        return jsonify({
            "detections": [serialize.environment_detection(d, width) for d in detections],
            "image_width": width,
            "image_height": height,
            "inference_time_ms": round(inference_ms),
            "detection_count": len(detections),
        })

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
