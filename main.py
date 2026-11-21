import os
import cv2
import time
from vision.camera import open_camera
from vision.detector import ObjectDetector
from vision.spatial import get_horizontal_position
from vision.barcode_scan import format_scan, scan_frame
from vision.identification import identify_detection, format_identification
from vision.ocr import TextRecognizer
from vision.products.catalog import ProductCatalog
from vision.products.matcher import ProductMatcher

FDA_CACHE = "vision/products/data/otc_ndc_2026-10-30.json"

camera = open_camera()
detector = ObjectDetector()

# Expensive objects are created once; OCR only runs when "i" is pressed
recognizer = matcher = catalog = None
if os.path.exists(FDA_CACHE):
    print("Loading OCR model and FDA catalog...")
    recognizer = TextRecognizer()
    catalog = ProductCatalog.from_file(FDA_CACHE)
    matcher = ProductMatcher(catalog)
    print("Ready. Press 'i' to identify the bottle in view, 'b' to scan a barcode, 'q' to quit.")
else:
    print(f"FDA cache not found ({FDA_CACHE}); identification and barcode lookup disabled. "
          "Build it with: python -m vision.products.build_cache")


def run_identification(frame, detections):
    """Print the selected bottle's details, OCR text and match result (on 'i')."""
    if matcher is None:
        print("Identification unavailable: FDA cache not loaded.")
        return
    try:
        identified = identify_detection(frame, detections, recognizer, matcher)
    except Exception as error:  # keep the camera loop running
        print(f"Identification failed: {error}")
        return
    if identified is None:
        print("No bottle detected.")
    else:
        position = get_horizontal_position(identified[0]["box"], frame.shape[1])
        print(format_identification(*identified, position=position))


def run_barcode_scan(frame):
    """Print the barcodes in the clean frame and their FDA lookup (on 'b')."""
    if catalog is None:
        print("Barcode lookup unavailable: FDA cache not loaded.")
        return
    try:
        print(format_scan(scan_frame(frame, catalog)))
    except Exception as error:  # keep the camera loop running
        print(f"Barcode scan failed: {error}")


previous_time = time.time()

while True:
    success, frame = camera.read()
    if not success:
        break
    result = detector.detect(frame)
    detections = detector.get_detections(result)
    frame_height, frame_width, channels = frame.shape
    
    # Draw on a copy so frame stays clean for OCR
    annotated_frame = result.plot() if detections else frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(v) for v in detection["box"])
        cv2.putText(
            annotated_frame,
            get_horizontal_position(detection["box"], frame_width),
            (x1 + 5, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    current_time = time.time()

    fps = 1/ max(current_time - previous_time, 1e-6) # prevent division by zero
    previous_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    cv2.imshow("Visual Assistant", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("i"):
        run_identification(frame, detections)
    if key == ord("b"):
        run_barcode_scan(frame)

camera.release()
cv2.destroyAllWindows()
