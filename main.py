import os
import cv2
import time
from vision.camera import open_camera
from vision.detector import ObjectDetector
from vision.spatial import get_horizontal_position
from vision.identification import identify_detection, format_identification
from vision.ocr import TextRecognizer
from vision.products.catalog import ProductCatalog
from vision.products.matcher import ProductMatcher

FDA_CACHE = "vision/products/data/otc_ndc_2026-10-30.json"

camera = open_camera()
detector = ObjectDetector()

# Expensive objects are created once; OCR only runs when "i" is pressed
recognizer = matcher = None
if os.path.exists(FDA_CACHE):
    print("Loading OCR model and FDA catalog...")
    recognizer = TextRecognizer()
    matcher = ProductMatcher(ProductCatalog.from_file(FDA_CACHE))
    print("Ready. Press 'i' to identify the bottle in view, 'q' to quit.")
else:
    print(f"FDA cache not found ({FDA_CACHE}); identification disabled. "
          "Build it with: python -m vision.products.build_cache")


def run_identification(frame, detections):
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
        print(format_identification(*identified))


previous_time = time.time()

while True:
    success, frame = camera.read()
    if not success:
        break
    result = detector.detect(frame)
    detections = detector.get_detections(result)
    frame_height, frame_width, channels = frame.shape
    
    for detection in detections:
        position = get_horizontal_position(
                detection["box"],
                    frame_width
                )
        print(
                    detection["class_name"],
                    f"{detection['confidence']:.2f}",
                    position
                )


    # Draw on a copy so frame stays clean for OCR
    annotated_frame = result.plot() if detections else frame.copy()
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

camera.release()
cv2.destroyAllWindows()
