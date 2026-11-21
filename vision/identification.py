"""User-triggered product identification on a YOLO detection.

YOLO detection -> crop of the clean frame -> TextRecognizer -> ProductMatcher.

The helpers here take the recognizer and matcher as arguments, so the
expensive objects are created once by the caller (see main.py).
"""
from vision.products.matcher import Status

TARGET_CLASS = "bottle"
MAX_CANDIDATES_SHOWN = 3


def select_best_detection(detections, class_name=TARGET_CLASS):
    """Highest-confidence detection of class_name, or None."""
    matches = [d for d in detections if d["class_name"] == class_name]
    return max(matches, key=lambda d: d["confidence"], default=None)


def clamp_box(box, width, height):
    """Clip (x1, y1, x2, y2) to the image; None if nothing is left."""
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_box(frame, box):
    """Crop of frame inside box (clamped), or None if the box is empty."""
    height, width = frame.shape[:2]
    clamped = clamp_box(box, width, height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    return frame[y1:y2, x1:x2]


def identify_detection(frame, detections, recognizer, matcher, class_name=TARGET_CLASS):
    """Run OCR + matching on the best detection of class_name.

    frame must be the clean camera frame (no boxes or FPS text drawn on it).
    Returns (detection, ocr_results, match_result), or None if there is no
    such detection or its box is empty.
    """
    detection = select_best_detection(detections, class_name)
    if detection is None:
        return None
    crop = crop_box(frame, detection["box"])
    if crop is None:
        return None
    ocr_results = recognizer.read(crop)
    return detection, ocr_results, matcher.match(ocr_results)


def format_identification(detection, ocr_results, result, position=None):
    lines = ["--- PRODUCT IDENTIFICATION ---", ""]
    where = f" {position}" if position else ""
    lines.append(f"{detection['class_name']} {detection['confidence']:.2f}{where} box={detection['box']}")
    lines += ["", "OCR:"]
    if ocr_results:
        width = max(len(r["text"]) for r in ocr_results) + 2
        lines += [f"{r['text']:<{width}}{r['confidence']:.2f}" for r in ocr_results]
    else:
        lines.append("(no text read)")

    lines += ["", "Result:"]
    if result.status == Status.NO_MATCH:
        lines += ["NO_MATCH", "No supported product identity found."]
    elif result.status == Status.VERIFIED:
        lines.append(f"VERIFIED / {result.level.value}")
        candidate = result.candidates[0]
        lines.append(" + ".join(f"{n} {s}" for n, s in candidate.ingredients))
        lines.append("Brands: " + ", ".join(candidate.brand_names[:3]))
        if result.package:
            lines.append(f"Package: NDC {result.package['package_ndc']} - {result.package['description']}")
    else:
        lines += ["UNCERTAIN", result.reason]
        for candidate in result.candidates[:MAX_CANDIDATES_SHOWN]:
            missing = candidate.missing_ingredients + candidate.base_only_ingredients
            note = f" (not observed: {', '.join(missing)})" if missing else ""
            if candidate.brand_hint_records:
                note += f" [{candidate.brand_hint_records} records match the observed brand hint]"
            lines.append("  - " + " + ".join(f"{n} {s}" for n, s in candidate.ingredients) + note)
        if len(result.candidates) > MAX_CANDIDATES_SHOWN:
            lines.append(f"  ... {len(result.candidates) - MAX_CANDIDATES_SHOWN} more")
    return "\n".join(lines)
