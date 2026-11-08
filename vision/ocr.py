import easyocr

class TextRecognizer:
    def __init__(self, languages=("en",), min_confidence=0.0, gpu=False):
        # Loading the EasyOCR model is slow, so do it once here
        self.reader = easyocr.Reader(list(languages), gpu=gpu)
        self.min_confidence = min_confidence

    def read(self, image):
        # image: NumPy BGR frame or ROI crop (e.g. frame[y1:y2, x1:x2])
        # box coordinates are relative to the image passed in
        results = []
        for points, text, confidence in self.reader.readtext(image):
            confidence = float(confidence)
            if confidence < self.min_confidence:
                continue

            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            results.append({
                "text": text,
                "confidence": confidence,
                "box": (min(xs), min(ys), max(xs), max(ys))
            })

        return results

    def read_text(self, image):
        return " ".join(result["text"] for result in self.read(image))

if __name__ == "__main__":
    import cv2

    image = cv2.imread("vision/ocr_test.jpg")
    if image is None:
        raise FileNotFoundError("Couldn't read vision/ocr_test.jpg (run from the repo root).")

    recognizer = TextRecognizer()
    for result in recognizer.read(image):
        print(f"{result['text']!r} {result['confidence']:.2f} {result['box']}")
