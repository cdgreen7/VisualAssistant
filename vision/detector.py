from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolo26n.pt"):
        self.model =YOLO(model_path)
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0]
    def get_detections(self, result):
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1, y1, x2, y2)
            })

        return detections