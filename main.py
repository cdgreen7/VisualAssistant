import cv2
import time
from vision.camera import open_camera
from vision.detector import ObjectDetector
from vision.spatial import get_horizontal_position
camera = open_camera()
detector = ObjectDetector()

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

    annotated_frame = result.plot() if detections else frame
    current_time = time.time()

    fps = 1/(current_time - previous_time) # how long one loop took 
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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()