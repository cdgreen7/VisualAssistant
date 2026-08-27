import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Couldn't open camera")
    exit()

previous_time = time.time()

while True:
    success, frame = camera.read()

    if not success:
        print("Couldn't read frame")
        break

    results = model(frame, verbose=False) #Model Inference put verbose false so it dont detstory the terminal with output 
    annotated_frame = results[0].plot()
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
    cv2.imshow("YOLO Lice Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()