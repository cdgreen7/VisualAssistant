import cv2
try:
    from vision.camera import open_camera
except Exception:
    # Fallback to local camera module path if package import fails during development
    from camera import open_camera
camera = open_camera()
while True:
    success, frame = camera.read()

    if not success:
        break
    cv2.imshow("OCR capture Test: ", frame)
    key = cv2.waitKey(1) & 0xFF
    #Press C to capture

    if key == ord("c"):
        cv2.imwrite("vision/ocr_test.jpg", frame)
        print("image captured!")
        break

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()