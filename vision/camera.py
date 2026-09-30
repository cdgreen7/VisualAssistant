import cv2 
def open_camera(index=0):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        raise RuntimeError("Couldn't open Camera.")
    return camera