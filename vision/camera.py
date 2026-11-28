"""Opening and reading the camera (OpenCV; AVFoundation on macOS).

Which camera: index 0 by default, or the VA_CAMERA_INDEX environment variable,
e.g. `VA_CAMERA_INDEX=1 python main.py`. A Mac can list several cameras (the
built-in one and iPhones through Continuity Camera), and index 0 is not always
the built-in camera.

Cameras often need a moment after opening before they deliver frames, and can
drop a frame now and then. open_camera() therefore retries for a short time,
read_frame() retries a bounded number of failed reads, and both raise
CameraError with an explanation instead of failing silently. The capture
resolution and backend are OpenCV's defaults, unchanged.
"""
import os
import sys
import time

import cv2

CAMERA_INDEX_ENV = "VA_CAMERA_INDEX"
DEFAULT_INDEX = 0
OPEN_TIMEOUT_S = 3.0          # how long open_camera() keeps trying to get a first frame
OPEN_RETRY_INTERVAL_S = 0.2
MAX_FAILED_READS = 20         # consecutive failed reads before read_frame() gives up
READ_RETRY_INTERVAL_S = 0.05  # so at most about 1 s of retrying


class CameraError(RuntimeError):
    """The camera could not be opened or stopped delivering frames."""


def camera_index(environ=None):
    """VA_CAMERA_INDEX as a non-negative int, or 0 when it isn't set."""
    value = (os.environ if environ is None else environ).get(CAMERA_INDEX_ENV, "").strip()
    if not value:
        return DEFAULT_INDEX
    if not value.isdigit():
        raise CameraError(f"{CAMERA_INDEX_ENV} must be a camera number such as 0 or 1, not {value!r}")
    return int(value)


def _advice(index):
    tips = [f"Try another camera with {CAMERA_INDEX_ENV}=1 (or 2) if several are connected; "
            f"camera {index} may be a phone or virtual camera."]
    if sys.platform == "darwin":
        tips.insert(0, "On macOS, allow camera access for the app running Python (System Settings > "
                       "Privacy & Security > Camera, e.g. Terminal or VS Code), then quit and reopen that app.")
    tips.append("Close other apps that may be using the camera.")
    return " ".join(tips)


def _release(camera):
    try:
        camera.release()
    except Exception:
        pass


def open_camera(index=None, timeout=OPEN_TIMEOUT_S, capture=cv2.VideoCapture, sleep=time.sleep,
                clock=time.monotonic):
    """An opened camera that has delivered a frame, or CameraError.

    index: camera number; None means VA_CAMERA_INDEX or 0. Keeps retrying for
    `timeout` seconds (re-creating the capture if it won't open, re-reading if it
    opened but has no frame yet). Anything it opened is released on failure.
    """
    index = camera_index() if index is None else index
    deadline = clock() + timeout
    camera, opened_once = None, False
    while True:
        if camera is None:
            camera = capture(index)
        if camera.isOpened():
            opened_once = True
            success, frame = camera.read()
            if success and frame is not None:
                return camera
        else:
            _release(camera)
            camera = None
        if clock() >= deadline:
            break
        sleep(OPEN_RETRY_INTERVAL_S)
    if camera is not None:
        _release(camera)
    if opened_once:
        raise CameraError(f"Camera {index} opened but delivered no frame within {timeout:g} s. {_advice(index)}")
    raise CameraError(f"Couldn't open camera {index}. {_advice(index)}")


def read_frame(camera, max_failures=None, sleep=time.sleep):
    """The next frame, retrying up to max_failures consecutive failed reads
    (default MAX_FAILED_READS, READ_RETRY_INTERVAL_S apart); CameraError after that."""
    max_failures = MAX_FAILED_READS if max_failures is None else max_failures
    for attempt in range(max_failures + 1):
        success, frame = camera.read()
        if success and frame is not None:
            return frame
        if attempt < max_failures:
            sleep(READ_RETRY_INTERVAL_S)
    raise CameraError(f"The camera stopped delivering frames ({max_failures + 1} failed reads in a row). "
                      "It may have been disconnected, put to sleep or taken by another app.")
