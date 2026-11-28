"""vision/camera.py: camera selection, start-up retries, bounded read retries,
error messages and release. Fake captures and a fake clock; no camera is used."""
import unittest
from unittest import mock

import numpy as np

from vision import camera as cam

FRAME = np.zeros((4, 6, 3), np.uint8)


class FakeCapture:
    """Scripted cv2.VideoCapture: `opened` and a list of read results (True/False)."""
    def __init__(self, index, opened=True, reads=(True,)):
        self.index, self.opened, self.reads, self.released = index, opened, list(reads), 0

    def isOpened(self):
        return self.opened

    def read(self):
        ok = self.reads.pop(0) if self.reads else self.reads_default
        return (True, FRAME) if ok else (False, None)

    reads_default = True

    def release(self):
        self.released += 1


class Clock:
    """Fake monotonic clock that advances when sleep() is called."""
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def factory(*captures):
    """capture(index) returning the given captures in order (the last one repeats)."""
    made = []

    def capture(index):
        if len(made) < len(captures):
            cap = captures[len(made)]
        else:                                   # further attempts behave like the last scripted one
            cap = FakeCapture(index, captures[-1].opened, [])
        cap.index = index
        made.append(cap)
        return cap
    capture.made = made
    return capture


class CameraIndexTests(unittest.TestCase):
    def test_default_and_environment(self):
        self.assertEqual(cam.camera_index({}), 0)
        self.assertEqual(cam.camera_index({"VA_CAMERA_INDEX": ""}), 0)
        self.assertEqual(cam.camera_index({"VA_CAMERA_INDEX": "2"}), 2)
        self.assertEqual(cam.camera_index({"VA_CAMERA_INDEX": " 1 "}), 1)
        for bad in ("-1", "one", "1.5", "0x1"):
            with self.assertRaises(cam.CameraError) as caught:
                cam.camera_index({"VA_CAMERA_INDEX": bad})
            self.assertIn("VA_CAMERA_INDEX", str(caught.exception))

    def test_open_camera_uses_the_environment_only_when_no_index_is_given(self):
        clock = Clock()
        capture = factory(FakeCapture(None))
        with mock.patch.dict("os.environ", {"VA_CAMERA_INDEX": "1"}):
            cam.open_camera(capture=capture, sleep=clock.sleep, clock=clock)
            cam.open_camera(index=3, capture=capture, sleep=clock.sleep, clock=clock)
        self.assertEqual([c.index for c in capture.made], [1, 3])
        with mock.patch.dict("os.environ", {}, clear=True):
            capture = factory(FakeCapture(None))
            cam.open_camera(capture=capture, sleep=clock.sleep, clock=clock)
            self.assertEqual(capture.made[0].index, 0)                    # backward compatible default


class UnchangedCaptureSettingsTests(unittest.TestCase):
    def test_plain_videocapture_without_setting_resolution_or_backend(self):
        import inspect
        import cv2
        self.assertIs(inspect.signature(cam.open_camera).parameters["capture"].default, cv2.VideoCapture)
        with open(cam.__file__) as f:
            source = f.read()
        self.assertNotIn(".set(", source)                       # no resolution, FPS or format changes
        self.assertNotIn("CAP_", source)                        # no explicit backend: OpenCV's default, as before


class OpenCameraTests(unittest.TestCase):
    def open(self, *captures, timeout=cam.OPEN_TIMEOUT_S):
        clock = Clock()
        capture = factory(*captures)
        try:
            return cam.open_camera(0, timeout, capture, clock.sleep, clock), capture, clock
        except cam.CameraError as error:
            return error, capture, clock

    def test_opens_at_once(self):
        result, capture, clock = self.open(FakeCapture(0))
        self.assertIs(result, capture.made[0])
        self.assertEqual((clock.sleeps, result.released), ([], 0))

    def test_retries_until_the_camera_opens(self):
        result, capture, clock = self.open(FakeCapture(0, opened=False), FakeCapture(0, opened=False), FakeCapture(0))
        self.assertIs(result, capture.made[2])
        self.assertEqual([c.released for c in capture.made], [1, 1, 0])     # failed attempts released
        self.assertEqual(len(clock.sleeps), 2)

    def test_waits_for_the_first_frame(self):
        result, capture, clock = self.open(FakeCapture(0, reads=[False, False, False, True]))
        self.assertIs(result, capture.made[0])
        self.assertEqual(len(capture.made), 1)                               # same capture, read again

    def test_never_opens(self):
        with mock.patch.object(cam.sys, "platform", "darwin"):
            error, capture, clock = self.open(FakeCapture(0, opened=False), timeout=1.0)
        message = str(error)
        self.assertIsInstance(error, cam.CameraError)
        self.assertIsInstance(error, RuntimeError)                          # as before: a RuntimeError
        self.assertIn("Couldn't open camera 0", message)
        self.assertIn("Privacy & Security > Camera", message)
        self.assertIn("VA_CAMERA_INDEX=1", message)
        self.assertTrue(all(c.released == 1 for c in capture.made))
        self.assertLessEqual(clock.now, 1.0 + cam.OPEN_RETRY_INTERVAL_S)   # bounded

    def test_opens_but_never_delivers_a_frame(self):
        stuck = FakeCapture(0, reads=[False] * 1000)
        error, capture, clock = self.open(stuck, timeout=1.0)
        self.assertIn("opened but delivered no frame within 1 s", str(error))
        self.assertEqual(stuck.released, 1)

    def test_permission_advice_only_on_macos(self):
        with mock.patch.object(cam.sys, "platform", "linux"):
            error, _, _ = self.open(FakeCapture(0, opened=False), timeout=0.2)
        self.assertNotIn("Privacy & Security", str(error))
        self.assertIn("VA_CAMERA_INDEX", str(error))


class ReadFrameTests(unittest.TestCase):
    def test_returns_frames(self):
        self.assertIs(cam.read_frame(FakeCapture(0), sleep=lambda s: None), FRAME)

    def test_survives_temporary_failures(self):
        sleeps = []
        camera = FakeCapture(0, reads=[False, False, False, True])
        self.assertIs(cam.read_frame(camera, sleep=sleeps.append), FRAME)
        self.assertEqual(sleeps, [cam.READ_RETRY_INTERVAL_S] * 3)

    def test_gives_up_after_a_bounded_number_of_failures(self):
        sleeps = []
        camera = FakeCapture(0, reads=[False] * 100)
        with self.assertRaises(cam.CameraError) as caught:
            cam.read_frame(camera, max_failures=5, sleep=sleeps.append)
        self.assertIn("6 failed reads in a row", str(caught.exception))
        self.assertEqual(len(sleeps), 5)
        self.assertEqual(len(camera.reads), 100 - 6)                        # exactly max_failures + 1 reads
        with self.assertRaises(cam.CameraError):
            cam.read_frame(FakeCapture(0, reads=[False] * 100), sleep=lambda s: None)
        self.assertLessEqual(cam.MAX_FAILED_READS * cam.READ_RETRY_INTERVAL_S, 1.0)   # about a second at most

    def test_each_call_starts_a_new_count(self):
        camera = FakeCapture(0, reads=[False, False, True, False, False, True])
        self.assertIs(cam.read_frame(camera, max_failures=2, sleep=lambda s: None), FRAME)
        self.assertIs(cam.read_frame(camera, max_failures=2, sleep=lambda s: None), FRAME)


if __name__ == "__main__":
    unittest.main()
