"""Importing YOLO must not change what happens when an image can't be opened."""
import io
import unittest
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

from vision.safe_image import keep_pillow_open


class SafeImageTests(unittest.TestCase):
    def test_bad_image_is_an_error_not_a_package_install(self):
        from ultralytics.utils import checks, patches       # what loading the detector imports
        self.addCleanup(keep_pillow_open)
        Image.open = patches.image_open                      # the state right after that import
        self.assertTrue(keep_pillow_open())
        self.assertIs(Image.open, patches._image_open)
        self.assertFalse(keep_pillow_open())                 # nothing left to undo
        with patch.object(checks, "check_requirements", side_effect=AssertionError("tried to install a package")):
            with self.assertRaises(UnidentifiedImageError):
                Image.open(io.BytesIO(b"broken"))

    def test_good_images_still_open(self):
        keep_pillow_open()
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, "PNG")
        buffer.seek(0)
        self.assertEqual(Image.open(buffer).size, (4, 4))

    def test_loading_the_detector_through_the_web_services_restores_pillow(self):
        from ultralytics.utils import patches
        from web import services
        self.addCleanup(keep_pillow_open)
        Image.open = patches.image_open
        with patch("vision.detector.ObjectDetector", return_value="detector"):
            self.assertEqual(services._load_detector(), "detector")
        self.assertIs(Image.open, patches._image_open)


if __name__ == "__main__":
    unittest.main()
