"""Barcode decoding only (no FDA catalog). Images are generated symbols, not photos."""
import unittest

import cv2
import numpy as np
import zxingcpp

from vision.barcode import decode_barcodes

F = zxingcpp.BarcodeFormat


def symbol(fmt, text, width=300, height=120):
    return np.array(zxingcpp.write_barcode(fmt, text, width, height, quiet_zone=20))


def stack(*images):
    width = max(i.shape[1] for i in images)
    rows = []
    for image in images:
        rows.append(cv2.copyMakeBorder(image, 20, 20, 0, width - image.shape[1], cv2.BORDER_CONSTANT, value=255))
    return np.vstack(rows)


def values(observations):
    return [(o["type"], o["value"], o["decoded"]) for o in observations]


class DecodeTests(unittest.TestCase):
    def test_upc_a_value_is_kept_exactly(self):
        observations = decode_barcodes(symbol(F.UPCA, "305730150200"))
        self.assertEqual(values(observations), [("UPC_A", "305730150200", True)])  # 12 digits, not padded to 13
        x1, y1, x2, y2 = observations[0]["box"]
        self.assertTrue(x2 > x1 and y2 > y1)

    def test_each_supported_format(self):
        for fmt, text, expected in ((F.EAN13, "4006381333931", "EAN_13"), (F.Code128, "LOT-A123", "CODE_128")):
            self.assertEqual(values(decode_barcodes(symbol(fmt, text))), [(expected, text, True)])
        matrix = symbol(F.DataMatrix, "0100305730150205", 200, 200)
        self.assertEqual(values(decode_barcodes(matrix)), [("DATA_MATRIX", "0100305730150205", True)])

    def test_bgr_photo_like_input(self):
        scene = np.full((480, 640), 200, np.uint8)
        code = symbol(F.UPCA, "305730150200")
        scene[100:100 + code.shape[0], 150:150 + code.shape[1]] = code
        matrix = cv2.getRotationMatrix2D((320, 240), 12, 1.0)
        scene = cv2.warpAffine(scene, matrix, (640, 480), borderValue=200)
        scene = cv2.GaussianBlur(cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR), (0, 0), 1)
        self.assertEqual(values(decode_barcodes(scene)), [("UPC_A", "305730150200", True)])

    def test_no_barcode(self):
        self.assertEqual(decode_barcodes(np.full((200, 300, 3), 255, np.uint8)), [])
        noise = (np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8)
        self.assertEqual(decode_barcodes(noise), [])

    def test_multiple_barcodes_top_to_bottom(self):
        image = stack(symbol(F.UPCA, "305730150200"), symbol(F.Code128, "LOT-A123"), symbol(F.UPCA, "012345678905"))
        self.assertEqual(values(decode_barcodes(image)), [("UPC_A", "305730150200", True),
                                                          ("CODE_128", "LOT-A123", True),
                                                          ("UPC_A", "012345678905", True)])

    def test_unsupported_format_is_reported_without_a_value(self):
        observations = decode_barcodes(symbol(F.QRCode, "hello", 200, 200))
        self.assertEqual(values(observations), [("QRCODE", None, False)])
        self.assertEqual(observations[0]["reason"], "unsupported format")

    def test_damaged_barcode_is_never_guessed(self):
        code = symbol(F.UPCA, "305730150200")
        code[:, 140:200] = 255  # wipe out several digits
        for observation in decode_barcodes(code):
            self.assertFalse(observation["decoded"])
            self.assertIsNone(observation["value"])

    def test_empty_image_is_an_error(self):
        with self.assertRaises(ValueError):
            decode_barcodes(None)
        with self.assertRaises(ValueError):
            decode_barcodes(np.zeros((0, 0, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
