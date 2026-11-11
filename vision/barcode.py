"""Standalone barcode decoding (not part of the live loop).

decode_barcodes(image) returns one observation per barcode found:

    {"type": "UPC_A", "value": "305730150200", "decoded": True, "box": (x1, y1, x2, y2)}

The value is exactly what the decoder read: no padding, no added or
corrected digits. A barcode value is not an NDC; mapping it to FDA records
is a separate step (vision/products/barcode_lookup.py).

Barcodes that are located but can't be used are still reported, with
decoded False, value None and a reason (a format outside SUPPORTED_FORMATS,
or a symbol the decoder flagged as invalid, e.g. a failed checksum).
"""
import zxingcpp

SUPPORTED_FORMATS = {
    zxingcpp.BarcodeFormat.UPCA: "UPC_A",
    zxingcpp.BarcodeFormat.EAN13: "EAN_13",
    zxingcpp.BarcodeFormat.Code128: "CODE_128",
    zxingcpp.BarcodeFormat.DataMatrix: "DATA_MATRIX",
}


def _box(position):
    corners = (position.top_left, position.top_right, position.bottom_right, position.bottom_left)
    xs, ys = [p.x for p in corners], [p.y for p in corners]
    return min(xs), min(ys), max(xs), max(ys)


def decode_barcodes(image):
    """All barcodes in a BGR or grayscale NumPy image, top to bottom."""
    if image is None or getattr(image, "size", 0) == 0:
        raise ValueError("decode_barcodes needs a non-empty image")

    observations = []
    for found in zxingcpp.read_barcodes(image, return_errors=True):
        name = SUPPORTED_FORMATS.get(found.format)
        observation = {"type": name or found.format.name.upper(), "value": None,
                       "decoded": False, "box": _box(found.position)}
        if name is None:
            observation["reason"] = "unsupported format"
        elif not found.valid:
            observation["reason"] = f"invalid symbol: {found.error}"
        else:
            observation.update(value=found.text, decoded=True)
        if observation not in observations:
            observations.append(observation)
    return sorted(observations, key=lambda o: (o["box"][1], o["box"][0]))
