"""Keep Pillow's own image opener.

Importing Ultralytics (YOLO) replaces PIL.Image.open with a version that, the
first time any image fails to open, tries to `pip install pi-heif` from the
internet. In this project that meant a damaged upload could trigger a package
installation attempt on the server and come back as a 500 error instead of
"invalid image". We never pass file paths to YOLO, so its patch is not needed.
"""
import sys

from PIL import Image


def keep_pillow_open():
    """Undo Ultralytics' PIL.Image.open patch if it is installed. Safe to call repeatedly."""
    patches = sys.modules.get("ultralytics.utils.patches")
    original = getattr(patches, "_image_open", None)
    if original is not None and Image.open is getattr(patches, "image_open", None):
        Image.open = original
        return True
    return False
