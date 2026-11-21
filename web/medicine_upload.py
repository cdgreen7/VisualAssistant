"""Bounded, format-validated label crop decoding; no OCR downscaling or storage."""
import io
import warnings
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from web.services import ServiceError


def read_crop(upload):
    if upload is None:
        raise ServiceError(400, 'missing_image', 'Upload a label crop in the image field.')
    data = upload.read(8*1024*1024+1)
    if len(data) > 8*1024*1024:
        raise ServiceError(413, 'image_too_large', 'Medicine crops must be at most 8 MiB.')
    if not data:
        raise ServiceError(400, 'missing_image', 'The label crop is empty.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if (image.format, upload.mimetype) not in (('JPEG','image/jpeg'),('PNG','image/png')):
                    raise ValueError()
                if image.width*image.height > 7_000_000:
                    raise ServiceError(413, 'image_too_large', 'Medicine crops must be at most 7 megapixels.')
                image.load()
                return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ServiceError(413, 'image_too_large', 'Medicine crop has too many pixels.') from None
    except (OSError, ValueError, UnidentifiedImageError):
        raise ServiceError(400, 'invalid_image', 'Upload a readable JPEG or PNG label crop.') from None
