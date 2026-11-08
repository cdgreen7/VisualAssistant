"""NDC and UPC code normalization. Pure functions, no product data.

Canonical forms:
  package NDC -> 11 digits (5-4-2), e.g. "0573-0150-20" -> "00573015020"
  product NDC -> 9 digits (5-4),    e.g. "0573-0150"    -> "005730150"

An undashed 10-digit package NDC (or 8-digit product NDC) can be read more
than one way, so the *_candidates functions return every reading. A candidate
only means something once it matches a known code in the catalog.
"""
import re

PACKAGE_WIDTHS = (5, 4, 2)
PRODUCT_WIDTHS = (5, 4)

# Segment layouts that appear in the FDA NDC Directory
PACKAGE_FORMATS = ((4, 4, 2), (5, 3, 2), (5, 4, 1))
PRODUCT_FORMATS = ((4, 4), (5, 3), (5, 4))


def _clean(value):
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"^NDC[\s:#]*", "", text)
    return re.sub(r"\s+", "", text)


def _pad(parts, widths):
    return "".join(part.zfill(width) for part, width in zip(parts, widths))


def _split(digits, layout):
    parts, start = [], 0
    for width in layout:
        parts.append(digits[start:start + width])
        start += width
    return parts


def _normalize(value, formats, widths):
    text = _clean(value)
    total = sum(widths)
    if re.fullmatch(r"\d+(-\d+)+", text):
        parts = text.split("-")
        layout = tuple(len(part) for part in parts)
        if layout not in formats and layout != widths:
            return None
        return _pad(parts, widths)
    if re.fullmatch(r"\d{%d}" % total, text):
        return text
    return None


def _candidates(value, formats, widths):
    text = _clean(value)
    raw_length = sum(widths) - 1
    if re.fullmatch(r"\d{%d}" % raw_length, text):
        candidates = []
        for layout in formats:
            if sum(layout) != raw_length:
                continue
            candidate = _pad(_split(text, layout), widths)
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates

    normalized = _normalize(value, formats, widths)
    return [normalized] if normalized else []


def normalize_package_ndc(value):
    """Dashed or 11-digit package NDC -> 11 digits; None if ambiguous/invalid."""
    return _normalize(value, PACKAGE_FORMATS, PACKAGE_WIDTHS)


def normalize_product_ndc(value):
    """Dashed or 9-digit product NDC -> 9 digits; None if ambiguous/invalid."""
    return _normalize(value, PRODUCT_FORMATS, PRODUCT_WIDTHS)


def package_ndc_candidates(value):
    """Every 11-digit reading of a package NDC (up to 3 for undashed 10 digits)."""
    return _candidates(value, PACKAGE_FORMATS, PACKAGE_WIDTHS)


def product_ndc_candidates(value):
    """Every 9-digit reading of a product NDC (up to 2 for undashed 8 digits)."""
    return _candidates(value, PRODUCT_FORMATS, PRODUCT_WIDTHS)


def _gtin_check_digit_ok(digits):
    body, check = digits[:-1], int(digits[-1])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == check


def normalize_upc(value):
    """UPC-A (12) or EAN-13 (13) with a valid check digit -> 13 digits; else None."""
    digits = re.sub(r"[\s-]", "", str(value or ""))
    if not re.fullmatch(r"\d{12}|\d{13}", digits):
        return None
    if not _gtin_check_digit_ok(digits):
        return None
    return digits.zfill(13)


def ndc_candidates_from_upc(value):
    """Package NDC candidates embedded in a UPC, or [] when there can't be one.

    Only a valid UPC-A in number system 3 (EAN-13 "03...") can carry an NDC,
    and even then many don't. Candidates are unverified until a catalog
    lookup finds a matching package.
    """
    upc = normalize_upc(value)
    if upc is None or not upc.startswith("03"):
        return []
    return package_ndc_candidates(upc[2:12])
