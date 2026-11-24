"""Vision results -> plain JSON-safe dicts (no NumPy types, enums or FDA record dumps)."""
from vision.spatial import get_horizontal_position

MAX_CANDIDATES = 10
MAX_RECORDS = 5
MAX_BRANDS = 5

# Shown when a barcode was read but no source could say what the product is
SCAN_FRONT_LABEL = "scan_front_label"
UNRESOLVED_BARCODE_MESSAGE = ("Barcode scanned successfully. We couldn\u2019t identify this product in our "
                              "database. Try scanning the front label instead.")


def _next_step(identification):
    """A presentation hint only: it never changes or upgrades a status."""
    unresolved = identification.final_status.value == "UNRESOLVED"
    return SCAN_FRONT_LABEL if identification.observation["decoded"] and unresolved else None


def _box(box):
    return [int(v) for v in box] if box is not None else None


def _record(record):
    return {
        "brand_name": record["brand_name"],
        "labeler_name": record["labeler_name"],
        "product_ndc": record["product_ndc"],
        "ingredients": [{"name": i["name"], "strength": i["strength"]} for i in record["active_ingredients"]],
    }


def _package(package):
    if not package:
        return None
    return {"package_ndc": package["package_ndc"], "description": package.get("description")}


def barcode_identification(identification):
    observation, fda, entry = identification.observation, identification.fda, identification.registry_entry
    next_step = _next_step(identification)
    return {
        "next_step": next_step,
        "message": UNRESOLVED_BARCODE_MESSAGE if next_step else None,
        "barcode": {
            "type": observation["type"],
            "value": observation["value"],
            "decoded": bool(observation["decoded"]),
            "reason": observation.get("reason"),
            "box": _box(observation.get("box")),
        },
        "fda": {
            "status": fda.status.value,
            "reason": fda.reason,
            "record": _record(fda.record) if fda.record else None,
            "package": _package(fda.package),
            "label_record_count": len(fda.label_records),
            "label_records": [_record(r) for r in fda.label_records[:MAX_RECORDS]],
            "ndc_matches": [dict(_record(r), package_ndc=p["package_ndc"]) for r, p in fda.ndc_matches[:MAX_RECORDS]],
        },
        "registry": {
            "status": identification.registry_status,
            "source": entry["source"] if entry else None,
            "date_verified": entry["date_verified"] if entry else None,
            "package_description": entry["package_description"] if entry else None,
        },
        "product_name": identification.product_name,
        "ingredients": [{"name": name, "strength": strength} for name, strength in identification.ingredients],
        "package_ndc": identification.package_ndc,
        "package_ndc_source": identification.package_ndc_source,
        "source": identification.source,
        "final_status": identification.final_status.value,
        "reason": identification.reason,
    }


def detection(found, frame_width=None):
    data = {"class_name": str(found["class_name"]), "confidence": float(found["confidence"]),
            "box": _box(found["box"])}
    if frame_width:
        data["position"] = get_horizontal_position(found["box"], frame_width)
    return data


def environment_detection(found, frame_width, track_id=None):
    """One object for /api/detect; the box is in the submitted image's pixels.

    track_id is a temporary per-session number, or None when the request is
    not tracked or the object could not be associated reliably.
    """
    return {"class_name": str(found["class_name"]), "confidence": float(found["confidence"]),
            "position": get_horizontal_position(found["box"], frame_width),
            "bounding_box": _box(found["box"]),
            "track_id": int(track_id) if track_id is not None else None}


def ocr_results(results):
    return [{"text": str(r["text"]), "confidence": float(r["confidence"]), "box": _box(r.get("box"))}
            for r in results]


def _candidate(candidate):
    return {
        "ingredients": [{"name": name, "strength": strength} for name, strength in candidate.ingredients],
        "brand_names": candidate.brand_names[:MAX_BRANDS],
        "record_count": len(candidate.records),
        "not_observed": list(candidate.missing_ingredients) + list(candidate.base_only_ingredients),
        "brand_hint_records": int(candidate.brand_hint_records),
    }


def match_result(result):
    return {
        "status": result.status.value,
        "level": result.level.value,
        "reason": result.reason,
        "confidence": float(result.confidence),
        "package": _package(result.package),
        "candidate_count": len(result.candidates),
        "candidates": [_candidate(c) for c in result.candidates[:MAX_CANDIDATES]],
        "evidence": [{"kind": e.kind, "value": e.value, "confidence": float(e.confidence),
                      "corrected_from": e.corrected_from} for e in result.evidence],
        "conflicts": [{"value": c.evidence.value, "reason": c.reason} for c in result.conflicts],
        "ignored": [o.text for o in result.ignored],
    }
