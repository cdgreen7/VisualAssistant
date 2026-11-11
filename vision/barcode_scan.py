"""User-triggered barcode scan of one clean camera frame.

frame -> decode_barcodes -> identify_barcode (FDA catalog + local UPC registry)
      -> printable text.

The catalog and registry are passed in, so they are loaded once by the caller
(see main.py). A decoded barcode is never reported as ProductMatcher's
VERIFIED; see vision/products/barcode_identify.py for the final statuses.
"""
from vision.barcode import decode_barcodes
from vision.products.barcode_identify import identify_barcode
from vision.products.barcode_lookup import conflicting

NO_BARCODE = "No barcode detected. Move closer, improve lighting, or rotate the package."
MAX_RECORDS = 5
REGISTRY_NOTES = {
    "VERIFIED": "VERIFIED (confirmed against the package by a person)",
    "PENDING": "PENDING (not yet confirmed; not used for identification)",
    "NOT_FOUND": "NOT_FOUND",
    "NOT_LOADED": "NOT_LOADED (registry file missing or invalid)",
}


def describe_record(record):
    ingredients = " + ".join(f"{i['name']} {i['strength']}" for i in record["active_ingredients"])
    return f"{record['brand_name']} ({record['labeler_name']}) - {ingredients} [product NDC {record['product_ndc']}]"


def _fda_lines(fda):
    lines = [fda.status.value, fda.reason]
    if fda.record:
        lines.append(describe_record(fda.record))
        if fda.package:
            lines.append(f"Package: NDC {fda.package['package_ndc']} - {fda.package['description']}")
        return lines
    if fda.label_records:
        lines.append("Listed on FDA labels:")
        lines += ["  - " + describe_record(r) for r in fda.label_records[:MAX_RECORDS]]
        if len(fda.label_records) > MAX_RECORDS:
            lines.append(f"  ... {len(fda.label_records) - MAX_RECORDS} more")
    if fda.ndc_matches:
        lines.append("NDC read from the UPC digits:")
        lines += [f"  - {describe_record(r)}, package {p['package_ndc']}" for r, p in fda.ndc_matches[:MAX_RECORDS]]
    return lines


def format_identification(identification):
    observation = identification.observation
    value = observation["value"] if observation["decoded"] else f"(not decoded: {observation['reason']})"
    lines = ["--- BARCODE IDENTIFICATION ---", "",
             "Barcode:", f"Type: {observation['type'].replace('_', '-')}", f"Value: {value}", "",
             "FDA lookup:", *_fda_lines(identification.fda), "",
             "Local registry:", REGISTRY_NOTES[identification.registry_status], "",
             "Product:", identification.product_name or "Not identified", "",
             "Ingredients:"]
    lines += [f"{name} {strength}" for name, strength in identification.ingredients] or ["Not established"]
    if identification.package_ndc:
        lines += ["", "Package NDC:", f"{identification.package_ndc} (from {identification.package_ndc_source})"]
    entry = identification.registry_entry
    source = identification.source or "None"
    if entry and identification.source:
        source += f" - registry entry: {entry['source']}, verified {entry['date_verified']}"
    lines += ["", "Source:", source, "", "Final status:", identification.final_status.value, identification.reason]
    return "\n".join(lines)


def scan_frame(frame, catalog, registry=None):
    """One BarcodeIdentification per barcode in frame (empty list if there are none).

    frame must be the clean camera frame (no boxes or FPS text drawn on it).
    """
    return [identify_barcode(observation, catalog, registry) for observation in decode_barcodes(frame)]


def format_scan(identifications):
    if not identifications:
        return NO_BARCODE
    lines = ["\n\n".join(format_identification(i) for i in identifications)]
    if conflicting([i.fda for i in identifications]):
        lines += ["", "The barcodes in this frame name different FDA records: CONFLICT"]
    return "\n".join(lines)
