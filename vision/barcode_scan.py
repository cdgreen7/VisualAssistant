"""User-triggered barcode scan of one clean camera frame.

frame -> decode_barcodes -> lookup_barcode (FDA catalog) -> printable text.

The catalog is passed in, so it is loaded once by the caller (see main.py).
A decoded barcode is never reported as VERIFIED; the statuses are the
lookup's own: RESOLVED, CANDIDATE, AMBIGUOUS, CONFLICT, UNRESOLVED.
"""
from vision.barcode import decode_barcodes
from vision.products.barcode_lookup import conflicting, lookup_barcode

NO_BARCODE = "No barcode detected. Move closer, improve lighting, or rotate the package."
MAX_RECORDS = 5


def describe_record(record):
    ingredients = " + ".join(f"{i['name']} {i['strength']}" for i in record["active_ingredients"])
    return f"{record['brand_name']} ({record['labeler_name']}) - {ingredients} [product NDC {record['product_ndc']}]"


def format_lookup(lookup):
    observation = lookup.observation
    if not observation["decoded"]:
        return "\n".join(["Barcode found but not decoded", f"Type: {observation['type']}",
                          f"Reason: {observation['reason']}", "", "Status: UNRESOLVED"])

    lines = ["Barcode detected", f"Type: {observation['type']}", f"Value: {observation['value']}", "", "FDA lookup:"]
    if lookup.record:
        lines.append(describe_record(lookup.record))
        if lookup.package:
            lines.append(f"Package: NDC {lookup.package['package_ndc']} - {lookup.package['description']}")
    elif lookup.label_records or lookup.ndc_matches:
        if lookup.label_records:
            lines.append("Listed on FDA labels:")
            lines += ["  - " + describe_record(r) for r in lookup.label_records[:MAX_RECORDS]]
            if len(lookup.label_records) > MAX_RECORDS:
                lines.append(f"  ... {len(lookup.label_records) - MAX_RECORDS} more")
        if lookup.ndc_matches:
            lines.append("NDC read from the UPC digits:")
            lines += [f"  - {describe_record(r)}, package {p['package_ndc']}" for r, p in lookup.ndc_matches[:MAX_RECORDS]]
    else:
        lines.append("No reliable product mapping found.")
    lines += [lookup.reason, "", f"Status: {lookup.status.value}"]
    return "\n".join(lines)


def scan_frame(frame, catalog):
    """One BarcodeLookup per barcode in frame (empty list if there are none).

    frame must be the clean camera frame (no boxes or FPS text drawn on it).
    """
    return [lookup_barcode(observation, catalog) for observation in decode_barcodes(frame)]


def format_scan(lookups):
    if not lookups:
        return NO_BARCODE
    lines = ["--- BARCODE SCAN ---", "", "\n\n".join(format_lookup(lookup) for lookup in lookups)]
    if conflicting(lookups):
        lines += ["", "The barcodes in this frame name different FDA records: CONFLICT"]
    return "\n".join(lines)
