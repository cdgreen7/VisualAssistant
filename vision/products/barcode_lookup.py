"""Map a decoded barcode observation to FDA catalog records.

A UPC is not an NDC. Two independent sources can link a UPC to a product:

  label   - openFDA lists the UPC on a drug label (catalog.find_by_upc).
            Label-level: it names records, never a package size.
  ndc     - a number-system-3 UPC may carry a 10-digit NDC; a reading counts
            only if it is a known package NDC (catalog.find_by_upc_ndc).

Statuses (none of them is ProductMatcher's VERIFIED):
  RESOLVED   - both sources agree on one record and one package
  CANDIDATE  - one source only, pointing to a single record
  AMBIGUOUS  - several records or packages fit
  CONFLICT   - the two sources point to different records
  UNRESOLVED - nothing reliable; includes formats with no established mapping
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from vision.products.ndc import normalize_upc

UPC_TYPES = ("UPC_A", "EAN_13")


class LookupStatus(Enum):
    RESOLVED = "RESOLVED"
    CANDIDATE = "CANDIDATE"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICT = "CONFLICT"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class BarcodeLookup:
    status: LookupStatus
    reason: str
    observation: dict
    label_records: list = field(default_factory=list)  # records whose FDA label lists the UPC
    ndc_matches: list = field(default_factory=list)    # (record, package) from the embedded NDC
    record: Optional[dict] = None                      # set for RESOLVED and CANDIDATE
    package: Optional[dict] = None                     # set only when a package NDC matched


def _unique(records):
    seen = {}
    for record in records:
        seen.setdefault(record["product_id"], record)
    return list(seen.values())


def lookup_barcode(observation, catalog):
    def result(status, reason, **fields):
        return BarcodeLookup(status, reason, observation, label_records, ndc_matches, **fields)

    label_records, ndc_matches = [], []
    if not observation.get("decoded"):
        return result(LookupStatus.UNRESOLVED, "barcode was not decoded")
    if observation["type"] not in UPC_TYPES:
        return result(LookupStatus.UNRESOLVED, f"no established FDA mapping for {observation['type']} values")
    if normalize_upc(observation["value"]) is None:
        return result(LookupStatus.UNRESOLVED, "not a valid UPC-A/EAN-13 (length or check digit)")

    label_records = _unique(catalog.find_by_upc(observation["value"]))
    ndc_matches = catalog.find_by_upc_ndc(observation["value"])
    ndc_records = _unique(record for record, _ in ndc_matches)
    label_ids = {record["product_id"] for record in label_records}

    if not label_records and not ndc_matches:
        return result(LookupStatus.UNRESOLVED, "UPC is not in the FDA label data and carries no known package NDC")

    if label_records and ndc_matches:
        agreeing = [(r, p) for r, p in ndc_matches if r["product_id"] in label_ids]
        if not agreeing:
            return result(LookupStatus.CONFLICT,
                          "FDA label data and the NDC inside the UPC point to different records")
        if len(agreeing) == 1:
            record, package = agreeing[0]
            return result(LookupStatus.RESOLVED,
                          "UPC is listed on this record's FDA label and carries one of its package NDCs",
                          record=record, package=package)
        return result(LookupStatus.AMBIGUOUS, "the UPC's digits match more than one package NDC")

    if ndc_matches:
        if len(ndc_matches) == 1:
            record, package = ndc_matches[0]
            return result(LookupStatus.CANDIDATE,
                          "UPC carries a known package NDC, but FDA label data does not list this UPC",
                          record=record, package=package)
        return result(LookupStatus.AMBIGUOUS, "the UPC's digits match more than one package NDC")

    if len(label_records) == 1:
        return result(LookupStatus.CANDIDATE,
                      "UPC is listed on one FDA label; package size is not established",
                      record=label_records[0])
    return result(LookupStatus.AMBIGUOUS, f"UPC is listed on {len(label_records)} FDA records")


def conflicting(lookups):
    """True when two barcodes on one image name different records."""
    named = [lookup.record["product_id"] for lookup in lookups if lookup.record]
    return len(set(named)) > 1
