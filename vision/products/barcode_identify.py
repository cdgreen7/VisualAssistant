"""Reconcile the FDA barcode lookup with the local UPC registry.

Final statuses (none is ProductMatcher's VERIFIED):
  IDENTIFIED_FROM_FDA      - FDA lookup RESOLVED; no verified registry entry
  IDENTIFIED_FROM_REGISTRY - verified registry entry; FDA has nothing, or only
                             an ambiguous listing that includes the same formulation
  IDENTIFIED_FROM_BOTH     - FDA points to one record with the registry's formulation
  CANDIDATE                - FDA has a single unconfirmed record; no verified entry
  AMBIGUOUS                - FDA lists several records; no verified entry
  CONFLICT                 - FDA and the registry (or FDA's own sources) disagree
  UNRESOLVED               - neither source identifies the product

Agreement means the same set of active ingredients and strengths. Brand names
are not compared (FDA brand strings differ from package wording). A PENDING
registry entry is reported but never used.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from vision.products.barcode_lookup import UPC_TYPES, BarcodeLookup, LookupStatus, lookup_barcode
from vision.products.catalog import strength_key
from vision.products.matcher import ingredient_key
from vision.products.ndc import normalize_package_ndc
from vision.products.upc_registry import VERIFIED

SOURCE_FDA, SOURCE_REGISTRY, SOURCE_BOTH = "FDA", "Local Verified Registry", "Both"
REGISTRY_NOT_FOUND, REGISTRY_NOT_LOADED = "NOT_FOUND", "NOT_LOADED"


class FinalStatus(Enum):
    IDENTIFIED_FROM_FDA = "IDENTIFIED_FROM_FDA"
    IDENTIFIED_FROM_REGISTRY = "IDENTIFIED_FROM_REGISTRY"
    IDENTIFIED_FROM_BOTH = "IDENTIFIED_FROM_BOTH"
    CANDIDATE = "CANDIDATE"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICT = "CONFLICT"
    UNRESOLVED = "UNRESOLVED"


@dataclass
class BarcodeIdentification:
    observation: dict
    fda: BarcodeLookup
    registry_status: str                    # VERIFIED, PENDING, NOT_FOUND or NOT_LOADED
    final_status: FinalStatus
    reason: str
    source: Optional[str] = None            # set only when a product is named
    product_name: Optional[str] = None
    ingredients: list = field(default_factory=list)  # [(name, strength)]
    package_ndc: Optional[str] = None
    package_ndc_source: Optional[str] = None
    registry_entry: Optional[dict] = None   # the VERIFIED entry, if any


def _formulation(ingredients):
    return frozenset((ingredient_key(i["name"]), strength_key(i["strength"])) for i in ingredients)


def _from_record(record):
    return {"product_name": record["brand_name"],
            "ingredients": [(i["name"], i["strength"]) for i in record["active_ingredients"]]}


def _fda_only(fda):
    """Outcome when there is no verified registry entry."""
    fields = {}
    if fda.record:
        fields = _from_record(fda.record)
        if fda.package:
            fields.update(package_ndc=fda.package["package_ndc"], package_ndc_source=SOURCE_FDA)
    if fda.status == LookupStatus.RESOLVED:
        return FinalStatus.IDENTIFIED_FROM_FDA, fda.reason, dict(fields, source=SOURCE_FDA)
    if fda.status == LookupStatus.CANDIDATE:
        return FinalStatus.CANDIDATE, "single FDA record, not independently confirmed", dict(fields, source=SOURCE_FDA)
    return FinalStatus[fda.status.value], fda.reason, {}


def _with_registry(fda, entry, catalog):
    """Outcome when a person has confirmed this UPC in the registry."""
    truth = _formulation(entry["ingredients"])
    fields = {"product_name": entry["product_name"], "source": SOURCE_REGISTRY,
              "ingredients": [(i["name"], i["strength"]) for i in entry["ingredients"]]}
    if entry["package_ndc"]:
        fields.update(package_ndc=entry["package_ndc"], package_ndc_source=SOURCE_REGISTRY)

    if fda.status == LookupStatus.CONFLICT:
        return FinalStatus.CONFLICT, "FDA's own sources disagree about this UPC", {}
    if entry["package_ndc"]:
        if fda.package and (normalize_package_ndc(fda.package["package_ndc"])
                            != normalize_package_ndc(entry["package_ndc"])):
            return FinalStatus.CONFLICT, "registry and FDA give different package NDCs", {}
        known = catalog.find_by_package_ndc(entry["package_ndc"])
        if known and not any(_formulation(r["active_ingredients"]) == truth for r, _ in known):
            return (FinalStatus.CONFLICT,
                    "the registry's package NDC belongs to a different formulation in the FDA catalog", {})

    if fda.status == LookupStatus.UNRESOLVED:
        return FinalStatus.IDENTIFIED_FROM_REGISTRY, "confirmed in the local registry; FDA has no mapping", fields

    records = [fda.record] if fda.record else fda.label_records + [r for r, _ in fda.ndc_matches]
    consistent = [r for r in records if _formulation(r["active_ingredients"]) == truth]
    if not consistent:
        return FinalStatus.CONFLICT, "registry and FDA give different active ingredients or strengths", {}
    if fda.record:
        fields["source"] = SOURCE_BOTH
        if fda.package and not entry["package_ndc"]:
            fields.update(package_ndc=fda.package["package_ndc"], package_ndc_source=SOURCE_FDA)
        return FinalStatus.IDENTIFIED_FROM_BOTH, "registry and FDA give the same active ingredients and strengths", fields
    return (FinalStatus.IDENTIFIED_FROM_REGISTRY,
            f"confirmed in the local registry; FDA data is ambiguous ({len(consistent)} of {len(records)} "
            "FDA records have the same formulation)", fields)


def identify_barcode(observation, catalog, registry=None):
    """FDA lookup + registry lookup for one barcode observation, reconciled."""
    fda = lookup_barcode(observation, catalog)
    entry = None
    if registry is None:
        registry_status = REGISTRY_NOT_LOADED
    else:
        if observation.get("decoded") and observation["type"] in UPC_TYPES:
            entry = registry.find(observation["value"])
        registry_status = entry["verification_status"] if entry else REGISTRY_NOT_FOUND

    verified = entry if entry and entry["verification_status"] == VERIFIED else None
    status, reason, fields = _with_registry(fda, verified, catalog) if verified else _fda_only(fda)
    return BarcodeIdentification(observation, fda, registry_status, status, reason,
                                 registry_entry=verified, **fields)
