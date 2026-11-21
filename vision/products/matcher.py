"""Match OCR evidence from medicine packaging against the FDA product catalog.

Status:
  VERIFIED   - evidence is strong enough to name a formulation or package
  UNCERTAIN  - product evidence exists but is ambiguous, incomplete or conflicting
  NO_MATCH   - no product evidence

Level (what a VERIFIED result identifies):
  FORMULATION - one active ingredient + strength combination; it may span many
                FDA records (companies, repackagers, product NDCs, packages)
  PACKAGE     - one package NDC
  NONE

Observations (what OCR read) are kept separate from catalog facts. A
candidate never rewrites an observation; it lists what was not observed.

Missing ingredients never remove a candidate: OCR may simply have missed them.
So "ADVIL + IBUPROFEN + 200 mg" keeps Advil PM and cold/sinus products as
competitors and stays UNCERTAIN.

There is no fuzzy matching in V1: "ADV1L" is not read as "ADVIL". An
observed base name ("DIPHENHYDRAMINE") keeps salt forms as candidates but
can't verify them; the salt must be read.

OCR corrections are limited to a letter O inside a numeric strength
("2oomg" -> 200mg). Corrected evidence keeps its original text, can narrow
candidates, but never satisfies VERIFIED on its own.

Store-brand words used by few labelers (EQUATE) become "brand_hint"
evidence, and brands named in "Compare to ..." text become "compared_brand"
evidence. Neither narrows candidates or counts toward VERIFIED.
"""
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from vision.products.catalog import name_key, strength_key

# Prototype engineering thresholds, NOT probabilities. OCR confidence is
# EasyOCR's own score; these values need empirical calibration on real photos.
MIN_CONFIDENCE = 0.3     # below this an OCR result is ignored entirely
VERIFY_CONFIDENCE = 0.6  # evidence used to VERIFY must reach this

# A brand phrase counts as distinctive only if it contains a word that is not
# generic and appears in at most this many FDA brand names ("ADVIL": 66,
# "PAIN": 2313, "RELIEF": 4263). Also a prototype value.
MAX_BRAND_WORD_FREQUENCY = 100

# An exact FDA brand phrase that fails the rule above but has a word used by
# at most this many labelers is kept as a "brand_hint" (e.g. store brands:
# "EQUATE" is in 341 brand names but from 11 labelers; "ANTACID" from 130).
# Hints are reported and used to order candidates; they never narrow
# candidates or count toward VERIFIED. Prototype value.
MAX_BRAND_HINT_LABELERS = 20

MAX_PHRASE_WORDS = 6

STOP_WORDS = {"AND", "OF", "THE", "WITH", "FOR", "IN", "TO", "A", "AN", "OR", "PLUS"}
UNITS = {"MG", "MCG", "G", "ML"}
SYNONYMS = {"HCL": "HYDROCHLORIDE"}

# Not preceded by a digit, "." or letter O, so fragments of a misread number
# ("1O.5 MG" -> "5 MG", "BO0MG" -> "0MG") are not read as strengths.
STRENGTH_RE = re.compile(r"(?<![0-9.O])(\d+(?:\.\d+)?)\s*(MG|MCG|G|ML)\b")
# A strength whose number starts with a digit but has letter O in place of
# zero ("2OOMG", "20OMG"). Only this pattern is corrected; brands, NDCs and
# other text never are.
CONFUSED_STRENGTH_RE = re.compile(r"(?<![A-Z0-9.])(\d[0-9O]*(?:\.[0-9O]+)?)\s*(MG|MCG|G|ML)\b")
NDC_RE = re.compile(r"\b\d{4,5}-\d{3,4}-\d{1,2}\b|NDC[\s:#]*\d{10,11}\b")


class Status(str, Enum):
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    NO_MATCH = "NO_MATCH"


class Level(str, Enum):
    FORMULATION = "FORMULATION"
    PACKAGE = "PACKAGE"
    NONE = "NONE"


@dataclass(frozen=True)
class Observation:
    """One OCR result exactly as read."""
    index: int
    text: str
    confidence: float


@dataclass(frozen=True)
class Evidence:
    """A product-relevant phrase found in one or two observations.

    value is the normalized form of the observed text, not a catalog value.
    corrected_from holds the original text when value is an OCR correction
    ("2OOMG" -> "200mg"); corrected evidence can't satisfy VERIFIED.
    """
    kind: str  # "brand" | "brand_hint" | "compared_brand" | "ingredient" | "strength" | "ndc"
    value: str
    observations: Tuple[Observation, ...]
    corrected_from: Optional[str] = None

    @property
    def confidence(self):
        return min(o.confidence for o in self.observations)


@dataclass
class Conflict:
    evidence: Evidence
    reason: str


@dataclass
class FormulationCandidate:
    """FDA records sharing one active ingredient + strength combination."""
    ingredients: Tuple[Tuple[str, str], ...]  # catalog (name, strength)
    records: List[dict]
    missing_ingredients: List[str]            # catalog ingredients not observed
    base_only_ingredients: List[str]          # observed only by base name, salt not read
    missing_strengths: List[str]              # catalog strengths not observed
    missing_brand_words: Dict[str, List[str]]  # catalog brand -> words not observed
    brand_hint_records: int = 0               # records whose brand contains an observed brand hint

    @property
    def product_ids(self):
        return [r["product_id"] for r in self.records]

    @property
    def brand_names(self):
        return sorted({r["brand_name"] for r in self.records if r["brand_name"]})


@dataclass
class MatchResult:
    status: Status
    level: Level
    candidates: List[FormulationCandidate]
    evidence: List[Evidence]
    conflicts: List[Conflict]
    confidence: float  # lowest OCR confidence the decision relied on; not a probability
    reason: str
    package: Optional[dict] = None  # {"product_id", "package_ndc", "description"} for PACKAGE
    ignored: List[Observation] = field(default_factory=list)  # below MIN_CONFIDENCE


def ingredient_key(name):
    return " ".join(SYNONYMS.get(word, word) for word in name_key(name).split())


def _formulation_key(record):
    return frozenset((ingredient_key(i["name"]), strength_key(i["strength"])) for i in record["active_ingredients"])


def _compatible(observed, catalog):
    """Observed ingredient matches a catalog one exactly, or names its base
    ("DIPHENHYDRAMINE" vs "DIPHENHYDRAMINE CITRATE")."""
    return catalog == observed or catalog.startswith(observed + " ")


class ProductMatcher:
    def __init__(self, catalog, min_confidence=MIN_CONFIDENCE, verify_confidence=VERIFY_CONFIDENCE,
                 max_brand_word_frequency=MAX_BRAND_WORD_FREQUENCY, max_brand_hint_labelers=MAX_BRAND_HINT_LABELERS):
        self.catalog = catalog
        self.min_confidence = min_confidence
        self.verify_confidence = verify_confidence
        self.max_brand_word_frequency = max_brand_word_frequency
        self.max_brand_hint_labelers = max_brand_hint_labelers

        self._records = catalog.records
        self._position = {r["product_id"]: i for i, r in enumerate(self._records)}

        # Brand words -> record positions; used only to expand an already
        # matched complete brand phrase into its family (ADVIL -> ADVIL PM, ...)
        self._brand_word_index = defaultdict(set)
        brand_word_labelers = defaultdict(set)
        for i, record in enumerate(self._records):
            for word in name_key(record["brand_name"]).split():
                self._brand_word_index[word].add(i)
                brand_word_labelers[word].add(name_key(record["labeler_name"]))
        self._brand_word_labelers = {word: len(labelers) for word, labelers in brand_word_labelers.items()}

        self._ingredient_index = defaultdict(set)
        self._strength_index = defaultdict(set)
        for i, record in enumerate(self._records):
            for ingredient in record["active_ingredients"]:
                self._ingredient_index[ingredient_key(ingredient["name"])].add(i)
                self._strength_index[strength_key(ingredient["strength"])].add(i)

        self._generic_words = set(STOP_WORDS) | UNITS
        for key in self._ingredient_index:
            self._generic_words.update(key.split())
        for record in self._records:
            self._generic_words.update(name_key(record["dosage_form"]).split())

    # Observations -> evidence

    def _is_distinctive_brand(self, phrase):
        return any(
            word not in self._generic_words and not word.isdigit()
            and len(self._brand_word_index[word]) <= self.max_brand_word_frequency
            for word in phrase.split()
        )

    def _is_brand_hint(self, phrase):
        return (not self._is_distinctive_brand(phrase)
                and any(word not in self._generic_words and not word.isdigit()
                        and self._brand_word_labelers.get(word, 0) <= self.max_brand_hint_labelers
                        for word in phrase.split())
                and bool(self.catalog.find_by_brand(phrase)))

    @staticmethod
    def _strengths(text):
        """(value, corrected_from) for each strength in upper-case text."""
        found = [(strength_key(f"{number} {unit}"), None) for number, unit in STRENGTH_RE.findall(text)]
        for match in CONFUSED_STRENGTH_RE.finditer(text):
            number, unit = match.groups()
            if "O" in number:
                found.append((strength_key(f"{number.replace('O', '0')} {unit}"), match.group(0)))
        return found

    def _phrases(self, words, accept):
        """Longest accepted word n-grams of one line, without overlaps."""
        found = []
        for size in range(min(MAX_PHRASE_WORDS, len(words)), 0, -1):
            for start in range(len(words) - size + 1):
                span = set(range(start, start + size))
                if any(span & taken for _, taken in found):
                    continue
                phrase = " ".join(words[start:start + size])
                if accept(phrase):
                    found.append((phrase, span))
        return [phrase for phrase, _ in found]

    def extract_evidence(self, observations):
        evidence = []
        for obs in observations:
            text = obs.text.upper()
            words = name_key(obs.text).split()

            brands = self._phrases(words, lambda p: self._is_distinctive_brand(p) and self.catalog.find_by_brand(p))
            evidence += [Evidence("brand", p, (obs,)) for p in brands]
            evidence += [Evidence("brand_hint", p, (obs,)) for p in self._phrases(words, self._is_brand_hint)]

            ingredients = self._phrases(words, lambda p: ingredient_key(p) in self._ingredient_index)
            evidence += [Evidence("ingredient", p, (obs,)) for p in ingredients]

            for value, corrected_from in self._strengths(text):
                evidence.append(Evidence("strength", value, (obs,), corrected_from))
            for match in NDC_RE.findall(text):
                evidence.append(Evidence("ndc", re.sub(r"^NDC[\s:#]*", "", match), (obs,)))

        # A number and its unit split across two OCR results: "200" + "mg"
        for first, second in zip(observations, observations[1:]):
            unit = re.match(r"(MG|MCG|G|ML)\b", second.text.strip().upper())
            if not unit:
                continue
            number = re.search(r"(\d+(?:\.\d+)?)$", first.text.strip())
            if number:
                evidence.append(Evidence("strength", strength_key(f"{number.group(1)} {unit.group(1)}"), (first, second)))
                continue
            confused = re.search(r"(?<![A-Z0-9.])(\d[0-9O]*(?:\.[0-9O]+)?)$", first.text.strip().upper())
            if confused and "O" in confused.group(1):
                value = strength_key(f"{confused.group(1).replace('O', '0')} {unit.group(1)}")
                evidence.append(Evidence("strength", value, (first, second), f"{confused.group(1)} {unit.group(1)}"))

        # Store brands print "Compare to Advil(R) active ingredient". A brand
        # named in that text is not the product's brand, and OCR often splits
        # the badge into several results, so once "COMPARE" is seen anywhere no
        # brand phrase is trusted as the product's brand.
        if any("COMPARE" in name_key(o.text).split() for o in observations):
            evidence = [Evidence("compared_brand", e.value, e.observations) if e.kind == "brand" else e
                        for e in evidence]
        return evidence

    # Evidence -> catalog record positions

    def _records_for(self, item):
        if item.kind == "brand":
            sets = [self._brand_word_index.get(word, set()) for word in item.value.split()]
            return set.intersection(*sets) if sets else set()
        if item.kind == "ingredient":
            matches = set()
            observed = ingredient_key(item.value)
            for key, positions in self._ingredient_index.items():
                if _compatible(observed, key):
                    matches |= positions
            return matches
        if item.kind == "ndc":
            return {self._position[r["product_id"]] for r, _ in self.catalog.find_by_package_ndc(item.value)}
        if item.kind == "strength":
            return set(self._strength_index.get(item.value, set()))
        return set()

    def _narrow(self, evidence, kinds):
        """Intersect the records each evidence item allows, in order.

        Brand, ingredient and NDC evidence propose candidates; strength only
        narrows. An item that would leave nothing is recorded as a conflict
        and skipped, so the candidates from the other evidence stay visible.
        """
        conflicts = []
        positions = None
        for kind in kinds:
            for item in (e for e in evidence if e.kind == kind):
                matched = self._records_for(item)
                if not matched:
                    conflicts.append(Conflict(item, f"{item.kind} {item.value!r} matches no catalog record"))
                elif positions is None:
                    if kind != "strength":
                        positions = matched
                elif positions & matched:
                    positions &= matched
                else:
                    conflicts.append(Conflict(item, f"{item.kind} {item.value!r} is not consistent with the other evidence"))
        return positions or set(), conflicts

    # Matching

    def match(self, ocr_results):
        """ocr_results: TextRecognizer.read() output, [{"text", "confidence", ...}]."""
        observations = [Observation(i, str(r.get("text", "")), float(r.get("confidence", 0.0)))
                        for i, r in enumerate(ocr_results or [])]
        usable = [o for o in observations if o.confidence >= self.min_confidence]
        ignored = [o for o in observations if o.confidence < self.min_confidence]
        evidence = self.extract_evidence(usable)

        # Formulation candidates come from the OCR text alone, so a (possibly
        # misread) NDC can't be what breaks a formulation ambiguity. The NDC
        # is applied separately for package identification.
        text_positions, text_conflicts = self._narrow(evidence, ("brand", "ingredient", "strength"))
        all_positions, all_conflicts = self._narrow(evidence, ("brand", "ingredient", "strength", "ndc"))

        if not (text_positions or all_positions):
            no_evidence = not any(e.kind not in ("strength", "brand_hint", "compared_brand") for e in evidence)
            return MatchResult(Status.NO_MATCH, Level.NONE, [], evidence, all_conflicts, 0.0,
                               "no product evidence matched the catalog" if no_evidence
                               else "product evidence matched no catalog record", ignored=ignored)

        package_result = self._verify_package(self._group(sorted(all_positions), evidence),
                                              evidence, all_conflicts, ignored)
        if package_result:
            return package_result

        positions = text_positions or all_positions
        candidates = self._group(sorted(positions), evidence)

        # Would the candidates differ without OCR-corrected evidence? Then the
        # decision depends on a correction and can't be VERIFIED.
        decisive_correction = False
        if any(e.corrected_from for e in evidence):
            plain = [e for e in evidence if not e.corrected_from]
            plain_positions, _ = self._narrow(plain, ("brand", "ingredient", "strength"))
            groups = lambda ps: {_formulation_key(self._records[i]) for i in ps}
            decisive_correction = groups(plain_positions or all_positions) != groups(positions)
        return self._decide_formulation(candidates, evidence, all_conflicts, ignored, decisive_correction)

    def _group(self, positions, evidence):
        groups = defaultdict(list)
        for i in positions:
            groups[_formulation_key(self._records[i])].append(self._records[i])

        observed_ingredients = [ingredient_key(e.value) for e in evidence if e.kind == "ingredient"]
        observed_strengths = {e.value for e in evidence if e.kind == "strength"}
        observed_words = {w for e in evidence if e.kind == "brand" for w in e.value.split()}
        hints = [set(e.value.split()) for e in evidence if e.kind == "brand_hint"]

        candidates = []
        for records in groups.values():
            ingredients = tuple(sorted({(i["name"], i["strength"]) for i in records[0]["active_ingredients"]}))
            missing_brand_words = {}
            for record in records:
                words = [w for w in name_key(record["brand_name"]).split() if w not in observed_words]
                if words:
                    missing_brand_words[record["brand_name"]] = words
            candidates.append(FormulationCandidate(
                ingredients=ingredients,
                records=records,
                missing_ingredients=[n for n, _ in ingredients
                                     if not any(_compatible(o, ingredient_key(n)) for o in observed_ingredients)],
                base_only_ingredients=[n for n, _ in ingredients
                                       if ingredient_key(n) not in observed_ingredients
                                       and any(_compatible(o, ingredient_key(n)) for o in observed_ingredients)],
                missing_strengths=[s for _, s in ingredients if strength_key(s) not in observed_strengths],
                missing_brand_words=missing_brand_words,
                brand_hint_records=sum(1 for r in records
                                       if any(h <= set(name_key(r["brand_name"]).split()) for h in hints)),
            ))
        # Order only: candidates matching a brand hint first, then by size
        candidates.sort(key=lambda c: (c.brand_hint_records == 0, -len(c.records), c.ingredients))
        return candidates

    def _strong(self, items):
        return all(e.confidence >= self.verify_confidence for e in items)

    def _verify_package(self, candidates, evidence, conflicts, ignored):
        ndc_items = [e for e in evidence if e.kind == "ndc"]
        if not ndc_items or conflicts or not self._strong(ndc_items):
            return None
        packages = {(r["product_id"], p["package_ndc"]): (r, p)
                    for e in ndc_items for r, p in self.catalog.find_by_package_ndc(e.value)}
        if len(packages) != 1 or len(candidates) != 1 or len(candidates[0].records) != 1:
            return None
        record, package = next(iter(packages.values()))

        # Support must name this exact product: its full brand, or all of its
        # active ingredients. Brand-family containment is not enough, because a
        # one-digit misread turns 0573-0150-20 (Advil) into 0573-0190-20
        # (Advil Multi-Symptom Cold and Flu).
        brand_support = [e for e in evidence if e.kind == "brand" and e.value == name_key(record["brand_name"])]
        ingredient_support = [e for e in evidence if e.kind == "ingredient"]
        candidate = candidates[0]
        all_ingredients = (record["active_ingredients"] and not candidate.missing_ingredients
                           and not candidate.base_only_ingredients)
        support = brand_support or (ingredient_support if all_ingredients else [])
        if not support or not self._strong(support):
            return None

        used = ndc_items + support
        return MatchResult(
            Status.VERIFIED, Level.PACKAGE, candidates, evidence, conflicts,
            min(e.confidence for e in used),
            f"package NDC {package['package_ndc']} resolves to one package, supported by observed "
            + ", ".join(sorted({e.value for e in support})),
            package={"product_id": record["product_id"], "package_ndc": package["package_ndc"],
                     "description": package["description"]},
            ignored=ignored,
        )

    def _decide_formulation(self, candidates, evidence, conflicts, ignored, decisive_correction=False):
        brands = [e for e in evidence if e.kind == "brand"]
        used = [e for e in evidence if e.kind != "ndc"]
        confidence = min((e.confidence for e in used), default=min(e.confidence for e in evidence))

        blockers = []
        if conflicts:
            blockers.append("conflicting evidence: " + "; ".join(c.reason for c in conflicts))
        if len(candidates) != 1:
            blockers.append(f"{len(candidates)} formulations remain")
        if not brands:
            blockers.append("no distinctive brand observed")
        if decisive_correction:
            blockers.append("candidates depend on OCR-corrected text")

        if len(candidates) == 1:
            candidate = candidates[0]
            if not candidate.ingredients:
                blockers.append("catalog lists no active ingredients")
            if candidate.missing_ingredients:
                blockers.append("active ingredients not observed: " + ", ".join(candidate.missing_ingredients))
            if candidate.base_only_ingredients:
                blockers.append("active ingredients observed only by base name: " + ", ".join(candidate.base_only_ingredients))
            if candidate.missing_strengths:
                blockers.append("strengths not observed: " + ", ".join(candidate.missing_strengths))
            plain_strengths = {e.value for e in evidence if e.kind == "strength" and not e.corrected_from}
            corrected_only = [s for _, s in candidate.ingredients
                              if s not in candidate.missing_strengths and strength_key(s) not in plain_strengths]
            if corrected_only:
                blockers.append("strengths read only through OCR correction: " + ", ".join(corrected_only))
            required = brands + [
                e for e in evidence
                if (e.kind == "ingredient" and ingredient_key(e.value) in {ingredient_key(n) for n, _ in candidate.ingredients})
                or (e.kind == "strength" and not e.corrected_from
                    and e.value in {strength_key(s) for _, s in candidate.ingredients})
            ]
            if required and not self._strong(required):
                blockers.append(f"evidence below verify confidence {self.verify_confidence}")
            if not blockers:
                return MatchResult(
                    Status.VERIFIED, Level.FORMULATION, candidates, evidence, conflicts,
                    min(e.confidence for e in required),
                    "one formulation remains; brand, every active ingredient (with salt) and every strength observed",
                    ignored=ignored,
                )

        compared = sorted({e.value for e in evidence if e.kind == "compared_brand"})
        if compared:
            blockers.append(f"{', '.join(compared)} appears in store-brand comparison text (not used as the brand)")
        hints = sorted({e.value for e in evidence if e.kind == "brand_hint"})
        if hints:
            blockers.append(f"brand hint {', '.join(hints)} observed (not used for verification)")
        return MatchResult(Status.UNCERTAIN, Level.NONE, candidates, evidence, conflicts,
                           confidence, "; ".join(blockers), ignored=ignored)
