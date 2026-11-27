# Phase 1: Matcher safety improvements

Commit `dffc40d` ("Phase-1 of testing", 2026-11-26) on branch
`feature/matcher-safety-improvements`, based on `main` at `d6020fa`.
Pushed; no pull request yet; not merged into `main`.

## 1. Objective

`ProductMatcher` turns OCR text from a medicine package into one of three answers:
**VERIFIED** (safe to name a formulation or package), **UNCERTAIN** (some
evidence, not enough) or **NO_MATCH**. For a blind user, a wrong VERIFIED is the
worst failure: the system would confidently name the wrong medicine. The matcher
on `main` had a crash bug and several ways to produce a wrong VERIFIED. The
improved matcher already existed on the research branch but had been removed
from `main` by a revert. This phase restores it, without the web app or Qwen.

## 2. Previous implementation

`main.py` → press `i` → `vision/identification.py` crops the most confident
YOLO "bottle" → EasyOCR reads it → `ProductMatcher.match()`:

1. **Evidence extraction.** OCR results under confidence 0.3 are ignored. Phrases
   of up to 6 words are compared with catalog brand names, ingredient names,
   strengths (regex such as `200 mg`) and NDC codes.
2. **Narrowing.** Each evidence item maps to the set of catalog records that
   contain it, and the candidate set is the intersection. An item that would
   empty the set becomes a *conflict* instead of being applied.
3. **Decision.** VERIFIED/FORMULATION needs exactly one remaining formulation, a
   distinctive brand, every active ingredient (with its salt), every strength,
   no conflicts and evidence confidence ≥ 0.6. VERIFIED/PACKAGE needs a printed
   package NDC that resolves to one package and is supported by a brand or by
   all ingredients. Everything else is UNCERTAIN.

Problems found (each reproduced by a new test that fails on `main`):

- **Crash.** Some catalog ingredients have no strength (`None`), and
  `", ".join(...)` raised `TypeError`. `main.py` caught it and printed only
  "Identification failed".
- **Liquids VERIFIED as tablets.** The strength index was keyed by the full
  catalog strength. OCR usually reads "15 mg" without "/5 mL", so the syrup
  (`15 mg/5mL`) was excluded and a tablet (`15 mg/1`) was VERIFIED. There were
  six real catalog cases, e.g. Pepto-Bismol, DayQuil and two cough syrups.
- **Non-brand words treated as brands.** Packaging words (ANTIHISTAMINE, OUTDOOR
  ALLERGIES), English words (HAD) and company names (BAYER, KROGER) counted as
  brand evidence, creating false narrowing or false conflicts.
- **Inactive or claim words treated as ingredients.** "No water needed" or
  "sodium 20 mg" produced WATER or SODIUM ingredient evidence that conflicted
  with the real active ingredient.
- **Salts.** "HBr" wasn't expanded to HYDROBROMIDE. A base name without its salt
  (CETIRIZINE) wasn't recognized at all.

## 3. Changes

The only production change is `vision/products/matcher.py`, restored unchanged
from the research branch (its commit `022aede` and earlier). The public API
(`ProductMatcher`, `MatchResult`, `Status`, `Level`, `ingredient_key`) is the same,
so `main.py`, `identification.py`, `evaluate.py` and `matcher_demo.py` need no change.

| Change | Why |
|---|---|
| A missing catalog strength is shown as "(not listed in catalog)" | Fixes the crash |
| Strength index keyed by amount only (`15 mg/5mL` → `15mg`); verification still needs the full strength | A liquid stays a candidate but can't be VERIFIED from "15 mg" alone |
| Word lists that can never be a brand's distinctive word: packaging vocabulary, function words, salt abbreviations, frequent generic-name words, labeler (company) name words | Removes false brand evidence |
| `HBR` → HYDROBROMIDE; `HCI` (OCR's misreading of "HCl") is not expanded | Correct salts without trusting a likely misread |
| Salt base names (CETIRIZINE) become *base-only* evidence | Keeps salt forms as candidates; base-only evidence can never verify |
| Common substances (WATER, SODIUM, …) next to a real ingredient become `ingredient_hint` | Reported, but never narrows, conflicts or verifies |
| **Superset guard**: before VERIFIED/FORMULATION, look for a related product with the same ingredients plus more | BAYER + naproxen must not exclude Aleve PM (naproxen + diphenhydramine) |

Every new rule can only keep a candidate or block a VERIFIED. None can create one.

## 4. Technical approach

**Distinctive brand word.** A word *w* in an observed brand phrase is distinctive if:

- *w* is not generic, not a number and not a word of any labeler (company) name, and
- *w* appears in at most 100 FDA brand names (`MAX_BRAND_WORD_FREQUENCY`).

A word is *generic* if it is a stop word, a unit, packaging vocabulary, a word of
an ingredient or dosage-form name, or appears in the FDA `generic_name` field of
≥ 10 records from ≥ 3 different labelers (`MIN_GENERIC_NAME_COUNT`,
`MIN_GENERIC_NAME_LABELERS`). The labeler condition keeps a brand that one company
also puts in its generic names (DIAL: 11 records, 1 labeler).

**Strength matching.** `strength_key` normalizes `"200 mg/1"` and `"200 mg"` to
`200mg`. The candidate index uses only the amount (`key.split("/")[0]`), so
`15mg` finds both `15 mg/1` and `15 mg/5mL`. Verification still compares the full
key, so the syrup remains UNCERTAIN until its concentration is read.

**Superset guard.** Let *F* be the ingredient-and-strength set of the one remaining
candidate. Collect every record whose brand name shares a distinctive word with the
main part of the candidate's brand name (before any comma). If any such record has
a formulation *F′* with *F ⊂ F′* (strictly more ingredients), and its brand name
contains none of the observed distinctive brand words, the result stays UNCERTAIN.
This check is conservative by design, not a search of every combination product.

## 5. File summary

| File | Status | Purpose and connection |
|---|---|---|
| `vision/products/matcher.py` | modified (+161/−9) | `ProductMatcher`, used by `main.py` through `identification.py`, and by `evaluate.py` |
| `tests/test_matcher_safety.py` | new (16 tests) | Crash fix, company and packaging words, superset guard; synthetic and real-catalog cases |
| `tests/test_matcher_evidence.py` | new (19 tests) | Salts (HBr, HCI, base names), common-substance hints, real DailyMed front-panel cases |
| `tests/test_matcher_concentration.py` | new (8 tests) | Liquids vs tablets: the six real false-VERIFIED cases plus three formatting cases |
| `tests/test_ocr_corrections.py` | modified (1 assertion) | The Equate photo now keeps 10 candidates, not 7: liquids and FDA formatting variants are no longer wrongly excluded |

## 6. Testing

```sh
.venv/bin/python -m unittest tests.test_matcher_safety tests.test_matcher_evidence \
    tests.test_matcher_concentration tests.test_ocr_corrections
```

Real-catalog tests need `vision/products/data/otc_ndc_2026-10-30.json`
(build it with `python -m vision.products.build_cache`). Without it they are
skipped, not passed.

Results at `dffc40d` (2026-11-26, catalog present, nothing skipped):

- All 13 test modules: **151 of 151 pass**. On `main` it was 108 of 108; the 43 new tests are the three new files.
- The new tests run against `main`'s matcher: **34 failures and 2 errors**. This confirms they detect the old behaviour.

## 7. Experimental findings

The matcher was replayed on **saved** OCR text from the DailyMed pilot: the same
10 official label images, the same pinned catalog (SHA-256 `a895d419…`) and the
same labels file (`d5e2b723…`). This compares matcher versions on identical inputs.
It is **not an accuracy result**: the labels were transcribed by an assistant and
are unreviewed, the images are label artwork rather than camera photos, and n = 10.

| Input | Correct VERIFIED (main → Phase 1) | False VERIFIED | Truth among candidates |
|---|---|---|---|
| Perfect OCR (label text) | 5 → 7 | 0 → 0 | 9 → 10 |
| OCR on full artwork | 1 → 2 | 0 → 0 | 7 → 9 |
| OCR on front panel | 2 → 3 | 0 → 0 | 9 → 9 |

- **Gained:** Benadryl (all three inputs), Zyrtec (perfect OCR, front panel) and Mucinex DM (perfect OCR).
- **Lost:** Aleve under perfect OCR and on the front panel. It is now UNCERTAIN because of the superset guard; this conservative cost is intended.
- **Reproducible:** `main`'s matcher reproduced the original 19 Nov run exactly, and the Phase 1 matcher reproduced the archived 19 Nov replay row for row.

## 8. Limitations

- No accuracy claim is possible yet: the pilot labels are unreviewed, and there are no reviewed real photos (see Phase 2).
- Real OCR still rarely verifies: 7 of 10 full artworks stay UNCERTAIN, and Excedrin gets NO_MATCH. These are mostly OCR problems (misread strengths and NDCs, small print).
- Known weak spots:
  - MUCINEX and BAYER count only as brand hints.
  - Motrin IB never verifies from real OCR.
- Thresholds (0.3, 0.6, 100, 10/3) are prototype values that still need calibration on reviewed photos.

## 9. Next phase

[Phase 2](../phase-02-evaluation-baseline/README.md) builds the review-gated
evaluation needed to measure this matcher, and later changes such as multi-frame
identification, on real photos rather than unreviewed artwork.
