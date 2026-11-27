# Phase 2: Evaluation baseline

Branch `feature/evaluation-baseline`, branched from the Phase 1 commit `dffc40d`.
In review; not yet committed. Production code (`main.py`, OCR, YOLO, matcher,
barcode lookup, identification, TTS) is unchanged from Phase 1.

## 1. Objective

Before changing the pipeline (Phases 3–8: target selection through multi-frame
aggregation and final integration), we need to be
able to measure it reproducibly and honestly. That means separate measurements
of each stage (detection, OCR, matching), results that only count annotations
checked by people, and runs that someone else can repeat exactly.

## 2. Previous implementation

- `vision/products/evaluate.py` scored one labeled image (`evaluation/labels.json`,
  author unrecorded) and printed counts. It had no detection metrics, no notion of
  whether a label had been checked, no negative (non-medicine) examples, no saved
  OCR, and no record of model or data versions.
- The research branch had a DailyMed pilot and a latency benchmark, but they were
  not on `main`, and the benchmark required the Flask web app and Qwen.
- Its results files were never committed, and the one "confirmed" photo annotation
  (`medicine-v1`) had inconsistent fields.
- The photos in `VA-photos/` had only model predictions (`drafts.json`), not annotations.

## 3. Changes

| What | Why |
|---|---|
| `evaluate.py` extended (not a new framework) with review-level gating, detection and target-selection scoring, three OCR inputs, negatives, product-level statistics, saved raw results and replay | One evaluator for every dataset |
| DailyMed pilot tooling restored; its duplicate `classify` now calls `evaluate.classify` | Reuse existing work; one scoring definition |
| `research/benchmark.py` restored and trimmed to local components (decode, YOLO, crop, OCR, matcher, barcode decode and lookup), with percentile gates | Latency without Flask or Qwen |
| `research/annotation_review.py`: independent two-person annotation and boxes, then `reconcile` | The only path from UNREVIEWED to a scored level |
| `research/datasets/photos-v1/labels.json`: the 7 existing photos, metadata only, all UNREVIEWED | Ready for annotation without invented truth |
| `.gitignore`: photos, credentials, results, previews, drafts | Keep private data and generated outputs out of Git |

## 4. Technical approach

**Review levels.** UNREVIEWED → descriptive only (statuses, candidate counts,
detections, timings; nothing scored). SINGLE_REVIEWED (one person) → exploratory.
REVIEWED (two different people who agree) → primary. Levels and groups
(`photo` vs official `artwork`) are reported separately and never pooled.
`load_labels` refuses software names ("assistant", "Claude", "yolo", …) as reviewers.

**Detection.** Intersection over union of two boxes A and B:
`IoU = area(A ∩ B) / area(A ∪ B)`. An annotated bottle counts as found when a YOLO
`bottle` box overlaps it with IoU ≥ 0.5. Matching is one-to-one: pairs are sorted
by IoU and taken greedily, so one detection can't satisfy two bottles. Target
selection asks which object `main.py`'s own `select_best_detection` (the most
confident bottle) would read: the medicine, another object, or nothing.

**OCR inputs.**
- `full`: the whole image.
- `truth_crop`: the reviewed medicine box, or the artwork's front panel. This isolates OCR from detection.
- `yolo_crop`: the YOLO-selected box. If YOLO finds no bottle, the outcome is NO_REGION (counted, not skipped).
- `oracle`: the reviewed front-panel text fed directly to the matcher (perfect OCR).

**Matcher metrics** (medicine images):
- candidate recall = images whose true formulation is among the candidates / images
- verification coverage = VERIFIED / images
- verification precision = correct VERIFIED / (correct + false VERIFIED)

On a non-medicine image any VERIFIED is false, and NO_MATCH is a correct rejection.

**Physical products.** Several photos of one bottle are not independent, so every
metric is given per image and per physical product. A product's outcome takes the
worst safety result first: any false VERIFIED view makes it FALSE_VERIFIED;
otherwise any correct VERIFIED view counts. Other product successes mean "in at
least one view". 95% intervals are computed over products:

- Wilson interval for product rates:
  `(p + z²/2n ± z·√(p(1−p)/n + z²/4n²)) / (1 + z²/n)`, with z = 1.96 and n = products.
- Cluster bootstrap for image rates: resample products with replacement 2000 times
  (fixed seed) and take the 2.5th and 97.5th percentiles of the ratio.
- No interval below 5 products.

**Annotation agreement.**
- Each reviewer's answers go to their own file.
- `reconcile` compares them case- and punctuation-insensitively: brand words,
  ingredients and strengths, NDC, UPC, front-panel text, and the negative and
  readable answers.
- Boxes agree at IoU ≥ 0.8, and the stored box is their mean.
- Product IDs may be named differently, but the grouping of photos must be identical.
- Any disagreement leaves the image UNREVIEWED, with the differing fields listed.

**Latency.** Each component runs in its own process: load time, the first (cold)
call and warm calls are reported separately. Median is reported only from n ≥ 3,
p90 from 10 and p95 from 20 warm calls.

**Reproducibility.** Each saved run (`--save`; never overwritten) records:

- image SHA-256, checked against the label
- labels, catalog, YOLO-weights and EasyOCR-model hashes
- raw OCR and YOLO detections
- Git commit, branch and changed-files flag
- package versions and torch thread counts

`--replay` re-scores a saved run with new annotations or a new matcher without
re-running the models. Saved OCR is reused only for the same pixels: the same image
file (SHA-256) and, for `truth_crop`, the same box. Any other row is marked stale,
keeps no OCR, isn't scored, and is excluded from every metric until the evaluation
is run again. Images that now have a box the saved run never read are listed as
needing OCR.

Details: [research/EVALUATION.md](../../research/EVALUATION.md).

## 5. File summary

| File | Status | Purpose and connection |
|---|---|---|
| `vision/products/evaluate.py` | modified (+761/−91) | The evaluator: gating, detection, OCR inputs, outcomes, statistics, provenance, replay. Uses `identification.select_best_detection`, `ProductMatcher`, `ObjectDetector`, `TextRecognizer`; not imported by `main.py` |
| `tests/test_evaluate.py` | modified (+398; 33 tests) | Gating, negatives, IoU and target selection, three inputs, replay (including refusal to reuse OCR from a changed crop or file), product statistics, hashes |
| `research/annotation_review.py` | new | `status`, `annotate`, `preview`, `reconcile` for independent human review |
| `tests/test_annotation_review.py` | new (15 tests) | Agreement, conflicts, box merging, grouping, software reviewers refused, scripted `annotate` |
| `research/dailymed_pilot.py` | restored; `classify` deduplicated, unreviewed notice added | DailyMed `fetch`, `verify`, regression `run`/`replay` |
| `tests/test_dailymed_pilot.py` | restored (18 tests) | Pilot tooling with synthetic files; no network |
| `research/benchmark.py` | restored and trimmed | Per-component latency; no Flask, Qwen or tracking components. Peak memory needs the Unix `resource` module; on Windows it is reported as unavailable |
| `tests/test_benchmark.py` | restored and updated (13 tests) | Statistics, percentile gates, real fast component runs, no web code, runs without the Unix-only `resource` module (Windows) |
| `research/__init__.py` | restored | Makes `research` importable |
| `research/EVALUATION.md` | new | Reference for tools, metrics and reproducibility |
| `research/datasets/photos-v1/labels.json` | new | 7 photos: metadata only, truth empty, UNREVIEWED |
| `research/datasets/photos-v1/README.md` | new | Annotation instructions and checklist |
| `research/datasets/dailymed-pilot/` (README, labels, retrieval record, `.gitignore`, `images/.gitkeep`) | restored | 10 artwork labels (assistant-transcribed, UNREVIEWED); images ignored |
| `.gitignore` | modified (+10) | Ignores `VA-photos/`, `vision/advil_front.jpg`, `web/certs/`, `research/results/`, previews, medicine-v1 images and drafts |
| `docs/README.md`, `docs/phase-01-…`, `docs/phase-02-…`, `README.md` | new / link added | This documentation |

## 6. Testing

```sh
# everything tracked on this branch
.venv/bin/python -m unittest tests.test_annotation_review tests.test_barcode_lookup tests.test_barcode_scan \
  tests.test_barcode tests.test_benchmark tests.test_dailymed_pilot tests.test_evaluate tests.test_identification \
  tests.test_matcher_concentration tests.test_matcher_evidence tests.test_matcher_safety tests.test_ndc \
  tests.test_ocr_corrections tests.test_product_catalog tests.test_product_matcher tests.test_spatial
.venv/bin/python -m research.dailymed_pilot verify
```

Results (2026-11-26):

- **Unit tests:** 226 of 226 pass, with nothing skipped (local catalog present; rerun 2026-11-27).
- **DailyMed verify:** "10 labels, 0 problems".
- **Untested on screen:** the OpenCV box-drawing window is covered only with scripted answers and boxes.
- **End-to-end run:** real YOLO and EasyOCR on the 7 photos (`--inputs full,truth_crop,yolo_crop --save`) completed. Its saved file recorded every hash, version and Git field. Replaying it gave identical statuses.

## 7. Experimental findings

There are **no accuracy results yet**: every annotation is UNREVIEWED. The real run
on the 7 photos is descriptive only:

| Input | VERIFIED | UNCERTAIN | NO_MATCH | NO_REGION |
|---|---|---|---|---|
| full | 0 | 3 | 4 | 0 |
| yolo_crop | 0 | 1 | 3 | 3 |

YOLO found no bottle in 3 of the 7 photos, including the close-up
`vision/advil_front.jpg`. Whether the bottles it did find were the medicine can't be
scored until boxes are reviewed. Other unreviewed observations from the inventory,
useful for planning but not results:

- The 7 photos show at most two medicine bottles. Four views (IMG_9766–9769) appear to show one 100-count Equate ibuprofen bottle. `advil_front.jpg` appears to show a different, 500-count bottle.
- A water bottle appears in all six iPhone photos.
- zxing decodes UPC `681131699372` from IMG_9767.

Latency has not been re-measured on this branch. Older latency figures on the
research branch measured the since-removed web endpoint, not this system, and are
not used here.

## 8. Limitations

- **No accuracy claim.** Nothing is reviewed yet. At most 2 medicine products and 1 non-medicine object exist, which is below the 5 products needed for an interval.
- **Not webcam input.** The photos are 24 MP iPhone stills, not webcam frames like `main.py` uses, and there are no frame sequences for multi-frame work.
- **Optimistic product counting.** "Any view" success at product level answers "could some view identify it"; false VERIFIED always dominates.
- **Independence by convention.** Reviewer files are readable by anyone.
- **Untried paths.** The review tool supports DailyMed labels (`--labels … --no-boxes`) but hasn't been tried on them.
- **Unrecorded conversion.** The HEIC → JPEG conversion tool wasn't recorded; hashes of both files are stored.
- **Location of `evaluate.py`.** It lives in `vision/products/` but is evaluation-only.

## 9. Next phase

**Phase 3: intelligent target selection.** Phase 2 found that `main.py` reads the
most confident YOLO "bottle", which may not be the medicine. In the real run YOLO
found two bottles in each of two photos. Whether it picked the medicine can't be
scored until boxes are reviewed, and target selection is now a measured metric. Phase 3 will improve
that choice and measure it on reviewed boxes. The full roadmap is in [docs/README.md](../README.md).

**In parallel: first reviewed baseline and data collection.**

1. You and your teammate annotate the 7 photos independently ([steps](../../research/datasets/photos-v1/README.md)), then run `reconcile`.
2. Run the evaluation again to read the new boxes (`truth_crop`). `--replay` re-scores only
   rows whose pixels are unchanged, and lists the rows that need a new run.
3. Measure latency with `research/benchmark.py --trials 20` on an idle machine.
4. Collect the minimum new data: about 8 medicines (single-ingredient, combination, a liquid, a same-ingredient pair), 2 non-medicine bottles and 3 empty scenes. Take 4 views and a short **webcam** clip per medicine, plus package NDC and UPC read by two people.

