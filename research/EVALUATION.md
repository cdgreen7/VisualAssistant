# Evaluation baseline (Phase 2)

Measurement infrastructure for the YOLO + EasyOCR + ProductMatcher pipeline in
`main.py`. Nothing here changes production behaviour: `main.py`, OCR, YOLO, the
matcher, barcode lookup and TTS are untouched. No Flask, Qwen, web or JavaScript
dependency is used. Phase summary and context:
[docs/phase-02-evaluation-baseline](../docs/phase-02-evaluation-baseline/README.md).

## Branch dependency

`feature/evaluation-baseline` is branched from commit `dffc40d` ("Phase-1 of
testing") on `feature/matcher-safety-improvements`, the matcher pull request,
which is not yet merged into `main`. Merge (or review) the matcher PR first. This
branch's diff against `main` otherwise also shows the matcher commit. The
evaluation code runs with either matcher, but the DailyMed regression numbers
and the matcher tests assume the Phase 1 matcher.

## Tools (one per job, no duplicates)

| Tool | Job |
|---|---|
| `vision/products/evaluate.py` | Accuracy: detection, OCR and matcher metrics, review-level gating, saved raw OCR, replay |
| `research/annotation_review.py` | Independent human annotation and boxes; `reconcile` is the only thing that sets review levels |
| `research/dailymed_pilot.py` | DailyMed artwork: `fetch`, `verify` against the catalog, matcher regression `run`/`replay` |
| `research/benchmark.py` | Latency: per-component load, cold and warm times in separate processes, median/p95 |

`dailymed_pilot` uses evaluate.py's scoring and `classify` (one definition). Its
labels are UNREVIEWED (transcribed by an assistant), so its run/replay counts are
a regression comparison of matcher versions on fixed OCR, not accuracy. Review-
gated metrics for the same labels come from
`python -m vision.products.evaluate --labels research/datasets/dailymed-pilot/labels.json --inputs full,truth_crop`.

## Groups and review levels

- Groups (`label["group"]`): `photo` (real camera photographs, `photos-v1`) and
  `artwork` (official label artwork, `dailymed-pilot`). Never pooled.
- Levels: UNREVIEWED (descriptive only), SINGLE_REVIEWED (exploratory),
  REVIEWED (primary). Reported separately, never pooled, never promoted
  automatically. A reviewed label must name people (software names such as
  "assistant" or "Claude" are refused), REVIEWED needs two different people, and
  a reviewed label needs a physical product ID and, for a medicine, its complete
  formulation. `load_labels` refuses files that break these rules.

All current annotations are UNREVIEWED: the 7 photographs (empty truth), the 10
DailyMed labels (assistant transcription), and `evaluation/labels.json`
(author unrecorded).

## Metrics

For each group, level, input (`oracle` = perfect OCR from the reviewed front-panel
text; `full`; `truth_crop` = reviewed medicine box or artwork front panel;
`yolo_crop` = the box main.py's `select_best_detection` picks) and preprocessing variant:

| Metric | Definition |
|---|---|
| YOLO bottle recall / medicine recall | annotated boxes matched one-to-one by a YOLO `bottle` box with IoU >= 0.5 |
| Target selection | MEDICINE_SELECTED, WRONG_OBJECT_SELECTED, NO_BOTTLE_DETECTED (and, on non-medicine photos, NON_MEDICINE_SELECTED / NOTHING_SELECTED); also on the subset with several bottles. Since Phase 3 also for the original baseline and every method in `vision/target_selection.py` on the same detections (`selection_choices`, every image; `selection_by_method`, reviewed images) |
| Intended target | Separate from medicine boxes: whether each method read the object the person meant (`intended_target`, recorded by reviewers as a box, `none` or unknown). Computed only for images whose intent was annotated; never inferred |
| OCR | brand words, every ingredient, every strength (plain; with the matcher's O->0 correction) found in the OCR text |
| Outcomes | CORRECT_VERIFIED, FALSE_VERIFIED, VERIFIED_PACKAGE_UNSCORED, UNCERTAIN, NO_MATCH, NO_REGION (yolo_crop found no bottle) |
| Candidate recall | true formulation among the candidates |
| Verification coverage | VERIFIED results / medicine images |
| Verification precision | correct / (correct + false) VERIFIED |
| Non-medicine | false VERIFIED (any VERIFIED) and correct rejection (NO_MATCH) |

Every metric is given per image and per physical product. Intervals (95%) are over
products: Wilson for product-level rates, a seeded cluster bootstrap for
image-level rates, and none with fewer than 5 products. Product outcome: any
FALSE_VERIFIED view, else any CORRECT_VERIFIED view, then VERIFIED_PACKAGE_UNSCORED,
UNCERTAIN, NO_MATCH, NO_REGION; other product-level successes mean "in at least one view".

Latency: `research/benchmark.py --images DIR --trials 20` (median from n >= 3,
p90 from 10, p95 from 20 warm calls; load and cold call separate; peak memory
needs the Unix `resource` module and is reported as unavailable on Windows). evaluate.py
records single-pass `yolo_ms`, `ocr_ms` and `match_ms` per row for context only.

## Reproducibility

A saved evaluation (`--save`, `research/results/evaluation-<time>.json`, never
overwritten) records: each image's SHA-256 (checked against the label's), the
labels file hash, the catalog file, hash and export date, the YOLO weights hash,
the EasyOCR model file hashes, raw OCR text and confidences, YOLO detections,
Git commit, branch and whether tracked files had changes, Python, platform,
torch threads and versions of torch, torchvision, ultralytics, easyocr,
opencv-python, numpy, zxing-cpp and Pillow. `--replay` re-scores a saved run
with the current matcher and annotations without re-running OCR or YOLO. OCR is
reused only for the same image file (SHA-256) and, for `truth_crop`, the same box;
other rows are marked stale, kept out of every metric and listed for a new run.

## Kept out of Git

`VA-photos/`, `vision/advil_front.jpg`, DailyMed images, `research/results/`,
review previews, `research/datasets/medicine-v1/` images and drafts, and
`web/certs/` are ignored. Labels and reviewer annotation files contain text,
boxes and hashes only.
