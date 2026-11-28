# Phase 3: Intelligent target selection

Branch `feature/intelligent-target-selection`, branched from the Phase 2 commit
`d86243f`. In review; not yet committed. `main.py`'s default behaviour is unchanged:
it still reads the highest-confidence bottle. Every new method is opt-in.

## 1. Objective

When the user presses `i`, `main.py` reads one detected bottle. It has always chosen
the one YOLO is most confident about. Detection certainty is not user intent: a
water bottle in the background can be detected more confidently than the medicine
the user is holding up. The research question:

> Can spatial and geometric information (box size and position) improve which object
> is selected, compared with confidence-only selection?

This phase builds the selector and the measurement for it. It does **not** answer
the question for real scenes, because no reviewed bounding boxes exist yet (§8).

## 2. Previous implementation and its limits

- `ObjectDetector.get_detections()` (`vision/detector.py`) returns one dict per box:
  `{"class_id", "class_name", "confidence", "box": (x1, y1, x2, y2)}`, in integer frame
  pixels. `main.py` creates `ObjectDetector()` with its default threshold 0.5, which is
  passed to YOLO, so lower-confidence boxes never reach selection.
- `identification.select_best_detection()` returns the `"bottle"` with maximum confidence.
  On ties, the first in YOLO's order wins. `identify_detection()` crops that box (clamped
  to the frame), runs EasyOCR on the crop and passes the text to `ProductMatcher`.
- `main.py` reports the box's LEFT/CENTER/RIGHT position (`vision/spatial.py`, frame thirds).

Limitations:

- **Ignores user intent.** Position and size play no part in the choice.
- **Invalid boxes win.** A zero-area or off-frame box with the highest confidence is still
  chosen; the crop is then empty and the user hears "No bottle detected".
- **Measured but not changeable.** Phase 2 scored target selection, for this single method only.

## 3. Method

For each eligible detection i (class `bottle`, valid confidence, box overlapping the frame):

```
S_i = wc·C_i + wa·A_i + wp·P_i          wc, wa, wp ≥ 0,   wc + wa + wp = 1
```

| Term | Definition | Range | Notes |
|---|---|---|---|
| C | YOLO confidence, unchanged | [0, 1] | Non-finite or outside [0, 1] → rejected |
| A | (w·h)/(W·H) of the box clipped to the frame | (0, 1] | Usually small: a bottle covering 10% of the frame gives A = 0.1, so area has little influence unless wa is large |
| P | 1 − √(dx² + dy²)/√2, with dx = (cx − W/2)/(W/2), dy = (cy − H/2)/(H/2) | [0, 1] | 1 at the centre, 0 at a corner, 0.29 at an edge midpoint. Normalized per axis, so the same relative position scores the same at any resolution or aspect ratio |

(cx, cy) is the centre of the clipped box, and W × H is the frame size.

- **Selected detection:** the one with the highest S. Ties (S equal to 12 decimals) go to the higher C, then to the earlier detection.
- **Rejected detections** are listed with their reason (wrong class, invalid confidence, box not four numbers, non-finite or zero area, outside the frame).
- **No changes to detections:** they are never modified, re-scored or re-thresholded.
- **Not a probability:** S is only for ranking. It is not the probability that an object is a medicine and is unrelated to `ProductMatcher`'s VERIFIED decision.

**Two confidence-only selectors, kept apart:**

| Name | What it is |
|---|---|
| `confidence_original` (env value `confidence`) | `identification.select_best_detection`, unchanged. The baseline and `main.py`'s default |
| `confidence_validated` | `vision/target_selection.py` with weights (1, 0, 0). Same choice whenever every box and confidence is valid (checked on 500 random lists), but invalid detections are skipped instead of chosen |

Any difference between these two comes from input validation, not from geometric
scoring. The geometric methods are:

| Method | (wc, wa, wp) |
|---|---|
| `center` | (0, 0, 1) |
| `area` | (0, 1, 0) |
| `weighted` | (0.40, 0.25, 0.35): illustrative starting values, **not fitted or validated**; opt-in |

**Why a heuristic, not a trained model.** No reviewed data exists to train or validate a
model of user intent: we have no reviewed boxes and two physical medicine bottles at most.
A learned ranker would add a dependency and a second model to validate. A three-term
formula is transparent: every choice can be explained from its score breakdown, it runs in
microseconds, and it can be checked against reviewed data later. If reviewed data shows the
heuristic is inadequate, that data is also what a learned approach would need first.

## 4. Changes and files

| File | Status | Purpose and connection |
|---|---|---|
| `vision/target_selection.py` | new (208 lines) | `Weights`, `METHODS`, `BASELINE`, `select_target()` (choice and per-detection breakdown), `make_selector()`, `describe()`. No OCR, barcode, TTS or model imports |
| `vision/identification.py` | modified (+8/−2) | `identify_detection(..., selector=None)`: without a selector it calls `select_best_detection` exactly as before |
| `main.py` | modified (+9/−1) | Opt-in `VA_TARGET_SELECTION=weighted` (or `center`, `area`, `confidence_validated`) `python main.py`: prints the score table and passes the selector. The default (`confidence`) is the original. An unknown name stops before the camera opens |
| `vision/products/evaluate.py` | modified (+158/−14) | `confidence_original` and every method run on the same saved detections. `selection_choices` (which box, every image); `selection_by_method` (medicine box selected) and `intent_by_method` (intended target selected, only where annotated) on reviewed images. `--selection-weights` adds a custom set. Replay rebuilds all of these |
| `research/annotation_review.py` | modified (+61/−2) | Reviewers record the intended target (box number, `none`, or unknown); reconcile compares it by box geometry |
| `research/target_selection_study.py` | new (255 lines) | Synthetic scenarios, weight-sensitivity grid, descriptive comparison on saved real detections, latency through the benchmark |
| `research/benchmark.py` | modified (+28/−1) | Latency components `selection_baseline` and `selection_weighted` (10 fixed synthetic detections) |
| `tests/test_target_selection.py` | new (24 tests) | Scoring, edge cases, ties, resolutions, weights, original vs validated, intent, evaluator, study |
| `tests/test_main_integration.py` | new (8 tests) | The real `main.py` run with mocked camera, YOLO and OCR; the detection threshold |
| `tests/test_annotation_review.py`, `tests/test_benchmark.py` | modified (+48/−1, +13/−1) | Intent annotation; selection latency components |
| `docs/…`, `research/EVALUATION.md`, `research/datasets/photos-v1/README.md` | new / updated | Documentation and phase status |

Not changed: `ProductMatcher`, EasyOCR, the YOLO model and threshold, barcode code, the
threaded speaker, camera capture, and `requirements.txt`.

## 5. Experimental methodology

Three sources, never pooled:

1. **Synthetic scenarios** (10, in `research/target_selection_study.py`). Hand-made boxes
   in a 1920×1080 frame, each with an intended target defined by the scenario. They
   include cases built to make the geometric methods fail. They show behaviour, not accuracy,
   and **must not be used to choose weights**: their answers were written by hand, so fitted
   weights would only reproduce those choices.
2. **Weight sensitivity.**
   - Every (wc, wa, wp) on a 0.05 grid: 231 points.
   - For each point, the synthetic scenarios where the intended target is chosen.
   - The exact decision boundary for the two-bottle case.
3. **Saved real detections, unreviewed.**
   - Sources: the Phase 2 native-resolution run (`evaluation-20261126-124213.json`) and
     the older `drafts.json` predictions (1200×1600, local only).
   - The same detections go to every method; the output is only which box each picks.

**Reviewed evaluation** (`vision/products/evaluate.py`) separates two questions:

- **Medicine selected:** did the method read a box annotated as a medicine (any medicine)?
  This is split into several-bottle and single-bottle images, and records what it read on
  images without a medicine.
- **Intended target selected:** did it read the object the person meant? Reviewers record
  that separately (`intended_target`: a box, `none`, or unknown), because a photo can show
  two medicines, or the user can mean another object. Intent is never inferred from medicine
  boxes; images without it are simply not counted for this metric.

Weights must not be tuned on the reviewed images used to report results.

```sh
.venv/bin/python -m research.target_selection_study --evaluation research/results/evaluation-<time>.json --save
.venv/bin/python -m vision.products.evaluate --labels research/datasets/photos-v1/labels.json \
    --replay research/results/evaluation-<time>.json          # per-method choices; scored once reviewed
```

## 6. Measured results (2026-11-27)

**Synthetic scenarios** (behaviour only; the scenarios were written for this phase, so the
counts say nothing about real scenes):

| Scenario | Intended | confidence_original | confidence_validated | center | area | weighted |
|---|---|---|---|---|---|---|
| single bottle | 0 | ✓ | ✓ | ✓ | ✓ | ✓ |
| confident water bottle at side, medicine centred | 1 | ✗ | ✗ | ✓ | ✗ | ✓ |
| large bottle at side, small medicine centred | 1 | ✗ | ✗ | ✓ | ✗ | ✓ |
| medicine held at the edge, other bottle centred | 0 | ✓ | ✓ | ✗ | ✓ | ✗ |
| medicine close-up, confident background bottle | 1 | ✗ | ✗ | ✓ | ✓ | ✓ |
| medicine small in background, large bottle in front | 1 | ✓ | ✓ | ✗ | ✗ | ✗ |
| three bottles | 1 | ✗ | ✗ | ✓ | ✓ | ✓ |
| confident bottle mostly outside the frame | 1 | ✗ | ✗ | ✓ | ✓ | ✓ |
| zero-area box with highest confidence | 1 | ✗ | ✓ | ✓ | ✓ | ✓ |
| no bottle | none | ✓ | ✓ | ✓ | ✓ | ✓ |
| **total** | | **4/10** | **5/10** | 8/10 | 6/10 | 8/10 |

- **Invalid boxes:** `confidence_validated` scores 5/10 against the original's 4/10 only because of the zero-area-box scenario. It skips the invalid box, while the original chooses it and identification then finds no bottle. This is input validation, **not** geometric scoring; a test checks that it is the only scenario where the two differ.
- **Different failures:** each geometric method fails where its assumption fails.

**Weight sensitivity.**

- **No single best setting:** no weight setting gets all 10 scenarios right.
  - The maximum, 9 of 10, is reached at only 2 of 231 points: (0.70, 0.10, 0.20) and (0.75, 0.00, 0.25).
  - The illustrative (0.40, 0.25, 0.35) gets 8 of 10.
  - Medicine at the edge is right at 13% of grid points, and medicine small in the background at 14%.
- **The weights were left unchanged:** these counts depend entirely on which scenarios were written, so they can't choose the weights. `weighted` stays opt-in.
- **Two-bottle boundary:**
  - **Setup:** equal box sizes, the medicine centred at C = 0.70 and the other bottle at C = 0.75, 620 px to the side.
  - **Score differences:** ΔC = −0.05 and ΔP = +0.457, both medicine minus the other bottle.
  - **Condition:** with wa = 0, the centred bottle is chosen whenever wp/wc > 0.11.
  - **Margin:** at the illustrative weights it wins by 0.14.

**Saved real detections** (unreviewed: which box is the medicine, or was intended, is not known):

| Source | Photo | Bottle confidences | confidence_original | confidence_validated | center | area | weighted |
|---|---|---|---|---|---|---|---|
| Phase 2 run, 4284×5712 | IMG_9766 | #0 0.800, #1 0.759 | #0 | #0 | #0 | #1 | #0 |
| Phase 2 run, 4284×5712 | IMG_9767 | #1 0.731, #2 0.679 | #1 | #1 | #1 | #2 | #1 |
| drafts.json, 1200×1600 | IMG_9766 | #1 0.754, #3 0.701 | #1 | #1 | #3 | #1 | #3 |
| drafts.json, 1200×1600 | IMG_9767 | #1 0.705, #3 0.633 | #1 | #1 | #1 | #3 | #1 |

Only these two photos have two bottle detections. In the Phase 2 run, 2 photos had one bottle and 3 had none.

**Experimental observation, not a general conclusion: confidence order and resolution.**

- **What was seen:** in one photo (IMG_9766), the two bottle boxes (matched across the two runs by position after scaling ×3.57) had their YOLO confidence order reversed:
  - 0.70 vs 0.75 when detected at 1200×1600
  - 0.80 vs 0.76 at 4284×5712
- **What follows:** confidence-only and weighted selection disagree in the 1200×1600 run and agree at full resolution.
- **Why it can't be generalized:**
  - It is one photo pair and two detection runs, with unreviewed boxes.
  - It doesn't show which box is the medicine or what the user intended.
  - The two runs also differ in their decoding pipeline, not only in resolution.
  - `main.py`'s webcam resolution was not tested.
- **What would test it:** a controlled test over several reviewed scenes and resolutions.

**Selection latency** (`research/benchmark.py`, separate process, 10 detections, 200 warm calls, Apple M4; two runs):

| Component | Warm median | p95 | Max |
|---|---|---|---|
| `select_best_detection` (original) | < 0.01 ms | < 0.01 ms | ≤ 0.34 ms |
| `select_target`, weighted, with breakdown | 0.03–0.05 ms | 0.03–0.07 ms | ≤ 0.45 ms |

The benchmark rounds to 0.01 ms. Both are negligible next to YOLO and OCR.

The generated study results are kept locally, not in Git:

- `research/results/target-selection-20261127-141647.json`
- `research/results/target-selection-20261127-144320.json`
- `evaluation-20261126-124213.json`

All three are copied, with SHA-256 checksums, to `../visual-assistant-backups/2026-11-27-phase3-results/`.

## 7. Testing

```sh
.venv/bin/python -m unittest tests.test_target_selection tests.test_main_integration \
    tests.test_identification tests.test_evaluate tests.test_annotation_review tests.test_benchmark
```

The full suite (18 modules: those in [Phase 2](../phase-02-evaluation-baseline/README.md#6-testing)
plus `tests.test_target_selection` and `tests.test_main_integration`) gives **263 of 263 passed**,
nothing skipped: 226 before this phase, 24 selection, 8 `main.py` integration, 4 review-tool
and 1 benchmark test.

**`main.py` without a camera.** `tests/test_main_integration.py` runs the real `main.py` with
a fake camera frame, fake YOLO detections (water bottle 0.75 at the side, medicine 0.70 centred,
a person), a recording OCR stub and the fixture catalog. It presses `i` once, then `q`. It checks:

- **Default:** OCR receives exactly the pixels of the higher-confidence box, nothing new is printed, and the reported position is RIGHT.
- **`VA_TARGET_SELECTION=confidence`:** the same as the default.
- **`VA_TARGET_SELECTION=weighted`:** OCR receives exactly the pixels of the centred box from the clean frame (no drawn overlays), the score table is printed, and the reported position is CENTER.
- **`VA_TARGET_SELECTION=confidence_validated`:** the original's choice on valid boxes.
- **An unknown method:** an error before the camera opens.
- **Threshold:** `main.py` constructs `ObjectDetector()` with its 0.5 default; `detect()` passes `conf=0.5` to YOLO; selection only ever returns one of its input detections.

**Bug found while testing:** a string box such as `"1234"` was read as four numbers. Such boxes are now rejected.

## 8. Limitations and what remains unvalidated

- **No real-world accuracy result.**
  - No boxes or intended targets have been reviewed, so the evaluator records each method's choice but doesn't score it.
  - Nothing here shows that the weighted method selects the medicine, or the intended object, more often than confidence-only.
- **Weights are unvalidated and untuned.** Tuning them needs reviewed images, separate from those used to report results. With at most 2 physical medicine bottles there aren't enough yet, and the synthetic scenarios must not be used.
- **The geometry assumes the user centres the medicine.**
  - Centre weighting fails when the medicine is off-centre or small, as the synthetic failure cases show.
  - Area tends to favour whatever is nearest the camera.
- **The resolution observation in §6 comes from a single photo pair.**
- **Only "bottle" is eligible**, as before; cartons and other packages are not candidates.
- **Selection is per frame, so it can switch** between bottles from frame to frame when scores are close.
- **`main.py` has been tested with mocked frames only,** not with a live camera.

## 9. Next phase: object tracking

Phase 4 will follow the same physical object across frames. Phase 3 provides:

- a deterministic per-frame choice with an inspectable score breakdown
- resolution-independent geometry
- a pluggable `selector` in `identify_detection`
- evaluator records of every method's choice, separated into medicine-box and intended-target outcomes

Tracking can then keep the selected target stable across frames instead of re-choosing every frame, measured with the same metrics.
