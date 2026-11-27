# Visual Assistant development documentation

The Visual Assistant helps blind and low-vision users: a camera frame goes through
YOLO object detection, a left/centre/right position, and, on request, EasyOCR
label reading or barcode decoding. The text is then matched against a local copy
of the FDA's over-the-counter drug catalog (openFDA NDC data). `main.py` is the
entry point. This folder records each development phase: what changed, why, how
it was tested and what is still unproven.

## Roadmap and progress

| Phase | Goal | Status |
|---|---|---|
| [1. Matcher safety](phase-01-matcher-safety/README.md) | Restore the improved, more conservative medicine matcher with regression tests | Done. Commit `dffc40d` on branch `feature/matcher-safety-improvements` (pushed; no pull request yet; not in `main`) |
| [2. Evaluation baseline](phase-02-evaluation-baseline/README.md) | Reproducible, review-gated measurement of detection, OCR and matching before changing the pipeline | Implemented on `feature/evaluation-baseline`; in review, not yet committed |
| 3. Intelligent target selection | Read the medicine package rather than simply the most confident YOLO bottle (Phase 2 scores this as target selection) | Planned |
| 4. Object tracking | Follow the same physical object from frame to frame | Planned |
| 5. Frame quality assessment | Score frames (sharpness, size, exposure) so OCR runs on usable ones | Planned |
| 6. Multi-frame evidence aggregation | Combine matcher evidence from several frames of one tracked package | Planned |
| 7. Barcode and OCR reconciliation | Combine barcode lookup and label matching; disagreement must never become VERIFIED | Planned |
| 8. Final integration and evaluation | Bring the phases into `main.py` with speech output; final review-gated evaluation | Planned |

**In parallel (not a phase):** human annotation of the existing photos with the
Phase 2 review workflow, and collection of new photographs and webcam clips. Each
later phase is measured on reviewed data from this track.

Phases 3–8 are plans, and each gets its own README only once it is implemented.
Restoring reverted research code in any phase (for example earlier tracking or UPC
registry code) needs explicit approval; see the constraints below.

## Architectural constraints (apply to every phase)

The approved system is a local Python application:

| Part | Implementation |
|---|---|
| Camera and display | OpenCV (`vision/camera.py`, `main.py`) |
| Object detection | Ultralytics YOLO, `yolo26n.pt` (`vision/detector.py`) |
| Text reading | EasyOCR (`vision/ocr.py`) |
| Product identification | `ProductMatcher` on the local FDA NDC catalog from openFDA (`vision/products/`) |
| Barcodes | zxing-cpp decoding and FDA catalog lookup (`vision/barcode.py`, `vision/products/barcode_lookup.py`) |
| Speech | the threaded TTS speaker (`vision/audio/speaker.py`: macOS `say`, otherwise pyttsx3) |

Every phase must keep to these rules:

1. **No Qwen, Ollama or other LLM or vision-language model** for identification or
   anything else: no imports, API calls, model loading or runtime dependency.
2. **No Flask, web server, browser interface or JavaScript.**
3. **No restoring reverted experimental code without explicit approval.** The
   integration merged in pull request #2 was reverted in #3. Approved, reviewed
   pieces are restored file by file (`git checkout <branch> -- <path>`), never by
   merging or reverting the revert. Phase 1 (matcher) and Phase 2 (DailyMed pilot,
   trimmed benchmark) are the approved restorations so far.
4. **Production behaviour changes only in a phase that says so**, with tests. Don't
   modify a teammate's implementation (e.g. the speaker, `README.md` content)
   without their agreement; add, don't rewrite.
5. **No new entries in `requirements.txt`** unless a phase needs them and the team
   approves. Evaluation code uses only what the pipeline already installs.
6. **Network use is opt-in and research-only.** Running `main.py`, the tests and the
   evaluator needs no network. Downloads (catalog build, DailyMed `fetch`) are
   explicit commands.
7. **Nothing is shown as VERIFIED unless the matcher's rules are met.** Unreviewed or
   AI-generated annotations are never treated as ground truth (see Phase 2 review levels).
8. **Private data stays out of Git:** photographs, certificates and keys, generated
   results.
9. **Each phase is checked before commit:** a search for excluded technologies, an
   import check, the full test suite, and a `git merge-tree` conflict check against
   `main` and `dev`.

Documents may mention excluded technologies only as history (what was removed and
why), never as part of the current system.

## Phase dependencies

```
main (d6020fa)
 └─ Phase 1: feature/matcher-safety-improvements (dffc40d)
     └─ Phase 2: feature/evaluation-baseline (branched from dffc40d)
         └─ Phase 3  target selection      (measured by Phase 2 target-selection metrics)
             └─ Phase 4  object tracking
                 └─ Phase 5  frame quality assessment
                     └─ Phase 6  multi-frame aggregation (uses tracks from 4, frames chosen by 5)
                         └─ Phase 7  barcode and OCR reconciliation
                             └─ Phase 8  final integration and evaluation

Parallel track: annotation (Phase 2 review workflow) and data collection,
needed to measure Phases 3–8 on reviewed data.
```

Merge Phase 1 before Phase 2. Phase 2's diff against `main` otherwise also
contains the matcher commit.

## Related documents

- [research/EVALUATION.md](../research/EVALUATION.md): evaluation tools, metrics and reproducibility (reference)
- [research/datasets/photos-v1/README.md](../research/datasets/photos-v1/README.md): real photographs and the human-review steps
- [research/datasets/dailymed-pilot/README.md](../research/datasets/dailymed-pilot/README.md): official label artwork pilot

## Background

Phase 1 started from a read-only audit of the `main`, `dev` and
`Visual-assistant-research` branches. The research branch had been merged into
`main` (pull request #2) and then reverted (#3), so its commits are already in
`main`'s history while their changes are not. `git merge` of that branch does
nothing. Each phase therefore restores selected files with
`git checkout origin/Visual-assistant-research -- <paths>` and reviews them
individually. The web application and the Qwen vision-language model from that
branch are deliberately not brought back.
