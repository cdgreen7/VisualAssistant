# Local batch annotation assistant

## Commands

Use the existing virtual environment from the repository root:

```sh
cd /path/to/visual-assistant
.venv/bin/python -m research.annotate_medicine batch VA-photos/
.venv/bin/python -m research.annotate_medicine review
.venv/bin/python -m research.annotate_medicine report
```

Absolute paths also work. Relative source and manifest paths resolve from the
repository root, not an arbitrary shell directory. When the virtual environment
is activated, `python3 -m research.annotate_medicine ...` is equivalent.
Existing `add IMAGE`, `edit ID`, `report`, and global `--manifest PATH` remain.
Use `review --no-open` only when inspecting images in your own local viewer.
Batch requires no questions per image; only review asks for approval.

## What batch does

- Recursively discovers JPG/JPEG/PNG/HEIC/HEIF, case-insensitively, including
  nested `VA-photos/VA-photos/optimized/` and `converted/`. Non-images are ignored.
- SHA-256 identifies byte-identical files, keeping alternate paths as aliases.
  Already-registered originals are recognized without inference or overwrites.
- HEIC is decoded with `pillow_heif` when it is installed, otherwise with macOS's
  built-in `sips`, which writes a temporary JPEG copy that is read and deleted.
  The stored hash is always of the original HEIC bytes. Without either decoder a
  HEIC file is reported as an error; converted JPEGs still work. Nothing is
  installed or downloaded. Source images are never modified.
- Processes an EXIF-oriented, aspect-preserving working copy (1600-pixel maximum
  side; configurable with `batch ... --ocr-side 1280`, allowed 320–2560).
- Reuses locally installed `yolo26n.pt`, English CPU EasyOCR and ProductMatcher.
  Model instances are reused across images. EasyOCR downloads are disabled;
  absent YOLO weights fail clearly instead of triggering a download.
- Reads the whole working image to avoid silently dropping a second visible
  ingredient/label, then reads each detected bottle again (up to three, most
  confident first) cut from the **original** pixels with a 4% margin, because
  label print on a 24 MP photograph is usually too small in the reduced frame.
  Every OCR line records its region (`frame`, `bottle_crop_1`, ...) and each
  crop's box in original-image coordinates is saved. When YOLO finds more than
  one bottle, `bottles_detected` says so: text may come from different objects.
- A single ingredient with exactly one confidently read amount anywhere on the
  label gets that amount as a suggestion, and its source says it was not on the
  same line. Two different amounts, or two ingredients, leave strength unknown. YOLO bottle detections, OCR lines/confidences, working-image
  coordinates, model names, matcher status and per-field suggestion sources are
  recorded. Ingredient names come from matcher evidence; amounts are associated
  only with isolated, sufficiently confident OCR ingredient observations.
- Ambiguous/missing strengths stay unknown. A detected bottle is not medicine
  proof; missing bottle detection is not proof of absence. No ingredient evidence
  means medicine/negative status stays unknown. Readability is only a suggestion.
- Product-name suggestions are the first confidently read wordy OCR lines
  (bottle crops first), explicitly not an exact
  product identity. Brand evidence is not package verification. Dosage forms are
  OCR word observations. Multiple ingredients generate a combination suggestion;
  all six categories can be selected/corrected during review. Same-formulation
  brands and non-medicine status require human judgment; they are not inferred
  automatically from a generic bottle. Qwen is not called unless you pass
  `batch ... --qwen` (slow, needs Ollama running); its transcription is then
  stored as an observation only and never fills a suggested field.
- `batch ... --refresh` re-inspects drafts nobody has reviewed yet, for example
  after the reader improves. Saved entries, confirmed duplicates and drafts with
  human edits are never re-inspected.

Drafts live in ignored `research/datasets/medicine-v1/drafts.json` with
`truth.confirmed=false`. They are not evaluation manifest entries. Batch updates
are atomic after each item; rerunning resumes without reprocessing completed
items. Failed items are retried (useful after a decoder is installed). Interrupted
inference may repeat the current item, but cannot duplicate approved ingestion.

## Quick review

The local viewer opens the original; larger images are reviewed before smaller
versions. Suggestions and their sources are displayed. At the edit prompt:

- Enter accepts suggested fields **as a draft**, not ground truth.
- `set brand "Text you independently read"` changes one field.
- `clear brand` rejects it and stores unknown.
- `set negative false` or `set category "combination"` changes a field.
- `set ingredients [{"name":"NAME FROM LABEL","strength":"AMOUNT","unit":"mg"}]`
  replaces the associated ingredient rows. Replace placeholders with actual
  independently inspected values; add one object for each active ingredient.
- `skip` leaves the image for a later review; `quit` stops. Field edits are
  checkpointed immediately, including across Ctrl-C/restart.

You then choose whether **all truth fields were independently verified** against
the physical package/another independent source. Default is No. Accepting AI
suggestions, a VERIFIED matcher result, or clicking Save cannot set confirmation.

If No, the manifest's `truth` remains unknown/unconfirmed; accepted/corrected
suggestions are stored separately as `reviewed_suggestions`. If Yes, you must
supply checker/source and confirm the complete ingredient–strength associations.
Unknown required fields fail validation rather than being guessed. Exact product
identity defaults to null; it may only be entered manually as an `exact_product`
object with valid package NDC and independent verification source, then passed
through the explicit independent-confirmation step. UPC alone cannot verify it.

Supply a physical item ID only when you know the physical item. Reuse it for
other views of that same item; do not group merely because brands match. Blank
means unknown. IDs derived from file hashes identify photographs, not products.
Finally choose a split (default unassigned), confirm privacy/local-use permission,
and explicitly approve Save. Refusal saves no new original, derivative or manifest
entry; the preliminary draft stays available locally for later review.

Image/annotation text and EXIF may contain private information. The helper does
not reliably detect/redact that information. Review it before approving retention.
Originals, derivatives, raw VA-photos and draft queues are ignored by Git. Draft
OCR is local and never transmitted; review files before sharing the repository.

## Duplicate relationships and product splits

Filename stems and oriented thumbnail dHash/aspect/color similarity suggest
resized/reformatted copies. These are **not proof**, so review asks whether to
mark an image as a duplicate of an offered representative, explicitly declare it
a distinct photograph, skip, or quit. Declared duplicates do not become additional
manifest examples. Undecodable files cannot be representative evaluation inputs.
Existing confirmed entries are never silently replaced by higher-resolution copies.
Similarity can miss copies or flag different photographs; human review remains
necessary. Exact hashes alone do not identify re-encoded copies.

A saved annotation records `photo_group_id`, `source_sha256`, optional human-set
`physical_product_id`, and `split`. Assigning train/validation/test requires a
physical item ID. The shared manifest validator rejects a physical/photo/source
group spanning splits (including mixing assigned and unassigned views) and
multiple entries for one declared photograph. Keep all views of a product in the
same split from the beginning. If changing a multi-view group's split later,
update all its entries together in the manifest and validate it; sequential
single-row changes are intentionally rejected.

The evaluator selects one split, default **unassigned**, instead of mixing train
and test images:

```sh
.venv/bin/python -m research.medicine_eval --split test --trials 20
```

All unconfirmed annotations remain excluded from accuracy metrics. Unknown
physical relationships and mistaken human duplicate/group decisions cannot be
solved by software alone; resolve them before making train/test claims.

## Originals versus endpoint derivatives

Offline defaults are 100 MiB per encoded image and 60 megapixels, configurable:

```sh
VA_RESEARCH_MAX_BYTES=157286400 VA_RESEARCH_MAX_PIXELS=80000000 \
  .venv/bin/python -m research.annotate_medicine batch VA-photos/
```

5712×4284 research originals are supported. Save copies the original bytes to
ignored `images/`, preserving its hash and metadata. It also writes a **separate**
EXIF-oriented JPEG under ignored `derived/`: aspect-preserving LANCZOS resize to
at most 2560 pixels on its long side, quality 92, reduced further if necessary
to stay within the unchanged 7 MP/8 MiB production endpoint limits. No crop is
applied. Source dimensions, orientation, resize description, output dimensions,
quality and both hashes are recorded. This is a full-frame derivative, not a
claim of successful YOLO bottle localization.

`medicine_eval.prepare` validates original and derivative hashes and submits the
recorded derivative to the same endpoint. Legacy entries with no derivative keep
the previous behavior. Offline safety limits do not alter production upload limits.
Using a derived image changes the evaluation input and must be documented when
comparing experiments; no performance or accuracy improvement is claimed here.

## Files and validation

Added: `research/dataset_images.py`, `research/annotation_batch.py`,
`tests/test_annotation_batch.py`, `tests/test_annotation_reader.py`, and this guide.
Updated: `research/annotate_medicine.py`, `research/medicine_eval.py`,
`research/datasets/medicine-v1/README.md`, its `.gitignore`, root `.gitignore`,
and the research README. No production inference or verification file changed.

Tests cover nested/relative/absolute discovery, decoding and HEIC failure,
high-resolution originals and bounded derivatives, configurable limits, exact
and suggested duplicates, human duplicate confirmation, local suggestion
provenance/combination amounts, explicit independent confirmation, rejection of
private retention, saved-entry preservation, interrupted edits, resume, legacy
compatibility, hash tampering and cross-split leakage prevention.

Actual local discovery on November 12, 2026: six HEIC originals, six converted
JPEGs and one optimized JPEG. The HEIC files decode through `sips`.
An existing confirmed manifest entry was preserved and recognized by its exact
hash. The queue contains **twelve pending drafts** (each HEIC and its converted
JPEG are flagged as possible copies of each other by filename), one
already-saved image reference and no errors. No new annotation was approved or added
by the assistant. These counts and model observations are not accuracy results.
