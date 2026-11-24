# Medicine endpoint dataset v1

Target: **40 independently annotated images**, scalable to 30–50. No images
have been collected or downloaded by this task. `images/` is ignored by Git.
`manifest.json` is the accuracy dataset; currently empty.
`latency-only.json` references one existing user-authorized project photograph,
with unconfirmed truth and a fixed test ROI. It is not an accuracy example.
`annotation-template.json` is an unfilled template, never a benchmark input.

Suggested primary groups (assign each image one group, add optional tags):

| Category value | Target | Purpose |
|---|---:|---|
| single_ingredient | 10 | Ingredient and strength correctness |
| combination | 8 | All ingredients with correctly associated strengths |
| same_ingredients_different_brand | 8 | Brand distinctions with identical formulation |
| unclear_label | 6 | Blur, glare, small print; abstention |
| non_medicine_bottle | 4 | Medicine false positives |
| no_bottle | 4 | Identification rejection when candidate gating errs |

Counts are a collection plan, not labels or measured sample sizes. Prefer
multiple physical products and both phone families. Record related views of the
same product; they are not independent products. Do not use model predictions,
search candidates, Qwen text or filename guesses as ground truth.

## Annotation schema

Copy the template into `manifest.json`'s `images` array only after provenance
and permission are established. Paths are relative to the manifest directory.
The manifest has `schema_version: 1`, `dataset_id`, and `images`.

| Field | Type / requirement |
|---|---|
| id | Unique nonempty string; do not put names or private identifiers here |
| path | Local JPEG/PNG path relative to manifest |
| category | One primary group from the table |
| capture_device | `iphone-safari`, `android-chrome`, or null/other documented source |
| input_kind | Prefer `phone_crop`; otherwise describe full-frame/test-ROI input |
| roi | null for already-cropped image; otherwise integer `[x1,y1,x2,y2]` in stored pixels |
| provenance.source | Explain origin and permission confirmation |
| provenance.local_evaluation_permitted | Must be true to enter a run |
| truth.confirmed | Boolean; false excludes **every** accuracy metric |
| truth.confirmed_by | Required checker identifier for confirmed annotations |
| truth.confirmation_source | Required independent source, preferably physical package inspection |
| truth.confirmation_date | Recommended ISO date |
| truth.object_category | Required for confirmed examples: actual object, e.g. bottle/box/none |
| truth.brand | Exact readable brand text, or null (brand metric excluded) |
| truth.ingredients | Array of `{ "name": string, "strength": string, "unit": string }`; preserve every active ingredient |
| truth.dosage_form | Tablet/capsule/liquid/etc., if independently known; recorded, not separately scored |
| truth.exact_product | null, or `{ "package_ndc": string, "verification_source": string, "upc": string-or-null }` |
| truth.negative | Required boolean: not a medicine identification target |
| truth.readable | Required boolean: whether adequate identifying evidence is readable in this image |

Strength is a numeric string; units retain denominators. For example the
*format* `"strength":"<amount>", "unit":"mg/5 mL"` denotes concentration;
never substitute a tablet amount. These are schema descriptions, not labels.
A confirmed positive needs its complete ingredient list. A negative may have
an empty list. If the physical formulation cannot be established, leave the
whole annotation unconfirmed instead of inventing partial truth.

For exact identity, check the package identifier independently. A candidate
returned by the matcher is not that check. The endpoint verifies package NDCs;
a UPC-only annotation without a confirmed package NDC is insufficient for the
exact-product metric in this runner. No mappings are written to the registry.

## Capture preparation and privacy

Prefer the original JPEG/PNG high-resolution phone crop (no drawn boxes).
Stored crop bytes are submitted unchanged. An optional explicit ROI is decoded,
cropped without resizing, and encoded at JPEG quality 92 **outside** the server
request timer. This is a repeatable fixture preparation, not a measurement of
Safari canvas encoding or YOLO localization. Annotate in stored-pixel orientation;
EXIF orientation is not automatically changed by the deployed decoder.

Do not include patient names, prescription labels, faces without permission or
sensitive backgrounds. Review annotations/results before sharing. Keep image
files local; do not add them to Git. No images, OCR strings or model descriptions
are saved by the evaluation runner. Result files include IDs, hashes, counts,
statuses, timing, error codes and configuration. Model/source paths may appear
in initialization failures: review before publishing.

## Local annotation helper

From the repository root:

```sh
# Opens the ORIGINAL image locally in Preview on macOS; never uploads it.
.venv/bin/python -m research.annotate_medicine add "/absolute/path/to/crop.jpg"

# Inspect coverage/integrity as JSON (read-only).
.venv/bin/python -m research.annotate_medicine report

# Reopen an existing image and replace its annotation after a fresh review.
.venv/bin/python -m research.annotate_medicine edit YOUR_IMAGE_ID

# Explicitly inspect in your own viewer instead of launching Preview:
.venv/bin/python -m research.annotate_medicine add "/absolute/path/to/crop.png" --no-open

# Another manifest: global option goes BEFORE the subcommand.
.venv/bin/python -m research.annotate_medicine \
  --manifest research/datasets/medicine-v1/latency-only.json report
```

The helper imports `load_manifest` and `CATEGORIES` from the existing evaluator;
it does not replace the evaluation framework. It additionally checks numeric
strengths, unit syntax, per-ingredient associations, identifiers, image format
and configurable offline processing limits (default 100 MiB/60 MP). Original research images are not constrained by production upload limits. No inference or network call is made.
Requires the existing Pillow dependency and a POSIX system (macOS/Linux), with
Preview on macOS or `xdg-open` on Linux. No new dependency/model is installed.

### Example workflow (not prefilled ground truth)

1. Open a permitted local photograph with `add`. Examine its pixels; inspect
   the physical product independently when confirming medicine information.
2. Choose a nonprivate image ID, such as `bottle-a-front-01`, and physical-item
   ID, such as `bottle-a`. Reuse `bottle-a` for other views of that **same item**.
   A separate bottle gets another ID even if its brand/formulation is identical.
   Leave product ID blank for unknown identity/no physical product. Unknown IDs
   are reported separately, not treated as independent products.
3. Choose the category and describe capture device, input type and provenance.
   Confirm permission for local use. Enter negative/readable status (`?` allowed).
4. Enter only information established independently: object category, brand,
   dosage form, then **one ingredient at a time with its own amount and unit**.
   For a combination, repeat for every active ingredient; do not enter an
   unordered list of amounts. Blank ingredient name finishes the list.
5. Review the displayed ingredient/strength rows against the physical label.
   Confirm the full list only if all active ingredients and amounts are known.
   Units support mg, mcg, g, %, IU, units, and concentrations such as mg/5 mL.
   Unsupported notation is rejected rather than silently normalized; preserve
   the true concentration. No OCR/Qwen/search result can populate these fields.
6. Record checker ID and independent source. Choose **No** to full confirmation
   to save a draft if anything required is unknown. Unknown fields are null;
   all draft metrics remain excluded by the evaluator. An unreadable image can
   still have confirmed truth from the physical package: mark readable=false.
   Negatives can be confirmed without medicine ingredients.
7. Exact product identity is optional. Record it only after a separate explicit
   confirmation, with valid package NDC and independent source; optional UPC
   must pass its check digit. Otherwise exact_product stays null. A brand or
   formulation match does not establish exact package identity.
8. Inspect image pixels, **EXIF metadata**, and annotation text for private
   information. The tool reports EXIF presence, but does not detect all private
   information or strip metadata. If the original is unsuitable, cancel and
   prepare an appropriate separate image yourself. The original is never changed.
9. Review the final JSON and confirm Save. The tool makes a byte-identical copy
   under ignored `images/<sha256>.jpg|png`, then atomically updates the manifest.
   Declining or Ctrl-C before saving writes nothing. `edit` replaces the existing
   annotation after a full review without adding another image copy; re-enter
   its physical-product ID to retain grouping. It cannot change image bytes/ID.

### Integrity and report interpretation

SHA-256 catches byte-identical files even when renamed. It does **not** detect
visually similar or recompressed images; physical-product IDs remain essential.
Every existing image is checked before adding/editing: missing, damaged or
changed files block writes until repaired. Hashes stored by the helper reveal
later modifications. Legacy entries without hashes can be inspected/reported;
their hash is established when they are edited.

The manifest update is protected by a local file lock and atomic replacement.
A failed manifest write removes only the new copy made by that operation. An
existing destination is never overwritten. Do not manually edit the manifest
while an interactive annotation session is active. Lock files are Git-ignored.

`report` provides:

- Total image entries and confirmed labeled images, plus counts by category.
- Usable confirmed images (valid annotations with present, readable, unchanged
  files); duplicate groups remain separately flagged and should be resolved.
- Unique **recorded physical-item IDs**, and images missing an ID. This is not
  an inferred count of brands/formulations/SKUs or a claim about unknown items.
- Counts of confirmed positive images with complete ingredient/strength labels,
  and confirmed exact-product annotations. These are image-level coverage counts,
  not independent product counts, model accuracy or numbers of ingredients.
- Unconfirmed/missing annotations, field-level annotation gaps, invalid rows,
  missing/invalid/changed files and duplicate image hash groups.

Optional unknown brand fields appear as gaps without invalidating otherwise
confirmed annotations. Missing/invalid files are listed even if their annotation
was confirmed, so total labeled images can exceed usable confirmed images.
Different confirmed formulations/brands/package identifiers on the same physical
item ID are rejected on save. This detects obvious association/grouping conflicts;
only independent human inspection can detect an otherwise plausible swapped
strength or mistaken ground-truth label.

### Validation

`tests/test_annotate_medicine.py` uses synthetic temporary images and human-input
fixtures, never the real dataset. It covers copies preserving original bytes,
compatibility with `load_manifest`, duplicates, missing/unsupported/corrupt images,
privacy/permission checks, partial drafts, combination association review,
invalid units/strengths, exact identity, negatives, editing, physical-item grouping,
conflicting truth, report integrity, rollback, viewer invocation and cancellation.
The real confirmed manifest remains empty until the collector adds images.

Edits preserve any existing evaluation ROI; when editing such an entry the
helper prints its coordinates so readability is judged within the evaluated
region. The inspected source hash is checked again before saving: if the source
changed while the questionnaire was open, saving is rejected and reinspection
is required. The helper never recrops or re-encodes the source.


## Automated batch suggestions and review

See [batch workflow and integrity rules](../../ANNOTATION_AUTOMATION.md).
`batch` writes local, ignored preliminary drafts; it never adds model-generated
truth to the confirmed manifest. `review` requires approval before copying an
original and creating a separate bounded endpoint image. Existing `add`, `edit`
and `report` commands remain available. Use batch/review for high-resolution
images that need endpoint derivatives; legacy manual add keeps its original
input behavior and does not automatically create an endpoint derivative.
