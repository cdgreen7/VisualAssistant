# DailyMed pilot (10 US OTC label images)

A small pilot before the 30–40 image benchmark. It measures **OCR + ProductMatcher
only**, on official label artwork. It says nothing about YOLO, camera blur, glare
or real phone photos; those remain the job of `medicine-v1` (real photographs).

| File | Tracked | Purpose |
|---|---|---|
| `labels.json` | yes | Ground truth, in `vision/products/evaluate.py` format plus optional keys |
| `retrieval.json` | yes | Written by `fetch`: source URL, SPL set ID, media name, UTC time, SHA-256, bytes |
| `images/` | **no** (gitignored) | The downloaded JPEGs, named `<id>.jpg` |

## Reproduce

```sh
.venv/bin/python -m research.dailymed_pilot fetch    # ~5.4 MB, only the 10 listed images
.venv/bin/python -m research.dailymed_pilot verify   # labels vs. local openFDA catalog + hashes
.venv/bin/python -m research.dailymed_pilot run      # writes research/results/dailymed-pilot-<time>.json
```

To compare a changed matcher without re-running OCR, replay the OCR text saved
in an earlier run (OCR results stay exactly as recorded; each row keeps its
previous outcome and reason):

```sh
.venv/bin/python -m research.dailymed_pilot replay --results research/results/dailymed-pilot-20261119-143441.json
```

`fetch` never overwrites: an existing image must match its recorded hash, and if
DailyMed later serves different bytes it stops so the labels can be reviewed.
The existing evaluator also reads these labels (full images only):
`.venv/bin/python -m vision.products.evaluate --labels research/datasets/dailymed-pilot/labels.json`.

## Label keys (backward-compatible additions)

`evaluate.py` reads `image, product, brand_words, ingredients, formulation,
package_ndc, visible_text` and ignores everything else. Added, all optional:

- `id`, `category` (the `medicine-v1` category names)
- `roi`: `[x1, y1, x2, y2]` of the **front panel** in stored pixels; `null` = whole image is the front panel
- `barcodes`: UPC digits as printed under the bars (truth for a later barcode run; not scored here)
- `provenance`: `dataset`, `spl_set_id`, `media_name`, `source_url`

Conventions:

- `formulation` uses catalog names and strengths (`"200 mg/1"`) and must equal a
  formulation of the SPL set (`verify` checks this).
- `ingredients` uses the full catalog name, salt included, with the printed amount.
  The OCR "ingredients" score therefore means "the matcher recognized the complete
  ingredient", e.g. `HBr` printed on Mucinex DM does not count as HYDROBROMIDE.
- `visible_text` is the front panel only (the input for evaluate.py's perfect-OCR "oracle").
- `package_ndc` only when an NDC is **printed as text** on the artwork and is a
  package of the same SPL set. NDCs inferred from UPC digits are not used as truth.

## Truth and its limits

Selection and transcription were done by the assistant (Claude) on 2026-11-19 by
viewing each image and cross-checking the local catalog
(`otc_ndc_2026-10-30.json`) by SPL set ID. **A person should re-check
`labels.json` against the images before these labels are relied on.**

DailyMed images are unfolded carton artwork (dielines), not photographs: they
include Drug Facts, barcodes and printer marks. Use them locally for research;
label artwork belongs to the manufacturers, so don't redistribute the images.
