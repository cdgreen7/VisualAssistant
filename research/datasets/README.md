# Datasets

No dataset is stored here yet. This file records what exists, what it can and
cannot support, and what must be collected before any accuracy number is
reported.

## What exists today

| Source | Contents | Ground truth available |
|--------|----------|------------------------|
| `evaluation/labels.json` | **1** image: `vision/advil_front.jpg`, the front of an Equate Ibuprofen 200 mg bottle | Brand words, ingredient, strength, formulation, visible text. **No** package NDC, **no** UPC, **no** object class or box. The label was read from the photo, not from the physical bottle; the file itself says it needs confirming. |
| `tests/fixtures/openfda_ndc_sample.json` | 12 FDA records | Catalog data for unit tests, not images. |
| `vision/products/data/verified_upcs.json` | 1 UPC (`681131699372`) marked PENDING with every product field empty | None: it is a placeholder. |
| Unit-test barcodes | Generated symbols | The encoded value, but they are synthetic, not photographs. |

Ground truth by question:

| Question | Labels available |
|----------|------------------|
| Object category | None |
| Product brand | 1 image |
| Ingredient | 1 image |
| Strength | 1 image |
| Barcode value | None (no photographed barcode with a recorded value) |
| Exact product (NDC/UPC) | None |

With n = 1, no accuracy, precision, recall or error rate can be stated for any
component. `python -m vision.products.evaluate` prints counts out of n for this
reason.

## What must be collected

A set that can separate the five quality levels in `methodology.md`:

- **Products:** at least 30 distinct OTC medicines, including pairs that differ
  only in formulation (e.g. Advil and Advil PM), store brands with
  "Compare to ..." text, liquids as well as tablets, and 10 non-medicine
  bottles and containers as negatives.
- **Views per product:** front label, Drug Facts panel, barcode, and one
  cluttered scene containing the product with other objects and people.
- **Conditions:** two distances, good and poor light, and both phone families
  (iPhone Safari, Android Chrome) through the web app, since that is the real
  capture path.
- **Scenes for counting and awareness:** 30 images with 0-6 people and common
  objects, with the true count and object list.

That is roughly 300-400 images. Fewer than about 100 labelled images per
condition will not support percentages to better than +/- 10 points.

## Label format

One JSON object per image, written **from the physical package**, never copied
from what a model or the FDA lookup returned:

```json
{
  "image": "research/datasets/v1/advil_pm_front_01.jpg",
  "captured_with": "iphone-safari",
  "view": "front",
  "objects": [{"class": "bottle", "box": [412, 180, 1310, 1040]}],
  "people_count": 0,
  "is_medicine": true,
  "brand_words": ["Advil", "PM"],
  "ingredients": [["IBUPROFEN", "200 mg"], ["DIPHENHYDRAMINE CITRATE", "38 mg"]],
  "formulation": [["IBUPROFEN", "200 mg/1"], ["DIPHENHYDRAMINE CITRATE", "38 mg/1"]],
  "package_ndc": "0573-0164-20",
  "upc": "305730164207",
  "visible_text": ["Advil PM", "Ibuprofen 200 mg", "Diphenhydramine citrate 38 mg"],
  "label_source": "Read from the carton by <initials>, <date>",
  "second_checker": "<initials>"
}
```

The NDC and UPC in this example are placeholders showing the format; they are
not real values.

`brand_words`, `ingredients`, `formulation`, `package_ndc` and `visible_text`
match what `vision/products/evaluate.py` already reads. Use `null` for anything
that cannot be read from the package; do not guess.

## Privacy

Photographs can contain faces, rooms and prescription labels. Keep people who
have not agreed out of the frame, never photograph a prescription label with a
name on it, and decide before collecting whether the images may be committed
or must stay local (this directory can be git-ignored).

## Medicine endpoint dataset structure

Use [medicine-v1](medicine-v1/README.md) for the new 30–50 image evaluation.
It has an empty confirmed manifest, a blank annotation template, and a separate
user-authorized latency-only manifest. The existing photograph remains excluded
from accuracy scoring until its ground truth is independently confirmed.
