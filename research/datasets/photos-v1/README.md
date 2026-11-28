# photos-v1: real camera photographs for human annotation

Real photographs only. Official label artwork (DailyMed) is a separate group
(`research/datasets/dailymed-pilot/`) and is never pooled with these.

`labels.json` lists the seven existing project photographs. It was prepared from
file metadata only: path, SHA-256, the HEIC original and its hash, EXIF device,
capture time, orientation and size as OpenCV loads it. **Every truth field is
empty and every entry is UNREVIEWED.** No product identity, bounding box or label
text was filled in by the assistant, and model predictions
(`research/datasets/medicine-v1/drafts.json`) are not annotations.

The images themselves are not in Git (`VA-photos/` and `vision/advil_front.jpg`
are ignored). Labels refer to them by path and hash; an annotation made on a
different file is ignored.

| id | image | what to establish |
|---|---|---|
| photo-IMG_9764 | VA-photos/converted/IMG_9764.jpg | Non-medicine bottle? Box(es) |
| photo-IMG_9765 | VA-photos/converted/IMG_9765.jpg | Non-medicine bottle? Box(es) |
| photo-IMG_9766 | VA-photos/converted/IMG_9766.jpg | Medicine identity, front text, medicine and other-bottle boxes |
| photo-IMG_9767 | VA-photos/converted/IMG_9767.jpg | Same bottle as 9766? UPC; boxes |
| photo-IMG_9768 | VA-photos/converted/IMG_9768.jpg | Same bottle? Readable? Boxes |
| photo-IMG_9769 | VA-photos/converted/IMG_9769.jpg | Same bottle? Readable? Boxes (small, far away) |
| photo-advil_front | vision/advil_front.jpg | A different bottle (500 count?) or the same? Identity, front text, box |

Duplicate copies (`VA-photos/optimized`, `VA-photos/VA-photos/optimized/IMG_9764.jpg`,
`research/datasets/medicine-v1/images/55cf…jpg`) are resized copies of IMG_9764
and are deliberately not listed: they are not independent images.

## Review levels

| Level | Set by `reconcile` when | Used for |
|---|---|---|
| UNREVIEWED | nobody annotated it, or reviewers disagree | descriptive results only |
| SINGLE_REVIEWED | one person annotated it | exploratory metrics |
| REVIEWED | two (or more) people annotated it independently and agree | primary metrics |

`physical_product_id` groups views of the same physical package. Metrics are
reported per image and per physical product, and confidence intervals are
computed over products, so four views of one bottle count as one sample.

## How to review (each person on their own)

Do it with the physical bottles in hand if you still have them. Don't look at the
other reviewer's file (`reviews/<name>.json`), `drafts.json`, the older
`evaluation/labels.json`, or any matcher output before you finish.

```sh
.venv/bin/python -m research.annotation_review status
.venv/bin/python -m research.annotation_review annotate photo-IMG_9766 --reviewer "RJ"
.venv/bin/python -m research.annotation_review preview  photo-IMG_9766 --reviewer "RJ"   # check your own boxes
```

`annotate` asks, in order:

1. Does the photo show a medicine package to identify? (No for the water-bottle photos.)
2. A short ID for the main package/object. Use the same ID for every photo of the
   same physical item (e.g. `P1` for the Equate bottle in all its views, `N1` for
   the water bottle). Your IDs don't need to match your teammate's; the grouping does.
3. Did you check against the physical package?
4. Is the identifying text (brand, ingredient, strength) readable in this photo?
5. For a medicine: product name, brand words, each active ingredient and strength
   as printed (`Ibuprofen | 200 mg`; liquids `Acetaminophen | 160 mg/5 mL`; salts as
   printed, e.g. `Diphenhydramine HCl | 25 mg`), package NDC as printed (leave
   empty if you can't read it; never guess), UPC digits, and the front-panel text
   line by line (this is the "perfect OCR" input; leave empty to skip).
6. Boxes, in an OpenCV window: first the medicine package (exactly one box), then
   every other bottle in the photo (the water bottle appears in all six iPhone
   photos). Drag a box, press ENTER or SPACE to keep it, ESC when finished. Draw
   tightly around the visible object, cap included.
7. The intended target: which numbered box the person taking the photo meant the
   system to read, `none` if nothing, or empty if you don't know. This is separate
   from the medicine box (a photo can show two medicines, or the person may have
   meant something else) and is never filled in from it. Only someone who knows
   why the photo was taken can answer; leave it empty otherwise.

Then, once both of you are done:

```sh
.venv/bin/python -m research.annotation_review reconcile --dry-run   # see agreements and disagreements
.venv/bin/python -m research.annotation_review reconcile             # write the levels into labels.json
```

On a disagreement the image stays UNREVIEWED and `status` lists the fields.
Look at the package together, then the person who was wrong re-runs `annotate`
with `--replace`, and you reconcile again. Boxes agree when they overlap by
IoU >= 0.8; the stored box is the mean of the two.

## Checklist

- [ ] Are IMG_9766–9769 one physical bottle? (Same ID for all four.)
- [ ] Is `vision/advil_front.jpg` a different bottle? (It shows "500 coated tablets";
      IMG_9766 appears to show 100. Verify, don't assume.)
- [ ] For each Equate bottle: brand words, ingredient and strength as printed, NDC
      (only if you can read every digit), UPC (zxing decodes `681131699372` in
      IMG_9767; confirm it against the printed digits yourself).
- [ ] Front-panel text for the front views (9766, advil_front).
- [ ] Readable yes/no for every view; the side, back and top views may be "no".
- [ ] Boxes for every bottle in every photo, including the water bottle in the background.
- [ ] The intended target for each photo, if you know what it was taken to show.
- [ ] `vision/advil_front.jpg` shows part of a person: decide whether it may be
      used and shared; it is never committed.

## Running the evaluation

```sh
.venv/bin/python -m vision.products.evaluate --labels research/datasets/photos-v1/labels.json \
    --inputs full,truth_crop,yolo_crop --save
# after more reviews, re-score the saved OCR and detections without re-running models:
.venv/bin/python -m vision.products.evaluate --labels research/datasets/photos-v1/labels.json \
    --replay research/results/evaluation-<time>.json --save
```

Results go to `research/results/` (ignored by Git). Replay reuses saved OCR only
for the same image file and, for `truth_crop`, the same box. Other rows are marked
stale, not scored, and listed; run the evaluation again for them. A run made
before boxes existed has no `truth_crop` OCR, so run it again after `reconcile`.
