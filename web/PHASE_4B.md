# Phase 4B: automatic OCR identification and full-screen camera

## Result

A stable Phase 4A candidate now automatically sends its detailed bottle crop
to EasyOCR and the existing ProductMatcher. The user does not need to tap
Identify medicine. A compact, collapsed medicine card shows the status and
formulation explanation. Expand it for the raw recognized label, catalog
support, exact-package evidence, crop preview, timing, and Rescan label.
Automatic spoken medicine information is not implemented.

The user's physical iPhone tests validated Phase 4A for two bottles:
Equate (477×894 crop) and Kroger combination medicine (675×1527 crop), both
from 1080×1920 frames. Those exact images were not saved in this workspace.
Phase 4B has automated and local backend validation; its new behavior still
requires a physical iPhone test.

## Camera layout

The app, video and overlay have explicit viewport-sized bounds, no maximum
width, no margins, and centered `object-fit: cover`. `visualViewport` resize
and scroll events update the available CSS height/width and redraw boxes;
`100vh` / `100dvh` remain fallbacks. Safari safe-area padding belongs to the
header/dock/cards, not the camera pixels. The header and dock are shorter.
Versioned CSS and app URLs force a fresh asset load after deployment.

The 54-pixel collapsed medicine card is an expandable native `details` element.
Its expanded height is limited to 36dvh in portrait and 48dvh in landscape.
The preview stays behind every card. Ask AI's existing one-card-at-a-time
behavior remains intact. The More sheet and manual result sheet retain their
own controls. No camera constraints, OCR capture size, JPEG quality, or
bounding-box coordinates are reduced for this layout change.

`view.js` is reused unchanged. `View.coverTransform` and `View.mapBox` account
for image scale and centered cropping. Video and overlay share identical
bounds. Device rotation invalidates old candidate images and ignores detection
results whose aspect ratio no longer matches. The existing front-camera view
is intentionally unmirrored (`PREVIEW_MIRRORED = false`); the box helper's
mirroring behavior is tested, and no CSS mirroring is introduced.

Browser geometry checks used a synthetic preview at 320×568, 375×667,
390×844, 430×932 and 844×390. In each, the stage, video and overlay exactly
matched the viewport with origin (0,0), and the video used `cover`. At 390×844,
the collapsed card measured 54 px; expanded fixture content measured 177 px
while the video remained 844 px high. This is a browser viewport test, not a
real iPhone or Safari-toolbar/safe-area hardware test. No Mac webcam was used.
A synthetic screenshot is saved at `/tmp/visual-assistant-phase4b-layout.jpg`.

## Automatic identification workflow

1. The existing default-off automatic-scanning setting enables Phase 4A's
   stable bottle gate: three tracked or five untracked sightings, spatial
   stability, ambiguity checks, and cooldown/continuous-object suppression.
2. Phase 4A derives its 640-pixel detection image from the same frozen native
   frame used to prepare the bottle crop. Its 2560-pixel maximum side and
   JPEG quality 0.92 are unchanged. Mapping uses independent x/y scale factors.
3. `MedicineIdentification` submits only the crop to
   `POST /api/medicine/identify-crop`, once per capture. It sends session and
   capture IDs and whether the optional reader is requested.
4. The endpoint validates JPEG/PNG content and MIME type, an 8 MiB byte limit,
   and 7 MP pixel limit. It retains native uploaded pixels. Multipart image
   streams and decoded images stay in memory; no images are saved by the app.
5. `VisionServices.identify_crop` reuses `identify_detection`, with a whole-crop
   ROI adapter, then the existing EasyOCR and ProductMatcher. It does not run
   bottle detection again inside an already localized crop. The existing
   identification/crop/matcher algorithm is not duplicated.
6. The response carries OCR, the existing serialized match, evidence-level
   explanations, timings, and echoed capture/session IDs. The card uses text
   nodes, never model/OCR HTML.

Example request, from the project root with a real crop available:

```bash
curl --cacert "$(mkcert -CAROOT)/rootCA.pem" \
  -F 'image=@/path/to/label-crop.jpg;type=image/jpeg' \
  -F 'session_id=example_session_1234' -F 'capture_id=example_capture_1234' \
  -F 'assist=false' https://localhost:5443/api/medicine/identify-crop
```

The existing `/api/identify` request/response remains unchanged. It continues
to run YOLO on a manually captured image, select the best bottle, and use the
same OCR/matcher helper. Barcode scanning, UPC registry and FDA catalog paths
are unchanged.

## Evidence levels and combinations

The card distinguishes three levels without inventing new matcher statuses:

- **Label recognized:** EasyOCR observations, preserved even when noisy or
  conflicting. Confidence and normalized matcher evidence are also returned.
- **Formulation:** a catalog-compatible ingredient/strength formulation when
  every necessary ingredient/strength is observed and there are no conflicts.
  This is informative support, not a status upgrade. Missing ingredients,
  missing strengths, conflicting evidence and unresolved combinations remain
  explicit. Multiple dosage forms are not guessed into a single tablet type.
- **Exact package:** shown as verified only when the existing matcher returns
  `VERIFIED / PACKAGE` with package data. `VERIFIED / FORMULATION` does not
  verify a package. Otherwise this field says **Not verified**.

`ProductMatcher` is unchanged. Statuses remain VERIFIED, UNCERTAIN and NO_MATCH.
A stricter presentation guard can downgrade a combination FORMULATION result
to UNCERTAIN when ingredient-to-strength associations are not supported.
The matcher treats observed strengths as an unordered set, so the presentation
requires every combination ingredient and its uncorrected amount to appear
in a high-confidence OCR observation containing just that active ingredient
(and its printed amount) before claiming that formulation
is supported. Separately printed ingredient/amount lines can therefore stay
UNCERTAIN. Exact-package NDC evidence still uses the existing package rules.
This guard never upgrades or weakens a decision.

Both acetaminophen and ibuprofen observations remain in the response and card,
including conflicting or missing amounts. No combination is simplified to a
single-ingredient product. Store-brand hints and comparison brands remain
subject to the existing conservative matcher rules.

## Optional local Qwen reader

The **Qwen second reader (slow; unverified)** setting in More is unchecked by
default. It causes one optional reading per captured crop, after primary OCR
and matching. It never reads ordinary awareness frames. `QwenLabelReader`
uses the installed `qwen2.5vl:3b`, or `QWEN_VISION_MODEL`, through loopback
Ollama with cloud models, proxies and redirects excluded by the local API
adapter. It uses a 1024-pixel copy for Qwen only; EasyOCR retains the full crop.
The second reader has a 25-second timeout and a bounded structured response.

Qwen lines, token-level agreements, and unconfirmed/conflicting observations
are returned separately. Token agreement is a diagnostic, not validation of
a product or the printed ingredient-to-strength relationship. Qwen output
never reaches ProductMatcher or the official catalog. Timeout/unavailable/
invalid-response errors leave the primary OCR/matcher result available.
There is no Qwen-derived product verification and no speech of medicine data.

Experiment: on the existing `vision/advil_front.jpg` full image, EasyOCR read
the key brand/ingredient/strength text; the optional Qwen reader returned an
empty line list after **9.213 seconds**. This experiment showed extra latency
without improved extraction on that image, so assistance remains off by
default. It is a single-image result, not a model-wide accuracy benchmark.

## Responsiveness and stale-result handling

The awareness loop does not await the identification fetch. Its next YOLO tick
continues on the existing schedule. EasyOCR has a separate nonblocking crop
lock; concurrent crop requests receive HTTP 409 rather than queueing. Manual
OCR shares that OCR lock. Manual identification now releases YOLO's lock before
OCR. The loaded YOLO component can be retrieved without waiting on first-time
OCR initialization's load lock.

Capture IDs suppress repeated requests. A generation counter, abort signal,
and echoed session/capture IDs reject stale responses after camera changes,
rotation, awareness stop/restart, candidate replacement/departure, disabling
automatic scanning or dismissal. **Rescan label** clears the candidate latch
and waits for a fresh stable frame. It does not reuse an old result.

Browser cancellation does not forcibly terminate EasyOCR already executing
on the server. A rescan may receive a busy response until that work completes;
the user can rescan again. Client requests time out after 90 seconds. CPU/RAM
are still shared, so independent locks do not guarantee zero YOLO slowdown.
A failed awareness poll does not automatically cancel identification or erase
its cooldown, avoiding repeated work from transient detection failures.

## Measured OCR and matching results

Hardware: existing Apple M4 MacBook Pro with 24 GB memory. Benchmark process
used CPU EasyOCR with four PyTorch threads. App defaults are unchanged, so
phone/Wi-Fi latency and CPU scheduling can produce different times.

A 730×1080 crop of the existing Equate photo was JPEG encoded at quality 0.92
and submitted through Flask's actual crop endpoint. No new camera image was
captured and no image was stored.

| Run | OCR | Matcher | Endpoint processing total | Result |
|---|---:|---:|---:|---|
| Cold service | 2992.21 ms | 13.65 ms | 7079.22 ms | UNCERTAIN; package not verified |
| Warm service | 2375.31 ms | 15.45 ms | 2391.46 ms | UNCERTAIN; package not verified |

Cold total includes loading the recognizer/catalog/matcher. Totals exclude
network transfer and browser capture time; the card also reports client round
trip. Recognized key text: `equate`, `Ibuprofen`, `Tablets 200mg`. Other OCR
noise is preserved in raw evidence. Supported formulation text was
`IBUPROFEN 200 mg/1`; no exact package was verified.

The user's stated Equate and Kroger text was also fed as high-confidence OCR
fixtures to the **installed full FDA catalog** (not passed off as real-image
OCR). Equate retained seven formulation candidates and UNCERTAIN. Kroger
retained five candidates and UNCERTAIN, preserved both `ACETAMINOPHEN` and
`IBUPROFEN`, and displayed “Combination label observed; formulation uncertain.”
The readable text retains acetaminophen 250 mg and ibuprofen 125 mg. The exact
physical Kroger crop is unavailable here, so real-image Kroger OCR accuracy
has not been measured.

## Official medicine labeling lookup: follow-up milestone

A live lookup is deliberately deferred so it does not disrupt the working
Phase 4B path. The offline FDA NDC catalog remains the matching source.

The [openFDA Drug Label endpoint](https://open.fda.gov/apis/drug/label/how-to-use-the-endpoint/)
provides search results, and its [documented fields](https://open.fda.gov/apis/drug/label/searchable-fields/)
include label text and harmonized identifiers. The
[DailyMed v2 SPL API](https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_api.cfm)
provides listing/filtering; [SPL-to-NDC resources](https://dailymed.nlm.nih.gov/dailymed/webservices-help/v2/spls_setid_ndcs_api.cfm)
provide identifier associations. These are lookup interfaces, not proof that
the photographed package is the returned record.

A reliable next milestone needs a separate exact-identifier-first service:
prefer a trusted package/product NDC or SPL Set ID; otherwise search all
candidate brand/ingredient records with bounded pagination. Validate every
active ingredient, associated strength, dosage form and package identifier;
retain ambiguous matches instead of choosing the first result. Parse the
relevant SPL ingredient/amount fields and validate units rather than assuming
unstructured labeling text is normalized catalog data. Cache only successful,
validated records by identifier and record version with a bounded TTL; handle
429/Retry-After, network failures and bounded retries without blocking scans.
Never send images or Qwen guesses as authoritative evidence, and exclude
personalized dosing. This needs its own fixtures and source-schema tests.
No fragile search-result shortcut or live network dependency was added.

## Files changed in this phase

Created:
- `web/medicine_upload.py`: crop validation without downscaling/storage.
- `web/medicine_result.py`: evidence-level presentation and conservative
  combination-association guard.
- `web/qwen_label.py`: optional untrusted local second reader.
- `web/static/medicine-identification.js`: independent capture request lifecycle.
- `tests/test_medicine_api.py`, `tests/test_qwen_label.py`,
  `tests/js/medicine-identification.test.js`: new test suites.
- `web/PHASE_4B.md`: this report.

Modified:
- `web/services.py`: crop adapter, timing, separate OCR lock, manual lock scope.
- `web/server.py`: dedicated crop endpoint, optional reader, in-memory uploads.
- `web/static/app.js`: automatic submission, compact evidence card, rescan,
  lifecycle cancellation, visual viewport synchronization.
- `web/templates/index.html`, `web/static/style.css`: full-screen camera,
  collapsed card, optional reader setting, versioned assets.
- `tests/js/medicine.test.js`, `tests/js/ask-ai.test.js`: updated app integration
  mocks and automatic-crop expectation.
- `tests/js/view.test.js`: multiple iPhone geometries and mirrored mappings.
- `tests/test_frontend_js.py`: run the new client test suite.
- `web/README.md`: entry point to this guide.

No Caden files, YOLO/tracking algorithms, ProductMatcher implementation,
FDA catalog data or UPC registry rules were modified. Existing unrelated
working-tree changes were preserved. Nothing was committed or pushed.

## Test on the iPhone

1. Restart Flask from the project directory:

   ```bash
   .venv/bin/python -m web.server --host 0.0.0.0 \
     --cert web/certs/cert.pem --key web/certs/key.pem
   ```

2. On the same Wi-Fi, open
   `https://Microadmins-MacBook-Pro-2.local:5443` in iPhone Safari. Reload the
   page (close/reopen the tab if old layout assets remain cached). Use the
   existing trusted HTTPS certificate setup.
3. Open **More**. Enable **Automatic medicine scanning (experimental)**.
   Leave **Qwen second reader** off initially. Close More and start awareness
   using the **Camera** dock control. This uses the iPhone camera only.
4. Hold the Equate label steady and large in view. After stable capture,
   expect “Reading medicine label…” followed by a matcher status. No tap on
   Identify medicine is needed. Expand the medicine card and check label text,
   formulation, exact package and timings. UNCERTAIN is an expected valid result.
5. Try Kroger. Confirm both acetaminophen 250 mg and ibuprofen 125 mg remain
   visible in label evidence. Missing/conflicting/ambiguous text must not be
   reported as a verified plain-ibuprofen package.
6. Keep the bottle in view: it must not rescan on every YOLO frame. Expand the
   card and tap **Rescan label** for a fresh scan; wait for a stable new capture.
7. Rotate the phone, switch cameras, or stop/restart awareness while a scan is
   pending: the old result must not populate the new session. Check overlay
   alignment at portrait/landscape and that cards leave the preview visible.
8. Try unreadable text and verify NO_MATCH or UNCERTAIN with raw evidence.
   Verify manual identification and barcode scans after stopping awareness.
9. Optionally enable Qwen second reader, trigger a fresh rescan and compare its
   separately labeled output/timing. It must not change the match or package
   verification. Disable it again if it adds no value.

## Remaining limitations

OCR can confuse brands/numbers, miss a second ingredient, or read comparison
text. No system can establish that all active ingredients are visible in a
partial front label; package verification remains strictly evidence-based.
The catalog can omit or represent combination formulations differently, as
seen with the Kroger text. Brand, ingredient names, strengths and corrected
observations are not equivalent to exact package identity. The compact card
may truncate its summary; expand it for full evidence. Independent execution
can still contend for CPU. New Phase 4B physical iPhone behavior, blur/glare,
Safari speech restrictions for existing features and device memory behavior
remain to be tested on hardware. There is no automatic medicine speech,
medical advice, live official lookup or Qwen-derived verification.

## Final test results

- `.venv/bin/python -m unittest discover tests`: **231 tests passed**.
- `node --test tests/js/*.test.js`: **60 tests passed**.
- JavaScript syntax checks and `git diff --check` passed.
- New coverage includes real matcher fixtures for Equate and combination
  labels, installed-catalog combination uncertainty, missing/conflicting and
  swapped amounts, unreadable labels, NO_MATCH, exact NDC package support,
  native crop pixels, byte/pixel limits, optional-reader conflicts/failures,
  duplicate and concurrent requests, cancellation/late results, session IDs,
  actual detector method access while mocked OCR waits, and five viewport
  sizes plus portrait/landscape/mirroring geometry.
- Existing manual identification, barcode, UPC registry, FDA catalog, matcher,
  tracking, voice, Ask AI and Phase 4A tests passed. Warnings from deliberately
  invalid PNG fixtures and existing unclosed test resources did not fail tests.
- The actual camera was never opened by the agent; the only UI preview used
  for layout verification was synthetic. Phase 4B remains ready for the user's
  real iPhone test, not claimed as already tested on that device.
