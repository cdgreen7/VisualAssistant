# Reproducible medicine-crop baseline

This iteration adds evaluation only. It does not optimize or change deployed
recognition, verification, camera handling, UI, or API behavior.

## Actual workflow

`POST /api/medicine/identify-crop` receives the already prepared high-resolution
crop and session/capture identifiers. It does not detect bottles with YOLO.

1. Validate identifiers and upload bounds.
2. `read_crop`: validate JPEG/PNG format against MIME; decode with Pillow;
   enforce 8 MiB/7 MP; convert RGB to BGR. No resizing or disk image storage.
3. `VisionServices.identify_crop`: acquire nonblocking OCR lock; get cached
   recognizer/matcher; construct a full-image region.
4. Existing `identify_detection` selects that region and calls `crop_box`.
5. EasyOCR reads it; ProductMatcher matches the observations to the local FDA
   catalog. Models are loaded once and reused, not per request.
6. `medicine_result.explain` serializes existing decisions and applies the
   existing stricter combination association presentation guard.
7. Optional Qwen assistance (off for this baseline) cannot become matcher evidence.
8. Serialize JSON with the echoed IDs and timing fields.

Failures: invalid IDs or missing/corrupt/mismatched-format image (400), upload
limits (413), OCR already in progress (409), missing/corrupt models/catalog
(503), unexpected processing failure (500). Optional Qwen failures are reported
inside the successful primary result. Client-side generation/identifier checks
can discard replies, outside this server evaluation.

Verification remains unchanged: VERIFIED/FORMULATION and VERIFIED/PACKAGE are
separate; required ingredients/strengths/brand evidence, conflicts, corrected
characters, ambiguous candidates and package identifiers retain their existing
rules. UNCERTAIN and NO_MATCH are never promoted by the evaluator. Model
architecture and model files are untouched.

## Run

```sh
# Empty until independently annotated images have been added:
.venv/bin/python -m research.medicine_eval \
  --manifest research/datasets/medicine-v1/manifest.json --trials 20

# One authorized existing image, excluded from every accuracy denominator:
.venv/bin/python -m research.medicine_eval \
  --manifest research/datasets/medicine-v1/latency-only.json \
  --trials 20 --timeout 180
```

Results are timestamped JSON under `research/results/`. The runner starts one
fresh spawned process per image; it does not connect to or restart the user's
server. Each process exercises the actual Flask route with its test client and
uses unchanged service/decoder/matcher code. EasyOCR downloads are disabled in
the runner; missing installed weights produce a recorded failure, not a download.
Qwen, external lookups and YOLO are not invoked.

Before a controlled run, stop other inference/benchmark work yourself and keep
power/thermal conditions comparable. The runner does not stop unrelated processes.
Use the same image hashes, ROI, manifest, catalog, dependencies, thread settings
and machine for comparisons. Configuration records code hashes, catalog hash,
manifest hash, Git branch/commit/dirty state, versions, hardware, and thread counts.

## Timing and memory interpretation

- `fixture_prepare_ms`: file I/O plus optional fixture ROI preparation. Once per
  image, outside request latency. It is **not phone crop/capture time**.
- `load_ms`: preload of recognizer and matcher/catalog in a fresh process;
  includes library/model initialization. Excludes full application startup and
  unused YOLO/registry preload. OS disk caches are not forcibly emptied.
- `cold`: first endpoint call after preload; does not include `load_ms`.
- `warm`: subsequent endpoint calls using the same initialized instances.
- `decode_ms`, `server_total_ms`: existing Server-Timing header measurements.
- `crop_ms`: evaluation-only timing wrapper around the unchanged `crop_box` call.
  Server crop is a full-region array view, not repeated high-resolution capture.
- `ocr_ms`, `match_ms`: existing service timing; OCR includes EasyOCR internal
  preprocessing/detection/recognition, not separately instrumented internals.
- `request_ms`: Flask test-client wall time, including multipart setup/handling;
  excludes real network, TLS, camera, speech, and browser rendering.
- `peak_rss_mb`: process high-water RSS in MiB, including models/libraries. It is
  not current RSS, incremental allocation per request, or a whole-system total.
  Fresh process per image prevents previous image memory contaminating the peak.

Report median only at n>=3, p90 at n>=10 and p95 at n>=20. These minimum gates
prevent tiny-sample percentiles; even n=20 gives only a preliminary tail estimate.
Keep per-image distributions rather than treating repeated copies as independent
images. One cold observation is retained raw, without a cold median or p95.
Each event (input preparation/startup/request) has the configured watchdog timeout;
a timed-out worker is terminated and partial events are retained. HTTP errors,
Python exceptions, worker failure/exit codes and timeouts remain explicit.

## Correctness

Scores apply only to independently confirmed labels. Every request retains its
scores, but aggregate counts use one successful warm outcome per image (cold
fallback if no warm success). Repeated trials do not multiply the denominator.
Failures remain separate and must be reported alongside those conditional scores.

- Ingredient recognition: exact normalized set, including extra ingredients.
- Strength recognition: exact uncorrected strengths; combinations additionally
  require ingredient-associated observations, rejecting swapped amounts.
- Brand recognition: expected brand words present in OCR, not verified identity.
- Formulation/combination verified: endpoint says VERIFIED and returned first
  formulation exactly matches the complete ground truth. UNCERTAIN counts false
  for this metric even if the true formulation remains a candidate.
- Exact product: VERIFIED/PACKAGE with matching independently known NDC and
  formulation. Unknown exact truth is excluded, not counted correct.
- False VERIFIED: a verified wrong formulation or known wrong package; any
  VERIFIED on a negative is false. Package claims with unknown package truth
  remain unscorable rather than assumed safe. The numerator counts errors.
- Negative rejection: NO_MATCH. UNCERTAIN is not counted as correct rejection.
- Appropriate uncertainty: UNCERTAIN or NO_MATCH on a confirmed unreadable
  positive. Abstention on readable positives is not a correct identification.

A no-bottle example cannot assess YOLO recall with this endpoint; it measures
what identification does when fed a non-target crop. Small counts are descriptive,
not population accuracy estimates. No trial's output becomes annotation truth.

## Dataset status

The 40-image structure and annotation requirements are in
`datasets/medicine-v1/README.md`. At creation there are **zero confirmed images**
and **one user-authorized latency-only photograph**. Existing Equate annotations
require physical-package confirmation; combination, same-formulation other-brand,
unclear, negative bottle and no-bottle labeled examples are still missing.

Accuracy and false-verification rates therefore cannot yet be measured. A result
of zero evaluated examples must never be reported as 0% errors or 100% accuracy.

## Measured endpoint baseline — November 12, 2026

Primary result: `results/medicine-endpoint-20261112-133018.json`.
Earlier repeat: `results/medicine-endpoint-20261112-132751.json`.
Hardware supplement: `results/medicine-hardware-20261112.json` (sandboxed sysctl
inside the runner returned empty strings; subsequent read-only sysctl confirmed
Apple M4 and 24 GiB physical memory). Raw results are preserved unchanged.

Installed EasyOCR English CPU weights and their hashes are recorded in the
primary run, along with torch 2.8.0, EasyOCR 1.7.2, Flask 3.1.3, NumPy 2.0.2,
4 torch intra-op threads and 10 inter-op threads. No thread settings were changed.

Input: one 1920×1080 existing photograph, explicit `[585,0,1315,1080]` ROI,
730×1080 submitted JPEG at quality 92. Fixture preparation: 14.386 ms. This ROI
is not an independently validated YOLO localization. All truth fields remain
unconfirmed and all accuracy denominators are zero.

| Stage (ms) | First request after preload | Warm median | Warm p90 | Warm p95 |
|---|---:|---:|---:|---:|
| Decode/validate | 4.5 | 2.55 | 2.90 | 2.91 |
| Server full-region crop | 0.0075 | 0.005 | 0.006 | 0.007 |
| EasyOCR | 2,224.17 | 1,057.20 | 1,535.40 | 1,985.19 |
| ProductMatcher | 9.25 | 7.81 | 9.31 | 9.95 |
| Total server | 2,240.90 | 1,069.80 | 1,550.31 | 1,999.94 |
| Flask test-client wall time | 2,242.69 | 1,070.27 | 1,551.01 | 2,000.51 |

Preload: **3,058.36 ms**, measured separately from the first request. It excludes
process bootstrap/import work before preload; it is not total application cold
startup. Initialization high-water RSS: **1,258.92 MiB**. Overall worker peak:
**4,989.33 MiB (4.87 GiB)**. This excludes unrelated processes and a deployed
server's optional YOLO/Qwen memory. No memory optimization is claimed.

20 warm trials plus one cold trial; all HTTP 200, UNCERTAIN/NONE; zero request
failures or worker timeouts. Earlier run: warm total median 1,052.80 ms and p95
1,985.07 ms; first request 3,096.30 ms; peak RSS 5,002.42 MiB. The difference in
cold timings shows why one cold sample is not a reliable startup distribution.
No system-wide quiescence, thermal state, or clock-frequency control was enforced;
no other benchmark/test suite was intentionally run concurrently with inference.

These results measure actual endpoint code, **not phone end-to-end latency**.
They do not imply improvement over Phase 1's 5.38-second full-frame OCR: the input
region and processing scope differ. They do not validate iPhone/Android capture,
request cancellation, Wi-Fi, browser rendering, or the candidate detector.

## Validation and remaining work

**295 Python tests and 90 JavaScript tests pass.** Fourteen new evaluation tests
cover abstention vs verification, complete combination truth, missing/swapped
ingredients/strengths, package claims with missing/wrong truth, negatives,
unreadable labels, provenance, percentile gates, unique-image denominators,
actual route with mocked OCR/real matcher, invalid upload limits, process failures
and watchdog termination with partial output. Existing web/API/mobile logic tests
remain passing. No phone hardware test was performed.

Changed/created files in this iteration: `research/medicine_eval.py`,
`tests/test_medicine_eval.py`, `research/README.md`, `research/datasets/README.md`,
this report, `research/datasets/medicine-v1/{README.md,manifest.json,latency-only.json,annotation-template.json,.gitignore,images/.gitkeep}`,
and the three result files listed above. No production files were edited.

**First optimization hypothesis, not implemented:** after obtaining independently
confirmed readable, unclear and combination crops, compare native crops with one
fixed modest OCR input-size cap (for example 960 pixels on the long side). OCR
is the measured dominant stage, so this is worth testing, but there is no current
evidence that it preserves small-print/strength recognition. Do not adopt it
without the same labeled inputs, false-VERIFIED checks and matched run conditions.
The immediate next action remains collecting/confirming the evaluation dataset.
