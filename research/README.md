# Visual Assistant research

**Question.** Can adaptive selection of computer-vision and language-model
components improve identification accuracy and latency compared with always
using a vision-language model?

**Phase 1 (this directory so far).** Measure the existing system: where the
time goes, and what data exists to judge accuracy. No new models, no routing
logic, no accuracy claims.

| File | Purpose |
|------|---------|
| `methodology.md` | Hypothesis, baselines, what each timing means, controls, metrics, limitations |
| `benchmark.py` | Runs each existing component on a directory of images and saves JSON + CSV |
| `results/` | Benchmark output, one timestamped pair per run |
| `datasets/README.md` | What labelled data exists (one image) and what must be collected |

Related code outside this directory: `web/timing.py` (server stage times in a
`Server-Timing` header) and `web/static/timing.js` (browser-side times).

## Reproduce the component benchmark

From the repository root, with the project's virtual environment:

```bash
.venv/bin/python -m research.benchmark --images vision --trials 10 --vlm-trials 3 --max-sides 0,1280,640
```

- `--images` is any directory of `.jpg`/`.png` files. `vision` currently holds
  the only real photograph (`advil_front.jpg`, untracked).
- The Qwen components need Ollama running with `qwen2.5vl:3b` already
  installed. If not, they are recorded as `unavailable`; nothing is downloaded.
- Add `--online` to include the openFDA product-research component.
- A run takes about four minutes on the reference machine.

## Collect browser timings from a phone

```bash
.venv/bin/python -m web.server --host 0.0.0.0 --cert web/certs/cert.pem --key web/certs/key.pem --timing-log research/results/client-timings-iphone.jsonl
```

Open the page with `?research=1` added to the address (for example
`https://<mac-name>.local:5443/?research=1`), use the app normally, then close
or background the page so the last batch is sent. Use a different log file name
for each device. Without `?research=1` nothing is uploaded; without
`--timing-log` the server refuses uploads.

On any device, `VATimings.summary()` in the browser console shows median, p90
and p95 for what has been recorded in that page.

## Baseline results (measured)

Run `results/benchmark-20261112-130744.json`: Apple M4 (10 cores), Python
3.9.6, torch 2.8.0, ultralytics 8.4.130, EasyOCR 1.7.2, commit `f50f75c` plus
uncommitted instrumentation. **One input image**, 1920x1080. Warm figures are
medians over 10 calls (3 for Qwen); p95 in brackets.

| Component | Load | Cold call | Warm median (p95) | Peak memory |
|-----------|------|-----------|-------------------|-------------|
| Image decode, 1920x1080 | - | 5 ms | 5 ms (11) | 70 MB |
| YOLO `yolo26n`, CPU, 1920x1080 | 0.8 s | 31 ms | 21 ms (25) | 280 MB |
| YOLO, input reduced to 640 px | 0.6 s | 36 ms | 21 ms (24) | 285 MB |
| Tracking, 10 synthetic boxes | - | <0.1 ms | 0.13 ms | 65 MB |
| Crop | - | <0.1 ms | <0.01 ms | 65 MB |
| **EasyOCR, CPU, whole 1920x1080 frame** | 3.0 s | 6.1 s | **5.4 s (6.1 s)** | **7.8 GB** |
| EasyOCR, reduced to 1280 px | 3.1 s | 1.3 s | 1.3 s (1.5 s) | 5.4 GB |
| EasyOCR, reduced to 640 px | 3.2 s | 0.41 s | 0.37 s (0.39 s) | 1.9 GB |
| ProductMatcher, one fixed OCR reading | 1.7 s | 14 ms | 14 ms (22) | 445 MB |
| Barcode decode, 1920x1080 (no barcode in image) | - | 23 ms | 22 ms | 72 MB |
| Barcode FDA + registry lookup, 2 UPCs | 0.9 s | 0.2 ms | 0.03 ms | 398 MB |
| `/api/detect` in-process, 1920x1080 upload | 5.0 s | 48 ms | 29 ms (31) | 1.36 GB |
| `/api/barcode` in-process | 4.7 s | 35 ms | 29 ms (29) | 1.33 GB |
| **Qwen scene, as deployed** | - | 10.4 s | **9.0 s (9.0 s)** | in Ollama |
| Qwen scene, model kept loaded, unseen image | - | 8.9 s | 7.8 s (7.9 s) | in Ollama |
| **Qwen label reader, as deployed** | - | 6.6 s | **7.9 s (8.4 s)** | in Ollama |

Stage breakdown of `/api/detect` (warm medians): decode 4.2 ms, YOLO 23.6 ms,
tracking <0.05 ms, JSON <0.05 ms, total 28.5 ms.

Where Qwen's time goes (Ollama's own counters, warm medians, as deployed):
loading the model 1.3 s, processing the image and prompt 6.5 s, generating 36
output tokens 1.0 s.

No failures and no timeouts occurred in this run.

### Smaller measurements, also real but less controlled

- **OCR reading vs resolution, one image.** At 1920, 1280, 960 and 640 px wide,
  EasyOCR read the brand, ingredient and strength of the one labelled image
  every time, and the matcher gave the same answer (UNCERTAIN, 7 formulations,
  true one ranked first). This is one bottle with large print; it says nothing
  about small print.
- **Qwen and repeated images.** With the model kept loaded, a second request
  for an image it had just seen took 1.1 s instead of 7.2 s. Ollama reuses its
  work for an identical image; a new image pays the full cost.
- **Browser side, desktop simulation.** A desktop browser on the same Mac
  (loopback, phone-size window, a canvas standing in for the camera), 27
  awareness frames: capture and JPEG encode 12 ms median, round trip 41 ms,
  of which 36 ms was the server, leaving about 5 ms of transfer; rendering
  0.5 ms; whole cycle 54 ms. One spoken announcement started 4 ms after its
  response. The first request of the session took 423 ms because the
  connection was being set up. **These are not phone numbers.**

### Not measured

- Anything on a real iPhone or Android phone, over Wi-Fi, with TLS.
- The OCR path of `/api/identify` through the server: YOLO finds no bottle in
  the one image at the deployed threshold, so that endpoint returned
  `NO_BOTTLE` on every trial and never ran OCR.
- Product research latency (needs `--online`; not run).
- Any accuracy figure.

## What the measurements say about bottlenecks

| Suspect | Finding |
|---------|---------|
| Model inference | **Yes, for OCR and Qwen.** They are 200-400 times slower than YOLO. Everything else is under 30 ms. |
| Model loading | **Partly.** Qwen is unloaded after every call (`keep_alive: 0`), which costs about 1.3 s of each 9 s request. Server start-up loads everything once in about 5 s. |
| Image resolution | **Yes for OCR, no for YOLO.** OCR time falls 15-fold from 1920 to 640 px; YOLO is flat because it resizes internally. |
| OCR preprocessing | Not measured in this phase. |
| Network transfer | **Unknown.** About 5 ms on loopback; the phone-over-Wi-Fi figure has not been measured. |
| Sequential processing | **Small today.** In `/api/detect` the stages sum to the total. For identification, YOLO then OCR run one after the other, but OCR dominates. |
| Browser rendering | **No.** About 0.5 ms of script time per awareness frame on desktop. |
| Speech initiation | **Unknown on phones.** 4 ms on desktop, one sample. |

## Recommended next experiment

Collect the labelled set described in `datasets/README.md` first; without it
neither half of the research question can be answered. Then, on that set:

1. Run OCR on the YOLO bottle crop at several resolutions and record both time
   and whether brand, ingredient and strength were read. This tests whether the
   15-fold OCR speed-up survives small print.
2. Run Qwen with the model kept loaded and with a smaller image, recording the
   same quality counts, as the always-VLM baseline.
3. Record phone timings with `?research=1` on one iPhone and one Android phone
   to learn what transfer and speech start really cost.

## Deployed medicine-crop evaluation

The separate [medicine endpoint baseline](MEDICINE_ENDPOINT_BASELINE.md) exercises
`/api/medicine/identify-crop` through the unchanged Flask route, including decode,
full-region crop, EasyOCR, matching, presentation and JSON. Its
[40-image collection structure](datasets/medicine-v1/README.md) separates confirmed
annotations from explicitly permitted latency-only inputs. Run it with
`python -m research.medicine_eval`; no production behavior is changed.

To collect annotations without editing JSON by hand, use
`python -m research.annotate_medicine add /path/to/image.jpg` and
`python -m research.annotate_medicine report`. See the
[local helper workflow](datasets/medicine-v1/README.md#local-annotation-helper)
for drafts, combination labels, privacy review, exact identity and physical-item grouping.

For automatic local preliminary annotations, use
`python -m research.annotate_medicine batch VA-photos/` followed by
`python -m research.annotate_medicine review`. See
[batch annotation and review](ANNOTATION_AUTOMATION.md) for high-resolution
originals, independent confirmation, duplicate review and split safeguards.
