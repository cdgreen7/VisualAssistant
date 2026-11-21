# Methodology

## Research question

Can adaptive selection of computer-vision and language-model components improve
identification accuracy and latency compared with always using a
vision-language model (VLM)?

## Initial hypothesis

For the tasks this assistant performs, a router that sends each request to the
cheapest component able to answer it (detector, barcode decoder, OCR + catalog
matcher) and calls the VLM only when those cannot, will be

1. several times faster at the median than calling the VLM for every request, and
2. no less accurate on product identity, with fewer unsupported claims,
   because the matcher and barcode lookup answer from FDA records and can say
   "uncertain", while a VLM answers from the picture alone.

This is a hypothesis. Phase 1 measures latency only; it does not test accuracy.

## Baseline systems

| ID | System | Status in this repository |
|----|--------|---------------------------|
| B0 | Always-VLM: every request goes to Qwen (`qwen2.5vl:3b` through Ollama) | Component exists (`web/qwen_scene.py`, `web/qwen_label.py`). The scene prompt is deliberately restricted to generic categories, so as deployed it cannot name a product. An always-VLM identification baseline needs its own prompt and is not built yet. |
| B1 | Classical only: YOLO -> crop -> EasyOCR -> ProductMatcher, plus barcode -> FDA lookup -> UPC registry | Exists and is what `/api/identify` and `/api/barcode` run. |
| B2 | Fixed hybrid as deployed: B1, with Qwen as an optional "second reader" whose output never reaches the matcher | Exists (`/api/medicine/identify-crop` with `assist=true`). |
| A1 | Adaptive selection (the proposal) | Not built. Out of scope for Phase 1. |

## What is measured, and where

All times are milliseconds from monotonic clocks (`time.perf_counter` /
`time.monotonic` on the server, `performance.now()` in the browser).

**Component level** (`research/benchmark.py`): each component runs in its own
process on images from a directory.

- `load_ms`: constructing the component (model load) in a fresh process.
- `cold_ms`: the first call after loading.
- warm: every later call; reported as median, p90, p95, min, max, n.
- `peak_rss_mb`: peak resident memory of that process. For Qwen the model lives
  in the Ollama process, so this number does not include it.
- failures (with the error) and timeouts (per component process) are recorded.

**Server level** (`web/timing.py`): every HTTP response carries a
`Server-Timing` header with the stages that ran:
`decode`, `yolo`, `tracking`, `identify` (select + crop + OCR + match),
`ocr`, `match`, `barcode`, `qwen`, `research`, `json`, and `total`.
Crop/selection time is `identify - ocr - match`.
`total` starts when Flask begins handling the request, after the upload has
been received, and ends when the response object is built.

**Client level** (`web/static/timing.js`): per request the browser records

| Field | Meaning |
|-------|---------|
| `prepare_ms` | camera frame -> JPEG blob |
| `roundtrip_ms` | `fetch()` called -> response headers received |
| `parse_ms` | headers -> JSON parsed (response available) |
| `render_ms` | response available -> page updated (script time, not paint) |
| `total_ms` | start of capture -> page updated |
| `server.*` | the server's own stages, copied from `Server-Timing` |
| `transfer_ms` | `roundtrip_ms - server.total` |
| `speech_ms` | response available -> the speech engine's `start` event |

`transfer_ms` is an **estimate by subtraction**. It contains upload, download,
TLS, connection setup, the server's upload-receiving time and browser
scheduling; these cannot be separated from inside a web page.

Server times alone are not end-to-end latency. End-to-end is the client's
`total_ms` (plus `speech_ms` for spoken results).

## Client identification

The browser labels itself from its User-Agent (`iphone-safari`,
`android-chrome`, `desktop-chrome`, ...), and the server labels the same
request independently (`client_from_server`). Both are stored with each
uploaded record so they can be cross-checked. A desktop browser emulating a
phone reports the phone's label; real-device runs must be noted by hand in the
results file name.

## Experimental controls

- One machine, one component per process, nothing else heavy running.
- The cold call is reported separately and never mixed into warm statistics.
- Inputs are files in a directory; the same files are used for every component
  and every run. Image names, sizes and byte counts are saved with the results.
- The resolution sweep (`--max-sides`) resizes the same images with
  `cv2.INTER_AREA`; it never enlarges.
- Model names, thresholds, device, library versions, git commit and whether the
  working tree had uncommitted changes are saved with every run.
- Qwen is run exactly as deployed (`keep_alive: 0`), and separately with the
  model kept loaded. In the second variant the image is shifted by a few
  pixels each call, because Ollama reuses its work for an image it has just
  seen, which would otherwise understate the time by about six seconds.
- Offline by default. The openFDA research component runs only with `--online`
  and clears its cache before each call so the network is what is timed.
- No model is downloaded. If Ollama or the model is missing, the Qwen
  components are recorded as `unavailable`.

## Evaluation metrics

Latency: median, p90, p95 per component and per stage; cold and load times;
failure and timeout counts; peak memory.

Quality (to be measured once a labelled dataset exists; see
`datasets/README.md`), each as a count out of n with the level stated:

1. **General object detection**: the labelled object class is among the detections.
2. **Specific object identification**: the correct kind of product is named.
3. **Medicine formulation**: the true set of active ingredients and strengths
   is the single reported formulation (for the matcher: `VERIFIED/FORMULATION`,
   or rank 1 among candidates, reported separately).
4. **Exact product**: the correct package (NDC or UPC).
5. **Unsupported claims**: any stated brand, ingredient, strength or package
   that is not on the label. For the matcher this is a false `VERIFIED`; for a
   VLM it is a hallucinated fact.

Abstention ("uncertain", "no match", "cannot tell") is counted separately from
errors: a system that abstains is not wrong, but it is not right either.

## Known limitations

- **One image.** The only real photograph in the repository is
  `vision/advil_front.jpg` (1920x1080). Latency numbers describe the components
  on this one input; they are not a distribution over real use. YOLO finds no
  bottle in it at the deployed 0.5 threshold, so the `/api/identify` benchmark
  measures only the no-bottle path and never reaches OCR.
- **No real-phone measurements yet.** Browser-side numbers so far come from a
  desktop browser on the same Mac as the server (loopback, no Wi-Fi, no TLS).
- **Cold is not disk-cold.** The operating system's file cache is warm after
  the first run, so `load_ms` understates a first-ever start.
- **`transfer_ms` is derived**, as described above.
- **`render_ms` is script time**, not time to pixels.
- **`speech_ms`** relies on the browser's speech `start` event, whose timing
  accuracy on iOS is unverified.
- **Single machine** (Apple M4, 10 cores). YOLO and EasyOCR run on the CPU.
- **Qwen memory** is not captured (it is in the Ollama process).
- **The fixed OCR reading and fixed UPCs** used for the matcher and lookup
  benchmarks are inputs chosen for repeatability, not samples of real use.
