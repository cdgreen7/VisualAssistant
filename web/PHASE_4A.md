# Phase 4A — automatic medicine candidate image preparation

This phase detects a possible container and prepares images only. A YOLO
`bottle` is not evidence that something contains medicine. No automatic OCR,
product matching, verification, medical advice, Qwen call, image upload, or new
server endpoint is introduced. The feature is disabled on every page load.

## Existing integration points

- `/api/detect` decodes the browser JPEG and runs the existing `VisionServices`
  detector. Its response boxes refer to uploaded image pixels. LEFT/CENTER/RIGHT
  is computed independently from those boxes.
- `TrackSessions` keeps one `ObjectTracker` per awareness session. IDs are
  associated by same-class box overlap/center proximity; ambiguous matches can
  have no ID. IDs expire and are not permanent object identities. Repeated or
  old frame numbers return `tracking: stale`.
- `awarenessTick()` captures the phone preview, sends at most one 640-pixel
  frame at a time, then pauses 1.5 seconds before the next request.
- Manual `/api/identify` runs YOLO on its uploaded frame, selects the
  highest-confidence bottle using `select_best_detection()`, clamps/crops
  that box using `crop_box()`, then calls EasyOCR and ProductMatcher.
- `AnnouncementManager` selects environmental phrases after repeated sightings;
  the existing `Speaker` never builds a speech queue. Phase 4A adds no speech.
- Ask AI captures a separate resized preview JPEG and sends it to
  `/api/scene/describe`; `QwenSceneService` checks local model capabilities and
  sends base64 image data to local Ollama. Phase 4A does not involve Qwen.

## Candidate algorithm

`medicine-candidates.js` contains a pure, independently testable `CandidateGate`.
Defaults are configurable through the constructor / `DEFAULTS`:

| Rule | Default |
|---|---|
| Generic class | `bottle` only |
| Number of bottles in response | Exactly one; ambiguous scenes are skipped |
| Confidence | At least 0.65 |
| Box size | At least 2.5% of full frame area; valid in-frame coordinates |
| Consecutive tracked sightings | 3 |
| Consecutive untracked sightings | 5 |
| Minimum stable duration | 2 seconds |
| Geometric stability | Normalized box IoU at least 0.65 between observations |
| Maximum interval between stable observations | 4.5 seconds |
| Departure / candidate forgetting | 5 seconds without a qualifying observation |
| Capture cooldown | 30 seconds globally, with per-ID cooldown memory |

A missing frame breaks the stability streak. A missing or changing tracking
ID restarts the streak; with missing IDs the gate falls back to the stricter
five-sighting spatial rule. Stale responses cannot trigger capture. A captured
region remains suppressed while continuously occupied, including through ID
changes and brief detection gaps. This prevents capture every time the cooldown
expires. If an object leaves, a later stable candidate may trigger after the
cooldown. The short per-ID history is pruned and bounded to 128 entries. Session
restart, camera stop/change, rotation, page hiding, a failed awareness request,
or disabling the setting clears state. The setting is not persisted in local storage.

Only one image preparation runs at a time. The app also suppresses preparation
while a manual request or Ask AI is busy. This phase issues **zero identification
requests**, so it cannot create concurrent automatic identification work.

## Image capture and coordinates

When enabled, `medicine-capture.js` freezes the existing video preview into a
private canvas **before** uploading the awareness frame. It then derives the
usual 640-pixel JPEG from that canvas. If that response completes a stable
candidate, the same frozen canvas is used to prepare the image and crop. There
is no assumption that a live frame captured after inference still matches an
old detection.

- Camera request remains the existing ideal 1920×1080 stream. Safari chooses
  actual resolution; no new camera or Mac webcam is opened.
- Full still: native video dimensions, downscaled only if the longest side
  exceeds **2560 px**. No upscaling, CSS crop, mirroring or stretching.
- Awareness: longest side **640 px**, JPEG quality **0.70**, unchanged.
- Prepared full frame and bottle crop: JPEG quality **0.92**. Each is capped
  at **8 MiB**, below the existing 10 MiB server limit for a later single-image
  request. Oversized encoding is rejected, never silently over-uploaded.
- A maximum frame is at most 6.56 MP, below the existing 25 MP server limit.
- Coordinates use `x * captureWidth / detectionWidth` and
  `y * captureHeight / detectionHeight`, with floor/ceil and clipping. The
  aspect ratios must match. The OCR crop has its own zero-based coordinates,
  just like the existing Python `crop_box()` output.
- Portrait dimensions are taken from `videoWidth/videoHeight`; Safari's
  oriented pixels are drawn directly. Rotation/video resize invalidates
  pending work, drops the saved packet and waits for fresh detections.

One packet is retained as `medicineScanner.packet`: full JPEG `image`, JPEG
`crop`, and `metadata`. Metadata includes original detection box/dimensions,
class/confidence, track ID or null, session/frame number, capture timestamp,
original video dimensions, prepared dimensions, transformed box, crop size,
and `product_identity_verified: false`. It is suitable input for a future OCR
stage, but Phase 4A does not run that stage or pass its box to `/api/identify`.
The existing manual endpoint continues to detect/crop its own uploaded image.

Frames/crops remain in browser memory. There are no disk writes or uploads.
There is at most one current awareness snapshot and one saved packet. Preview
object URLs are revoked when replaced/discarded. The packet is cleared when
the candidate leaves/is replaced, the setting is disabled, awareness stops,
the camera changes, the page hides, or orientation changes. Generation checks
reject late encoding results after reset. Capture failure falls back to normal
awareness rather than stopping YOLO. A failed candidate capture remains
suppressed until departure/reset, avoiding a retry loop.

## UI and iPhone test procedure

1. Start/restart the existing Flask server on the Mac:

   ```bash
   .venv/bin/python -m web.server --host 0.0.0.0 \
     --cert web/certs/cert.pem --key web/certs/key.pem
   ```

2. On the same Wi-Fi, open
   `https://Microadmins-MacBook-Pro-2.local:5443` in **iPhone Safari**. Reload
   to load the new scripts. Use the trusted certificate setup in `web/README.md`.
3. Open **More** and verify **Automatic medicine scanning (experimental)**
   is unchecked. Enable it and close the menu. The candidate status panel
   appears only while the feature is enabled. This setting does
   not enable the camera or awareness by itself.
4. Tap the **Camera** dock control to start awareness and grant access to the iPhone camera. Keep a single
   bottle large enough in view, label facing the camera, and hold steady for
   at least three completed detections (normally several seconds).
5. Expect **Possible medicine container detected**, then **Preparing label
   scan**, then **Image ready for identification**, with a crop thumbnail,
   frame/crop dimensions and tracking ID. No medicine identity should appear.
6. Leave the bottle in view beyond 30 seconds: it should not capture repeatedly.
   Remove it for at least five seconds / several detections. A new stable
   candidate may be captured after the cooldown. Try two bottles together:
   automatic capture should be skipped.
7. Rotate the phone: the previous crop should clear. Hold steady again and
   check that a new crop lines up with the bottle. Try **Discard image**,
   disable the setting, and stop/restart awareness. Reloading defaults to off.
8. Confirm voice, tracking overlays and Ask AI still work. Manual **Identify
   medicine** and **Capture barcode** retain their existing behavior: stop
   awareness before using them. Automatic candidate preparation never invokes
   either action.

## Files

Created:
- `web/static/medicine-candidates.js` — pure stability/cooldown policy.
- `web/static/medicine-capture.js` — frozen-frame capture, scaling and crop.
- `web/static/medicine-scanner.js` — single-work-item lifecycle coordinator.
- `tests/js/medicine.test.js` — policy/capture/lifecycle/app integration tests.
- `web/PHASE_4A.md` — this report.

Modified:
- `web/static/app.js` — awareness hook, setting, metadata preview and cleanup.
- `web/templates/index.html` — default-off setting, status, preview, scripts.
- `web/static/style.css` — compact status/preview styles.
- `tests/test_frontend_js.py` — run the new Node suite with Python tests.
- `tests/js/ask-ai.test.js` — provide new module globals in its simulated app.
- `web/README.md` — link to this guide.

No `vision/` modules, Flask APIs, medicine rules, Caden's files or existing
unrelated working-tree changes were edited for Phase 4A. Nothing was committed
or pushed.

## Tests and limitations

The new suite covers stable/single-frame/repeated detections, missing/changing
IDs, cooldown/departure, ambiguous scenes, movement, invalid boxes, disabled
state, concurrent work, native and portrait capture, size limits, coordinate
mapping, rotation/reset, late results, capture failures, metadata, and app
integration. The integration test confirms only `/api/detect` is used and
its image stays at 640 pixels, with no OCR/Qwen requests.

Physical iPhone Phase 4A testing is still outstanding. The user previously
confirmed Ask AI works on the iPhone; that does not validate this new feature.
Automated browser tests use canvas/video mocks, not an actual Safari camera.

Generic bottle detection will include drinks and cosmetics and miss medicine
boxes/blisters. Geometric stability is not a blur, focus, glare, label visibility
or OCR-readability assessment. The crop may be poor even when marked ready;
“ready” means prepared, not readable or identified. Native camera resolution
may be lower than requested. Multiple bottles are deliberately skipped. The
spatial fallback can confuse similar objects swapped into the same area.
The saved snapshot can be several seconds old by the time inference returns.
Freezing a detailed canvas every enabled awareness cycle adds phone memory/CPU
work; encoding full-resolution JPEGs happens only on a candidate trigger.
No automatic identification, verification, Qwen-based selection or medical
advice has been implemented (Phases 4B/4C remain future work).

Final validation: **216 Python unittest cases passed** using
`.venv/bin/python -m unittest discover tests`; **55 JavaScript tests passed**
using `node --test tests/js/*.test.js`, including 18 new Phase 4A cases.
JavaScript syntax checks and `git diff --check` passed. The concurrent UI
redesign's template compatibility assertion was updated by that work and
passed in the final run. The Phase 4A app integration harness loads the shared
UI module and verifies the redesigned app. Existing warnings concerned open
fixture resources and intentionally invalid PNG inputs, not test failures.
`pytest` was also attempted but is not installed in this virtual environment;
all project unittest tests and all Node test files were run successfully.
No physical iPhone Phase 4A test was run.
