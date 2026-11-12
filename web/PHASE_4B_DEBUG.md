# Phase 4B served-build investigation — November 11, 2026

## Findings and limits

The listener at investigation start was PID 53017, started at 15:08:20. Its cwd
was `/Users/ridayjana/Riday-Personal/school-work/visual-assistant`. Command:

```
/Applications/Xcode.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python -m web.server --host 0.0.0.0 --cert web/certs/cert.pem --key web/certs/key.pem
```

Loaded dependencies were from this repository's `.venv`. Its Python executable
is the underlying Xcode Python 3.9 executable used by that virtual environment.
The repository source and every fetched CSS/JS asset matched byte-for-byte.
The actual HTML already contained the compact `<details>` medicine card,
medicine-identification.js, current research integration and **Dismiss**.
The automatic crop endpoint returned its expected validation error, not 404.
There was no evidence of a stale backend at the time of inspection.

The reported **Discard image** screen differs from this served HTML. An older
Safari document/assets or a capture made before a server restart is consistent
with that observation, but the original iPhone page/request history was not
available. Do not claim the screenshot alone proves which cache/process was old.
No service-worker registration or service-worker code exists in this project.
The old HTML had no explicit no-store policy; most script URLs had no build
version, and Flask caches compiled templates in the running non-debug process.
Static responses previously had `no-cache` (revalidation), not `no-store`.

No narrow camera constraint was reproduced in the actual served DOM. At
375×667, 390×844, 430×932 and landscape 844×390, stage/video/overlay all occupied
the viewport. The video computed `object-fit: cover`, `position: absolute`, and
`max-width: none`; stage was fixed. Safe-area padding affects overlays, not the
video. No additional CSS overrides were added. Existing coverTransform/mapBox
and rotation/mirroring tests remain unchanged and pass. These desktop browser
viewport measurements do not establish actual iOS camera/Safari behavior.

## Changes (debugging only)

- `web/server.py`: startup fingerprint of Python/templates/CSS/JS, health/build
  response header, common template build value, no-store HTML/assets/API/health;
  safe crop dimensions/status/timing logs without OCR text or identifiers.
- `web/templates/index.html`: every CSS/JS URL carries the same build; More panel
  shows build/loading state and camera layout measurements.
- `web/static/app.js`: confirms page/backend build agreement; reports measured
  viewport, stage, video, source size and object-fit; logs identification events;
  replaces misleading permanent “not identified” crop caption with “captured
  label image”. Camera/video resize updates the measurements.
- `web/static/medicine-identification.js`: diagnostics for request start, HTTP
  status, cancellation, stale identifiers and rendering. Missing sessions and
  mismatched identifiers now display recovery messages rather than silently
  stopping. This is a concrete error-reporting gap found in the code, not proof
  that it caused the original screenshot.
- `tests/test_web_build.py`: actual Flask HTML/assets/build/cache contract test.
- `tests/js/medicine-identification.test.js`: missing-session and stale-response
  diagnostics regressions.
- This report.

No CSS, YOLO, ProductMatcher, barcode, OCR algorithms or Caden files were changed.
No commits or pushes. No webcam opened. No new recognition/research features.

## Running server

Only the confirmed Visual Assistant listener was stopped/restarted. Final PID:
**56618**, launched explicitly with `.venv/bin/python -u -m web.server` and the
same host/certificate/key arguments and cwd. Final build:
**4b-debug-60d3788c53**. Output: `/tmp/visual-assistant-5443-debug.log`.
The replacement runs independently of the old terminal; do not start another
server on 5443 while this one is listening.

The certificate SAN covers localhost, 127.0.0.1,
Microadmins-MacBook-Pro-2.local and 192.168.40.116, but **not 10.129.100.236**.
Browser inspection therefore used the same listener through valid localhost.
Certificate files/trust were not altered. If Safari reports a certificate error,
use an accessible hostname already covered by your trusted certificate or renew
that certificate for the current address; don't bypass a security warning.

## Automatic identification and research validation

1. Downloaded the actual served JS/CSS and compared each with source.
2. Ran 27 capture/identification/research tests against downloaded assets,
   including stable detections -> detailed crop -> one automatic POST -> result.
3. Posted an in-memory crop from the existing `vision/advil_front.jpg` fixture
   to the live HTTPS `/api/medicine/identify-crop` endpoint. No new camera capture.
4. Received HTTP 200, UNCERTAIN, matching echoed capture/session IDs:
   OCR 1507.94 ms; matcher 12.88 ms; backend total 1521.40 ms; round trip 1.56 s.
5. Replayed that real response through the downloaded app integration test and
   asserted that the medicine card renders and includes **Search for product
   information**. Existing tests also cover NO_MATCH/manual/research entry points.

The UI request log now distinguishes “POST ... started”, “response HTTP ...”,
“response discarded: ...”, and “result rendered: ...”. Source photos, OCR text,
barcodes, product names and capture/session identifiers are not added to these
new diagnostic logs. The existing bounded activity log holds client messages.

Full suites: **257 Python unittest cases and 67 JavaScript tests pass**.
The live API and desktop served-page checks are not a real iPhone validation.
We cannot establish whether the old iPhone session sent a request or discarded
its result without its new activity log.

## Verify on the iPhone

1. Close the old Visual Assistant tab. Open your existing HTTPS address with
   `/?check=4b-debug-60d3788c53` appended, then open **More**. If the address shows
   a certificate warning, use the certificate guidance above.
2. Require **Build 4b-debug-60d3788c53 · JavaScript ready · backend matches**.
   If the build is absent/different, this is not the verified page. Reload or
   open a fresh private tab before testing recognition.
3. Start the iPhone camera, enable automatic medicine scanning, then awareness.
   Hold one bottle steady. The summary should progress to **Reading medicine
   label…**, then a matcher status. Expand the compact medicine card.
4. For UNCERTAIN/NO_MATCH, find **Search for product information** inside the
   expanded card. No research should run automatically.
5. If stuck, open **More → Status and activity**. Record whether POST started,
   the HTTP status, and any cancellation/discard/render message. Also record the
   build and layout line. Do not share full OCR text or images to diagnose this.
6. If margins persist on this build, compare stage/video sizes to viewport and
   confirm `fit cover`; repeat after rotation. These numbers distinguish a
   layout constraint from image content or browser-specific presentation.

If the current-build iPhone still fails, its diagnostics are the missing evidence;
the original phone-specific root cause has not been conclusively reproduced.
