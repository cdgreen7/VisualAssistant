# Ask AI: local Qwen scene descriptions

The iPhone remains the camera. Ask AI captures one frame from the existing
Safari preview; it never opens a camera on the backend. YOLO, tracking,
barcode lookup, FDA catalog and ProductMatcher retain their existing roles.

## Model and architecture

The default is `qwen2.5vl:3b`, configurable with `QWEN_VISION_MODEL` before
starting Flask. On November 11, 2026, local Ollama reported a 3.2 GB model,
3.8B parameters, Q4_K_M quantization, and `completion` plus `vision`
capabilities. The separate `qwen2.5-coder:14b` model lacks vision support.
No model downloads are performed by the app.

```
iPhone preview -> one JPEG -> Flask /api/scene/describe
 -> image validation/resizing -> QwenSceneService -> localhost Ollama
 -> validated generic observations -> short scene text -> panel/shared Speaker
```

The service checks `/api/show` before sending the image to `/api/chat`.
It uses only `http://127.0.0.1:11434`, disables HTTP proxies and redirects,
and rejects remote/cloud models. The service owns a separate nonblocking
lock: simultaneous AI requests receive HTTP 409 rather than queueing. This
lock is per Flask process; run the supplied single-process threaded server.
YOLO retains its own service and lock.

Images must be JPEG or PNG with matching MIME type and decoded format,
at most 4 MiB and 12 megapixels. The browser captures at most 1024 pixels on
the longest side; the backend also resizes to 1024 and re-encodes to JPEG,
removing metadata. Ask AI multipart file streams stay in memory rather than
Werkzeug's usual temporary upload files. No application image files or model
responses are logged or saved. API responses use `Cache-Control: no-store`.
The existing overall multipart request ceiling remains 10 MiB.

Ollama requests have a shared 60-second timeout budget; the browser aborts
after 75 seconds. Responses are bounded to 64 KiB and 256 generated tokens.
The model uses a 4096-token context and `keep_alive: 0` to release model
resources after completion. This favors memory availability over warm-request
latency. A browser abort cannot guarantee cancellation of inference inside
Ollama; a later request may receive a busy response until backend work ends.

## Output and speech

Qwen produces up to five structured observations: generic category, color,
and image left/center/right. Python validates every field against fixed
vocabularies and formats the result into short sentences. Arbitrary model text
is never displayed or spoken. This intentionally limits scene richness to
exclude product identities, ingredients, dosage, medical recommendations,
distance estimates, and safety-to-approach claims. Qwen does not modify or
verify any product-matcher result. Objects and positions can still be wrong.

The closable panel labels its result **AI-generated scene description** and
says product identity is not verified. Opening the panel cancels current
speech and pauses environmental announcements while visual awareness keeps
running. Closing it resumes environmental announcements. The shared Speaker
never queues speech. With voice enabled, the result attempts automatic
speech; **Read aloud** provides a direct user gesture for iOS Safari when
asynchronous speech is blocked. **Stop speech**, **Disable voice**, closing
the panel, stopping the camera, and hiding the page cancel speech.

## API examples

From the repository root, with Flask running using the existing certificate:

```bash
curl --cacert "$(mkcert -CAROOT)/rootCA.pem" \
  -F 'image=@vision/advil_front.jpg;type=image/jpeg' \
  https://localhost:5443/api/scene/describe
```

Actual successful response from the existing `vision/advil_front.jpg` photo
(timing varies by hardware and scene):

```json
{
  "description": "The image appears to show: white bottle in the center of the image.",
  "model": "qwen2.5vl:3b",
  "processing_time_ms": 11128,
  "label": "AI-generated scene description",
  "product_identity_verified": false
}
```

Errors retain the existing envelope:

```json
{"error":{"code":"model_unavailable","message":"Cannot reach local Ollama. Start Ollama on the Mac."}}
```

| HTTP | Error code | Meaning |
|---|---|---|
| 400 | `missing_image` / `invalid_image` | Missing, unreadable, or unsupported upload |
| 413 | `image_too_large` | Byte, pixel, or request limit exceeded |
| 409 | `scene_busy` | Another Ask AI request is running |
| 503 | `model_unavailable` | Ollama/model unavailable |
| 503 | `model_not_vision` / `model_not_local` | Unsuitable configured model |
| 504 | `model_timeout` | Local model request timed out |
| 502 | `invalid_model_response` | Invalid, incomplete, or disallowed model output |

## Test using the iPhone

1. Keep Ollama running on the Mac. Confirm `ollama list` includes
   `qwen2.5vl:3b`. If needed, start the Ollama app; `ollama serve` is an
   alternative when no daemon is already running.
2. From the repository root, start or restart Flask:

   ```bash
   QWEN_VISION_MODEL=qwen2.5vl:3b .venv/bin/python -m web.server \
     --host 0.0.0.0 --cert web/certs/cert.pem --key web/certs/key.pem
   ```

3. Put the iPhone and Mac on the same Wi-Fi. On the **iPhone**, open
   `https://Microadmins-MacBook-Pro-2.local:5443` in Safari. This hostname
   matches the current local certificate. Use the certificate trust setup in
   `web/README.md` if Safari does not already trust it. Reload an already-open
   page so it loads the updated controls.
4. Tap **Scan**, then **Start camera**, and allow the iPhone camera. Alternatively,
   **Start awareness** starts the preview and existing YOLO awareness.
5. Point at a simple scene and tap **Ask AI**. Expect a loading panel followed
   by a description while the preview stays visible. Repeated taps cannot
   start parallel AI requests.
6. Tap **Enable voice** before asking for automatic speech, or **Read aloud**
   after the result. Test **Stop speech** and **Close**. With awareness active,
   confirm visual detections continue and environmental speech resumes after
   closing the panel.
7. Try a bottle: the description may mention a generic bottle, color and
   position, but must not verify medicine identity or give dosing advice.
   Use the existing manual barcode/medicine controls for product matching.

## Files and automated checks

Created: `web/qwen_scene.py`, `web/static/ask-ai.js`,
`tests/test_qwen_scene.py`, `tests/js/ask-ai.test.js`, and this guide.
Modified: `web/server.py`, `web/static/app.js`, `web/static/style.css`,
`web/templates/index.html`, `tests/test_frontend_js.py`, `requirements.txt`
(explicit Pillow dependency), and `web/README.md`.
Existing unrelated working-tree changes are preserved.

Run `.venv/bin/python -m unittest discover tests`.
New tests mock Ollama and cover JPEG/PNG success, missing/invalid/mismatched
images, byte/pixel limits, resizing, metadata removal, unavailable models,
non-vision/cloud models, timeouts, malformed/unsafe responses, lock release,
and concurrent requests. JavaScript tests cover one-preview-frame capture,
loading/results, duplicate taps, late responses after close, error recovery,
voice off/on, speech cancellation, and environmental speech pause/resume.
The existing API, tracking, camera/voice wiring, and product tests also run.
These are automated simulated-browser checks, not physical iPhone tests.

## Limitations

This is a conservative scene-description experiment, not a navigation or
medical system. The fixed vocabulary omits many objects and all arbitrary
label text. A valid response can still hallucinate objects or colors. Inference
shares Mac memory/compute with YOLO, so awareness latency may rise even though
the service locks are independent. First load and model unloading add latency.
No simultaneous real YOLO/Qwen performance benchmark or physical iPhone Safari
test has been performed. Speech behavior still requires verification on the
actual iPhone.

## Validation results — November 11, 2026

- Hardware: Apple M4 MacBook Pro, 10 CPU cores, 24 GB unified memory.
- Full suite: **215 Python unittest cases passed**, including the Node suites
  invoked by `tests/test_frontend_js.py`. Existing warnings included unclosed
  fixture resources and deliberately malformed PNG inputs; there were no failures.
- A real image was submitted through Flask's test client to the actual
  `/api/scene/describe` route, using local Ollama rather than a mock. This tests
  the route, validation, resizing, model protocol and output formatting; it
  does not test Wi-Fi, TLS or physical iPhone Safari.
- Initial prompt returned an empty object list after 21.551 seconds. Providing
  an explicit JSON example and allowed category list in the prompt resolved
  that result on this image.
- Final HTTP 200 response: “The image appears to show: white bottle in the
  center of the image.” Model: `qwen2.5vl:3b`; backend processing: **11.128 s**.
  No brand, medicine identity or dosage was returned.
- Ollama timing: total 11.031 s, model load 3.447 s, prompt/image evaluation
  6.463 s, output generation 1.094 s (36 output tokens).
- Memory-pressure utility reported 39% system-wide free before inference and
  48% on the later check. These are system snapshots, not a controlled model
  memory benchmark. Ollama `/api/ps` was empty after completion, consistent
  with unloading. Peak model memory was not captured.
- No Mac webcam was opened, no model was downloaded by the implementation,
  and no commits or pushes were made.
