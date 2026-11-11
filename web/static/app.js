// Visual Assistant web client: camera preview, one still image per button press.
// No video is uploaded; all recognition runs on the Mac.
"use strict";

const REQUEST_TIMEOUT_MS = 90000;
const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), result = $("result");

let stream = null;
let facing = "environment";   // rear camera first
let busy = false;
let fallbackBarcode = null;   // barcode value kept on screen while the front label is scanned instead

// ---------- small DOM helpers (textContent only: server text is never parsed as HTML)

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function pill(text, tone) { return el("span", text, "pill" + (tone ? " " + tone : "")); }

function definitionList(pairs) {
  const dl = el("dl");
  for (const [term, value, mono] of pairs) {
    if (value === null || value === undefined || value === "") continue;
    dl.append(el("dt", term), el("dd", value, mono ? "mono" : ""));
  }
  return dl;
}

function log(message) {
  const item = el("li");
  item.append(el("time", new Date().toLocaleTimeString()), document.createTextNode(message));
  $("log").prepend(item);
}

function show(...nodes) { result.replaceChildren(...nodes); }
function showMessage(text, tone) { show(el("p", text, tone === "bad" ? "" : "muted")); }

// ---------- camera

function cameraError(error) {
  if (!window.isSecureContext) {
    return "This page is not on a secure (HTTPS) connection, so the browser blocks the camera. " +
           "Open the https:// address from the setup guide.";
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return "This browser does not support camera access.";
  switch (error && error.name) {
    case "NotAllowedError": return "Camera permission was denied. Allow camera access for this site in the browser settings, then try again.";
    case "NotFoundError": case "OverconstrainedError": return "No matching camera was found on this device.";
    case "NotReadableError": return "The camera is in use by another app. Close it and try again.";
    default: return "Could not start the camera: " + (error && error.message ? error.message : "unknown error");
  }
}

function stopCamera() {
  stopAwareness("camera stopped");
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  $("camera-message").hidden = false;
  $("camera-message").textContent = "Camera is off.";
  $("toggle-camera").textContent = "Start camera";
  updateButtons();
}

async function startCamera(deviceId) {
  stopCamera();
  const wanted = deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: facing } };
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("unsupported");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { ...wanted, width: { ideal: 1920 }, height: { ideal: 1080 } },
    });
  } catch (error) {
    const message = cameraError(error);
    $("camera-message").textContent = message;
    log("Camera error: " + message);
    updateButtons();
    return;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  $("camera-message").hidden = true;
  $("toggle-camera").textContent = "Stop camera";
  log("Camera started");
  updateButtons();
  listCameras();
}

async function listCameras() {
  const select = $("camera-select");
  try {
    const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const current = stream && stream.getVideoTracks()[0].getSettings().deviceId;
    select.replaceChildren(...cameras.map((camera, index) => {
      const option = el("option", camera.label || "Camera " + (index + 1));
      option.value = camera.deviceId;
      option.selected = camera.deviceId === current;
      return option;
    }));
    select.hidden = cameras.length < 2;
  } catch (error) {
    select.hidden = true;
  }
}

function updateButtons() {
  const on = Boolean(stream);
  // OCR and barcode scans are not run while awareness is on
  $("scan-barcode").disabled = $("identify").disabled = !on || busy || awareness.on;
  $("flip-camera").disabled = !on || busy;
  $("toggle-camera").disabled = busy;
  $("start-awareness").disabled = busy || awareness.on;
  $("stop-awareness").disabled = !awareness.on;
}

// ---------- capture and upload

// One JPEG still from the preview. maxSide shrinks the frame (keeping its shape) for awareness.
function captureBlob(maxSide, quality = 0.92) {
  if (!video.videoWidth) return Promise.reject(new Error("The camera has no picture yet. Wait a moment and try again."));
  const scale = maxSide ? Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight)) : 1;
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("Could not capture the image."))), "image/jpeg", quality));
}

async function send(path, label, render) {
  if (busy) return;
  busy = true;
  updateButtons();
  $("busy-text").textContent = label + "…";
  $("busy").hidden = false;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const started = performance.now();
  try {
    const blob = await captureBlob();
    const form = new FormData();
    form.append("image", blob, "capture.jpg");
    log(`${label}: sent ${canvas.width}×${canvas.height} image (${Math.round(blob.size / 1024)} KB)`);
    const response = await fetch(path, { method: "POST", body: form, signal: controller.signal });
    let body = null;
    try { body = await response.json(); } catch (error) { /* not JSON */ }
    if (!response.ok || !body) {
      throw new Error(body && body.error ? body.error.message : `Server error (${response.status})`);
    }
    render(body);
    log(`${label}: done in ${((performance.now() - started) / 1000).toFixed(1)} s`);
  } catch (error) {
    const message = error.name === "AbortError" ? "The server took too long to answer." :
      error instanceof TypeError ? "Could not reach the server. Check that it is running and on the same network." :
      error.message;
    showMessage(message, "bad");
    log(`${label} failed: ${message}`);
  } finally {
    clearTimeout(timer);
    busy = false;
    $("busy").hidden = true;
    updateButtons();
  }
}

// ---------- environmental awareness: one small frame at a time to /api/detect

const AWARENESS_INTERVAL_MS = 1500;   // pause between one answer and the next capture
const AWARENESS_TIMEOUT_MS = 15000;
const AWARENESS_MAX_SIDE = 640;       // YOLO works at 640 px, so larger uploads add nothing
const AWARENESS_MAX_FAILURES = 5;
const awareness = { on: false, controller: null, timer: null, failures: 0, last: null };
const overlay = $("overlay");

function setAwarenessState(text, tone) {
  $("awareness-state").textContent = text;
  $("awareness-state").className = "pill" + (tone ? " " + tone : "");
}

async function startAwareness() {
  if (awareness.on || busy) return;
  if (!stream) await startCamera();
  if (!stream) { setAwarenessState("camera unavailable", "bad"); return; }
  awareness.on = true;
  awareness.failures = 0;
  setAwarenessState("starting…");
  log("Awareness started");
  updateButtons();
  awarenessTick();
}

function stopAwareness(reason) {
  if (!awareness.on) return;
  awareness.on = false;
  clearTimeout(awareness.timer);
  if (awareness.controller) awareness.controller.abort();   // cancel the request in flight
  awareness.controller = awareness.last = null;
  drawOverlay();                                            // clear stale boxes
  $("awareness-list").replaceChildren();
  $("awareness-latency").textContent = "";
  setAwarenessState(reason ? "off – " + reason : "off", reason && reason !== "stopped" ? "warn" : "");
  log("Awareness stopped" + (reason ? ": " + reason : ""));
  updateButtons();
}

// Exactly one request at a time: the next capture is scheduled only after this one settles.
async function awarenessTick() {
  if (!awareness.on) return;
  const controller = awareness.controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, AWARENESS_TIMEOUT_MS);
  const started = performance.now();
  try {
    const form = new FormData();
    form.append("image", await captureBlob(AWARENESS_MAX_SIDE, 0.7), "frame.jpg");
    const response = await fetch("/api/detect", { method: "POST", body: form, signal: controller.signal });
    let body = null;
    try { body = await response.json(); } catch (error) { /* not JSON */ }
    if (!awareness.on || controller !== awareness.controller) return;
    if (!response.ok || !body) {
      const failure = new Error(body && body.error ? body.error.message : `Server error (${response.status})`);
      failure.code = body && body.error ? body.error.code : null;
      throw failure;
    }
    awareness.failures = 0;
    awareness.last = body;
    renderAwareness(body, Math.round(performance.now() - started));
  } catch (error) {
    if (!awareness.on || controller !== awareness.controller) return;   // stopped: the abort was ours
    awarenessFailed(error, timedOut);
  } finally {
    clearTimeout(timer);
  }
  if (awareness.on) awareness.timer = setTimeout(awarenessTick, AWARENESS_INTERVAL_MS);
}

function awarenessFailed(error, timedOut) {
  if (error.code === "busy") { setAwarenessState("server busy, retrying", "warn"); return; }   // not counted
  awareness.failures += 1;
  const message = timedOut ? "request timed out" :
    error instanceof TypeError ? "server unavailable" :
    error.code === "initialization_failed" ? "model failed to load" :
    error.code === "invalid_image" ? "frame was not a valid image" : error.message;
  awareness.last = null;
  drawOverlay();
  $("awareness-list").replaceChildren();
  if (error.code === "initialization_failed" || awareness.failures >= AWARENESS_MAX_FAILURES) {
    stopAwareness(message);
  } else {
    setAwarenessState(message + ", retrying", "bad");
  }
}

function renderAwareness(body, totalMs) {
  setAwarenessState(body.detection_count ? `${body.detection_count} object${body.detection_count === 1 ? "" : "s"}` : "no objects detected", "good");
  $("awareness-latency").textContent = `inference ${body.inference_time_ms} ms · round trip ${totalMs} ms · ${body.image_width}×${body.image_height}`;
  $("awareness-list").replaceChildren(...body.detections.map((d) => {
    const item = el("li");
    item.append(el("span", d.class_name, "name"), el("span", Math.round(d.confidence * 100) + "%", "percent"), pill(d.position));
    return item;
  }));
  drawOverlay();
}

// Where the video picture sits inside its box. The preview uses object-fit: cover, so the
// picture is scaled to fill the box and cropped equally on the overflowing sides.
function coverTransform(boxWidth, boxHeight, imageWidth, imageHeight) {
  const scale = Math.max(boxWidth / imageWidth, boxHeight / imageHeight);
  return { scale, dx: (boxWidth - imageWidth * scale) / 2, dy: (boxHeight - imageHeight * scale) / 2 };
}

function drawOverlay() {
  const ratio = window.devicePixelRatio || 1;
  const width = overlay.clientWidth, height = overlay.clientHeight;
  overlay.width = Math.round(width * ratio);
  overlay.height = Math.round(height * ratio);
  const context = overlay.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const body = awareness.last;
  if (!awareness.on || !body || !width) return;
  // The preview is never mirrored (front camera included), so image x maps straight to screen x.
  const t = coverTransform(width, height, body.image_width, body.image_height);
  context.lineWidth = 2;
  context.font = "600 13px -apple-system, sans-serif";
  context.textBaseline = "top";
  for (const d of body.detections) {
    const [x1, y1, x2, y2] = d.bounding_box;
    const x = x1 * t.scale + t.dx, y = y1 * t.scale + t.dy;
    context.strokeStyle = "#ffd60a";
    context.strokeRect(x, y, (x2 - x1) * t.scale, (y2 - y1) * t.scale);
    const label = `${d.class_name} ${Math.round(d.confidence * 100)}%`;
    const labelX = Math.max(0, x), labelY = Math.max(0, y);
    context.fillStyle = "rgba(0, 0, 0, 0.75)";
    context.fillRect(labelX, labelY, context.measureText(label).width + 8, 18);
    context.fillStyle = "#ffd60a";
    context.fillText(label, labelX + 4, labelY + 3);
  }
}

// ---------- rendering

const TONES = {
  IDENTIFIED_FROM_FDA: "good", IDENTIFIED_FROM_REGISTRY: "good", IDENTIFIED_FROM_BOTH: "good", VERIFIED: "good",
  CANDIDATE: "warn", AMBIGUOUS: "warn", UNCERTAIN: "warn", PENDING: "warn",
  CONFLICT: "bad",
};

function ingredientText(ingredients) {
  return ingredients.map((i) => `${i.name} ${i.strength}`).join(" + ");
}

function recordText(record) {
  return `${record.brand_name} (${record.labeler_name}) – ${ingredientText(record.ingredients)} [NDC ${record.product_ndc}]`;
}

function renderBarcode(body) {
  if (body.status === "NO_BARCODE") { showMessage(body.message); log("Barcode: none detected"); return; }
  const cards = body.barcodes.map((b) => {
    const card = el("div", null, "card");
    const title = el("h3");
    title.append(pill(b.final_status, TONES[b.final_status]));
    card.append(title, el("p", b.reason, "reason"));
    card.append(definitionList([
      ["Barcode", b.barcode.type.replace(/_/g, "-")],
      ["Value", b.barcode.decoded ? b.barcode.value : `not decoded (${b.barcode.reason})`, true],
      ["Product", b.product_name || "Not identified"],
      ["Ingredients", b.ingredients.length ? ingredientText(b.ingredients) : "Not established"],
      ["Package NDC", b.package_ndc ? `${b.package_ndc} (from ${b.package_ndc_source})` : null, true],
      ["Source", b.source || "None"],
      ["FDA lookup", `${b.fda.status} – ${b.fda.reason}`],
      ["Local registry", b.registry.status + (b.registry.source ? ` – ${b.registry.source}, verified ${b.registry.date_verified}` : "")],
    ]));
    const listed = b.fda.record ? [] : b.fda.label_records.map(recordText)
      .concat(b.fda.ndc_matches.map((r) => `${recordText(r)}, package ${r.package_ndc}`));
    if (listed.length) {
      const list = el("ul", null, "small");
      listed.forEach((text) => list.append(el("li", text)));
      const more = b.fda.label_record_count - b.fda.label_records.length;
      if (more > 0) list.append(el("li", `… ${more} more FDA records`, "muted"));
      card.append(el("p", "FDA records for this barcode:", "muted small reason"), list);
    }
    if (b.next_step === "scan_front_label") {
      const button = el("button", "Scan front label", "primary block");
      button.type = "button";
      button.addEventListener("click", () => startFrontLabelScan(b.barcode.value));
      title.after(el("p", b.message, "notice"), button);
    }
    log(`Barcode ${b.barcode.value || b.barcode.type}: ${b.final_status}`);
    return card;
  });
  if (body.barcodes_conflict) cards.push(el("p", "The barcodes in this image name different FDA records: CONFLICT", "reason"));
  show(...cards);
}

// ---------- front-label fallback after an unresolved barcode

function barcodeNote() {
  return fallbackBarcode ? [definitionList([["Barcode scanned", fallbackBarcode, true]])] : [];
}

function captureFrontLabelButton(text) {
  const button = el("button", text, "primary block");
  button.type = "button";
  button.addEventListener("click", captureFrontLabel);
  return button;
}

// Shows instructions only: nothing is captured until the user presses the capture button.
function startFrontLabelScan(barcodeValue) {
  fallbackBarcode = barcodeValue;
  const card = el("div", null, "card");
  card.append(el("h3", "Scan the front label"),
              el("p", "Point the camera at the front of the medicine bottle. Keep the entire bottle visible, " +
                      "then tap Capture front label.", "reason"),
              ...barcodeNote(), captureFrontLabelButton("Capture front label"));
  show(card);
  log("Front label scan started for barcode " + barcodeValue);
  if (!stream) startCamera();
  $("video").scrollIntoView({ behavior: "smooth", block: "start" });
}

function captureFrontLabel() {
  if (!stream) { startCamera(); return; }
  send("/api/identify", "Identification", renderIdentify);
}

function renderIdentify(body) {
  if (body.status === "NO_BOTTLE") {
    const seen = body.detections_seen.map((d) => `${d.class_name} ${d.confidence.toFixed(2)}`).join(", ");
    const nodes = [el("p", body.message + " Move the camera back and keep the entire bottle visible, then try again.", "notice"),
                   el("p", seen ? "Objects seen: " + seen : "No objects were detected.", "muted small")];
    if (fallbackBarcode) nodes.push(...barcodeNote(), captureFrontLabelButton("Capture front label again"));
    show(...nodes);
    log("Identify: no bottle detected");
    return;
  }
  const match = body.match, card = el("div", null, "card");
  const title = el("h3");
  title.append(pill(match.status + (match.status === "VERIFIED" ? " / " + match.level : ""), TONES[match.status]));
  card.append(title, el("p", match.status === "NO_MATCH" ? "No supported product identity found." : match.reason, "reason"));
  if (match.status === "UNCERTAIN") {
    card.append(el("p", "This is not an exact identification. The label text fits the possibilities listed below.", "notice"));
  }
  card.append(...barcodeNote());
  const d = body.detection;
  card.append(definitionList([
    ["Detected", `${d.class_name} ${d.confidence.toFixed(2)}, ${d.position}`],
    ["Package", match.package ? `NDC ${match.package.package_ndc} – ${match.package.description || ""}` : null],
  ]));

  if (match.candidates.length) {
    card.append(el("p", match.status === "VERIFIED" ? "Product:" : `Possible products (${match.candidate_count}):`, "muted small reason"));
    const list = el("ul");
    match.candidates.forEach((c) => {
      let text = ingredientText(c.ingredients);
      if (c.brand_names.length) text += ` — ${c.brand_names.join(", ")}`;
      if (c.not_observed.length) text += ` (not observed: ${c.not_observed.join(", ")})`;
      list.append(el("li", text));
    });
    if (match.candidate_count > match.candidates.length) {
      list.append(el("li", `… ${match.candidate_count - match.candidates.length} more`, "muted"));
    }
    card.append(list);
  }

  const ocrCard = el("div", null, "card");
  ocrCard.append(el("h3", "Text read from the label"));
  if (body.ocr.length) {
    const table = el("table");
    body.ocr.forEach((o) => {
      const row = el("tr");
      row.append(el("td", o.text), el("td", o.confidence.toFixed(2), "num"));
      table.append(row);
    });
    ocrCard.append(table);
  } else {
    ocrCard.append(el("p", "No text was read.", "muted"));
  }
  if (fallbackBarcode) card.append(captureFrontLabelButton("Capture front label again"));
  show(card, ocrCard);
  log(`Identify: ${match.status}, ${match.candidate_count} candidate(s), ${body.ocr.length} text item(s)`);
}

// ---------- wiring

$("toggle-camera").addEventListener("click", () => (stream ? (stopCamera(), log("Camera stopped")) : startCamera()));
$("flip-camera").addEventListener("click", () => { facing = facing === "environment" ? "user" : "environment"; startCamera(); });
$("camera-select").addEventListener("change", (event) => startCamera(event.target.value));
$("scan-barcode").addEventListener("click", () => { fallbackBarcode = null; send("/api/barcode", "Barcode scan", renderBarcode); });
$("identify").addEventListener("click", () => { fallbackBarcode = null; send("/api/identify", "Identification", renderIdentify); });
$("start-awareness").addEventListener("click", startAwareness);
$("stop-awareness").addEventListener("click", () => stopAwareness("stopped"));
window.addEventListener("resize", drawOverlay);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) return;
  stopAwareness("page inactive");
  if (stream && !busy) { stopCamera(); log("Camera stopped (page hidden)"); }
});
window.addEventListener("pagehide", () => stopAwareness("page inactive"));

fetch("/health").then((r) => r.json()).then((health) => {
  $("server-state").textContent = "server ready";
  $("server-state").className = "pill good";
  if (health.registry_error) log("Local UPC registry not loaded: " + health.registry_error);
}).catch(() => {
  $("server-state").textContent = "server unreachable";
  $("server-state").className = "pill bad";
});
if (!window.isSecureContext) $("camera-message").textContent = cameraError();
