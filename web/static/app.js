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
let serverOk = null;          // null until /health answers

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

// Scan results appear in a sheet over the camera, never in the page below it.
function show(...nodes) {
  result.replaceChildren(...nodes);
  sheets.open("result-sheet", document.activeElement);
}

// ---------- sheets: the More menu and the result sheet (one open at a time; see ui.js)

const sheets = new UI.Sheets({
  show(name) {
    $(name).hidden = false;
    if (name === "tools") {
      $("sheet-backdrop").hidden = false;
      $("more-tools").setAttribute("aria-expanded", "true");
    }
  },
  hide(name) {
    $(name).hidden = true;
    if (name === "tools") {
      $("sheet-backdrop").hidden = true;
      $("more-tools").setAttribute("aria-expanded", "false");
    }
  },
  focusFirst(name) {
    const first = $(name).querySelector("button:not([disabled])");
    if (first) first.focus({ preventScroll: true });
  },
  focus(element) { if (element && element.isConnected && element.focus) element.focus({ preventScroll: true }); },
});

// Keep Tab inside the open modal menu.
function trapFocus(event) {
  if (event.key !== "Tab" || sheets.current !== "tools") return;
  const items = [...$("tools").querySelectorAll("button:not([disabled]), select:not([hidden]), input, summary")]
    .filter((node) => node.offsetParent !== null);
  if (!items.length) return;
  const first = items[0], last = items[items.length - 1];
  if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
  else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
}
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
  askAI.close();
  stopAwareness("camera stopped");
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  $("camera-message").hidden = true;       // the welcome screen says the camera is off
  $("camera-message").textContent = "";
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
    $("camera-message").hidden = false;
    log("Camera error: " + message);
    updateButtons();
    return;
  }
  video.srcObject = stream;
  await video.play().catch(() => {});
  $("camera-message").hidden = true;
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
    $("camera-select-row").hidden = cameras.length < 2;
  } catch (error) {
    $("camera-select-row").hidden = true;
  }
}

// Everything the controls show is derived from this one description of the app (ui.js).
function appState() {
  return {
    cameraOn: Boolean(stream), awarenessOn: awareness.on, busy, askBusy: askAI.busy,
    voiceOn: speaker.enabled, voiceSupported: speaker.supported, serverOk,
    awarenessText: $("awareness-state").textContent, awarenessTone: $("awareness-state").className.replace("pill", "").trim(),
  };
}

function updateButtons() {
  const state = appState(), view = UI.controls(state), chip = UI.statusChip(state);
  $("stage").classList.toggle("is-off", view.welcomeVisible);
  $("status-text").textContent = chip.text;
  $("status-chip").className = "chip" + (chip.tone ? " " + chip.tone : "");

  $("awareness-toggle").setAttribute("aria-pressed", String(view.awareness.pressed));
  $("awareness-toggle").setAttribute("aria-label", view.awareness.ariaLabel);
  $("awareness-toggle").disabled = view.awareness.disabled;
  $("ask-ai").disabled = view.askAI.disabled;

  $("voice-toggle").setAttribute("aria-pressed", String(view.voice.pressed));
  $("voice-toggle").disabled = view.voice.disabled;
  $("voice-toggle-state").textContent = view.voice.state;
  $("flip-camera").disabled = view.switchCamera.disabled;
  $("flip-hint").textContent = view.switchCamera.hint;
  $("identify").disabled = view.identify.disabled;        // OCR and barcode scans never run during awareness
  $("identify-hint").textContent = view.identify.hint;
  $("scan-barcode").disabled = view.barcode.disabled;
  $("barcode-hint").textContent = view.barcode.hint;
  $("toggle-camera").disabled = view.cameraPower.disabled;
  $("toggle-camera-label").textContent = view.cameraPower.label;
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
const SCENE_MAX_ITEMS = 3;            // objects listed in the scene panel before "+N more"
const PREVIEW_MIRRORED = false;       // the preview is shown as the camera sees it, front camera included
const awareness = { on: false, controller: null, timer: null, failures: 0, last: null, session: null, frame: 0 };
const overlay = $("overlay");

// A fresh random id per awareness run: the server keeps track IDs per session, so a
// restart begins again at #1 and two phones never share IDs.
function newSessionId() {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

function forgetSession(sessionId) {
  if (!sessionId) return;
  const form = new FormData();
  form.append("session_id", sessionId);
  fetch("/api/detect/reset", { method: "POST", body: form, keepalive: true }).catch(() => {});
}

// ---------- voice announcements (speech runs in this browser; see announcer.js)

const announcements = new Announcer.AnnouncementManager();
const speaker = new Announcer.Speaker(window.speechSynthesis, window.SpeechSynthesisUtterance, (reason) => {
  setVoiceState("voice error: " + reason, "bad");      // visual detection carries on regardless
  log("Voice error: " + reason);
});

function setVoiceState(text, tone) {
  $("voice-state").textContent = text;
  $("voice-state").className = "pill" + (tone ? " " + tone : "");
}

function updateVoiceButtons() { updateButtons(); }   // the Voice tile and status chip follow the speaker

// Called from a tap, which is what lets iOS Safari play speech afterwards.
function enableVoice() {
  if (!speaker.supported) {
    setVoiceState("voice not supported in this browser", "bad");
    log("Voice: SpeechSynthesis is not available in this browser");
    return;
  }
  speaker.volume = Number($("voice-volume").value);
  speaker.enable("Voice announcements on.");
  setVoiceState("voice on", "good");
  $("voice-last").textContent = awareness.on ? "" : "Start awareness to hear objects.";
  log("Voice enabled");
  updateVoiceButtons();
}

function disableVoice() {
  speaker.disable();
  setVoiceState("voice off");
  $("voice-last").textContent = "";
  log("Voice disabled");
  updateVoiceButtons();
}

// One detection result in, at most one spoken phrase out.
function announce(detections) {
  const text = announcements.update(detections, performance.now(), speaker.canSpeak && !askAI.active);
  if (!text || !speaker.say(text)) return;
  $("voice-last").textContent = "Said: " + text;
  log("Voice: " + text);
}

function setAwarenessState(text, tone) {
  $("awareness-state").textContent = text;
  $("awareness-state").className = "pill" + (tone ? " " + tone : "");
  updateButtons();                                       // mirror it in the status chip
}

async function startAwareness() {
  if (awareness.on || busy) return;
  if (!stream) await startCamera();
  if (!stream) { setAwarenessState("camera unavailable", "bad"); return; }
  awareness.on = true;
  awareness.failures = 0;
  awareness.session = newSessionId();
  awareness.frame = 0;
  announcements.reset();
  sheets.close();                               // free the preview
  $("scene").hidden = false;
  renderScene([]);
  $("scene-summary").textContent = "Looking…";
  setAwarenessState("starting…");
  log("Awareness started");
  updateButtons();
  awarenessTick();
}

function stopAwareness(reason) {
  medicineScanner.invalidate();
  if (!awareness.on) return;
  awareness.on = false;
  clearTimeout(awareness.timer);
  if (awareness.controller) awareness.controller.abort();   // cancel the request in flight
  awareness.controller = awareness.last = null;
  forgetSession(awareness.session);                         // drop the server's track IDs for this run
  awareness.session = null;
  speaker.cancel();                                         // stop talking, nothing left queued
  announcements.reset();
  if (speaker.enabled) $("voice-last").textContent = "";
  drawOverlay();                                            // clear stale boxes
  $("scene").hidden = true;
  renderScene([]);
  $("awareness-latency").textContent = "Awareness is off.";
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
  const medicineGeneration = medicineScanner.generation;
  let snapshot = null;
  let captureError = null;
  try {
    const form = new FormData();
    if (medicineScanner.enabled && !medicineScanner.busy && !busy && !askAI.busy) {
      try { snapshot = await medicineCapture.freeze(video); }
      catch (error) {
        // Candidate capture must never take down ordinary YOLO awareness.
        captureError = error;
        medicineScanner.invalidate();
        $("medicine-status").textContent = "Capture unavailable: " + error.message;
      }
    }
    form.append("image", snapshot ? snapshot.detectionBlob : await captureBlob(AWARENESS_MAX_SIDE, 0.7), "frame.jpg");
    form.append("session_id", awareness.session);
    form.append("frame", String(++awareness.frame));
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
    if (!captureError && medicineGeneration === medicineScanner.generation) {
      await medicineScanner.observe({...body, session_id: awareness.session, frame: awareness.frame}, snapshot, performance.now(), !busy && !askAI.busy &&
        View.sameShape(body.image_width, body.image_height, video.videoWidth, video.videoHeight));
      if (medicineScanner.packet && !medicineScanner.busy) void medicineIdentification.identify(medicineScanner.packet);
    }
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
  renderScene([]);
  $("scene-summary").textContent = "";
  if (error.code === "initialization_failed" || awareness.failures >= AWARENESS_MAX_FAILURES) {
    stopAwareness(message);
  } else {
    setAwarenessState(message + ", retrying", "bad");
  }
}

function renderAwareness(body, totalMs) {
  setAwarenessState("Awareness on", "good");
  $("awareness-latency").textContent = `Awareness: inference ${body.inference_time_ms} ms · round trip ${totalMs} ms · ` +
    `${body.image_width}×${body.image_height} frame` + (body.tracking === "ok" ? "" : ` · tracking ${body.tracking || "off"}`);
  renderScene(body.detections);
  drawOverlay();
  announce(body.detections);
}

// The compact panel at the bottom-left of the camera: who/what is where.
function renderScene(detections) {
  const maxItems = $("stage").clientHeight < 420 ? 2 : SCENE_MAX_ITEMS;   // less room in landscape
  const scene = View.sceneLines(detections, maxItems);
  const items = scene.items.map((text) => el("li", text));
  if (scene.more) items.push(el("li", scene.more, "more"));
  $("awareness-list").replaceChildren(...items);
  $("scene-summary").textContent = scene.summary;
}

// Boxes are in the uploaded frame's pixels; View.mapBox places them on the preview,
// which is scaled and cropped by object-fit: cover (see view.js).
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
  // After a rotation the last result has the old orientation: wait for the next one.
  if (!View.sameShape(body.image_width, body.image_height, video.videoWidth, video.videoHeight)) return;
  context.lineWidth = 3;
  context.font = "700 14px -apple-system, sans-serif";
  context.textBaseline = "top";
  for (const d of body.detections) {
    const box = View.mapBox(d.bounding_box, body.image_width, body.image_height, width, height, PREVIEW_MIRRORED);
    context.strokeStyle = "#ffd60a";
    context.strokeRect(box.x, box.y, box.width, box.height);
    const label = `${View.objectLabel(d)} ${Math.round(d.confidence * 100)}%`;
    const labelWidth = context.measureText(label).width + 10;
    const labelX = Math.min(Math.max(0, box.x), Math.max(0, width - labelWidth)), labelY = Math.max(0, box.y);
    context.fillStyle = "rgba(0, 0, 0, 0.78)";
    context.fillRect(labelX, labelY, labelWidth, 20);
    context.fillStyle = "#ffd60a";
    context.fillText(label, labelX + 5, labelY + 3);
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

// Research is explicit and separate from matching. Only OCR observations and
// decoded identifiers leave the Mac; camera images and Qwen text are excluded.
let productResearchBusy = false;
function productResearchCard(body, upc = "") {
  const card = el("div", null, "card");
  const button = el("button", "Search for product information", "secondary block");
  button.type = "button";
  const output = el("div");
  card.append(button, el("p", "Optional: sends label evidence/barcode to openFDA. No photo is sent.", "small muted"), output);
  button.addEventListener("click", async () => {
    if (productResearchBusy) { output.textContent = "Another product lookup is running. Try again shortly."; return; }
    productResearchBusy = true;
    button.disabled = true;
    output.textContent = "Searching official product labels…";
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 30000);
    try {
      const response = await fetch("/api/products/research", {
        method: "POST", headers: {"Content-Type": "application/json"}, signal: controller.signal,
        body: JSON.stringify({upc, evidence: body.match?.evidence || [], ocr: body.ocr || []}),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error?.message || "Product lookup failed.");
      if (!card.isConnected) return; // a rescan or camera reset discarded this result
      const nodes = [el("p", result.message, "notice"),
        el("p", "Observed label text: " + (result.observed.label.join(" · ") || "None; barcode only"), "small"),
        el("p", `${result.total_records} official record(s) · ${result.processing_ms} ms${result.cached ? " · cached" : ""}`, "small muted")];
      if (result.search_basis === "label_evidence_after_identifier_not_found") nodes.push(el("p", "Exact identifier search found no label. These are approximate label-text candidates.", "notice"));
      if (result.truncated) nodes.push(el("p", "More records exist; this is an incomplete result set.", "notice"));
      if (!result.records.length) nodes.push(el("p", "No official label found. Barcode remains unresolved; try scanning the front label."));
      for (const record of result.records) {
        const details = el("details", null, "card");
        details.append(el("summary", `${record.brands.join(" / ") || "Official label"} — ${record.comparison}`));
        details.append(definitionList([
          ["Retrieved ingredient text", record.active_ingredient_text.join(" · ") || "Unavailable"],
          ["Retrieved purpose (not a recommendation)", record.purpose.join(" · ") || "Unavailable"],
          ["Identifier association", record.exact_upc ? "Exact UPC listed on label" : record.exact_ndc ? "Exact package NDC listed on label" : "Text search candidate only"],
          ["Exact package", record.exact_package],
          ["Conflicts", record.conflicts.join(" · ") || "None detected; not proof of identity"],
          ["Missing evidence", record.missing.join(" · ") || "None in this comparison"],
          ["Source ID", record.source_id],
          ["Label effective date", record.effective_time || "Unavailable"],
        ]));
        for (const [label, url] of [["openFDA source record", record.source_url], ["DailyMed label", record.dailymed_url]]) {
          if (!url || !/^https:\/\/(api\.fda\.gov|dailymed\.nlm\.nih\.gov)\//.test(url)) continue;
          const link = el("a", label); link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer";
          details.append(link, el("br"));
        }
        nodes.push(details);
      }
      output.replaceChildren(...nodes);
    } catch (error) {
      if (card.isConnected) output.textContent = error.name === "AbortError" ? "Lookup timed out. Try again." : error.message;
    } finally {
      clearTimeout(timeout); productResearchBusy = false; button.disabled = false;
    }
  });
  return card;
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
    if (b.final_status === "UNRESOLVED" && b.barcode.decoded) card.append(productResearchCard({}, b.barcode.value));
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
}

function captureFrontLabel() {
  if (!stream) { startCamera(); return; }
  if (awareness.on) stopAwareness("stopped");   // OCR does not run during awareness
  $("result-title").textContent = "Medicine result";
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
  if (match.status === "UNCERTAIN" || match.status === "NO_MATCH") card.append(productResearchCard(body, fallbackBarcode || ""));
  show(card, ocrCard);
  log(`Identify: ${match.status}, ${match.candidate_count} candidate(s), ${body.ocr.length} text item(s)`);
}

// AI uses the same phone preview and Speaker, with an independent request state.
const askAI = new AskAI({
  ready: () => Boolean(stream) && !busy,
  capture: captureBlob,
  fetch: (...args) => fetch(...args),
  speaker,
  visible: () => !document.hidden,
  update: updateButtons,
  voiceUpdate: updateVoiceButtons,
  render: (text, loading) => {
    sheets.close();                                        // the card replaces any open sheet
    $("ai-panel").hidden = false;
    $("stage").classList.add("ai-open");
    $("ai-text").textContent = loading ? "Understanding the scene…" : text;
    $("ai-spinner").hidden = !loading;
    $("ai-panel").setAttribute("aria-busy", String(loading));
    $("ai-read").disabled = loading || !askAI.text || !speaker.supported;
  },
  hide: () => {
    $("ai-panel").hidden = true;
    $("stage").classList.remove("ai-open");
  },
});

// Capture coordinator prepares the crop; awarenessTick starts the independent OCR request.
const medicineCapture = new MedicineCapture.MedicineCapture(() => document.createElement("canvas"));
let medicinePreviewURL = null;
const medicineScanner = new MedicineScanner(new MedicineCandidates.CandidateGate(), medicineCapture, {
  status: text => { $("medicine-status").textContent = text; },
  clear: () => {
    if (medicinePreviewURL) URL.revokeObjectURL(medicinePreviewURL);
    medicinePreviewURL = null;
    $("medicine-preview").removeAttribute("src");
    $("medicine-packet").hidden = true;
    medicineIdentification.cancel();
  },
  ready: packet => {
    medicinePreviewURL = URL.createObjectURL(packet.crop);
    $("medicine-preview").src = medicinePreviewURL;
    const m = packet.metadata;
    $("medicine-details").textContent = `${m.capture_width}×${m.capture_height} frame · ` +
      `${m.crop_width}×${m.crop_height} crop · ` + (m.track_id === null ? "untracked candidate" : `track #${m.track_id}`) +
      " · captured label image";
    $("medicine-packet").hidden = false;
  },
});
function resetMedicineOrientation() { medicineScanner.invalidate(); }
const medicineIdentification = new MedicineIdentification({
  diagnostic: text => log("Medicine: " + text),
  fetch: (...args) => fetch(...args),
  assist: () => $("qwen-label-assist").checked,
  status: text => { $("medicine-status").textContent = text; },
  clear: () => { $("medicine-evidence").replaceChildren(); },
  result: body => {
    const rows = [["Matcher status", body.match.status], ["Label recognized", body.label_recognized],
      ["Formulation", body.formulation], ["Exact package", body.exact_package]];
    const details = definitionList(rows);
    const timing = body.timing;
    const nodes = [details, el("p", body.match.reason, "small muted"),
      el("p", `OCR ${timing.ocr_ms} ms · match ${timing.matcher_ms} ms · Qwen ${timing.qwen_ms} ms · total ${timing.round_trip_ms} ms`, "small muted")];
    if (body.qwen.enabled) {
      nodes.push(el("p", "Optional Qwen reader — unverified, excluded from matching", "small"));
      nodes.push(el("p", body.qwen.error || (body.qwen.lines || []).join(" · "), "small"));
      if ((body.qwen.unconfirmed_or_conflicting || []).length)
        nodes.push(el("p", "Unconfirmed or conflicting: " + body.qwen.unconfirmed_or_conflicting.join(" · "), "small"));
    }
    if (body.match.status === "UNCERTAIN" || body.match.status === "NO_MATCH") nodes.push(productResearchCard(body));
    $("medicine-evidence").replaceChildren(...nodes);
    $("medicine-status").textContent = body.match.status + " · " + body.formulation;
  },
});

// ---------- wiring

$("auto-medicine").addEventListener("change", event => {
  $("medicine-panel").hidden = !event.target.checked;    // no status line on screen while the feature is off
  medicineScanner.enable(event.target.checked);
});
$("medicine-rescan").addEventListener("click", () => medicineScanner.invalidate());
$("medicine-discard").addEventListener("click", () => {
  medicineScanner.clear();
  $("medicine-status").textContent = "Candidate image discarded";
});
window.addEventListener("orientationchange", resetMedicineOrientation);
video.addEventListener("resize", resetMedicineOrientation);
$("ask-ai").addEventListener("click", () => askAI.ask());
$("ai-close").addEventListener("click", () => askAI.close());
$("ai-read").addEventListener("click", () => askAI.speak());
$("ai-stop").addEventListener("click", () => speaker.cancel());
function scan(path, label, title, render) {
  sheets.close();                 // the menu makes way for the camera, then for the result
  fallbackBarcode = null;
  $("result-title").textContent = title;
  send(path, label, render);
}

$("welcome-start").addEventListener("click", () => startCamera());
$("toggle-camera").addEventListener("click", () => {
  sheets.close();
  if (stream) { stopCamera(); log("Camera stopped"); } else startCamera();
});
$("flip-camera").addEventListener("click", () => {
  sheets.close();
  facing = facing === "environment" ? "user" : "environment";
  startCamera();
});
$("camera-select").addEventListener("change", (event) => startCamera(event.target.value));
$("scan-barcode").addEventListener("click", () => scan("/api/barcode", "Scanning barcode", "Barcode result", renderBarcode));
$("identify").addEventListener("click", () => scan("/api/identify", "Reading label", "Medicine result", renderIdentify));
$("voice-toggle").addEventListener("click", () => (speaker.enabled ? disableVoice() : enableVoice()));
$("voice-volume").addEventListener("input", (event) => { speaker.volume = Number(event.target.value); });
$("awareness-toggle").addEventListener("click", () => (awareness.on ? stopAwareness("stopped") : startAwareness()));
$("more-tools").addEventListener("click", () => sheets.toggle("tools", $("more-tools")));
$("tools-close").addEventListener("click", () => sheets.close("tools"));
$("sheet-backdrop").addEventListener("click", () => sheets.close("tools"));   // tap outside the menu
$("result-close").addEventListener("click", () => sheets.close("result-sheet"));
document.addEventListener("keydown", (event) => {
  if (sheets.key(event.key)) event.preventDefault();
  else trapFocus(event);
});
// Keep the boxes aligned when the preview changes size or the phone is rotated
window.addEventListener("resize", drawOverlay);
window.addEventListener("orientationchange", drawOverlay);
video.addEventListener("resize", drawOverlay);
if (window.ResizeObserver) new ResizeObserver(drawOverlay).observe($("stage"));
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) return;
  askAI.close();
  speaker.cancel();
  stopAwareness("page inactive");
  if (stream && !busy) { stopCamera(); log("Camera stopped (page hidden)"); }
});
window.addEventListener("pagehide", () => { askAI.close(); stopAwareness("page inactive"); });

fetch("/health").then((r) => r.json()).then((health) => {
  serverOk = true;
  const buildNode = $("app-build");
  const pageBuild = buildNode.dataset && buildNode.dataset.build;
  if (pageBuild) buildNode.textContent = `Build ${pageBuild} · JavaScript ready · ` +
    (health.build === pageBuild ? 'backend matches' : 'BACKEND CHANGED — reload this page');
  $("server-state").textContent = "server ready";
  $("server-state").className = "pill good";
  if (health.registry_error) log("Local UPC registry not loaded: " + health.registry_error);
  updateButtons();
}).catch(() => {
  serverOk = false;
  $("server-state").textContent = "server unreachable";
  $("server-state").className = "pill bad";
  updateButtons();
});
if (!window.isSecureContext) {
  $("camera-message").textContent = cameraError();
  $("camera-message").hidden = false;
}
updateButtons();

function syncCameraViewport() {
  const viewport = window.visualViewport;
  if (viewport) {
    $("stage").style.setProperty("--camera-height", `${viewport.height}px`);
    $("stage").style.setProperty("--camera-width", `${viewport.width}px`);
  }
  drawOverlay();
  const stageRect = $("stage").getBoundingClientRect(), videoRect = video.getBoundingClientRect();
  $("camera-layout").textContent = `Viewport ${Math.round(window.innerWidth)}×${Math.round(window.innerHeight)} · ` +
    `stage ${Math.round(stageRect.width)}×${Math.round(stageRect.height)} · video ${Math.round(videoRect.width)}×${Math.round(videoRect.height)} · ` +
    `source ${video.videoWidth}×${video.videoHeight} · fit ${getComputedStyle(video).objectFit}`;
}
if (window.visualViewport) {
  video.addEventListener("resize", syncCameraViewport);
  window.visualViewport.addEventListener("resize", syncCameraViewport);
  window.visualViewport.addEventListener("scroll", syncCameraViewport);
  syncCameraViewport();
}
