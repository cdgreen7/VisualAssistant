// Run with: node --test tests/js   (also run by tests/test_frontend_js.py)
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { Recorder, parseServerTiming, clientLabel, percentile } = require("../../web/static/timing.js");

// A recorder on a hand-driven clock; fetch advances it by `network` ms.
function setup(options) {
  const clock = { t: 1000 };
  const o = Object.assign({ network: 180, header: "decode;dur=3.1, yolo;dur=24.6, total;dur=29.4" }, options);
  const recorder = new Recorder(Object.assign({
    now: () => clock.t,
    fetch: async () => { clock.t += o.network; return { status: 200, headers: { get: (name) => (name === "Server-Timing" ? o.header : null) } }; },
  }, o.recorder));
  return { clock, recorder };
}

test("Server-Timing header is parsed into stage milliseconds", () => {
  assert.deepEqual(parseServerTiming("decode;dur=3.1, yolo;dur=24.6, json;dur=0.0, total;dur=29.4"),
                   { decode: 3.1, yolo: 24.6, json: 0, total: 29.4 });
  assert.deepEqual(parseServerTiming('cache;desc="hit";dur=2, bad, x;dur=abc'), { cache: 2 });
  assert.deepEqual(parseServerTiming(null), {});
  assert.deepEqual(parseServerTiming(""), {});
});

test("client family distinguishes iPhone Safari from Android Chrome", () => {
  const iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1";
  const android = "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 Chrome/130.0.0.0 Mobile Safari/537.36";
  assert.equal(clientLabel(iphone), "iphone-safari");
  assert.equal(clientLabel(android), "android-chrome");
  assert.equal(clientLabel(iphone.replace("Version/18.0", "CriOS/130.0")), "iphone-chrome");
  assert.equal(clientLabel("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15"), "desktop-safari");
  assert.equal(clientLabel("Mozilla/5.0 (Macintosh) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"), "desktop-chrome");
  assert.equal(clientLabel(undefined), "desktop-other");
});

test("a full trace separates client, transfer and server time", async () => {
  const { clock, recorder } = setup();
  const trace = recorder.begin("detect");
  clock.t += 12;   trace.mark("prepared");                 // capture + JPEG encode
  await trace.fetch("/api/detect", {});                    // 180 ms round trip, 29.4 of it on the server
  clock.t += 0.4;  trace.mark("parsed");
  clock.t += 1.5;  trace.mark("rendered");
  const record = trace.end({ image: "640x360" });
  assert.deepEqual(record, {
    kind: "detect", status: 200, image: "640x360",
    prepare_ms: 12, roundtrip_ms: 180, parse_ms: 0.4, render_ms: 1.5, total_ms: 193.9,
    server: { decode: 3.1, yolo: 24.6, total: 29.4 }, transfer_ms: 150.6,
  });
  assert.equal(recorder.records.length, 1);
  assert.equal(trace.end(), record);                       // ending twice records once
  assert.equal(recorder.records.length, 1);
});

test("server time is never invented when the header is missing", async () => {
  const { recorder } = setup({ header: null });
  const trace = recorder.begin("identify");
  await trace.fetch("/api/identify", {});
  const record = trace.end();
  assert.equal("server" in record, false);
  assert.equal("transfer_ms" in record, false);
  assert.equal(record.roundtrip_ms, 180);
  assert.equal("prepare_ms" in record, false);             // marks that were not taken leave no field
  assert.equal("render_ms" in record, false);
});

test("a request-only trace (no capture or render marks) still records the round trip", async () => {
  const { recorder } = setup();
  const trace = recorder.begin("medicine/identify-crop");
  await trace.fetch("/x", {});
  assert.deepEqual(trace.end(), { kind: "medicine/identify-crop", status: 200, roundtrip_ms: 180, total_ms: 180,
                                  server: { decode: 3.1, yolo: 24.6, total: 29.4 }, transfer_ms: 150.6 });
});

test("time from response available to speech start", async () => {
  const { clock, recorder } = setup();
  const trace = recorder.begin("detect");
  await trace.fetch("/api/detect", {});
  trace.mark("parsed");
  clock.t += 2; trace.mark("rendered");
  trace.end();
  recorder.expectSpeech(trace);
  clock.t += 140;                                          // the speech engine takes a moment to start
  recorder.speechStarted();
  assert.equal(recorder.records[0].speech_ms, 142);
  clock.t += 500;
  recorder.speechStarted();                                // a later, unrelated utterance changes nothing
  assert.equal(recorder.records[0].speech_ms, 142);
});

test("speech that never starts leaves no speech time", async () => {
  const { recorder } = setup();
  const trace = recorder.begin("detect");
  await trace.fetch("/api/detect", {});
  trace.end();
  recorder.expectSpeech(trace);
  assert.equal("speech_ms" in recorder.records[0], false);
});

test("only the newest records are kept", () => {
  const { recorder } = setup({ recorder: { max: 5 } });
  for (let i = 0; i < 12; i++) recorder.add({ kind: "detect", total_ms: i });
  assert.deepEqual(recorder.records.map((r) => r.total_ms), [7, 8, 9, 10, 11]);
});

test("nothing is uploaded unless an uploader is configured", () => {
  const { recorder } = setup();
  for (let i = 0; i < 50; i++) recorder.add({ kind: "detect", total_ms: i });
  assert.equal(recorder.pending.length, 0);
  recorder.flush();
});

test("uploads go in batches, and a refused upload stops further attempts", async () => {
  const batches = [];
  let accept = true;
  const { recorder } = setup({ recorder: { batchSize: 3, upload: async (batch) => { batches.push(batch.length); return accept; } } });
  for (let i = 0; i < 7; i++) recorder.add({ kind: "detect", total_ms: i });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(batches, [3, 3]);
  recorder.flush();                                        // page hidden: send the remainder
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(batches, [3, 3, 1]);
  accept = false;                                          // server started without --timing-log
  for (let i = 0; i < 3; i++) recorder.add({ kind: "detect", total_ms: i });
  await new Promise((resolve) => setImmediate(resolve));
  for (let i = 0; i < 9; i++) recorder.add({ kind: "detect", total_ms: i });
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(batches, [3, 3, 1, 3]);
  assert.equal(recorder.records.length, 19);               // local recording continues
});

test("summary gives median, p90 and p95 per kind, with server stages alongside", () => {
  const { recorder } = setup();
  for (let i = 1; i <= 100; i++) recorder.add({ kind: "detect", total_ms: i, server: { yolo: i / 2, total: i } });
  recorder.add({ kind: "barcode", total_ms: 70 });
  const summary = recorder.summary();
  assert.deepEqual(summary.detect.total_ms, { n: 100, median: 50.5, p90: 90.1, p95: 95.1 });
  assert.equal(summary.detect.server_yolo_ms.median, 25.3);
  assert.deepEqual(summary.barcode.total_ms, { n: 1, median: 70, p90: 70, p95: 70 });
  assert.equal(percentile([], 0.5), null);
});
