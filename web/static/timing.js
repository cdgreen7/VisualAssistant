// Browser-side latency measurement for the research baseline.
//
// One trace per request: marks are taken with performance.now() (monotonic),
// the server's own stage times arrive in the Server-Timing response header, and
// the finished record goes into a small ring buffer. Nothing is logged per
// frame. Records are only uploaded when the page is opened with ?research=1.
//
// What each number is:
//   prepare_ms     camera frame -> JPEG blob (canvas draw + encode)
//   roundtrip_ms   fetch() called -> response headers received (upload + server + download start)
//   parse_ms       headers -> JSON body parsed ("response available")
//   render_ms      response available -> page updated (script time, not paint)
//   total_ms       start of capture -> page updated
//   server         stage times measured ON THE SERVER (milliseconds)
//   transfer_ms    roundtrip_ms - server.total: network, TLS and browser overhead (an estimate)
//   speech_ms      response available -> the speech engine reports it started talking
(function (root) {
  "use strict";

  // "decode;dur=3.1, yolo;dur=24.6, total;dur=29.4" -> { decode: 3.1, yolo: 24.6, total: 29.4 }
  function parseServerTiming(header) {
    const stages = {};
    for (const part of String(header || "").split(",")) {
      const match = /^\s*([A-Za-z0-9_-]+)\s*;.*?\bdur=([0-9.]+)/.exec(part);
      if (match) stages[match[1]] = Number(match[2]);
    }
    return stages;
  }

  // Coarse client family, e.g. "iphone-safari", "android-chrome", "desktop-chrome".
  function clientLabel(userAgent) {
    const ua = String(userAgent || "");
    let device, browser;
    if (/iPhone|iPad|iPod/.test(ua)) {
      device = /iPad/.test(ua) ? "ipad" : "iphone";
      browser = /CriOS/.test(ua) ? "chrome" : /FxiOS/.test(ua) ? "firefox" : "safari";
    } else if (/Android/.test(ua)) {
      device = "android";
      browser = /Firefox/.test(ua) ? "firefox" : /SamsungBrowser/.test(ua) ? "samsung" : /Chrome/.test(ua) ? "chrome" : "other";
    } else {
      device = "desktop";
      browser = /Edg\//.test(ua) ? "edge" : /Chrome/.test(ua) ? "chrome" : /Firefox/.test(ua) ? "firefox" :
        /Safari/.test(ua) ? "safari" : "other";
    }
    return `${device}-${browser}`;
  }

  const round = (value) => Math.round(value * 10) / 10;

  function percentile(sorted, p) {
    if (!sorted.length) return null;
    const rank = (sorted.length - 1) * p, low = Math.floor(rank), high = Math.ceil(rank);
    return round(sorted[low] + (sorted[high] - sorted[low]) * (rank - low));
  }

  class Trace {
    constructor(recorder, kind) {
      this.recorder = recorder;
      this.kind = kind;
      this.marks = { start: recorder.now() };
      this.server = null;
      this.extra = {};
      this.record = null;
    }

    mark(name) { this.marks[name] = this.recorder.now(); return this; }

    // fetch() that notes when it was called, when headers arrived, and the server's stage times.
    async fetch(url, options) {
      this.mark("sent");
      const response = await this.recorder.fetch(url, options);
      this.mark("headers");
      this.extra.status = response.status;
      const header = response.headers && response.headers.get ? response.headers.get("Server-Timing") : null;
      if (header) this.server = parseServerTiming(header);
      return response;
    }

    // Close the trace and store its record. Missing marks simply leave fields out.
    end(extra) {
      if (this.record) return this.record;
      const m = this.marks, between = (a, b) => (a in m && b in m ? round(m[b] - m[a]) : undefined);
      const available = "parsed" in m ? "parsed" : "headers";
      const last = "rendered" in m ? "rendered" : available in m ? available : "sent";
      const record = Object.assign({ kind: this.kind }, this.extra, extra, {
        prepare_ms: between("start", "prepared"),
        roundtrip_ms: between("sent", "headers"),
        parse_ms: between("headers", "parsed"),
        render_ms: between(available, "rendered"),
        total_ms: between("start", last),
      });
      if (this.server) {
        record.server = this.server;
        if (record.roundtrip_ms !== undefined && typeof this.server.total === "number") {
          record.transfer_ms = round(record.roundtrip_ms - this.server.total);
        }
      }
      for (const key of Object.keys(record)) if (record[key] === undefined) delete record[key];
      this.availableAt = m[available];
      this.record = record;
      this.recorder.add(record);
      return record;
    }

    // Call when the speech engine starts speaking the result of this trace.
    speechStarted() {
      if (this.record && this.availableAt !== undefined && this.record.speech_ms === undefined) {
        this.record.speech_ms = round(this.recorder.now() - this.availableAt);
      }
    }
  }

  class Recorder {
    // options: now() monotonic ms, fetch, max records kept, upload(batch) or null, batch size
    constructor(options) {
      const o = options || {};
      this.now = o.now || (() => performance.now());
      this.fetch = o.fetch || ((...args) => fetch(...args));
      this.max = o.max || 300;
      this.upload = o.upload || null;
      this.batchSize = o.batchSize || 20;
      this.records = [];
      this.pending = [];
      this.awaitingSpeech = null;
    }

    begin(kind) { return new Trace(this, kind); }

    add(record) {
      this.records.push(record);
      if (this.records.length > this.max) this.records.shift();
      if (!this.upload) return;
      this.pending.push(record);
      if (this.pending.length >= this.batchSize) this.flush();
    }

    // Send what has been collected (research runs only). Failures stop further uploads.
    flush() {
      if (!this.upload || !this.pending.length) return;
      const batch = this.pending;
      this.pending = [];
      Promise.resolve().then(() => this.upload(batch)).then((ok) => { if (ok === false) this.upload = null; })
        .catch(() => { this.upload = null; });
    }

    // The next utterance to start belongs to this trace.
    expectSpeech(trace) { this.awaitingSpeech = trace; }
    speechStarted() {
      if (this.awaitingSpeech) this.awaitingSpeech.speechStarted();
      this.awaitingSpeech = null;
    }

    // Median / p90 / p95 per kind and field, for a quick look on the device.
    summary() {
      const groups = {};
      for (const record of this.records) {
        const group = groups[record.kind] || (groups[record.kind] = {});
        const flat = Object.assign({}, record);
        for (const [stage, value] of Object.entries(record.server || {})) flat["server_" + stage + "_ms"] = value;
        for (const [key, value] of Object.entries(flat)) {
          if (key.endsWith("_ms") && typeof value === "number") (group[key] || (group[key] = [])).push(value);
        }
      }
      const out = {};
      for (const [kind, fields] of Object.entries(groups)) {
        out[kind] = {};
        for (const [key, values] of Object.entries(fields)) {
          const sorted = values.slice().sort((a, b) => a - b);
          out[kind][key] = { n: sorted.length, median: percentile(sorted, 0.5), p90: percentile(sorted, 0.9), p95: percentile(sorted, 0.95) };
        }
      }
      return out;
    }
  }

  const api = { Recorder, Trace, parseServerTiming, clientLabel, percentile };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Timing = api;
})(typeof self !== "undefined" ? self : this);
