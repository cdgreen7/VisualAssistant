// Phase 4A policy only: no camera, network, OCR, model, DOM or medical identity.
(function(root) {
  "use strict";
  const DEFAULTS = {sightings: 3, fallbackSightings: 5, stableMs: 2000,
    maxGapMs: 4500, leaveMs: 5000, cooldownMs: 30000, confidence: 0.65,
    minArea: 0.025, overlap: 0.65};
  function iou(a, b) {
    const area = r => (r[2]-r[0])*(r[3]-r[1]);
    const intersection = Math.max(0, Math.min(a[2],b[2])-Math.max(a[0],b[0])) *
      Math.max(0, Math.min(a[3],b[3])-Math.max(a[1],b[1]));
    return intersection / (area(a)+area(b)-intersection);
  }
  class CandidateGate {
    constructor(options) { this.options = {...DEFAULTS, ...options}; this.reset(); }
    reset() { this.current = null; this.lastCapture = -Infinity; this.ids = new Map(); }
    update(body, now, allowed = true) {
      const o = this.options, width = body.image_width, height = body.image_height;
      const bottles = (body.detections || []).filter(d => d.class_name === 'bottle');
      if (body.tracking === 'stale') { this.current = null; return null; }
      // More than one bottle is ambiguous; do not choose among containers.
      const d = bottles.length === 1 ? bottles[0] : null;
      const b = d && d.bounding_box;
      const valid = d && d.confidence >= o.confidence && width > 0 && height > 0 &&
        Array.isArray(b) && b.length === 4 && b.every(Number.isFinite) &&
        b[0] >= 0 && b[1] >= 0 && b[2] <= width && b[3] <= height &&
        b[2] > b[0] && b[3] > b[1] && (b[2]-b[0])*(b[3]-b[1])/(width*height) >= o.minArea;
      if (!valid) {
        if (this.current) {
          this.current.hits = 0;
          this.current.since = now;
          if (now-this.current.lastSeen >= o.leaveMs) this.current = null;
        }
        return null;
      }
      const box = [b[0]/width,b[1]/height,b[2]/width,b[3]/height];
      const id = body.tracking === 'ok' && Number.isInteger(d.track_id) ? d.track_id : null;
      let c = this.current;
      const continuous = c && now-c.lastSeen <= o.leaveMs && iou(c.box, box) >= o.overlap;
      const stable = continuous && now-c.lastSeen <= o.maxGapMs && c.id === id && c.hits > 0;
      // Keep the captured latch across ID changes when the same region is occupied.
      c = this.current = {box, id, lastSeen: now, since: stable ? c.since : now,
        hits: stable ? c.hits+1 : 1, captured: Boolean(continuous && c.captured)};
      for (const [key, time] of this.ids) if (now-time >= o.cooldownMs) this.ids.delete(key);
      const required = id === null ? o.fallbackSightings : o.sightings;
      if (!allowed || c.captured || c.hits < required || now-c.since < o.stableMs ||
          now-this.lastCapture < o.cooldownMs || (id !== null && this.ids.has(id))) return null;
      c.captured = true;
      this.lastCapture = now;
      if (id !== null) {
        this.ids.set(id, now);
        if (this.ids.size > 128) this.ids.delete(this.ids.keys().next().value);
      }
      return {track_id: id, session_id: body.session_id || null, frame: body.frame || null,
        class_name: 'bottle', confidence: d.confidence,
        bounding_box: [...b], image_width: width, image_height: height};
    }
  }
  const api = {CandidateGate, DEFAULTS, iou};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MedicineCandidates = api;
})(typeof self !== 'undefined' ? self : this);
