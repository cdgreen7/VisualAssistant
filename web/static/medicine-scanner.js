// Phase 4A coordinator. A single in-memory packet, no upload or identification.
(function(root) {
  'use strict';
  class MedicineScanner {
    constructor(gate, capture, ui) {
      this.gate = gate; this.capture = capture; this.ui = ui;
      this.enabled = false; this.busy = false; this.packet = null; this.generation = 0;
    }
    clear() { this.packet = null; this.ui.clear(); }
    invalidate() {
      this.generation += 1;
      this.gate.reset();
      this.clear();
      this.ui.status(this.enabled ? 'Automatic scan waiting for awareness' : 'Automatic medicine scan off');
    }
    enable(value) { this.enabled = Boolean(value); this.invalidate(); }
    async observe(body, snapshot, now, allowed = true) {
      if (!this.enabled) return;
      const candidate = this.gate.update(body,now,allowed && !this.busy && Boolean(snapshot));
      if (this.packet && (!this.gate.current || !this.gate.current.captured)) this.clear();
      if (!candidate) {
        if (!this.busy && !this.packet) this.ui.status(this.gate.current && this.gate.current.hits ?
          'Possible medicine container detected' : 'Looking for a stable container');
        return;
      }
      this.busy = true;
      const generation = this.generation;
      this.ui.status('Possible medicine container detected');
      this.ui.status('Preparing label scan');
      try {
        const packet = await this.capture.prepare(snapshot,candidate);
        if (!this.enabled || generation !== this.generation) return;
        this.clear();
        this.packet = packet;
        this.ui.ready(packet);
        this.ui.status('Image ready for identification');
      } catch (error) {
        if (generation === this.generation) this.ui.status('Capture unavailable: ' + error.message);
      } finally { this.busy = false; }
    }
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = MedicineScanner;
  else root.MedicineScanner = MedicineScanner;
})(typeof self !== 'undefined' ? self : this);
