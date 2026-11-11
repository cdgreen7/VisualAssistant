// Voice announcements for environmental awareness.
//
// AnnouncementManager decides WHAT to say from successive /api/detect results.
// It has no DOM, timer or speech code, so it can be tested on its own.
// Speaker says it with the browser's SpeechSynthesis, on the phone itself.
//
// This is not object tracking. An "object" here is just a class name seen in
// one third of the picture (LEFT / CENTER / RIGHT) in successive results; two
// chairs on the right are one "chair on your right".
(function (root) {
  "use strict";

  const PHRASES = { LEFT: "on your left", CENTER: "ahead", RIGHT: "on your right" };
  const DEFAULTS = {
    cooldownMs: 10000,       // the same kind of object is announced at most once in this time
    forgetMs: 5000,          // absent this long -> forgotten, so it can be announced again
    requiredSightings: 2,    // consecutive results an object must appear in first
    priorityClasses: ["person"],
  };

  function phrase(className, position) {
    const where = PHRASES[position];
    if (!className || !where) return null;
    const name = String(className);
    return `${name.charAt(0).toUpperCase()}${name.slice(1)} ${where}.`;
  }

  class AnnouncementManager {
    constructor(options) {
      this.options = Object.assign({}, DEFAULTS, options);
      this.reset();
    }

    reset() {
      this.seen = new Map();          // "class|POSITION" -> { streak, lastSeen, announced }
      this.lastSpokenByClass = new Map();
      this.cycle = 0;
    }

    // Feed one detection result. Returns the single phrase to say now, or null.
    // With canSpeak false nothing is chosen (and nothing is marked as announced),
    // but sightings are still counted.
    update(detections, now, canSpeak = true) {
      const { cooldownMs, forgetMs, requiredSightings, priorityClasses } = this.options;
      this.cycle += 1;

      const present = new Map();      // key -> best detection for that class + position
      for (const detection of detections || []) {
        if (!phrase(detection.class_name, detection.position)) continue;
        const key = `${detection.class_name}|${detection.position}`;
        const best = present.get(key);
        if (!best || detection.confidence > best.confidence) present.set(key, detection);
      }

      for (const [key, state] of this.seen) {   // forget what has been gone long enough
        if (!present.has(key) && now - state.lastSeen >= forgetMs) this.seen.delete(key);
      }
      for (const key of present.keys()) {
        let state = this.seen.get(key);
        if (state && now - state.lastSeen >= forgetMs) state = null;   // it left and came back
        if (!state) state = { streak: 0, lastCycle: -1, lastSeen: now, announced: false };
        state.streak = state.lastCycle === this.cycle - 1 ? state.streak + 1 : 1;
        state.lastCycle = this.cycle;
        state.lastSeen = now;
        this.seen.set(key, state);
      }
      if (!canSpeak) return null;

      const ready = [...present.entries()].filter(([key, detection]) => {
        const state = this.seen.get(key);
        const spoken = this.lastSpokenByClass.get(detection.class_name);
        return state.streak >= requiredSightings && !state.announced &&
               (spoken === undefined || now - spoken >= cooldownMs);
      });
      if (!ready.length) return null;

      const rank = ([, d]) => (priorityClasses.includes(d.class_name) ? 0 : 1);
      ready.sort((a, b) => rank(a) - rank(b) || b[1].confidence - a[1].confidence ||
                           (a[0] < b[0] ? -1 : 1));
      const [key, chosen] = ready[0];           // at most one announcement per result
      this.seen.get(key).announced = true;
      this.lastSpokenByClass.set(chosen.class_name, now);
      return phrase(chosen.class_name, chosen.position);
    }
  }

  // Thin wrapper around SpeechSynthesis. Never queues: if something is still
  // being spoken, a new phrase is dropped rather than stacked up behind it.
  class Speaker {
    constructor(synth, Utterance, onError) {
      this.synth = synth || null;
      this.Utterance = Utterance || null;
      this.onError = onError || function () {};
      this.enabled = false;
      this.volume = 1;
    }

    get supported() { return Boolean(this.synth && this.Utterance); }
    get busy() { return Boolean(this.synth && (this.synth.speaking || this.synth.pending)); }
    get canSpeak() { return this.enabled && this.supported && !this.busy; }

    // Must be called from a tap: iOS Safari only allows speech that starts in a user gesture.
    enable(testPhrase) {
      if (!this.supported) return false;
      this.enabled = true;
      this.synth.cancel();
      if (testPhrase) this.say(testPhrase);
      return true;
    }

    disable() {
      this.enabled = false;
      this.cancel();
    }

    cancel() {
      if (this.synth) this.synth.cancel();    // also empties anything pending
    }

    say(text) {
      if (!text || !this.canSpeak) return false;
      try {
        const utterance = new this.Utterance(text);
        utterance.volume = this.volume;
        utterance.onerror = (event) => {
          const reason = event && event.error;
          if (reason !== "canceled" && reason !== "interrupted") this.onError(reason || "unknown error");
        };
        this.synth.speak(utterance);
        return true;
      } catch (error) {
        this.onError(error.message || "unknown error");
        return false;
      }
    }
  }

  const api = { AnnouncementManager, Speaker, phrase, DEFAULTS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Announcer = api;
})(typeof self !== "undefined" ? self : this);
