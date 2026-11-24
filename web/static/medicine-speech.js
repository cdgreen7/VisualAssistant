// Spoken medicine results through the shared Speaker (announcer.js).
//
// phrase() turns a matcher result into one sentence. It never upgrades a status:
//   VERIFIED / FORMULATION -> the verified ingredients and strengths (no brand:
//                             a formulation is not one product)
//   VERIFIED / PACKAGE     -> the package's single brand, then its ingredients
//   UNCERTAIN              -> "Not verified" first, then only product text that OCR
//                             read (brand and ingredient evidence); never a candidate,
//                             a strength or a "Compare to" brand
//   NO_MATCH               -> could not be identified
// Strengths are the label amounts the FDA lists; no dosage or advice is produced.
//
// MedicineSpeech speaks only when voice is on (the user's tap enabled it), does not
// talk over Ask AI, and does not repeat the same automatic result within repeatMs.
(function (root) {
  "use strict";

  const UNITS = { mg: ["milligram", "milligrams"], mcg: ["microgram", "micrograms"], g: ["gram", "grams"],
                  ml: ["milliliter", "milliliters"], "%": ["percent", "percent"] };
  const UNCERTAIN_TEXT_KINDS = new Set(["brand", "brand_hint", "ingredient"]);
  const MAX_TEXT_ITEMS = 3;
  const DEFAULTS = { repeatMs: 60000 };

  function amount(number, unit) {
    const words = UNITS[String(unit || "").toLowerCase()];
    if (!words) return null;
    return `${number} ${Number(number) === 1 ? words[0] : words[1]}`;
  }

  // "200 mg/1" -> "200 milligrams"; "15 mg/5mL" -> "15 milligrams per 5 milliliters"
  function spokenStrength(strength) {
    const m = /^\s*([\d.]+)\s*([a-zA-Z%]+)\s*(?:\/\s*([\d.]*)\s*([a-zA-Z%]*))?\s*$/.exec(String(strength || ""));
    if (!m) return String(strength || "").replace(/\//g, " per ");
    const main = amount(m[1], m[2]);
    if (!main) return String(strength).replace(/\//g, " per ");
    if (!m[4]) return main;                                  // "/1" or nothing: per unit
    const per = amount(m[3] || "1", m[4]);
    return per ? `${main} per ${per}` : main;
  }

  function lower(text) { return String(text || "").toLowerCase(); }
  function capitalized(text) { return text.charAt(0).toUpperCase() + text.slice(1); }

  function ingredients(candidate) {
    return candidate.ingredients.map((i) => `${lower(i.name)}, ${spokenStrength(i.strength)}`).join(", and ");
  }

  // Product words OCR actually read, as observed (not catalog values), without corrections.
  function textRead(match) {
    const seen = [];
    for (const e of match.evidence || []) {
      if (!UNCERTAIN_TEXT_KINDS.has(e.kind) || e.corrected_from) continue;
      const value = lower(e.value).trim();
      if (value && !seen.includes(value)) seen.push(value);
    }
    return seen.slice(0, MAX_TEXT_ITEMS);
  }

  function phrase(body) {
    if (!body) return null;
    if (body.status === "NO_BOTTLE") return "No bottle detected. Keep the whole bottle in view and try again.";
    const match = body.match;
    if (!match) return null;
    if (match.status === "VERIFIED") {
      const candidate = (match.candidates || [])[0];
      if (!candidate || !(candidate.ingredients || []).length) return null;
      if (match.level === "PACKAGE") {
        const brands = candidate.brand_names || [];
        const brand = brands.length === 1 ? `${brands[0]}. ` : "";
        return `Verified package: ${brand}${capitalized(ingredients(candidate))}.`;
      }
      if (match.level === "FORMULATION") return `Verified medicine: ${ingredients(candidate)}.`;
      return null;
    }
    if (match.status === "UNCERTAIN") {
      const read = textRead(match);
      return "Not verified. This medicine could not be confirmed." +
             (read.length ? ` Text read on the label: ${read.join(", ")}.` : "");
    }
    if (match.status === "NO_MATCH") return "Medicine could not be identified.";
    return null;
  }

  class MedicineSpeech {
    constructor(speaker, options) {
      this.speaker = speaker;
      this.options = Object.assign({}, DEFAULTS, options);
      this.now = this.options.now || (() => Date.now());
      this.last = null;
      this.lastAt = -Infinity;
    }

    // Returns the text spoken, or null. force: the user asked (Identify button).
    // blocked: something else owns the voice right now (Ask AI reading).
    announce(body, { force = false, blocked = false } = {}) {
      const text = phrase(body);
      const speaker = this.speaker;
      if (!text || blocked || !speaker || !speaker.enabled || !speaker.supported) return null;
      const now = this.now();
      if (!force && text === this.last && now - this.lastAt < this.options.repeatMs) return null;
      speaker.cancel();                     // a medicine result outranks an object announcement
      if (!speaker.say(text)) return null;
      this.last = text;
      this.lastAt = now;
      return text;
    }
  }

  const api = { MedicineSpeech, phrase, spokenStrength, DEFAULTS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MedicineSpeech = api;
})(typeof self !== "undefined" ? self : this);
