// Run with: node --test tests/js/medicine-speech.test.js (also run by tests/test_frontend_js.py)
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { Speaker } = require("../../web/static/announcer.js");
const { MedicineSpeech, phrase, spokenStrength } = require("../../web/static/medicine-speech.js");

class FakeSynth {
  constructor() { this.spoken = []; this.speaking = false; this.pending = false; this.cancels = 0; }
  speak(utterance) { this.spoken.push(utterance.text); this.speaking = true; }
  cancel() { this.cancels += 1; this.speaking = false; this.pending = false; }
}
class FakeUtterance { constructor(text) { this.text = text; } }
function setup(enabled = true) {
  const synth = new FakeSynth();
  const speaker = new Speaker(synth, FakeUtterance, () => {});
  if (enabled) { speaker.enable(); synth.spoken = []; }
  let clock = 0;
  const speech = new MedicineSpeech(speaker, { now: () => clock });
  return { synth, speaker, speech, advance: (ms) => { clock += ms; } };
}

const evidence = (kind, value, corrected_from = null) => ({ kind, value, confidence: 0.9, corrected_from });
const body = (match) => ({ match: Object.assign({ level: "NONE", candidates: [], evidence: [], package: null }, match) });

const VERIFIED_FORMULATION = body({
  status: "VERIFIED", level: "FORMULATION",
  candidates: [{ ingredients: [{ name: "DIPHENHYDRAMINE CITRATE", strength: "38 mg/1" }, { name: "IBUPROFEN", strength: "200 mg/1" }],
                 brand_names: ["ADVIL PM", "Advil PM"] }],
  evidence: [evidence("brand", "ADVIL PM"), evidence("ingredient", "IBUPROFEN")],
});
const VERIFIED_PACKAGE = body({
  status: "VERIFIED", level: "PACKAGE", package: { package_ndc: "50580-226-12", description: "12 TABLET" },
  candidates: [{ ingredients: [{ name: "DIPHENHYDRAMINE HYDROCHLORIDE", strength: "25 mg/1" }], brand_names: ["Benadryl"] }],
});
const UNCERTAIN_IBUPROFEN = body({
  status: "UNCERTAIN",
  // The candidates name products and strengths that were NOT confirmed; none may be spoken.
  candidates: [{ ingredients: [{ name: "IBUPROFEN", strength: "200 mg/1" }], brand_names: ["Advil", "Motrin IB"] },
               { ingredients: [{ name: "IBUPROFEN", strength: "200 mg/1" }, { name: "DIPHENHYDRAMINE CITRATE", strength: "38 mg/1" }], brand_names: ["Advil PM"] }],
  evidence: [evidence("ingredient", "IBUPROFEN"), evidence("strength", "200mg"), evidence("brand_hint", "EQUATE"),
             evidence("compared_brand", "ADVIL"), evidence("ingredient_hint", "WATER")],
});

test("strengths are spoken in words, with the denominator when there is one", () => {
  assert.equal(spokenStrength("200 mg/1"), "200 milligrams");
  assert.equal(spokenStrength("15 mg/5mL"), "15 milligrams per 5 milliliters");
  assert.equal(spokenStrength("1 g/1"), "1 gram");
  assert.equal(spokenStrength("400 mcg/1"), "400 micrograms");
});

test("VERIFIED formulation names every ingredient and strength, but no brand", () => {
  const text = phrase(VERIFIED_FORMULATION);
  assert.equal(text, "Verified medicine: diphenhydramine citrate, 38 milligrams, and ibuprofen, 200 milligrams.");
  assert.doesNotMatch(text, /advil/i);   // a formulation is not a product identity
});

test("VERIFIED package names its single brand", () => {
  assert.equal(phrase(VERIFIED_PACKAGE), "Verified package: Benadryl. Diphenhydramine hydrochloride, 25 milligrams.");
});

test("UNCERTAIN says so first and mentions only text read on the label", () => {
  const text = phrase(UNCERTAIN_IBUPROFEN);
  assert.equal(text, "Not verified. This medicine could not be confirmed. Text read on the label: ibuprofen, equate.");
  assert.match(text, /^Not verified\./);
  // No candidate brand, no strength, no comparison brand, no hint ingredient.
  for (const word of [/advil/i, /motrin/i, /200/, /milligram/i, /diphenhydramine/i, /water/i]) assert.doesNotMatch(text, word);
});

test("UNCERTAIN without readable product text and corrected text", () => {
  assert.equal(phrase(body({ status: "UNCERTAIN", evidence: [evidence("strength", "200mg", "2OOMG")] })),
               "Not verified. This medicine could not be confirmed.");
  assert.equal(phrase(body({ status: "UNCERTAIN", evidence: [evidence("ingredient", "IBUPROFEN"), evidence("ingredient", "IBUPROFEN")] })),
               "Not verified. This medicine could not be confirmed. Text read on the label: ibuprofen.");
});

test("NO_MATCH, missing bottle and malformed bodies", () => {
  assert.equal(phrase(body({ status: "NO_MATCH" })), "Medicine could not be identified.");
  assert.equal(phrase({ status: "NO_BOTTLE", match: null }), "No bottle detected. Keep the whole bottle in view and try again.");
  assert.equal(phrase(null), null);
  assert.equal(phrase({ match: { status: "SOMETHING_ELSE" } }), null);
  // A VERIFIED body without a candidate is not announced as verified.
  assert.equal(phrase(body({ status: "VERIFIED", level: "FORMULATION", candidates: [] })), null);
});

test("no dosage or advice words are ever produced", () => {
  for (const b of [VERIFIED_FORMULATION, VERIFIED_PACKAGE, UNCERTAIN_IBUPROFEN, body({ status: "NO_MATCH" })]) {
    assert.doesNotMatch(phrase(b), /\b(take|dose|dosage|every|hours|daily|tablets? a day|should)\b/i);
  }
});

test("speaks through the shared Speaker, interrupting an object announcement", () => {
  const { synth, speaker, speech } = setup();
  speaker.say("Person ahead.");
  synth.spoken = [];
  assert.equal(speech.announce(UNCERTAIN_IBUPROFEN), phrase(UNCERTAIN_IBUPROFEN));
  assert.deepEqual(synth.spoken, [phrase(UNCERTAIN_IBUPROFEN)]);
  assert.ok(synth.cancels >= 1);
});

test("respects voice off (mute) and unsupported speech", () => {
  const { synth, speech } = setup(false);
  assert.equal(speech.announce(VERIFIED_PACKAGE), null);
  assert.deepEqual(synth.spoken, []);
  const unsupported = new MedicineSpeech(new Speaker(null, null, () => {}));
  assert.equal(unsupported.announce(VERIFIED_PACKAGE), null);
});

test("does not interrupt when blocked (Ask AI reading)", () => {
  const { synth, speech } = setup();
  assert.equal(speech.announce(VERIFIED_PACKAGE, { blocked: true }), null);
  assert.deepEqual(synth.spoken, []);
});

test("the same automatic result is not repeated, a new result or a tap is", () => {
  const { synth, speech, advance } = setup();
  assert.ok(speech.announce(UNCERTAIN_IBUPROFEN));
  synth.speaking = false;
  advance(5000);
  assert.equal(speech.announce(UNCERTAIN_IBUPROFEN), null);                     // same text, automatic
  assert.ok(speech.announce(UNCERTAIN_IBUPROFEN, { force: true }));             // user pressed Identify
  synth.speaking = false;
  assert.ok(speech.announce(VERIFIED_PACKAGE));                                  // different result
  synth.speaking = false;
  advance(5000);
  assert.ok(speech.announce(UNCERTAIN_IBUPROFEN));                               // changed back: new information
  assert.equal(synth.spoken.length, 4);
});

test("the same result is spoken again after the repeat window", () => {
  const { synth, speech, advance } = setup();
  assert.ok(speech.announce(VERIFIED_PACKAGE));
  synth.speaking = false;
  advance(speech.options.repeatMs);
  assert.ok(speech.announce(VERIFIED_PACKAGE));
  assert.equal(synth.spoken.length, 2);
});

test("app.js speaks an Identify result only after the user turns voice on", () => {
  const vm = require("node:vm");
  const nodes = new Map(), spoken = [];
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, { hidden: false, disabled: false, value: "1", className: "", textContent: "",
      classList: { toggle() {}, add() {}, remove() {} }, addEventListener() {}, setAttribute() {}, removeAttribute() {},
      replaceChildren() {}, append() {}, prepend() {}, after() {}, querySelector: () => null, isConnected: true,
      getContext: () => ({ drawImage() {} }), toBlob: (cb) => cb(new Blob(["x"])) });
    return nodes.get(id);
  }
  const window = { addEventListener() {}, isSecureContext: true,
    speechSynthesis: { speaking: false, pending: false, cancel() {}, speak(u) { spoken.push(u.text); } },
    SpeechSynthesisUtterance: function (text) { this.text = text; } };
  const context = vm.createContext({ window, document: { getElementById: node, createElement: () => node(Symbol()),
      createTextNode: (t) => t, addEventListener() {}, hidden: false, activeElement: null },
    Announcer: require("../../web/static/announcer.js"), AskAI: require("../../web/static/ask-ai.js"),
    UI: require("../../web/static/ui.js"), Timing: require("../../web/static/timing.js"), View: {},
    MedicineCandidates: require("../../web/static/medicine-candidates.js"), MedicineCapture: require("../../web/static/medicine-capture.js"),
    MedicineScanner: require("../../web/static/medicine-scanner.js"), MedicineIdentification: require("../../web/static/medicine-identification.js"),
    MedicineSpeech: require("../../web/static/medicine-speech.js"),
    navigator: {}, console, Blob, FormData, AbortController, setTimeout: () => 1, clearTimeout() {}, performance,
    fetch: async () => ({ ok: true, json: async () => ({}) }) });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "../../web/static/app.js"), "utf8"), context);
  const identify = Object.assign({ detection: { class_name: "bottle", confidence: 0.9, position: "CENTER" }, ocr: [] }, UNCERTAIN_IBUPROFEN);
  identify.match = Object.assign({ reason: "7 formulations remain", candidate_count: 2 }, identify.match,
    { candidates: identify.match.candidates.map((c) => Object.assign({ not_observed: [] }, c)) });   // as serialize.py sends
  context.identifyBody = identify;

  vm.runInContext("renderIdentify(identifyBody)", context);
  assert.deepEqual(spoken, []);                                       // voice off: nothing, as before
  vm.runInContext("enableVoice()", context);                         // the user's tap
  spoken.length = 0;
  vm.runInContext("renderIdentify(identifyBody)", context);
  assert.deepEqual(spoken, [phrase(UNCERTAIN_IBUPROFEN)]);
  assert.doesNotMatch(spoken[0], /advil|motrin/i);
});

test("both identification paths in app.js speak, and the page loads the module", () => {
  const app = fs.readFileSync(path.join(__dirname, "../../web/static/app.js"), "utf8");
  const page = fs.readFileSync(path.join(__dirname, "../../web/templates/index.html"), "utf8");
  assert.match(app, /sayMedicine\(body, false\)/);   // automatic label scan
  assert.match(app, /sayMedicine\(body, true\)/);    // Identify button
  assert.ok(page.indexOf("medicine-speech.js") > -1 && page.indexOf("medicine-speech.js") < page.indexOf("app.js"));
});
