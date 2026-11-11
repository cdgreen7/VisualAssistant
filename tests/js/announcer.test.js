// Run with: node --test tests/js   (also run by tests/test_announcer_js.py)
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { AnnouncementManager, Speaker, phrase } = require("../../web/static/announcer.js");

const STEP = 1500;   // the awareness loop's interval
const d = (class_name, position, confidence = 0.9) => ({ class_name, position, confidence });

// Feeds one result per STEP and returns what was announced each time.
function run(manager, frames, startAt = 0, canSpeak = true) {
  return frames.map((detections, i) => manager.update(detections, startAt + i * STEP, canSpeak));
}

class FakeSynth {
  constructor() { this.spoken = []; this.speaking = false; this.pending = false; this.cancels = 0; }
  speak(utterance) { this.spoken.push(utterance); this.speaking = true; }
  cancel() { this.cancels += 1; this.speaking = false; this.pending = false; }
  finish() { this.speaking = false; }
}
class FakeUtterance { constructor(text) { this.text = text; } }
const speaker = (synth = new FakeSynth(), errors = []) =>
  [new Speaker(synth, FakeUtterance, (e) => errors.push(e)), synth, errors];

test("phrases", () => {
  assert.equal(phrase("person", "CENTER"), "Person ahead.");
  assert.equal(phrase("chair", "RIGHT"), "Chair on your right.");
  assert.equal(phrase("bottle", "LEFT"), "Bottle on your left.");
  assert.equal(phrase("cell phone", "LEFT"), "Cell phone on your left.");
  assert.equal(phrase("chair", "BEHIND"), null);
  assert.equal(phrase("", "LEFT"), null);
});

test("stable object is announced on its second consecutive sighting", () => {
  const m = new AnnouncementManager();
  assert.deepEqual(run(m, [[d("chair", "RIGHT")], [d("chair", "RIGHT")]]), [null, "Chair on your right."]);
});

test("one-frame false detection is ignored", () => {
  const m = new AnnouncementManager();
  assert.deepEqual(run(m, [[d("dog", "LEFT")], [], [], []]), [null, null, null, null]);
  // seen, missed, seen: never two results in a row
  const flicker = [[d("cup", "LEFT")], [], [d("cup", "LEFT")], [], [d("cup", "LEFT")]];
  assert.deepEqual(run(new AnnouncementManager(), flicker), [null, null, null, null, null]);
});

test("duplicate announcement is suppressed while the object stays", () => {
  const m = new AnnouncementManager();
  const said = run(m, Array(20).fill([d("chair", "RIGHT")]));   // 30 s, well past the cooldown
  assert.deepEqual(said.filter(Boolean), ["Chair on your right."]);
});

test("cooldown: the same kind of object is not announced again within 10 s", () => {
  const m = new AnnouncementManager();
  // person ahead, then the person moves to the right and stays there
  const frames = [[d("person", "CENTER")], [d("person", "CENTER")],
                  ...Array(8).fill([d("person", "RIGHT")])];
  const said = run(m, frames);
  assert.equal(said[1], "Person ahead.");                        // t = 1.5 s
  assert.deepEqual(said.slice(2, 8), Array(6).fill(null));       // t = 3 .. 10.5 s: inside cooldown
  assert.equal(said[8], "Person on your right.");                // t = 12 s: 10.5 s after the first
  assert.equal(said[9], null);
});

test("cooldown is configurable", () => {
  const m = new AnnouncementManager({ cooldownMs: 0 });
  const said = run(m, [[d("person", "CENTER")], [d("person", "CENTER")], [d("person", "RIGHT")], [d("person", "RIGHT")]]);
  assert.deepEqual(said, [null, "Person ahead.", null, "Person on your right."]);
});

test("rapidly changing positions do not cause chatter", () => {
  const m = new AnnouncementManager();
  const frames = [];
  for (let i = 0; i < 12; i++) frames.push([d("person", ["LEFT", "CENTER", "RIGHT"][i % 3])]);
  assert.deepEqual(run(m, frames).filter(Boolean), []);          // never in one place twice in a row
});

test("brief disappearance does not reset the announcement", () => {
  const m = new AnnouncementManager();
  const frames = [[d("chair", "RIGHT")], [d("chair", "RIGHT")],     // announced at 1.5 s
                  [], [],                                           // gone for 3 s (< 5 s)
                  ...Array(10).fill([d("chair", "RIGHT")])];         // back for 15 s
  assert.deepEqual(run(m, frames).filter(Boolean), ["Chair on your right."]);
});

test("disappearing for 5 s or more allows another announcement", () => {
  const m = new AnnouncementManager();
  const chair = [d("chair", "RIGHT")];
  // announced at 1.5 s; last seen 1.5 s; absent until 12 s (10.5 s); needs two sightings again
  const frames = [chair, chair, [], [], [], [], [], [], chair, chair];
  const said = run(m, frames);
  assert.equal(said[1], "Chair on your right.");
  assert.equal(said[8], null);
  assert.equal(said[9], "Chair on your right.");
});

test("a returning object still waits for the cooldown", () => {
  const m = new AnnouncementManager();
  const chair = [d("chair", "RIGHT")];
  // announced at 1.5 s, absent 6 s, back at 7.5 s and 9 s: forgotten, but only 7.5 s since it was spoken
  const said = run(m, [chair, chair, [], [], [], chair, chair, chair, chair]);
  assert.deepEqual(said.slice(5, 8), [null, null, null]);
  assert.equal(said[8], "Chair on your right.");                 // 12 s: cooldown over
});

test("multiple objects: one per cycle, person first, then by confidence", () => {
  const m = new AnnouncementManager();
  const scene = [d("chair", "RIGHT", 0.95), d("person", "CENTER", 0.6), d("bottle", "LEFT", 0.7), d("tv", "CENTER", 0.8)];
  const said = run(m, Array(6).fill(scene));
  assert.deepEqual(said, [null, "Person ahead.", "Chair on your right.", "Tv ahead.", "Bottle on your left.", null]);
});

test("several of the same thing in one place are one announcement", () => {
  const m = new AnnouncementManager();
  const scene = [d("chair", "RIGHT", 0.6), d("chair", "RIGHT", 0.9), d("chair", "RIGHT", 0.7)];
  assert.deepEqual(run(m, Array(4).fill(scene)).filter(Boolean), ["Chair on your right."]);
});

test("empty and malformed results announce nothing", () => {
  const m = new AnnouncementManager();
  assert.deepEqual(run(m, [[], null, undefined, [{ class_name: "x" }], [d("chair", "UP")], [d("chair", "UP")]]),
                   [null, null, null, null, null, null]);
});

test("no speech when voice is disabled, and nothing is used up", () => {
  const m = new AnnouncementManager();
  const chair = [d("chair", "RIGHT")];
  assert.deepEqual(run(m, Array(5).fill(chair), 0, false), Array(5).fill(null));
  assert.equal(m.update(chair, 5 * STEP, true), "Chair on your right.");   // said once voice is on
});

test("reset forgets everything", () => {
  const m = new AnnouncementManager();
  run(m, [[d("chair", "RIGHT")], [d("chair", "RIGHT")]]);
  m.reset();
  assert.deepEqual(run(m, [[d("chair", "RIGHT")], [d("chair", "RIGHT")]], 3000), [null, "Chair on your right."]);
});

test("speaker is off by default and says nothing", () => {
  const [s, synth] = speaker();
  assert.equal(s.enabled, false);
  assert.equal(s.say("Person ahead."), false);
  assert.equal(synth.spoken.length, 0);
});

test("enabling speaks a test phrase; disabling cancels and silences", () => {
  const [s, synth] = speaker();
  assert.equal(s.enable("Voice announcements on."), true);
  assert.deepEqual(synth.spoken.map((u) => u.text), ["Voice announcements on."]);
  s.disable();
  assert.equal(synth.speaking, false);
  assert.equal(s.say("Person ahead."), false);
  assert.equal(synth.spoken.length, 1);
});

test("speech never queues up behind an unfinished phrase", () => {
  const [s, synth] = speaker();
  s.enable();
  assert.equal(s.say("Person ahead."), true);
  assert.equal(s.canSpeak, false);
  assert.equal(s.say("Chair on your right."), false);            // dropped, not queued
  assert.equal(synth.spoken.length, 1);
  synth.finish();
  assert.equal(s.say("Chair on your right."), true);
});

test("no queued speech after stopping", () => {
  const [s, synth] = speaker();
  s.enable();
  s.say("Person ahead.");
  synth.pending = true;                                          // as if the browser had more waiting
  s.cancel();                                                    // what stopping awareness does
  assert.equal(synth.speaking || synth.pending, false);
  assert.equal(synth.spoken.length, 1);
});

test("a phrase that could not be spoken is not marked as announced", () => {
  const [s, synth] = speaker();
  const m = new AnnouncementManager();
  s.enable();
  s.say("something long");                                       // still speaking
  const chair = [d("chair", "RIGHT")];
  assert.equal(m.update(chair, 0, s.canSpeak), null);
  assert.equal(m.update(chair, STEP, s.canSpeak), null);         // ready, but the speaker is busy
  synth.finish();
  const text = m.update(chair, 2 * STEP, s.canSpeak);
  assert.equal(text, "Chair on your right.");
  assert.equal(s.say(text), true);
});

test("unsupported browser and speech errors are reported, not thrown", () => {
  const unsupported = new Speaker(undefined, undefined);
  assert.equal(unsupported.supported, false);
  assert.equal(unsupported.enable("test"), false);
  assert.equal(unsupported.say("Person ahead."), false);
  unsupported.cancel();

  const [s, synth, errors] = speaker();
  s.enable();
  s.say("Person ahead.");
  synth.spoken[0].onerror({ error: "canceled" });                // our own cancel is not an error
  synth.spoken[0].onerror({ error: "not-allowed" });
  assert.deepEqual(errors, ["not-allowed"]);

  const throwing = new FakeSynth();
  throwing.speak = () => { throw new Error("audio blocked"); };
  const [t, , thrown] = speaker(throwing);
  t.enable();
  assert.equal(t.say("Person ahead."), false);
  assert.deepEqual(thrown, ["audio blocked"]);
});

test("volume is applied to each phrase", () => {
  const [s, synth] = speaker();
  s.enable();
  s.volume = 0.4;
  s.say("Person ahead.");
  assert.equal(synth.spoken[0].volume, 0.4);
});
