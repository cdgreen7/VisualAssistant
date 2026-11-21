// Run with: node --test tests/js   (also run by tests/test_frontend_js.py)
// The interface's state rules (web/static/ui.js): what each control shows for a
// given app state, and how the More menu and result sheet open and close.
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { statusChip, controls, Sheets } = require("../../web/static/ui.js");

const OFF = { cameraOn: false, awarenessOn: false, busy: false, askBusy: false, voiceOn: false, voiceSupported: true, serverOk: true };
const state = (changes) => Object.assign({}, OFF, changes);

function sheets() {
  const events = [];
  const host = {
    show: (name) => events.push("show " + name), hide: (name) => events.push("hide " + name),
    focusFirst: (name) => events.push("focus-first " + name), focus: (element) => events.push("focus " + element),
  };
  return [new Sheets(host), events];
}

test("camera-off state: welcome shown, nothing that needs a picture is offered", () => {
  const view = controls(OFF);
  assert.equal(view.welcomeVisible, true);
  assert.equal(view.sceneVisible, false);
  assert.equal(view.askAI.disabled, true);
  assert.equal(view.switchCamera.disabled, true);
  assert.deepEqual(view.identify, { disabled: true, hint: "Start the camera first" });
  assert.deepEqual(view.barcode, { disabled: true, hint: "Start the camera first" });
  assert.equal(view.cameraPower.label, "Turn camera on");
  assert.equal(view.awareness.disabled, false);            // Camera can always be started
  assert.deepEqual(statusChip(OFF), { text: "Camera off", tone: "" });
});

test("camera started: welcome hidden, Ask AI and scans available", () => {
  const view = controls(state({ cameraOn: true }));
  assert.equal(view.welcomeVisible, false);
  assert.equal(view.askAI.disabled, false);
  assert.deepEqual(view.identify, { disabled: false, hint: "" });
  assert.deepEqual(view.barcode, { disabled: false, hint: "" });
  assert.deepEqual(view.switchCamera, { disabled: false, hint: "" });
  assert.equal(view.cameraPower.label, "Turn camera off");
  assert.deepEqual(statusChip(state({ cameraOn: true })), { text: "Camera on", tone: "" });
});

test("awareness on: Camera control is pressed and scans wait until it stops", () => {
  const on = state({ cameraOn: true, awarenessOn: true, awarenessText: "Awareness on", awarenessTone: "good" });
  const view = controls(on);
  assert.deepEqual(view.awareness, { pressed: true, ariaLabel: "Stop awareness", disabled: false });
  assert.equal(view.sceneVisible, true);
  assert.deepEqual(view.identify, { disabled: true, hint: "Stop awareness first" });
  assert.deepEqual(view.barcode, { disabled: true, hint: "Stop awareness first" });
  assert.equal(view.askAI.disabled, false);                // Ask AI works during awareness
  assert.deepEqual(statusChip(on), { text: "Awareness on", tone: "good" });
  assert.deepEqual(controls(state({ cameraOn: true })).awareness,
                   { pressed: false, ariaLabel: "Start awareness", disabled: false });
});

test("awareness problems show in the status chip", () => {
  const retrying = state({ cameraOn: true, awarenessOn: true, awarenessText: "server busy, retrying", awarenessTone: "warn" });
  assert.deepEqual(statusChip(retrying), { text: "server busy, retrying", tone: "warn" });
  assert.deepEqual(statusChip(state({ serverOk: false, cameraOn: true, awarenessOn: true })),
                   { text: "Server offline", tone: "bad" });
  assert.deepEqual(statusChip(state({ serverOk: null })), { text: "Camera off", tone: "" });   // still checking
});

test("voice toggle", () => {
  assert.deepEqual(controls(OFF).voice, { pressed: false, state: "Off", disabled: false });
  assert.deepEqual(controls(state({ voiceOn: true })).voice, { pressed: true, state: "On", disabled: false });
  assert.deepEqual(controls(state({ voiceSupported: false })).voice, { pressed: false, state: "Not supported", disabled: true });
  assert.equal(statusChip(state({ cameraOn: true, voiceOn: true })).text, "Camera on · Voice on");
});

test("a medicine or barcode scan in progress blocks other captures", () => {
  const view = controls(state({ cameraOn: true, busy: true }));
  assert.equal(view.identify.disabled, true);
  assert.equal(view.barcode.disabled, true);
  assert.equal(view.askAI.disabled, true);
  assert.equal(view.switchCamera.disabled, true);
  assert.equal(view.cameraPower.disabled, true);
  assert.equal(view.awareness.disabled, true);             // cannot start awareness mid-scan
  assert.equal(controls(state({ cameraOn: true, awarenessOn: true, busy: true })).awareness.disabled, false);  // can always stop
});

test("Ask AI is disabled while a description is loading", () => {
  assert.equal(controls(state({ cameraOn: true, askBusy: true })).askAI.disabled, true);
  assert.equal(controls(state({ cameraOn: true, askBusy: true })).identify.disabled, false);
});

test("More menu opens, takes focus, and returns focus when closed", () => {
  const [s, events] = sheets();
  s.open("tools", "more-button");
  assert.equal(s.current, "tools");
  assert.deepEqual(events, ["show tools", "focus-first tools"]);
  assert.equal(s.close(), true);
  assert.equal(s.current, null);
  assert.deepEqual(events.slice(2), ["hide tools", "focus more-button"]);
  assert.equal(s.close(), false);                          // nothing left to close
});

test("More button toggles the menu; Escape closes it", () => {
  const [s] = sheets();
  s.toggle("tools", "more-button");
  assert.equal(s.current, "tools");
  s.toggle("tools", "more-button");
  assert.equal(s.current, null);
  s.open("tools", "more-button");
  assert.equal(s.key("Tab"), false);
  assert.equal(s.current, "tools");
  assert.equal(s.key("Escape"), true);
  assert.equal(s.current, null);
  assert.equal(s.key("Escape"), false);                    // Escape with nothing open is not swallowed
});

test("only one sheet is open: a scan result replaces the menu", () => {
  const [s, events] = sheets();
  s.open("tools", "more-button");
  s.open("result-sheet", "identify-tile");
  assert.equal(s.current, "result-sheet");
  assert.deepEqual(events, ["show tools", "focus-first tools", "hide tools", "show result-sheet", "focus-first result-sheet"]);
  s.close();
  assert.equal(events.at(-1), "focus more-button");        // focus goes back to where the user started
});

test("closing a named sheet leaves a different open sheet alone", () => {
  const [s] = sheets();
  s.open("result-sheet");
  assert.equal(s.close("tools"), false);                   // e.g. a tap on the menu backdrop
  assert.equal(s.current, "result-sheet");
  assert.equal(s.close("result-sheet"), true);
});

test("re-opening the open sheet does nothing", () => {
  const [s, events] = sheets();
  s.open("result-sheet");
  s.open("result-sheet");
  assert.equal(events.length, 2);
});
