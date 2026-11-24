// What the interface shows for a given app state, and which sheet is open.
// Pure logic with no DOM access, so it is tested under Node (tests/js/ui.test.js);
// app.js applies the result to the page.
(function (root) {
  "use strict";

  // The single compact indicator in the top bar.
  function statusChip(state) {
    let chip;
    if (state.serverOk === false) chip = { text: "Server offline", tone: "bad" };
    else if (state.awarenessOn) chip = { text: state.awarenessText || "Awareness on", tone: state.awarenessTone || "good" };
    else if (state.cameraOn) chip = { text: "Camera on", tone: "" };
    else chip = { text: "Camera off", tone: "" };
    if (state.voiceOn) chip.text += " · Voice on";
    return chip;
  }

  // Every control's label and availability. Scans never run during awareness,
  // and nothing that needs a picture is offered while the camera is off.
  function controls(state) {
    const { cameraOn, awarenessOn, busy, askBusy, voiceOn, voiceSupported } = state;
    const scanReason = !cameraOn ? "Start the camera first" : awarenessOn ? "Stop awareness first" : "";
    return {
      welcomeVisible: !cameraOn,
      sceneVisible: Boolean(awarenessOn),
      awareness: {
        pressed: Boolean(awarenessOn),
        ariaLabel: awarenessOn ? "Stop awareness" : "Start awareness",
        disabled: Boolean(busy) && !awarenessOn,
      },
      askAI: { disabled: !cameraOn || Boolean(busy) || Boolean(askBusy) },
      voice: {
        pressed: Boolean(voiceOn),
        state: voiceSupported === false ? "Not supported" : voiceOn ? "On" : "Off",
        disabled: voiceSupported === false,
      },
      switchCamera: { disabled: !cameraOn || Boolean(busy), hint: cameraOn ? "" : "Start the camera first" },
      identify: { disabled: Boolean(scanReason) || Boolean(busy), hint: scanReason },
      barcode: { disabled: Boolean(scanReason) || Boolean(busy), hint: scanReason },
      cameraPower: { label: cameraOn ? "Turn camera off" : "Turn camera on", disabled: Boolean(busy) },
    };
  }

  // At most one sheet is open. host supplies show(name), hide(name), focusFirst(name)
  // and focus(element); the element that opened a sheet gets focus back when it closes.
  class Sheets {
    constructor(host) { this.host = host; this.current = null; this.opener = null; }

    open(name, opener) {
      if (this.current === name) return;
      if (this.current) this.host.hide(this.current);
      else this.opener = opener || null;
      this.current = name;
      this.host.show(name);
      this.host.focusFirst(name);
    }

    close(name) {
      if (!this.current || (name && name !== this.current)) return false;
      this.host.hide(this.current);
      this.current = null;
      const opener = this.opener;
      this.opener = null;
      if (opener) this.host.focus(opener);
      return true;
    }

    toggle(name, opener) {
      if (this.current === name) this.close();
      else this.open(name, opener);
    }

    // Escape closes the open sheet. Returns true when the key was used.
    key(key) { return key === "Escape" && this.close(); }
  }

  const api = { statusChip, controls, Sheets };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.UI = api;
})(typeof self !== "undefined" ? self : this);
