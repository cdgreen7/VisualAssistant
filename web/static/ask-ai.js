// One snapshot from the existing browser preview; no camera is opened here.
(function (root) {
  "use strict";
  class AskAI {
    constructor(ui) { this.ui = ui; this.busy = false; this.active = false; this.text = ""; }
    async ask() {
      if (this.busy || !this.ui.ready()) return;
      this.busy = true;
      this.active = true; // pause environmental speech while the panel is open
      this.text = "";
      this.ui.speaker.cancel();
      this.ui.render("Describing this frame…", true);
      this.ui.update();
      const controller = new AbortController();
      this.controller = controller;
      const timer = setTimeout(() => controller.abort(), 75000);
      try {
        const blob = await this.ui.capture(1024, 0.85);
        if (controller.signal.aborted) return;
        const form = new FormData();
        form.append("image", blob, "scene.jpg");
        const response = await this.ui.fetch("/api/scene/describe", { method: "POST", body: form, signal: controller.signal });
        const body = await response.json();
        if (!response.ok) throw new Error(body.error ? body.error.message : "Ask AI failed.");
        if (typeof body.description !== "string" || !body.description.trim()) throw new Error("No scene description was returned.");
        if (!this.active || controller.signal.aborted) return;
        this.text = body.description;
        this.ui.render(this.text, false);
        if (this.ui.speaker.enabled && this.ui.visible()) this.speak();
      } catch (error) {
        if (this.active) this.ui.render(error.name === "AbortError" ? "Ask AI timed out. Try again." : error.message, false);
      } finally {
        clearTimeout(timer);
        this.busy = false;
        this.controller = null;
        this.ui.update();
      }
    }
    speak() {
      if (!this.text || !this.active) return;
      const speaker = this.ui.speaker;
      speaker.cancel();
      if (!speaker.enabled) speaker.enable(); // Read aloud button is a user gesture
      speaker.say("AI-generated scene description. " + this.text);
      this.ui.voiceUpdate();
    }
    close() {
      this.active = false;
      this.text = "";
      if (this.controller) this.controller.abort();
      this.ui.speaker.cancel();
      this.ui.hide();
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = AskAI;
  else root.AskAI = AskAI;
})(typeof self !== "undefined" ? self : this);
