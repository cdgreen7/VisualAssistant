// Pure helpers for drawing detections over the camera preview and for the
// scene panel. No DOM access, so they are tested under Node (tests/js).
(function (root) {
  "use strict";

  const POSITION_WORDS = { LEFT: "Left", CENTER: "Ahead", RIGHT: "Right" };

  // Where the camera picture sits inside its box. The preview uses object-fit: cover,
  // so the picture is scaled to fill the box and cropped equally on the overflowing sides.
  function coverTransform(viewWidth, viewHeight, imageWidth, imageHeight) {
    const scale = Math.max(viewWidth / imageWidth, viewHeight / imageHeight);
    return { scale, dx: (viewWidth - imageWidth * scale) / 2, dy: (viewHeight - imageHeight * scale) / 2 };
  }

  // A box in image pixels -> a rectangle in preview (CSS) pixels.
  // mirrored: the preview is flipped left-to-right (a mirrored front camera).
  function mapBox(box, imageWidth, imageHeight, viewWidth, viewHeight, mirrored) {
    const t = coverTransform(viewWidth, viewHeight, imageWidth, imageHeight);
    const width = (box[2] - box[0]) * t.scale, height = (box[3] - box[1]) * t.scale;
    let x = box[0] * t.scale + t.dx;
    if (mirrored) x = viewWidth - x - width;
    return { x, y: box[1] * t.scale + t.dy, width, height };
  }

  // True when a detection result still matches the live picture's shape. After the
  // phone is rotated the old result has the other orientation and must not be drawn.
  function sameShape(imageWidth, imageHeight, videoWidth, videoHeight, tolerance = 0.03) {
    if (!imageWidth || !imageHeight || !videoWidth || !videoHeight) return false;
    return Math.abs(imageWidth / imageHeight - videoWidth / videoHeight) <= tolerance * (videoWidth / videoHeight);
  }

  // "person" + track 1 -> "Person #1"; no track id -> "Person"
  function objectLabel(detection) {
    const name = String(detection.class_name || "object");
    const id = Number.isInteger(detection.track_id) ? ` #${detection.track_id}` : "";
    return name.charAt(0).toUpperCase() + name.slice(1) + id;
  }

  // Text for the scene panel: at most maxItems objects, then "+N more", then a count.
  function sceneLines(detections, maxItems) {
    const all = detections || [];
    const shown = all.slice(0, maxItems).map((d) => `${objectLabel(d)} — ${POSITION_WORDS[d.position] || d.position}`);
    const more = all.length > shown.length ? `+${all.length - shown.length} more` : null;
    const summary = all.length === 0 ? "No objects detected" :
      `${all.length} object${all.length === 1 ? "" : "s"} detected`;
    return { items: shown, more, summary };
  }

  const api = { coverTransform, mapBox, sameShape, objectLabel, sceneLines, POSITION_WORDS };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.View = api;
})(typeof self !== "undefined" ? self : this);
