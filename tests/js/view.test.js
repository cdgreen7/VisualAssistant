// Run with: node --test tests/js   (also run by tests/test_frontend_js.py)
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { coverTransform, mapBox, sameShape, objectLabel, sceneLines } = require("../../web/static/view.js");

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-6, `${actual} != ${expected}`);
function assertBox(actual, expected) {
  for (const key of ["x", "y", "width", "height"]) near(actual[key], expected[key]);
}

test("same shape: the picture simply scales", () => {
  assert.deepEqual(coverTransform(320, 180, 640, 360), { scale: 0.5, dx: 0, dy: 0 });
  assertBox(mapBox([100, 50, 300, 250], 640, 360, 320, 180), { x: 50, y: 25, width: 100, height: 100 });
});

test("portrait phone showing a portrait frame taller than the camera: sides are cropped", () => {
  // 360x640 frame in a 390x780 preview: scale by height (1.21875), 24.375 px cut from each side
  const t = coverTransform(390, 780, 360, 640);
  near(t.scale, 1.21875); near(t.dx, -24.375); near(t.dy, 0);
  assertBox(mapBox([0, 0, 360, 640], 360, 640, 390, 780), { x: -24.375, y: 0, width: 438.75, height: 780 });
  // the centre of the image stays in the centre of the preview
  const centre = mapBox([170, 310, 190, 330], 360, 640, 390, 780);
  near(centre.x + centre.width / 2, 195); near(centre.y + centre.height / 2, 390);
});

test("landscape frame in a portrait preview: heavy side crop, still centred", () => {
  const box = mapBox([310, 170, 330, 190], 640, 360, 390, 780);   // image centre
  near(box.x + box.width / 2, 195); near(box.y + box.height / 2, 390);
  near(box.width, 20 * (780 / 360));
  assert.ok(mapBox([0, 0, 50, 50], 640, 360, 390, 780).x < 0);    // left edge of the image is off-screen
});

test("wide preview: top and bottom are cropped", () => {
  const t = coverTransform(1000, 400, 640, 480);
  near(t.scale, 1.5625); near(t.dx, 0); near(t.dy, -175);
});

test("mirrored preview flips boxes left-to-right only", () => {
  const plain = mapBox([0, 40, 100, 140], 640, 360, 640, 360, false);
  const mirrored = mapBox([0, 40, 100, 140], 640, 360, 640, 360, true);
  assertBox(plain, { x: 0, y: 40, width: 100, height: 100 });
  assertBox(mirrored, { x: 540, y: 40, width: 100, height: 100 });
  // with cropping too: mirrored x is the reflection about the preview's centre line
  const a = mapBox([100, 50, 300, 250], 640, 360, 390, 780, false);
  const b = mapBox([100, 50, 300, 250], 640, 360, 390, 780, true);
  near(a.x + a.width / 2 + b.x + b.width / 2, 390);
  near(a.y, b.y);
});

test("a result from before a rotation is not drawn on the rotated picture", () => {
  assert.equal(sameShape(640, 360, 1920, 1080), true);
  assert.equal(sameShape(360, 640, 1080, 1920), true);
  assert.equal(sameShape(640, 360, 1080, 1920), false);   // landscape result, portrait video
  assert.equal(sameShape(640, 480, 1920, 1080), false);
  assert.equal(sameShape(640, 360, 0, 0), false);         // camera has no picture
});

test("object labels show the tracking id only when there is one", () => {
  assert.equal(objectLabel({ class_name: "person", track_id: 1 }), "Person #1");
  assert.equal(objectLabel({ class_name: "cell phone", track_id: 12 }), "Cell phone #12");
  assert.equal(objectLabel({ class_name: "chair", track_id: null }), "Chair");
  assert.equal(objectLabel({ class_name: "chair" }), "Chair");            // older server without tracking
  assert.equal(objectLabel({ class_name: "chair", track_id: "7" }), "Chair");
});

test("scene panel lines", () => {
  const scene = [
    { class_name: "person", position: "CENTER", track_id: 1 },
    { class_name: "chair", position: "RIGHT", track_id: 2 },
    { class_name: "bottle", position: "LEFT", track_id: 3 },
  ];
  assert.deepEqual(sceneLines(scene, 4), {
    items: ["Person #1 — Ahead", "Chair #2 — Right", "Bottle #3 — Left"],
    more: null, summary: "3 objects detected",
  });
  assert.deepEqual(sceneLines(scene.slice(0, 1), 4).summary, "1 object detected");
  assert.deepEqual(sceneLines([], 4), { items: [], more: null, summary: "No objects detected" });
  assert.deepEqual(sceneLines(null, 4).summary, "No objects detected");
});

test("scene panel is limited and summarises the rest", () => {
  const many = Array.from({ length: 7 }, (_, i) => ({ class_name: "chair", position: "LEFT", track_id: i + 1 }));
  const lines = sceneLines(many, 4);
  assert.equal(lines.items.length, 4);
  assert.equal(lines.more, "+3 more");
  assert.equal(lines.summary, "7 objects detected");
  assert.equal(sceneLines(many, 2).more, "+5 more");
});

test("scene panel shows objects without a tracking id", () => {
  assert.deepEqual(sceneLines([{ class_name: "chair", position: "RIGHT", track_id: null }], 4).items, ["Chair — Right"]);
});

test('iPhone viewport cover geometry stays centered without margins across rotation and mirroring', () => {
  for (const [vw,vh] of [[320,568],[375,667],[390,844],[393,852],[430,932],[844,390]]) {
    for (const [iw,ih] of [[1080,1920],[1920,1080]]) {
      const t=coverTransform(vw,vh,iw,ih);
      assert.ok(iw*t.scale>=vw-1e-6 && ih*t.scale>=vh-1e-6);
      assert.ok(t.dx<=1e-6 && t.dy<=1e-6);
      for (const mirrored of [false,true]) {
        const box=mapBox([iw/4,ih/4,iw*3/4,ih*3/4],iw,ih,vw,vh,mirrored);
        assert.ok(Math.abs(box.x+box.width/2-vw/2)<1e-6);
        assert.ok(Math.abs(box.y+box.height/2-vh/2)<1e-6);
      }
    }
  }
});
