const {test} = require('node:test');
const assert = require('node:assert/strict');
const AskAI = require('../../web/static/ask-ai.js');
const {Speaker} = require('../../web/static/announcer.js');
function setup(fetcher) {
  const said = [], renders = [], captures = [];
  const synth = { speaking: false, pending: false, cancel() { this.speaking = false; },
    speak(u) { said.push(u.text); this.speaking = true; } };
  const speaker = new Speaker(synth, function(text) { this.text = text; });
  const ai = new AskAI({ready: () => true, capture: async (...args) => {
    captures.push(args); return new Blob(['frame'], {type: 'image/jpeg'});
  }, fetch: fetcher || (async (path, options) => {
    assert.equal(path, '/api/scene/describe');
    assert.equal(options.body.get('image').type, 'image/jpeg');
    return {ok: true, json: async () => ({description: 'A bottle in the center.'})};
  }), speaker, render: (...args) => renders.push(args), update() {}, visible: () => true,
    voiceUpdate() {}, hide() { renders.push('hidden'); }});
  return {ai, speaker, synth, said, renders, captures};
}
test('one existing preview capture, result, optional speech, close and cancel', async () => {
  const {ai, speaker, said, captures} = setup();
  speaker.enable();
  await ai.ask();
  assert.deepEqual(captures, [[1024, .85]]);
  assert.equal(said.length, 1);
  assert.match(said[0], /AI-generated/);
  assert.equal(speaker.canSpeak, false); // environmental speech cannot queue
  ai.close();
  assert.equal(speaker.busy, false);
  assert.equal(ai.active, false);
});
test('voice off stays silent; tap Read aloud enables shared speaker', async () => {
  const {ai, said} = setup();
  await ai.ask();
  assert.equal(said.length, 0);
  ai.speak();
  assert.equal(said.length, 1);
});
test('concurrent taps capture once and close ignores late result', async () => {
  let resolve;
  const waiting = new Promise(r => { resolve = r; });
  const {ai, captures} = setup(async () => { await waiting; return {ok:true, json:async () => ({description:'Late'})}; });
  const first = ai.ask();
  await ai.ask();
  assert.equal(captures.length, 1);
  ai.close();
  resolve();
  await first;
  assert.equal(ai.text, '');
  assert.equal(ai.busy, false);
});
test('clear server errors and retry', async () => {
  const {ai, renders} = setup(async () => ({ok:false, json:async () => ({error:{message:'Model unavailable'}})}));
  await ai.ask();
  assert.deepEqual(renders.at(-1), ['Model unavailable', false]);
  assert.equal(ai.busy, false);
});
test('app wiring captures the phone preview, keeps awareness, and gates its voice', async () => {
  const vm = require('node:vm');
  const fs = require('node:fs');
  const nodes = new Map(), drawn = [], requests = [], spoken = [];
  function node(id) {
    if (!nodes.has(id)) nodes.set(id, {hidden: false, disabled: false, value:'1', className:'',
      classList:{toggle(){},add(){},remove(){}},
      addEventListener() {}, setAttribute() {}, removeAttribute() {}, replaceChildren() {}, append() {}, prepend() {},
      getContext: () => ({drawImage: (...args) => drawn.push(args)}),
      toBlob: callback => callback(new Blob(['frame'], {type:'image/jpeg'}))});
    return nodes.get(id);
  }
  const preview = node('video');
  preview.videoWidth = 1920; preview.videoHeight = 1080;
  const document = {getElementById: node, createElement: () => node(Symbol()),
    createTextNode: text => text, addEventListener() {}, hidden:false};
  const window = {addEventListener() {}, isSecureContext:true,
    speechSynthesis: {speaking:false, pending:false, cancel() {}, speak(u) {spoken.push(u.text);}},
    SpeechSynthesisUtterance: function(text) {this.text = text;}};
  const context = vm.createContext({document, window, Announcer: require('../../web/static/announcer.js'),
    AskAI, UI: require('../../web/static/ui.js'), MedicineCandidates: require('../../web/static/medicine-candidates.js'),
    MedicineCapture: require('../../web/static/medicine-capture.js'),
    MedicineScanner: require('../../web/static/medicine-scanner.js'), MedicineIdentification:require('../../web/static/medicine-identification.js'), View:{}, navigator:{}, console, Blob, FormData, AbortController, setTimeout, clearTimeout,
    performance, fetch: async (url, options) => {
      requests.push([url, options]);
      return {ok:true, json:async () => url === '/health' ? {} : {description:'A chair.'}};
    }});
  vm.runInContext(fs.readFileSync(require.resolve('../../web/static/app.js'), 'utf8'), context);
  vm.runInContext('stream = {}; awareness.on = true;', context);
  await vm.runInContext('askAI.ask()', context);
  assert.equal(drawn[0][0], preview);
  assert.equal(node('canvas').width, 1024);
  assert.equal(node('canvas').height, 576);
  assert.equal(vm.runInContext('awareness.on', context), true);
  assert.equal(node('ai-text').textContent, 'A chair.');
  assert.equal(node('ai-panel').hidden, false);
  assert.equal(requests.filter(([url]) => url === '/api/scene/describe').length, 1);
  vm.runInContext('speaker.enable(); announce([{class_name:"person", position:"LEFT", confidence:1}]); announce([{class_name:"person", position:"LEFT", confidence:1}]);', context);
  assert.equal(spoken.length, 0); // panel suppresses environmental announcements
  vm.runInContext('askAI.close(); announce([{class_name:"person", position:"LEFT", confidence:1}]);', context);
  assert.equal(spoken.length, 1); // original announcement system resumes
});
