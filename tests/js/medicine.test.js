const {test} = require('node:test');
const assert = require('node:assert/strict');
const {CandidateGate} = require('../../web/static/medicine-candidates.js');
const {MedicineCapture, mapBox, dimensions} = require('../../web/static/medicine-capture.js');
const MedicineScanner = require('../../web/static/medicine-scanner.js');
const bottle = (id=1, box=[160,80,320,320]) => ({class_name:'bottle',confidence:.9,track_id:id,bounding_box:box});
const body = (detections=[bottle()], tracking='ok') => ({image_width:640,image_height:360,tracking,detections});
function stable(gate, start=0, id=1) {
  assert.equal(gate.update(body([bottle(id)]), start), null);
  assert.equal(gate.update(body([bottle(id)]), start+1500), null);
  return gate.update(body([bottle(id)]),start+3000);
}
test('three stable bottle sightings return immutable detection metadata', () => {
  const g = new CandidateGate();
  const candidate = stable(g);
  assert.equal(candidate.track_id,1);
  assert.deepEqual(candidate.bounding_box,[160,80,320,320]);
});
test('single-frame false detection and missing frames reset consecutive stability', () => {
  const g = new CandidateGate();
  g.update(body(),0);
  g.update(body([]),1500);
  assert.equal(g.update(body(),3000),null);
  assert.equal(g.update(body(),4500),null);
  assert.ok(g.update(body(),6000));
});
test('single-frame gaps retain suppression, continuous object never repeats after cooldown', () => {
  const g = new CandidateGate();
  assert.ok(stable(g));
  for (let t=4500;t<65000;t+=1500) {
    assert.equal(g.update(body(t===6000 ? [] : [bottle()]),t),null);
  }
});
test('tracking ID change requires fresh stability but does not duplicate captured region', () => {
  const g = new CandidateGate();
  g.update(body(),0); g.update(body(),1500);
  assert.equal(g.update(body([bottle(2)]),3000),null);
  assert.equal(g.update(body([bottle(2)]),4500),null);
  assert.ok(g.update(body([bottle(2)]),6000));
  for(let t=7500;t<45000;t+=1500) assert.equal(g.update(body([bottle(3)]),t),null);
});
test('missing IDs use five spatial sightings; changes between missing/present restart stability', () => {
  const g = new CandidateGate();
  for(let i=0;i<4;i++) assert.equal(g.update(body([bottle(null)]),i*1500),null);
  assert.equal(g.update(body([bottle(null)]),6000).track_id,null);
  const h = new CandidateGate();
  h.update(body(),0);h.update(body(),1500);
  assert.equal(h.update(body([bottle(null)]),3000),null);
  assert.equal(h.current.hits,1);
});
test('untracked API status ignores numeric ID; stale responses cannot trigger', () => {
  const g = new CandidateGate();
  g.update(body(),0);
  assert.equal(g.update(body([bottle()],'stale'),1500),null);
  assert.equal(g.current,null);
  for(let i=0;i<4;i++) assert.equal(g.update(body([bottle()],'off'),3000+i*1500),null);
  assert.equal(g.update(body([bottle()],'off'),9000).track_id,null);
});
test('cooldown survives departure and ID changes; leaving releases active candidate', () => {
  const g = new CandidateGate({cooldownMs:10000});
  stable(g);
  g.update(body([]),9000);
  assert.equal(g.current,null);
  assert.equal(g.update(body([bottle(2)]),9500),null);
  assert.equal(g.update(body([bottle(2)]),11000),null);
  assert.ok(g.update(body([bottle(2)]),14000));
});
test('busy state does not consume a candidate; independent gates do not share IDs', () => {
  const g = new CandidateGate();
  g.update(body(),0);g.update(body(),1500);
  assert.equal(g.update(body(),3000,false),null);
  assert.ok(g.update(body(),4500,true));
  assert.ok(stable(new CandidateGate()));
});
test('reject tiny, low-confidence, invalid, other-class, and multiple containers', () => {
  for(const detections of [[bottle(1,[1,1,3,3])],[{...bottle(),confidence:.4}],
    [{...bottle(),class_name:'cup'}],[bottle(1),bottle(2)], [bottle(1,[0,0,NaN,40])]]) {
    const g=new CandidateGate();
    for(let i=0;i<8;i++) assert.equal(g.update(body(detections),1500*i),null);
  }
});
test('movement, long gaps and reset discard stability', () => {
  const g=new CandidateGate();g.update(body(),0);g.update(body(),1500);
  assert.equal(g.update(body([bottle(1,[400,80,560,320])]),3000),null);
  assert.equal(g.current.hits,1);
  g.update(body(),10000);
  assert.equal(g.current.hits,1);
  g.reset();assert.equal(g.current,null);
});
test('bounding boxes use independent x/y scales and clamp to frame', () => {
  assert.deepEqual(mapBox([160,90,320,270],640,360,1920,1080),[480,270,960,810]);
  assert.deepEqual(mapBox([-10,0,650,360],640,360,1920,1080),[0,0,1920,1080]);
  assert.throws(()=>mapBox([0,0,10,10],640,360,1080,1920),/orientation/);
  assert.throws(()=>mapBox([10,10,5,5],640,360,1920,1080),/Empty/);
});
function canvases(size=100) {
  const list=[];
  return {list, create: () => {
    const canvas={width:0,height:0,calls:[],getContext(){return {drawImage:(...args)=>canvas.calls.push(args)};},
      toBlob(callback,type,quality){canvas.quality=quality;callback(new Blob(['x'.repeat(size)],{type}));}};
    list.push(canvas);return canvas;
  }};
}
test('high-res frame and 640 awareness use same source pixels; correct OCR crop', async () => {
  const c=canvases(), capture=new MedicineCapture(c.create), video={videoWidth:1920,videoHeight:1080};
  const snapshot=await capture.freeze(video);
  assert.equal(c.list[0].calls[0][0],video);
  assert.equal(c.list[1].calls[0][0],snapshot.frame);
  assert.deepEqual([snapshot.detectionWidth,snapshot.detectionHeight],[640,360]);
  assert.equal(c.list[1].quality,.7);
  const packet=await capture.prepare(snapshot,{...bottle(), image_width:640,image_height:360,
    bounding_box:bottle().bounding_box});
  assert.deepEqual(packet.metadata.capture_box,[480,240,960,960]);
  assert.equal(packet.metadata.crop_width,480);
  assert.equal(packet.image.type,'image/jpeg');
  assert.equal(c.list[0].quality,.92);
  assert.equal(c.list[2].calls[0][0],snapshot.frame);
  assert.equal(packet.metadata.product_identity_verified,false);
});
test('portrait rotation, max dimensions and no upscaling', async () => {
  assert.deepEqual(dimensions(4000,3000,2560),{width:2560,height:1920});
  assert.deepEqual(dimensions(640,480,2560),{width:640,height:480});
  const c=canvases(), capture=new MedicineCapture(c.create);
  const s=await capture.freeze({videoWidth:1080,videoHeight:1920});
  assert.deepEqual([s.detectionWidth,s.detectionHeight],[360,640]);
  const p=await capture.prepare(s,{image_width:360,image_height:640,bounding_box:[90,160,270,480]});
  assert.deepEqual(p.metadata.capture_box,[270,480,810,1440]);
});
test('invalid camera, size, and detection dimensions fail clearly', async () => {
  const c=canvases(100),capture=new MedicineCapture(c.create,{maxBytes:10});
  await assert.rejects(()=>capture.freeze({videoWidth:0,videoHeight:0}),/not ready/);
  const s=await capture.freeze({videoWidth:1920,videoHeight:1080});
  await assert.rejects(()=>capture.prepare(s,{image_width:360,image_height:640,bounding_box:[0,0,20,20]}),/match/);
  await assert.rejects(()=>capture.prepare(s,{image_width:640,image_height:360,bounding_box:[0,0,200,200]}),/8 MiB/);
});
function scanner(prepare) {
  const events=[];
  const s=new MedicineScanner(new CandidateGate(), {prepare: prepare || (async()=>({image:new Blob(),crop:new Blob()}))},
    {clear:()=>events.push('clear'),status:t=>events.push(t),ready:p=>events.push(p)});
  return {s,events};
}
test('disabled by default; UI stages and image released when object leaves', async () => {
  const {s,events}=scanner();
  await s.observe(body(),{},0); assert.equal(s.gate.current,null);
  s.enable(true);
  for(let i=0;i<3;i++) await s.observe(body(),{},i*1500);
  assert.ok(s.packet);
  assert.ok(events.includes('Possible medicine container detected'));
  assert.ok(events.includes('Preparing label scan'));
  assert.ok(events.includes('Image ready for identification'));
  await s.observe(body([]),{},9000);
  assert.equal(s.packet,null);
});
test('one preparation at a time; disabling or rotation discards late result', async () => {
  let release,calls=0;
  const {s}=scanner(async()=>{calls++;return new Promise(r=>{release=r;});});
  s.enable(true);await s.observe(body(),{},0);await s.observe(body(),{},1500);
  const pending=s.observe(body(),{},3000);
  await s.observe(body(),{},4500);
  assert.equal(calls,1); assert.equal(s.busy,true);
  s.invalidate();release({image:new Blob(),crop:new Blob()});await pending;
  assert.equal(s.packet,null);assert.equal(s.busy,false);
});
test('no snapshot/no permission produces no capture; preparation failure does not queue', async () => {
  let calls=0; const {s,events}=scanner(async()=>{calls++;throw new Error('encoding failed');});
  s.enable(true);
  for(let i=0;i<4;i++) await s.observe(body(),null,i*1500);
  assert.equal(calls,0);
  await s.observe(body(),{},6000,false);assert.equal(calls,0);
  await s.observe(body(),{},7500);
  assert.equal(calls,1);assert.equal(s.busy,false);
  assert.ok(events.includes('Capture unavailable: encoding failed'));
});
test('app integration keeps detection at 640, prepares native snapshot, automatically sends one crop without blocking awareness', async () => {
  const vm=require('node:vm'), fs=require('node:fs');
  const nodes=new Map(), c=canvases(), requests=[], revoked=[];
  let clock=0;
  function node(id) {
    if(!nodes.has(id)) nodes.set(id,{value:'1',hidden:false,disabled:false,checked:false,events:{},className:'',
      classList:{toggle(){},add(){},remove(){}},
      addEventListener(name,fn){const prior=this.events[name];this.events[name]=(...args)=>{if(prior)prior(...args);fn(...args);};},setAttribute(){},removeAttribute(){},
      replaceChildren(){},prepend(){},append(){},getContext(){return {drawImage(){},setTransform(){},clearRect(){}};},
      toBlob(cb){cb(new Blob(['frame'],{type:'image/jpeg'}));}});
    return nodes.get(id);
  }
  const video=node('video');video.videoWidth=1920;video.videoHeight=1080;
  const window={events:{},isSecureContext:true,addEventListener(name,fn){this.events[name]=fn;}};
  const ctx=vm.createContext({window,document:{getElementById:node,hidden:false,addEventListener(){},
    createTextNode:t=>t,createElement:tag=>tag==='canvas'?c.create():node(Symbol())},
    navigator:{},UI:require('../../web/static/ui.js'),Announcer:require('../../web/static/announcer.js'),
    AskAI:require('../../web/static/ask-ai.js'),View:require('../../web/static/view.js'),
    MedicineCandidates:require('../../web/static/medicine-candidates.js'),
    MedicineCapture:require('../../web/static/medicine-capture.js'),MedicineScanner, MedicineIdentification:require('../../web/static/medicine-identification.js'),
    URL:{createObjectURL:()=> 'blob:preview',revokeObjectURL:url=>revoked.push(url)},
    Blob,FormData,AbortController,performance:{now:()=>clock},
    setTimeout:()=>1,clearTimeout(){},crypto:{},fetch:async(url,options)=>{
      requests.push({url,options});return {ok:true,json:async()=>url==='/health'?{}:url==='/api/medicine/identify-crop'?{session_id:options.body.get('session_id'),capture_id:options.body.get('capture_id'),match:{status:'UNCERTAIN',reason:'fixture'},timing:{},qwen:{enabled:false},formulation:'Ibuprofen 200 mg',label_recognized:'Ibuprofen 200 mg',exact_package:'Not verified'}:body()};
    }});
  vm.runInContext(fs.readFileSync(require.resolve('../../web/static/app.js'),'utf8'),ctx);
  vm.runInContext('stream = {}; awareness.on = true; awareness.session = "session_test"; renderAwareness = () => {};',ctx);
  assert.equal(vm.runInContext('medicineScanner.enabled',ctx),false);
  await vm.runInContext('awarenessTick()',ctx);
  assert.equal(node('canvas').width,640); // default unchanged path
  vm.runInContext('medicineScanner.enable(true)',ctx);
  for(let i=0;i<3;i++){clock=1500*(i+1);await vm.runInContext('awarenessTick()',ctx);}
  assert.equal(vm.runInContext('medicineScanner.packet.metadata.capture_width',ctx),1920);
  assert.equal(vm.runInContext('medicineScanner.packet.metadata.track_id',ctx),1);
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(node('medicine-status').textContent,/UNCERTAIN/);
  assert.ok(requests.every(r=>['/health','/api/detect','/api/medicine/identify-crop'].includes(r.url)));
  assert.equal(vm.runInContext('awareness.on',ctx),true);
  assert.equal(requests.filter(r=>r.url==='/api/medicine/identify-crop').length,1);
  vm.runInContext('awareness.last = null',ctx);
  video.events.resize();
  assert.equal(vm.runInContext('medicineScanner.packet',ctx),null);
  assert.deepEqual(revoked,['blob:preview']);
  // A failed automatic capture falls back to the existing normal capture.
  vm.runInContext('medicineCapture.freeze = async () => {throw new Error("capture failed");}',ctx);
  clock+=1500;await vm.runInContext('awarenessTick()',ctx);
  assert.equal(vm.runInContext('awareness.failures',ctx),0);
  assert.equal(vm.runInContext('awareness.on',ctx),true);
  assert.match(node('medicine-status').textContent,/capture failed/);
});
