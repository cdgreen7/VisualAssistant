const {test}=require('node:test'), assert=require('node:assert/strict');
const Identify=require('../../web/static/medicine-identification.js');
function packet(frame=1){return {crop:new Blob(['jpeg'],{type:'image/jpeg'}),metadata:{session_id:'session_1234',frame,captured_at:1000+frame}};}
function setup(fetcher){
  const results=[],status=[];let calls=0;
  const service=new Identify({assist:()=>false,clear(){},status:t=>status.push(t),result:r=>results.push(r),fetch:async(url,opt)=>{
    calls++;assert.equal(url,'/api/medicine/identify-crop');assert.equal(opt.body.get('image').type,'image/jpeg');
    if(fetcher)return fetcher(opt);
    return {ok:true,json:async()=>({session_id:opt.body.get('session_id'),capture_id:opt.body.get('capture_id'),timing:{},match:{status:'UNCERTAIN'}})};
  }});return {service,results,status,calls:()=>calls};
}
test('automatic crop submission, identifiers, timing and repeated capture suppression',async()=>{
  const s=setup();await s.service.identify(packet());await s.service.identify(packet());
  assert.equal(s.calls(),1);assert.equal(s.results.length,1);assert.ok(s.results[0].timing.round_trip_ms>=0);
});
test('concurrent calls cannot queue; cancellation rejects late successful response',async()=>{
  let release;const wait=new Promise(r=>release=r);
  const s=setup(async opt=>{await wait;return {ok:true,json:async()=>({session_id:opt.body.get('session_id'),capture_id:opt.body.get('capture_id'),timing:{},match:{status:'VERIFIED'}})};});
  const first=s.service.identify(packet());await s.service.identify(packet(2));assert.equal(s.calls(),1);
  s.service.cancel();release();await first;assert.equal(s.results.length,0);assert.equal(s.service.busy,false);
});
test('wrong session or capture ID rejected after restarts',async()=>{
  const s=setup(async()=>({ok:true,json:async()=>({session_id:'old_session',capture_id:'wrong_capture',timing:{}})}));
  await s.service.identify(packet());assert.equal(s.results.length,0);
});
test('rescan with a fresh capture and failed request recovery',async()=>{
  const s=setup(async()=>({ok:false,json:async()=>({error:{message:'Another scan is running'}})}));
  await s.service.identify(packet());assert.equal(s.service.busy,false);assert.match(s.status.at(-1),/Another scan/);
  await s.service.identify(packet(2));assert.equal(s.calls(),2);
});
test('development diagnostics distinguish stale replies and expose a recovery status',async()=>{
  const s=setup(async()=>({status:200,ok:true,json:async()=>({session_id:'old_session',capture_id:'wrong_capture'})}));
  const events=[];s.service.ui.diagnostic=text=>events.push(text);
  await s.service.identify(packet());
  assert.match(events.join('\n'),/POST.*started/);
  assert.ok(events.includes('response HTTP 200'));
  assert.ok(events.includes('response discarded: identifiers differ'));
  assert.match(s.status.at(-1),/rescan/);
});
test('missing session is visible rather than silently leaving Image ready',async()=>{
  const s=setup();const p=packet();p.metadata.session_id=null;
  await s.service.identify(p);
  assert.equal(s.calls(),0);assert.match(s.status.at(-1),/missing session/);
});
