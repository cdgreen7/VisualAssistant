const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const source = fs.readFileSync('web/static/app.js', 'utf8');
function setup(fetch) {
  function el(tag, text) { return {tag, textContent:text, children:[], isConnected:true,
    append(...nodes){this.children.push(...nodes);}, replaceChildren(...nodes){this.children=nodes;},
    addEventListener(type,callback){this[type]=callback;}}; }
  const c = vm.createContext({el,fetch,AbortController,setTimeout,clearTimeout,definitionList:rows=>({rows})});
  vm.runInContext(source.slice(source.indexOf('let productResearchBusy'),source.indexOf('function renderBarcode')),c);
  return {make:(body={},upc='')=>c.productResearchCard(body,upc)};
}
function result(records=[]) {return {ok:true,json:async()=>({records,observed:{label:['IBUPROFEN 200 mg']},message:'Research only',total_records:records.length,processing_ms:2})};}
test('explicit click sends only original OCR/evidence, never Qwen or images',async()=>{
  let request, calls=0;
  const ui=setup(async(url,opts)=>{calls++;request=JSON.parse(opts.body);assert.equal(url,'/api/products/research');return result();});
  const card=ui.make({ocr:[{text:'IBUPROFEN',confidence:.9}],match:{evidence:[]},qwen:{lines:['guess']},image:'secret'});
  assert.equal(calls,0);await card.children[0].click();
  assert.equal(calls,1);assert.deepEqual(Object.keys(request).sort(),['evidence','ocr','upc']);
  assert.equal(card.children[0].disabled,false);
});
test('blocks duplicate requests across cards and ignores discarded result',async()=>{
  let release,calls=0;
  const ui=setup(()=>{calls++;return new Promise(r=>release=r);});
  const a=ui.make(),b=ui.make();const pending=a.children[0].click();
  await b.children[0].click();assert.equal(calls,1);
  a.isConnected=false;release(result());await pending;
  assert.equal(a.children[2].children.length,0);
});
test('errors displayed and retry enabled',async()=>{
  const card=setup(async()=>({ok:false,json:async()=>({error:{message:'Rate limited'}})})).make();
  await card.children[0].click();assert.equal(card.children[2].textContent,'Rate limited');assert.equal(card.children[0].disabled,false);
});
test('renders observed/retrieved distinction, conflicts and official citations',async()=>{
  const record={brands:['Example'],comparison:'CONFLICTING',active_ingredient_text:['IBUPROFEN 400 mg'],purpose:['Pain reliever'],exact_upc:true,exact_package:'Not verified',conflicts:['Strength mismatch'],missing:[],source_id:'123',source_url:'https://api.fda.gov/drug/label.json?search=id:123',dailymed_url:'https://evil.example/'};
  const card=setup(async()=>result([record])).make();await card.children[0].click();
  const output=card.children[2], details=output.children.at(-1);
  assert.match(output.children[1].textContent,/Observed label text/);
  assert.equal(details.children.filter(n=>n.tag==='a').length,1);
  assert.equal(details.children[1].rows[3][1],'Not verified');
});
test('manual, automatic and unresolved barcode entry points wired',()=>{
  assert.match(source,/card.append\(productResearchCard\(body, fallbackBarcode/);
  assert.match(source,/nodes.push\(productResearchCard\(body\)\)/);
  assert.match(source,/b.final_status === "UNRESOLVED".*productResearchCard/);
});
