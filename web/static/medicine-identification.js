// One crop request per capture; generation + echoed identifiers reject late results.
(function(root) {
  'use strict';
  class MedicineIdentification {
    constructor(ui) {this.ui=ui;this.generation=0;this.busy=false;this.seen=new Set();}
    diagnostic(text) {if(this.ui.diagnostic)this.ui.diagnostic(text);}
    cancel() {if(this.busy)this.diagnostic('cancelled: camera/session/candidate changed');this.generation++;if(this.controller)this.controller.abort();this.ui.clear();}
    async identify(packet) {
      if(this.busy || !packet) return;
      if(!packet.metadata.session_id){this.ui.status('Label scan missing session; restart awareness.');this.diagnostic('not sent: missing session');return;}
      const captureId = `${packet.metadata.session_id}-${packet.metadata.frame}-${packet.metadata.captured_at}`;
      // Server accepts bounded alphanumeric capture IDs independent of session length.
      const id = captureId.slice(-64);
      if(this.seen.has(id))return;
      this.seen.add(id);if(this.seen.size>128)this.seen.delete(this.seen.values().next().value);
      this.busy=true;
      const generation=this.generation, controller=this.controller=new AbortController();
      const timer=setTimeout(()=>controller.abort(),90000);
      this.ui.status('Reading medicine label…');
      this.diagnostic('POST /api/medicine/identify-crop started');
      const started=performance.now();
      try {
        const form=new FormData();form.append('image',packet.crop,'label.jpg');
        form.append('session_id',packet.metadata.session_id);form.append('capture_id',id);
        form.append('assist',String(this.ui.assist()));
        const response=await this.ui.fetch('/api/medicine/identify-crop',{method:'POST',body:form,signal:controller.signal});
        this.diagnostic('response HTTP '+response.status);
        const body=await response.json();
        if(generation!==this.generation || controller.signal.aborted){this.diagnostic('response discarded: cancelled generation');return;}
        if(!response.ok)throw new Error(body.error ? body.error.message : 'Label scan failed.');
        if(body.session_id!==packet.metadata.session_id || body.capture_id!==id){this.diagnostic('response discarded: identifiers differ');this.ui.status('Label result belongs to another capture; rescan.');return;}
        body.timing.round_trip_ms=Math.round(performance.now()-started);
        this.ui.status(body.match.status+' · label scan complete');
        this.ui.result(body);
        this.diagnostic('result rendered: '+body.match.status+' ('+body.timing.round_trip_ms+' ms)');
      }catch(error){
        this.diagnostic(controller.signal.aborted ? "request aborted" : "request or rendering failed");
        if(generation===this.generation)this.ui.status(error.name==='AbortError'?'Label scan timed out; rescan to retry.':error.message);
      }finally{clearTimeout(timer);this.busy=false;this.controller=null;}
    }
  }
  if(typeof module!=='undefined'&&module.exports)module.exports=MedicineIdentification;
  else root.MedicineIdentification=MedicineIdentification;
})(typeof self!=='undefined'?self:this);
