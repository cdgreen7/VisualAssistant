"""Resumable local model suggestions; independent human review is mandatory."""
import copy
import hashlib
import json
from pathlib import Path
import re
from datetime import date
from unittest.mock import patch
from research.dataset_images import ROOT,discover,read,fingerprint,related,derived
from research.annotate_medicine import atomic_json,locked,save_annotation,load_manifest,inspect_image,yes,text,validate_row


def queue_path(manifest):return Path(manifest).parent/'drafts.json'


def queue_load(manifest):
    p=queue_path(manifest)
    return json.loads(p.read_text()) if p.exists() else {'version':1,'items':[]}


def queue_report(manifest):
    from collections import Counter
    q=queue_load(manifest)
    return {'states':dict(Counter(i['state'] for i in q['items'])),
            'potential_duplicate_pairs':sum(len(i.get('possible_duplicates',[])) for i in q['items']),
            'errors':[{'source':i['source'],'error':i['error']} for i in q['items'] if i.get('error')]}


AMOUNT=re.compile(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|%)(?:\s*/\s*(\d+(?:\.\d+)?\s*m[lL]))?',re.I)
CROP_MARGIN=0.04      # extra border around the YOLO bottle box, as a fraction of its size
CROP_WORTHWHILE=0.7   # only re-read the bottle when its box is under this share of the frame


MAX_CROPS=3           # bottles re-read per photograph, most confident first


def bottle_crops(image,working_size,detections):
    """Each detected bottle cut from the FULL-resolution image: [(crop, box, confidence)].

    The whole frame is read at a reduced size, where label print on a 24 MP
    photograph is often too small for OCR. The same label cut from the
    original pixels is several times larger. Every bottle is read, not only
    the most confident one: a water bottle beside a medicine bottle is common.
    """
    crops=[]
    sx,sy=image.width/working_size[0],image.height/working_size[1]
    bottles=sorted((d for d in detections if d['class_name']=='bottle'),key=lambda d:-d['confidence'])
    for found in bottles[:MAX_CROPS]:
        x1,y1,x2,y2=found['box'];mx,my=(x2-x1)*CROP_MARGIN,(y2-y1)*CROP_MARGIN
        box=[max(0,int((x1-mx)*sx)),max(0,int((y1-my)*sy)),min(image.width,int((x2+mx)*sx)),min(image.height,int((y2+my)*sy))]
        if box[2]-box[0]<32 or box[3]-box[1]<32:continue
        if (box[2]-box[0])*(box[3]-box[1])>CROP_WORTHWHILE*image.width*image.height:continue
        crops.append((image.crop(box),box,found['confidence']))
    return crops


class LocalReader:
    def __init__(self,side=1600,qwen=False):
        if not 320<=side<=2560:raise ValueError('OCR side must be between 320 and 2560')
        self.side=side;self.services=None;self.qwen=qwen;self.label_reader=None
    def __call__(self,image):
        import numpy as np
        import easyocr
        from vision.detector import ObjectDetector
        from vision.safe_image import keep_pillow_open
        from web.services import VisionServices
        keep_pillow_open()   # an unreadable photo must be an error, never a package install
        if self.services is None:
            weights=ROOT/'yolo26n.pt'
            if not weights.exists():raise ValueError('Installed yolo26n.pt required; downloads disabled')
            self.services=VisionServices(loaders={'detector':lambda:ObjectDetector(str(weights))})
            reader=easyocr.Reader
            def installed(*a,**k):return reader(*a,**dict(k,download_enabled=False))
            with patch.object(easyocr,'Reader',installed):self.services.preload(('detector','recognizer','matcher'))
        working=image.copy();working.thumbnail((self.side,self.side))
        frame=np.array(working)[:,:,::-1].copy()
        detector=self.services._get('detector')
        detections=detector.get_detections(detector.detect(frame))
        bottles=[d for d in detections if d['class_name']=='bottle']
        # Read the entire working image so multiple visible labels are not silently discarded.
        recognizer=self.services._get('recognizer')
        ocr=[dict(r,region='frame') for r in recognizer.read(frame)]
        # Then read the detected bottle again from the original pixels, where its print is larger.
        crop_info=[]
        for number,(crop,box,confidence) in enumerate(bottle_crops(image,working.size,detections),1):
            crop.thumbnail((self.side,self.side))
            ocr+=[dict(r,region=f'bottle_crop_{number}') for r in recognizer.read(np.array(crop)[:,:,::-1].copy())]
            crop_info.append({'region':f'bottle_crop_{number}','box_in_original':box,'read_at':list(crop.size),
                              'detector_confidence':confidence,'margin':CROP_MARGIN})
        match=self.services._get('matcher').match(ocr)
        observations={'ocr':[{k:r[k] for k in ('text','confidence','region')} for r in ocr],
                      'detections':detections,'working_size':list(working.size),'bottle_crops':crop_info,
                      'matcher_status':match.status.value,
                      'models':{'detector':'yolo26n.pt','ocr':'EasyOCR English CPU','qwen':False}}
        if self.qwen:self.second_reader(frame,ocr,observations)
        return suggest(observations,match)

    def second_reader(self,frame,ocr,observations):
        """Optional local Qwen transcription. Stored as an observation only: it never
        fills a suggested field and never reaches the matcher."""
        from web.qwen_label import QwenLabelReader
        from web.services import ServiceError
        self.label_reader=self.label_reader or QwenLabelReader()
        try:
            out=self.label_reader.read(frame,ocr)
            observations['qwen']={'model':out['model'],'lines':out['lines'],'agrees_with_ocr':out['agrees_with_ocr'],
                                  'used_for_suggestions':False}
            observations['models']['qwen']=out['model']
        except ServiceError as failure:observations['qwen']={'error':failure.message,'used_for_suggestions':False}


def suggest(observations,match):
    from vision.products.matcher import ingredient_key
    names=sorted({e.value for e in match.evidence if e.kind=='ingredient'});lone_amount=False
    brands=sorted({e.value for e in match.evidence if e.kind=='brand'})
    rows=observations['ocr'];ingredients=[]
    for name in names:
        values=set()
        for row in rows:
            present=[n for n in names if ' '+ingredient_key(n)+' ' in ' '+ingredient_key(row['text'])+' ']
            if present==[name] and row['confidence']>=.6:
                values.update((a,b.lower(),c) for a,b,c in AMOUNT.findall(row['text']))
        value=next(iter(values)) if len(values)==1 else None
        if value is None and not values and len(names)==1:
            # One ingredient, and exactly one amount read with confidence anywhere on the label
            # (e.g. "Ibuprofen" and "USP; 200 mg" on separate lines). Still only a suggestion.
            seen={(a,b.lower(),c) for row in rows if row['confidence']>=.6 for a,b,c in AMOUNT.findall(row['text'])}
            if len(seen)==1:value=next(iter(seen));lone_amount=True
        ingredients.append({'name':name,'strength':value[0] if value else None,
                            'unit':(value[1]+('/'+value[2] if value[2] else '')) if value else None})
    bottle=any(d['class_name']=='bottle' for d in observations['detections'])
    forms=sorted({m.lower() for row in rows for m in re.findall(r'\b(tablets?|capsules?|caplets?|liquid|suspension)\b',row['text'],re.I)})
    strong=[r for r in rows if r['confidence']>=.6]
    # Confidently read lines with real words, bottle crops before the whole frame, no repeats.
    wordy=list(dict.fromkeys(r['text'].strip() for r in sorted(strong,key=lambda r:r.get('region','frame')=='frame')
                             if len(re.findall(r'[A-Za-z]',r['text']))>=3))
    fields={'category':'combination' if len(names)>1 else 'single_ingredient' if names else 'unclear_label' if bottle else 'no_bottle',
            'object_category':'bottle' if bottle else None,'brand':brands[0] if len(brands)==1 else None,
            'product_name':' · '.join(wordy[:3]) or None,'ingredients':ingredients,'dosage_form':forms[0] if len(forms)==1 else None,
            'negative':None,'readable':True if strong and names else None,
            'bottle_present':bottle,'bottles_detected':sum(d['class_name']=='bottle' for d in observations['detections']),
            'possible_medicine':True if names else None,'multiple_active_ingredients':len(names)>1,'exact_product':None}
    sources={k:'Conservative rule from local YOLO/OCR observations; unverified' for k in fields}
    sources.update(brand='ProductMatcher brand evidence from OCR; not a product identity',
                   ingredients='ProductMatcher ingredient evidence + '+('the only amount read on the label (single ingredient; not on the same line, so check it)' if lone_amount else 'isolated OCR amount associations; unknown if ambiguous'),
                   product_name='First confidently read OCR lines (bottle crops first); not a verified product name',negative='Unknown: bottle detection cannot determine medicine/negative truth')
    if fields['bottles_detected']>1:
        sources['bottles_detected']='YOLO found more than one bottle; label text may come from different objects, so check which one each field describes'
    return {'fields':fields,'field_sources':sources,'observations':observations}


def batch(manifest,folder,side=1600,reader=None,refresh=False,qwen=False):
    """refresh=True re-inspects drafts nobody has reviewed yet (e.g. after the reader improved).
    Saved entries, confirmed duplicates and drafts with human edits are never re-inspected."""
    manifest=Path(manifest);qpath=queue_path(manifest);reader=reader or LocalReader(side,qwen)
    with locked(qpath):
        queue=queue_load(manifest)
        for draft in queue['items']:draft.setdefault('truth',{'confirmed':False})
        known={r.get('source_sha256',r.get('sha256')):r['id'] for r in load_manifest(manifest)['images']}
        for source in discover(folder):
            digest=hashlib.sha256()
            with source.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
            raw_hash=digest.hexdigest()
            existing=next((i for i in queue['items'] if i['sha256']==raw_hash),None)
            if existing and raw_hash in known:
                existing.update(state='saved',manifest_id=known[raw_hash]);atomic_json(qpath,queue);continue
            untouched=existing and existing['state']=='pending' and 'reviewed_fields' not in existing and 'duplicates_reviewed' not in existing
            if existing and existing['state']!='error' and not (refresh and untouched and existing['source']==str(source)):
                if str(source) not in existing['aliases']:existing['aliases'].append(str(source))
                atomic_json(qpath,queue);continue
            item=existing or {'id':'photo-'+raw_hash[:16],'sha256':raw_hash,'source':str(source),'aliases':[str(source)],'state':'pending','truth':{'confirmed':False}}
            if not existing:queue['items'].append(item)
            if raw_hash in known:
                item.update(state='saved',manifest_id=known[raw_hash]);atomic_json(qpath,queue);continue
            try:
                _,info,image=read(source)
                item.update(info=info,fingerprint=fingerprint(image))
                item['suggestions']=reader(image)
                item['state']='pending';item.pop('error',None)
            except Exception as e:item.update(state='error',error=f'{type(e).__name__}: {e}')
            # Same stem and thumbnail similarity are hints, never auto-grouping.
            item['possible_duplicates']=[{'id':other['id'],'reason':why} for other in queue['items'] if other is not item
                                         for why in [related(item,other)] if why]
            atomic_json(qpath,queue)
            print(item['id'],item['state'],source.name,flush=True)
        # Symmetric hints, including later-discovered versions.
        for item in queue['items']:
            item['possible_duplicates']=[{'id':other['id'],'reason':why} for other in queue['items'] if other is not item
                                         for why in [related(item,other)] if why]
        atomic_json(qpath,queue)
    print(json.dumps(queue_report(manifest),indent=2))


def editable_fields(fields,ask=input,on_change=None):
    fields=copy.deepcopy(fields)
    while True:
        print(json.dumps(fields,indent=2))
        command=ask('Enter accepts suggestions as DRAFT; set FIELD JSON / clear FIELD / skip / quit: ').strip()
        if not command:return fields
        if command in ('skip','quit'):return command
        try:
            action,key,*value=command.split(' ',2)
            if key not in fields:raise ValueError('Unknown field')
            if action=='clear':fields[key]=[] if key=='ingredients' else None
            elif action=='set':fields[key]=json.loads(value[0])
            else:raise ValueError('Use set or clear')
            if on_change:on_change(fields)
        except (ValueError,IndexError) as e:print('Invalid edit:',e)


def make_row(item,fields,independent,checker=None,source=None):
    truth={'confirmed':False,'negative':None,'readable':None,'object_category':None,
           'brand':None,'ingredients':[],'exact_product':None}
    if independent:
        truth.update({k:fields.get(k) for k in ('negative','readable','object_category','brand','ingredients','dosage_form','exact_product')})
        truth.update(confirmed=True,confirmed_by=checker,confirmation_source=source,confirmation_date=date.today().isoformat())
    return {'id':item['id'],'path':'pending','physical_product_id':None,'split':'unassigned',
            'category':fields['category'],'provenance':{'source':'User-reviewed local photograph','local_evaluation_permitted':True},
            'truth':truth,'suggestions':item['suggestions'],'reviewed_suggestions':fields,
            'annotation_review':{'privacy_checked':True,'ingredient_pairs_checked':bool(independent)},
            'source_sha256':item['sha256'],'sha256':item['sha256'],'source_image':item.get('info'), 'photo_group_id':item['id'],
            'review_sources':{k:'human correction' if v!=item['suggestions']['fields'].get(k) else 'accepted model suggestion; independent verification only if truth.confirmed' for k,v in fields.items()}}


def review(manifest,ask=input,open_image=True):
    manifest=Path(manifest);qpath=queue_path(manifest)
    with locked(qpath):
        queue=queue_load(manifest)
        for item in sorted(queue['items'],key=lambda i:i.get('info',{}).get('width',0)*i.get('info',{}).get('height',0),reverse=True):
            if item['state'] not in ('pending','skipped'):continue
            # Recovery after manifest save but before queue update: never overwrite it.
            if any(r['id']==item['id'] or r.get('source_sha256',r.get('sha256'))==item['sha256'] for r in load_manifest(manifest)['images']):
                item['state']='saved';atomic_json(qpath,queue);continue
            try:
                _,info,image=read(item['source'])
                if info['sha256']!=item['sha256']:raise ValueError('Original changed; rerun batch')
                if open_image:inspect_image(Path(item['source']))
                print('\nReview',item['source'],'— model suggestions are NOT ground truth.')
                decodable={i['id'] for i in queue['items'] if i.get('suggestions') and i['state']!='duplicate'}
                duplicates=[d for d in item.get('possible_duplicates',[]) if d['id'] in decodable]
                if duplicates:
                    print('Potential copies (not proof):',json.dumps(duplicates,indent=2))
                    choice=ask('Duplicate of ID (confirm same photograph), distinct, skip, or quit: ').strip()
                    if choice=='quit':return
                    if choice=='skip':item['state']='skipped';atomic_json(qpath,queue);continue
                    if choice!='distinct':
                        if choice not in [x['id'] for x in duplicates]:raise ValueError('Choose an offered ID or distinct')
                        target=next(x for x in queue['items'] if x['id']==choice)
                        if target['state'] in ('duplicate','error'):raise ValueError('Choose a decodable representative, not another duplicate/error')
                        item.update(state='duplicate',duplicate_of=choice);atomic_json(qpath,queue);continue
                    item['duplicates_reviewed']='distinct'
                def progress(fields):
                    item['reviewed_fields']=copy.deepcopy(fields);atomic_json(qpath,queue)
                print('Suggestion sources:',json.dumps(item['suggestions']['field_sources'],indent=2))
                fields=editable_fields(item.get('reviewed_fields',item['suggestions']['fields']),ask,progress)
                if fields=='quit':return
                if fields=='skip':item['state']='skipped';atomic_json(qpath,queue);continue
                item['reviewed_fields']=fields;atomic_json(qpath,queue)
                independent=yes('Have you independently verified ALL truth fields against the physical package/another independent source (not just accepted AI)?',ask)
                checker=text('Checker ID',ask) if independent else None
                source=text('Independent verification source',ask) if independent else None
                if independent:
                    for key,prompt in [('negative','Is this a negative/non-medicine example?'),('readable','Is this label readable?')]:
                        if fields.get(key) is None:fields[key]=yes(prompt,ask,True)
                row=make_row(item,fields,independent,checker,source)
                if independent and not fields.get('negative'):
                    if not yes('All active ingredients and EACH associated strength/unit checked?',ask):raise ValueError('Leave this draft unconfirmed until ingredient pairs are checked')
                row['physical_product_id']=text('Physical item ID: reuse ONLY for known same physical item; blank stays unknown',ask)
                row['split']=ask('Split [unassigned/train/validation/test; Enter=unassigned]: ').strip() or 'unassigned'
                if not yes('Image, metadata and annotation are safe to retain locally, and you have permission?',ask):
                    item['state']='skipped';atomic_json(qpath,queue);continue
                validate_row(row,require_review=True)
                if not yes('Save original + separate endpoint derivative + reviewed annotation?',ask):
                    item['state']='skipped';atomic_json(qpath,queue);continue
                # Derived data is separate and bounded; source bytes remain identical.
                payload,transform=derived(image)
                folder=manifest.parent/'derived';folder.mkdir(exist_ok=True)
                dest=folder/(item['sha256']+'.jpg')
                if dest.exists() and dest.read_bytes()!=payload:raise ValueError('Existing derivative differs; original left unchanged')
                created=not dest.exists()
                if created:
                    with dest.open('xb') as handle:handle.write(payload)
                row['evaluation_image']={'path':dest.relative_to(manifest.parent).as_posix(),
                                         'sha256':hashlib.sha256(payload).hexdigest(),'transform':transform}
                try:save_annotation(manifest,item['source'],row)
                except BaseException:
                    if created:dest.unlink()
                    raise
                item['state']='saved';atomic_json(qpath,queue)
            except (ValueError,OSError,TypeError) as e:
                print('Not saved:',e);item['review_error']=str(e);atomic_json(qpath,queue)
