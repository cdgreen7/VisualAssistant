"""Offline evaluation of the unchanged medicine-crop Flask endpoint (no camera)."""
import argparse
from datetime import datetime
import hashlib
import io
import json
import multiprocessing as mp
import platform
from pathlib import Path
import resource
import statistics
import sys
import time
from unittest.mock import patch

CATEGORIES = {'single_ingredient', 'combination', 'same_ingredients_different_brand',
              'unclear_label', 'non_medicine_bottle', 'no_bottle'}
METRICS = ('ingredients', 'strengths', 'brand', 'combination_verified',
           'formulation_verified', 'exact_product', 'false_verified',
           'negative_rejected', 'appropriate_uncertainty')


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    if data.get('schema_version') != 1 or not isinstance(data.get('images'), list):
        raise ValueError('Expected schema_version 1 and images array')
    ids = set()
    splits = {}
    photos = set()
    for row in data['images']:
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
            raise ValueError('Unique nonempty image id required')
        ids.add(row['id'])
        if row.get('category') not in CATEGORIES: raise ValueError('Invalid category')
        if not isinstance(row.get('path'), str): raise ValueError('Image path required')
        split=row.get('split','unassigned')
        if split not in ('unassigned','train','validation','test'):raise ValueError('Invalid dataset split')
        if split!='unassigned' and not row.get('physical_product_id'):
            raise ValueError('Assign a verified physical item group before assigning a split')
        for key in ('physical_product_id','photo_group_id','source_sha256'):
            value=row.get(key)
            if value:
                group=(key,value)
                if group in splits and splits[group]!=split:raise ValueError('Product/photo group crosses dataset splits')
                splits[group]=split
        photo=row.get('photo_group_id')
        if photo and photo in photos:raise ValueError('One photograph cannot be multiple evaluation examples')
        if photo:photos.add(photo)
        provenance = row.get('provenance', {})
        if not provenance.get('source') or provenance.get('local_evaluation_permitted') is not True:
            raise ValueError('Document provenance and permission before including an image')
        truth = row.get('truth', {})
        if not isinstance(truth.get('confirmed'), bool): raise ValueError('truth.confirmed must be boolean')
        if truth.get('confirmed'):
            if not truth.get('confirmed_by') or not truth.get('confirmation_source'):
                raise ValueError('Confirmed truth needs a checker and independent source')
            if not isinstance(truth.get('negative'), bool) or not isinstance(truth.get('readable'), bool):
                raise ValueError('Confirmed labels need negative/readable booleans')
            if not truth.get('object_category'): raise ValueError('Object category required')
            if not isinstance(truth.get('ingredients'), list): raise ValueError('Ingredients array required')
            for ingredient in truth['ingredients']:
                if not all(isinstance(ingredient.get(k), str) and ingredient[k] for k in ('name','strength','unit')):
                    raise ValueError('Ingredient name, strength and unit required')
            if not truth['negative'] and not truth['ingredients']:
                raise ValueError('Confirmed positive medicine needs ingredient truth')
            exact = truth.get('exact_product')
            if exact and (not exact.get('package_ndc') or not exact.get('verification_source')):
                raise ValueError('Exact product needs independently verified package NDC/source')
    return data


def summary(values):
    values = sorted(values)
    n = len(values)
    def percentile(p):
        i=(n-1)*p;lo=int(i);hi=min(lo+1,n-1)
        return round(values[lo]+(values[hi]-values[lo])*(i-lo),3)
    return {'n':n, 'median':round(statistics.median(values),3) if n>=3 else None,
            'p90':percentile(.9) if n>=10 else None, 'p95':percentile(.95) if n>=20 else None,
            'min':min(values) if n else None, 'max':max(values) if n else None}


def score(row, body):
    """Score returned facts, never rerun matching or treat abstention as verification."""
    from vision.products.matcher import ingredient_key
    from vision.products.catalog import name_key, strength_key
    from vision.products.ndc import normalize_package_ndc
    from vision.products.evaluate import formulation_key
    result = dict.fromkeys(METRICS)
    truth = row.get('truth', {})
    if not truth.get('confirmed') or 'match' not in body: return result
    match=body['match']; verified=match['status']=='VERIFIED'
    if truth['negative']:
        result.update(false_verified=verified, negative_rejected=match['status']=='NO_MATCH',
                      appropriate_uncertainty=None)
        return result
    pairs=[(i['name'],i['strength']+' '+i['unit']) for i in truth['ingredients']]
    expected=formulation_key(pairs)
    actual=[formulation_key([(i['name'],i['strength']) for i in c['ingredients']]) for c in match['candidates']]
    observed={ingredient_key(e['value']) for e in match.get('evidence',[]) if e['kind']=='ingredient'}
    strengths={strength_key(e['value']) for e in match.get('evidence',[]) if e['kind']=='strength' and not e.get('corrected_from')}
    result['ingredients']=observed=={ingredient_key(n) for n,s in pairs}
    result['strengths']=strengths=={strength_key(s) for n,s in pairs}
    if len(pairs)>1:
        # Unordered strength sets cannot prove the association in a combination.
        from vision.products.matcher import ProductMatcher
        for name, strength in pairs:
            associated=set()
            for observation in body.get('ocr',[]):
                line=observation['text'];normalized=' '+ingredient_key(line)+' '
                present=[n for n,_ in pairs if ' '+ingredient_key(n)+' ' in normalized]
                if present==[name] and observation.get('confidence',0)>=.6:
                    associated.update(v for v,corrected in ProductMatcher._strengths(line.upper()) if not corrected)
            if associated!={strength_key(strength)}:result['strengths']=False
    brand=truth.get('brand')
    if brand:
        words=set(name_key(' '.join(r['text'] for r in body.get('ocr',[]))).split())
        result['brand']=set(name_key(brand).split()).issubset(words)
    correct=bool(actual and actual[0]==expected)
    result['formulation_verified']=verified and correct
    if len(pairs)>1: result['combination_verified']=verified and correct
    # Package-level claims with absent package truth are not countable as safe.
    result['false_verified']=not correct if verified else False
    exact=truth.get('exact_product')
    if exact:
        ndc=normalize_package_ndc(exact['package_ndc'])
        package=match.get('package') or {}
        same=bool(ndc and normalize_package_ndc(package.get('package_ndc'))==ndc)
        result['exact_product']=verified and match['level']=='PACKAGE' and same and correct
        if verified and match['level']=='PACKAGE': result['false_verified']=not (same and correct)
    elif verified and match['level']=='PACKAGE' and correct:
        result['false_verified']=None
    if not truth['readable']: result['appropriate_uncertainty']=match['status'] in ('UNCERTAIN','NO_MATCH')
    return result


def rss():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform=='darwin' else 1024),2)


def prepare(row, base):
    """Stored crop bytes unchanged; optional explicit fixture ROI, no resizing."""
    from PIL import Image
    start=time.perf_counter();path=(base/row['path']).resolve();raw=path.read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    derived_input=row.get('evaluation_image')
    if derived_input:
        if row.get('source_sha256')!=digest:raise ValueError('Original image hash changed')
        raw=(base/derived_input['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=derived_input['sha256']:raise ValueError('Derived image hash changed')
    with Image.open(io.BytesIO(raw)) as image:
        fmt=image.format; original=list(image.size)
        if row.get('roi') is not None:
            box=row['roi']
            if len(box)!=4 or not all(isinstance(x,int) for x in box) or not (0<=box[0]<box[2]<=image.width and 0<=box[1]<box[3]<=image.height):
                raise ValueError('ROI must be in original stored pixel coordinates')
            output=io.BytesIO();image.convert('RGB').crop(box).save(output,'JPEG',quality=92)
            raw=output.getvalue();fmt='JPEG'
    return raw, 'image/jpeg' if fmt=='JPEG' else 'image/png', {
        'source_sha256':digest,'submitted_sha256':hashlib.sha256(raw).hexdigest(),
        'original_size':([row['source_image']['width'],row['source_image']['height']] if row.get('source_image') else original),
        'submitted_size':original,'roi':row.get('roi'),'evaluation_image':derived_input,
        'fixture_prepare_ms':round((time.perf_counter()-start)*1000,3)}


def evaluate_image(row, base, trials, emit, services=None):
    from web.server import create_app
    from web.services import VisionServices
    import vision.identification as identification
    raw,mime,metadata=prepare(row,base)
    emit({'event':'input','metadata':metadata})
    # Models stay reused across all requests. Only endpoint dependencies preload;
    # YOLO is not run by this endpoint. No deployed code is modified.
    services=services or VisionServices()
    start=time.perf_counter();services.preload(('recognizer','matcher'))
    emit({'event':'initialization','load_ms':round((time.perf_counter()-start)*1000,3),'peak_rss_mb':rss()})
    import torch
    reader=getattr(services._get('recognizer'),'reader',None)
    directory=getattr(reader,'model_storage_directory',None)
    weights={}
    if directory:
        for path in sorted(Path(directory).glob('*.pth')):
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
            weights[path.name]=digest.hexdigest()
    emit({'event':'runtime','torch_threads':torch.get_num_threads(), 'torch_interop_threads':torch.get_num_interop_threads(),
          'installed_ocr_weight_sha256':weights})
    app=create_app(services=services);client=app.test_client()
    original=identification.crop_box
    for index in range(trials+1):
        crop_ms=[]
        def measured(*args,**kwargs):
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:crop_ms.append((time.perf_counter()-start)*1000)
        start=time.perf_counter()
        try:
            with patch.object(identification,'crop_box',measured):
                response=client.post('/api/medicine/identify-crop',data={
                    'session_id':'evaluation_session','capture_id':f'evaluation_capture_{index}',
                    'assist':'false','image':(io.BytesIO(raw),'crop.jpg',mime)})
            elapsed=(time.perf_counter()-start)*1000
            body=response.get_json();stages={}
            for part in response.headers.get('Server-Timing','').split(','):
                name,sep,value=part.strip().partition(';dur=')
                if sep:stages[name]=float(value)
            times=body.get('timing',{})
            record={'event':'trial','phase':'cold' if index==0 else 'warm','index':index,
                    'http_status':response.status_code,'success':response.status_code==200,
                    'request_ms':round(elapsed,3),'server_total_ms':stages.get('total'),
                    'decode_ms':stages.get('decode'),'crop_ms':round(sum(crop_ms),4),
                    'ocr_ms':times.get('ocr_ms'),'match_ms':times.get('matcher_ms'),
                    'peak_rss_mb':rss(),'scores':score(row,body),'status':body.get('match',{}).get('status'),
                    'level':body.get('match',{}).get('level'),'error':body.get('error',{}).get('code')}
            emit(record)
        except Exception as e:
            emit({'event':'trial','phase':'cold' if index==0 else 'warm','index':index,
                  'success':False,'error':type(e).__name__})


def worker(row, base, trials, connection):
    try:
        import easyocr
        reader=easyocr.Reader
        def installed_only(*args,**kwargs):
            kwargs['download_enabled']=False
            return reader(*args,**kwargs)
        with patch.object(easyocr,'Reader',installed_only):
            evaluate_image(row,Path(base),trials,connection.send)
    except Exception as e:connection.send({'event':'failure','error':type(e).__name__,'message':str(e)[:200]})
    finally:connection.close()


def run_worker(row, base, trials, timeout, target=worker):
    ctx=mp.get_context('spawn');receive,send=ctx.Pipe(duplex=False)
    process=ctx.Process(target=target,args=(row,str(base),trials,send));process.start();send.close()
    events=[];deadline=time.monotonic()+timeout;timed_out=False
    while True:
        if receive.poll(.1):
            try:events.append(receive.recv());deadline=time.monotonic()+timeout
            except EOFError:break
        if not process.is_alive():
            while receive.poll():
                try:events.append(receive.recv())
                except EOFError:break
            break
        if time.monotonic()>deadline:
            process.terminate();timed_out=True;break
    process.join(5)
    if process.is_alive():process.kill();process.join()
    receive.close()
    return {'id':row['id'],'category':row['category'],'truth_confirmed':row.get('truth',{}).get('confirmed',False),
            'timeout':timed_out,'exitcode':process.exitcode,'events':events}


def aggregate(runs):
    metrics={k:{'correct':0,'evaluated':0} for k in METRICS}
    for run in runs:
        # One score per image: first successful warm request, else cold. Repeats
        # are not extra independent accuracy examples; retain every trial score.
        trials=[e for e in run['events'] if e.get('event')=='trial' and e.get('success')]
        chosen=next((e for e in trials if e['phase']=='warm'),trials[0] if trials else None)
        if chosen:
            for k,v in chosen['scores'].items():
                if v is not None:metrics[k]['evaluated']+=1;metrics[k]['correct']+=int(v)
        for phase in ('cold','warm'):
            rows=[e for e in trials if e['phase']==phase]
            run[phase]={k:summary([e[k] for e in rows if e.get(k) is not None]) for k in
                        ('decode_ms','crop_ms','ocr_ms','match_ms','server_total_ms','request_ms','peak_rss_mb')}
    # For false_verified the numerator counts errors, not correct decisions.
    metrics['false_verified']['false']=metrics['false_verified'].pop('correct')
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='research/datasets/medicine-v1/manifest.json')
    parser.add_argument('--trials',type=int,default=20)
    parser.add_argument('--timeout',type=float,default=180)
    parser.add_argument('--output',default='research/results')
    parser.add_argument('--split',choices=('unassigned','train','validation','test'),default='unassigned')
    args=parser.parse_args()
    if args.trials<1 or args.timeout<=0:parser.error('Positive trials and timeout required')
    path=Path(args.manifest).resolve();manifest=load_manifest(path)
    from importlib.metadata import version
    import subprocess
    def command(args):
        p=subprocess.run(args,capture_output=True,text=True);return p.stdout.strip()
    config={'started':datetime.now().astimezone().isoformat(),
            'hardware_memory_bytes':command(['sysctl','-n','hw.memsize']) if sys.platform=='darwin' else None,
            'python':sys.version,'platform':platform.platform(),'cpu':command(['sysctl','-n','machdep.cpu.brand_string']) if sys.platform=='darwin' else platform.processor(),
            'versions':{p:version(p) for p in ('torch','easyocr','Flask','numpy')},
            'git_commit':command(['git','rev-parse','HEAD']),'git_branch':command(['git','branch','--show-current']),
            'git_status':command(['git','status','--short']),'split':args.split,'warm_trials':args.trials,'timeout_per_event_s':args.timeout,
            'endpoint':'/api/medicine/identify-crop','transport':'Flask test client; no network/TLS/browser',
            'ocr':'EasyOCR en CPU, installed weights only, unchanged defaults','assist':False,
            'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    # Hash code and local catalog for comparisons despite uncommitted changes.
    config['source_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('web','vision','research') for p in Path(folder).rglob('*.py')}
    from web.services import FDA_CACHE
    config['catalog_sha256']=hashlib.sha256(Path(FDA_CACHE).read_bytes()).hexdigest()
    runs=[run_worker(row,path.parent,args.trials,args.timeout) for row in manifest['images'] if row.get('split','unassigned')==args.split]
    report={'image_count':len(runs), 'confirmed_images':sum(bool(r['truth_confirmed']) for r in runs),
            'timeouts':sum(r['timeout'] for r in runs),
            'failed_trials':sum(e.get('event')=='trial' and not e.get('success') for r in runs for e in r['events']),
            'schema_version':1,'dataset_id':manifest['dataset_id'],'configuration':config,'runs':runs,'accuracy_counts':aggregate(runs),
            'limitations':['No independent phone/network/browser measurements','Peak RSS includes Python/models and is a process high-water mark',
                            'Percentiles describe repeats of individual images, not population accuracy','Only independently confirmed labels scored']}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    target=out/('medicine-endpoint-'+time.strftime('%Y%m%d-%H%M%S')+'.json');target.write_text(json.dumps(report,indent=2))
    print(target);print(json.dumps(report['accuracy_counts']))


if __name__=='__main__':main()
