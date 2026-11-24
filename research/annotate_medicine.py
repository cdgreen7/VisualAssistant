"""Local, human-only medicine dataset annotation and integrity reporting."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from PIL import Image
from research.medicine_eval import CATEGORIES, load_manifest
from vision.products.ndc import normalize_package_ndc, normalize_upc

DEFAULT = 'research/datasets/medicine-v1/manifest.json'
TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}')
UNIT = re.compile(r'(?:mg|mcg|g|%|iu|units)(?:/(?:1|(?:[0-9]+(?:\.[0-9]+)?\s*)?(?:ml|g)))?', re.I)


from research.dataset_images import image_info


def schema_check(data):
    # Use the evaluation runner's exact manifest validator; do not fork its schema.
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'manifest.json';path.write_text(json.dumps(data));load_manifest(path)


def validate_row(row, require_review=False):
    schema_check({'schema_version':1,'images':[row]})
    if not TOKEN.fullmatch(row['id']): raise ValueError('Use a short nonprivate image ID: letters, digits, _ or -.')
    product=row.get('physical_product_id')
    if product is not None and (not isinstance(product,str) or not TOKEN.fullmatch(product)):
        raise ValueError('Invalid physical product ID')
    truth=row['truth'];ingredients=truth.get('ingredients',[])
    if not isinstance(ingredients,list): raise ValueError('Ingredients must be a list')
    names=set()
    for item in ingredients:
        name=item.get('name')
        if not isinstance(name,str) or not name.strip(): raise ValueError('Each ingredient needs a name.')
        key=' '.join(name.upper().split())
        if key in names: raise ValueError('Duplicate ingredient; review associated amounts.')
        names.add(key)
        value=item.get('strength');unit=item.get('unit')
        if value is not None and (not isinstance(value,str) or not re.fullmatch(r'\d+(?:\.\d+)?',value) or float(value)<=0):
            raise ValueError('Strength must be a positive numeric string, or unknown.')
        if unit is not None and (not isinstance(unit,str) or not UNIT.fullmatch(unit)):
            raise ValueError('Unsupported unit; retain concentration denominator (e.g. mg/5 mL).')
        if truth['confirmed'] and (value is None or unit is None): raise ValueError('Confirmed ingredients require associated strength and unit.')
    for key in ('negative','readable'):
        if truth.get(key) is not None and not isinstance(truth[key],bool): raise ValueError(key+' must be boolean or unknown')
    if truth.get('negative') and ingredients: raise ValueError('Negative example cannot carry medicine ingredient truth.')
    if truth['confirmed']:
        if row['category'] in ('non_medicine_bottle','no_bottle') and not truth['negative']: raise ValueError('Negative category conflicts with annotation.')
        if row['category'] in ('single_ingredient','combination','same_ingredients_different_brand') and truth['negative']: raise ValueError('Medicine category conflicts with negative label.')
        if row['category']=='combination' and len(ingredients)<2: raise ValueError('Combination requires all active ingredients.')
        if row['category']=='single_ingredient' and len(ingredients)!=1: raise ValueError('Single ingredient category requires exactly one ingredient.')
        if not truth['negative'] and require_review and row.get('annotation_review',{}).get('ingredient_pairs_checked') is not True:
            raise ValueError('Independently review all ingredient-strength pairs before confirming.')
    exact=truth.get('exact_product')
    if exact:
        if truth.get('negative'):raise ValueError('Negative example cannot have an exact medicine identity.')
        if not truth['confirmed']: raise ValueError('Exact product identity cannot be recorded as verified on an unconfirmed annotation.')
        if not normalize_package_ndc(exact.get('package_ndc')) or not exact.get('verification_source'):
            raise ValueError('Exact identity needs unambiguous package NDC and independent verification source.')
        if exact.get('upc') and not normalize_upc(exact['upc']): raise ValueError('Invalid UPC check digit.')
    if require_review and row.get('annotation_review',{}).get('privacy_checked') is not True:
        raise ValueError('Privacy inspection of image, metadata and annotation is required.')


@contextmanager
def locked(path):
    with Path(str(path)+'.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def atomic_json(path,data):
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.annotation-',suffix='.json')
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(data,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def save_annotation(manifest,source,row,edit_id=None):
    manifest=Path(manifest).resolve()
    data,info=image_info(source)
    row=json.loads(json.dumps(row))
    if row.get('sha256') and row['sha256']!=info['sha256']:raise ValueError('Source image changed since inspection; inspect again.')
    row['sha256']=info['sha256']
    with locked(manifest):
        dataset=load_manifest(manifest);existing=dataset['images']
        if edit_id and not any(r['id']==edit_id for r in existing): raise ValueError('Image ID does not exist')
        for old in existing:
            old_path=(manifest.parent/old['path']).resolve()
            _,old_info=image_info(old_path) # fail closed for missing/broken existing files
            if old.get('sha256') and old['sha256']!=old_info['sha256']: raise ValueError('Existing image changed: '+old['id'])
            if old['id']==edit_id:
                if info['sha256']!=old_info['sha256'] or row['id']!=edit_id: raise ValueError('Editing annotations cannot replace the original image or ID')
                row['path']=old['path'];row['roi']=old.get('roi');continue
            if old['id']==row['id']: raise ValueError('Duplicate image ID')
            if old_info['sha256']==info['sha256']: raise ValueError('Duplicate image bytes already registered as '+old['id'])
            if row.get('physical_product_id') and row['physical_product_id']==old.get('physical_product_id'):
                a,b=row['truth'],old['truth']
                if a['confirmed'] and b['confirmed']:
                    pairs=lambda t: sorted((i['name'].upper(),i['strength'],i['unit'].lower()) for i in t['ingredients'])
                    if (a['negative']!=b['negative'] or pairs(a)!=pairs(b) or
                        (a.get('brand') and b.get('brand') and a['brand'].casefold()!=b['brand'].casefold()) or
                        (a.get('exact_product') and b.get('exact_product') and normalize_package_ndc(a['exact_product']['package_ndc'])!=normalize_package_ndc(b['exact_product']['package_ndc']))):
                        raise ValueError('Conflicting truth for the same physical product ID')
        destination=manifest.parent/'images'/(info['sha256']+info['extension'])
        if not edit_id:row['path']=destination.relative_to(manifest.parent).as_posix()
        validate_row(row,require_review=True)
        dataset['images']=[row if r['id']==edit_id else r for r in existing] if edit_id else existing+[row]
        schema_check(dataset)
        created=False
        try:
            if not edit_id:
                destination.parent.mkdir(exist_ok=True)
                with destination.open('xb') as handle:handle.write(data)
                created=True
            atomic_json(manifest,dataset)
        except BaseException:
            if created:destination.unlink()
            raise
    return row


def coverage(manifest):
    manifest=Path(manifest).resolve();data=json.loads(manifest.read_text())
    if not isinstance(data,dict) or not isinstance(data.get('images'),list):raise ValueError('Manifest needs images array')
    rows=data['images']
    result={'total_images':len(rows),'total_labeled_images':0,'usable_confirmed_images':0,'annotation_gaps':[],'by_category':dict.fromkeys(sorted(CATEGORIES),0),
            'unique_physical_products':0,'images_without_product_id':0,'confirmed_ingredient_labels':0,
            'confirmed_strength_labels':0,'confirmed_exact_product_identities':0,
            'missing_annotations':[],'invalid_annotations':[],'missing_files':[],'changed_files':[],
            'duplicate_images':[],'invalid_images':[]}
    hashes=defaultdict(list);products=set();ids=Counter(str(r.get('id')) for r in rows if isinstance(r,dict))
    for row in rows:
        if not isinstance(row,dict):
            result['invalid_annotations'].append({'id':'<missing>','error':'Annotation must be an object'});continue
        ident=row.get('id','<missing>');category=row.get('category')
        if category in result['by_category']:result['by_category'][category]+=1
        valid=True
        try:
            validate_row(row)
            if ids[str(ident)]>1:raise ValueError('Duplicate image ID')
        except (ValueError,KeyError,TypeError,AttributeError) as error:
            valid=False;result['invalid_annotations'].append({'id':ident,'error':str(error)})
        image_ok=True
        try:
            _,info=image_info(manifest.parent/row['path']);hashes[info['sha256']].append(ident)
            if row.get('sha256') and row['sha256']!=info['sha256']:
                result['changed_files'].append(ident);image_ok=False
        except FileNotFoundError:result['missing_files'].append(ident);image_ok=False
        except (OSError,ValueError,KeyError,TypeError) as error:
            result['invalid_images'].append({'id':ident,'error':str(error)});image_ok=False
        if isinstance(row.get('physical_product_id'),str) and row['physical_product_id']:products.add(row['physical_product_id'])
        else:result['images_without_product_id']+=1
        truth=row.get('truth',{})
        if not isinstance(truth,dict):truth={}
        if not truth.get('confirmed'):result['missing_annotations'].append(ident)
        gaps=[key for key in ('confirmed_by','confirmation_source','object_category','negative','readable') if truth.get(key) is None or truth.get(key)=='']
        if not truth.get('negative'):
            if not truth.get('brand'):gaps.append('brand (optional)')
            if not truth.get('ingredients'):gaps.append('ingredients')
            for index,item in enumerate(truth['ingredients'] if isinstance(truth.get('ingredients'),list) else []):
                if isinstance(item,dict):
                    gaps.extend(f'ingredients[{index}].{key}' for key in ('name','strength','unit') if not item.get(key))
        if gaps:result['annotation_gaps'].append({'id':ident,'fields':gaps})
        if valid and truth.get('confirmed'):
            if image_ok:result['usable_confirmed_images']+=1
            result['total_labeled_images']+=1
            if not truth['negative'] and truth['ingredients']:
                result['confirmed_ingredient_labels']+=1;result['confirmed_strength_labels']+=1
            if truth.get('exact_product'):result['confirmed_exact_product_identities']+=1
    result['unique_physical_products']=len(products)
    result['duplicate_images']=[{'sha256':h,'ids':identifiers} for h,identifiers in hashes.items() if len(identifiers)>1]
    return result


def text(prompt,ask=input):return ask(prompt+' [blank = unknown]: ').strip() or None


def yes(prompt,ask=input,unknown=False):
    while True:
        answer=ask(prompt+(' [y/n/?]: ' if unknown else ' [y/N]: ')).strip().lower()
        if answer in ('y','yes'):return True
        if answer in ('n','no') or (not answer and not unknown):return False
        if unknown and answer in ('','?'):return None
        print('Enter y, n'+(' or ?' if unknown else '')+'.')


def inspect_image(path):
    # Local viewer only. No model/API/web call; original file is not transformed.
    command=['open','-a','Preview',str(path)] if sys.platform=='darwin' else ['xdg-open',str(path)]
    subprocess.run(command,check=True)


def prompt_annotation(source,existing=None,ask=input,open_image=True):
    _,info=image_info(source)
    if open_image:inspect_image(Path(source).resolve())
    if existing and existing.get('roi'):print('Existing evaluation ROI is preserved:',existing['roi'],'— judge readability within this region.')
    print(f"Inspect original pixels: {info['width']}×{info['height']}; EXIF present: {info['has_exif']}.")
    print('Read the physical package independently. Do not copy model predictions. Blank fields mean unknown.')
    row={'id':existing['id'] if existing else text('Nonprivate image ID',ask), 'path':'pending',
         'physical_product_id':text('Physical item ID (reuse for every view of the SAME bottle; unknown stays ungrouped)',ask)}
    print('Categories: '+', '.join(sorted(CATEGORIES)))
    row['category']=ask('Primary category: ').strip()
    row['capture_device']=text('Capture device/browser',ask);row['input_kind']=text('Input kind (phone_crop/full_frame/other)',ask);row['roi']=None
    row['provenance']={'source':text('Image origin',ask),'local_evaluation_permitted':yes('Do you have permission for local evaluation?',ask)}
    negative=yes('Is this a negative/non-medicine example?',ask,True)
    truth={'confirmed':False,'negative':negative,'readable':yes('Is identifying label text readable in THIS image?',ask,True),
           'object_category':text('Actual object category (use none if no object)',ask),'brand':text('Brand',ask),
           'dosage_form':text('Dosage form',ask),'ingredients':[],'exact_product':None}
    if negative is not True:
        while True:
            name=text('Next active ingredient name (blank ends list)',ask)
            if name is None:break
            truth['ingredients'].append({'name':name,'strength':text('Amount for '+name,ask),
                                         'unit':text('Unit for '+name+' (retain denominator, e.g. mg/5 mL)',ask)})
    print('Review each associated ingredient and amount:',json.dumps(truth['ingredients'],indent=2))
    pairs_checked=yes('Have you checked that the complete ingredient list and each associated strength/unit match the physical label?',ask) if negative is not True else False
    truth['confirmed_by']=text('Checker initials or nonprivate ID',ask)
    truth['confirmation_source']=text('Independent confirmation source (physical package, not model output)',ask)
    truth['confirmation_date']=date.today().isoformat()
    truth['confirmed']=yes('Confirm this complete annotation as independently checked? No saves an unscored draft.',ask)
    if truth['confirmed'] and negative is not True and yes('Is exact package identity independently verified?',ask):
        truth['exact_product']={'package_ndc':text('Package NDC',ask),'upc':text('UPC, if known',ask),
                                'verification_source':text('Independent source for this exact package identifier',ask)}
    row['sha256']=info['sha256']
    if existing:
        row['roi']=existing.get('roi')
    row['truth']=truth
    row['annotation_review']={'ingredient_pairs_checked':pairs_checked,
        'privacy_checked':yes('Have you inspected pixels, EXIF metadata and annotation text for private information and confirmed this copy is appropriate to retain?',ask)}
    validate_row(row,require_review=True)
    print(json.dumps(row,indent=2))
    if not yes('Save this annotation and a byte-identical local image copy (or update existing annotation)?',ask):
        raise ValueError('Cancelled; nothing saved')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default=DEFAULT)
    sub=parser.add_subparsers(dest='command',required=True)
    add=sub.add_parser('add');add.add_argument('image');add.add_argument('--no-open',action='store_true',help='Inspect manually; still requires privacy/ground-truth review')
    edit=sub.add_parser('edit');edit.add_argument('id');edit.add_argument('--no-open',action='store_true')
    sub.add_parser('report')
    batch=sub.add_parser('batch');batch.add_argument('folder');batch.add_argument('--ocr-side',type=int,default=1600)
    batch.add_argument('--refresh',action='store_true',help='re-inspect drafts that have not been reviewed yet')
    batch.add_argument('--qwen',action='store_true',help='also record a local Qwen transcription as an observation (slow; needs Ollama)')
    review=sub.add_parser('review');review.add_argument('--no-open',action='store_true')
    args=parser.parse_args()
    from research.dataset_images import resolve
    manifest=resolve(args.manifest)
    try:
        if args.command in ('batch','review','report'):
            from research.annotation_batch import batch, review, queue_report
            if args.command=='batch':batch(manifest,args.folder,args.ocr_side,refresh=args.refresh,qwen=args.qwen);return
            if args.command=='review':review(manifest,open_image=not args.no_open);return
            result=coverage(manifest);result['draft_queue']=queue_report(manifest)
            print(json.dumps(result,indent=2));return
        existing=None
        if args.command=='edit':
            existing=next((r for r in load_manifest(manifest)['images'] if r['id']==args.id),None)
            if existing is None:raise ValueError('Unknown image ID')
            source=manifest.parent/existing['path']
        else:source=resolve(args.image)
        row=prompt_annotation(source,existing,open_image=not args.no_open)
        saved=save_annotation(manifest,source,row,existing['id'] if existing else None)
        print('Saved '+saved['id']+'; confirmed='+str(saved['truth']['confirmed']))
    except (ValueError,OSError,KeyError,TypeError,subprocess.SubprocessError) as error:
        parser.exit(1,'Not saved / report failed: '+str(error)+'\n')
    except (EOFError,KeyboardInterrupt):parser.exit(1,'Stopped; prior approved saves and batch/review progress are retained.\n')


if __name__=='__main__':main()
