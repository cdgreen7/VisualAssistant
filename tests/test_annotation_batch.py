import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from research.dataset_images import discover,read,derived,related,ROOT
from research.annotation_batch import batch,review,make_row,editable_fields,suggest
from research.annotate_medicine import load_manifest
from research.medicine_eval import prepare,score


def suggestions(image=None):
    fields={'category':'single_ingredient','object_category':'bottle','brand':'Test','product_name':'Test medicine',
            'ingredients':[{'name':'IBUPROFEN','strength':'200','unit':'mg'}],'negative':False,'readable':True,'dosage_form':'tablet','exact_product':None}
    return {'fields':fields,'field_sources':{k:'test model observation' for k in fields},'observations':{}}


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.images=self.base/'photos';self.images.mkdir();self.manifest=self.base/'manifest.json'
        self.manifest.write_text(json.dumps({'schema_version':1,'dataset_id':'test','images':[]}))
        self.image=self.images/'a.jpg';Image.new('RGB',(40,60),'red').save(self.image)

    def batch(self):
        with patch('builtins.print'):batch(self.manifest,self.images,reader=suggestions)
        return json.loads((self.base/'drafts.json').read_text())

    def test_discovery_nested_relative_absolute(self):
        sub=self.images/'converted';sub.mkdir();(sub/'ignored.txt').write_text('x')
        Image.new('RGB',(20,30)).save(sub/'B.PNG')
        self.assertEqual(2,len(discover(self.images)))
        self.assertEqual(discover(ROOT/'VA-photos'),discover('VA-photos'))

    def test_exact_duplicates_resume_without_inference(self):
        (self.images/'copy.jpeg').write_bytes(self.image.read_bytes());q=self.batch()
        self.assertEqual(1,len(q['items']));self.assertEqual(2,len(q['items'][0]['aliases']))
        with patch('builtins.print'):batch(self.manifest,self.images,reader=lambda i: self.fail('Already processed'))

    def test_similar_versions_require_review(self):
        sub=self.images/'optimized';sub.mkdir();Image.open(self.image).resize((20,30)).save(sub/'a.png')
        q=self.batch();self.assertEqual(2,len(q['items']))
        self.assertTrue(q['items'][0]['possible_duplicates']);self.assertEqual([],load_manifest(self.manifest)['images'])

    def test_heic_unavailable_recorded_without_download(self):
        p=self.images/'bad.HEIC';p.write_bytes(b'bad')
        q=self.batch();self.assertTrue(any(i['state']=='error' for i in q['items']))

    def test_high_resolution_original_and_derivative(self):
        Image.new('RGB',(5712,4284),'green').save(self.image)
        before=self.image.read_bytes();data,info,image=read(self.image);output,transform=derived(image)
        self.assertEqual((5712,4284),(info['width'],info['height']))
        self.assertEqual(before,data);self.assertEqual(before,self.image.read_bytes())
        self.assertLessEqual(transform['width']*transform['height'],7_000_000);self.assertLess(len(output),8*1024*1024)

    def test_model_acceptance_never_confirms_truth(self):
        item=self.batch()['items'][0];r=make_row(item,suggestions()['fields'],False)
        self.assertFalse(r['truth']['confirmed']);self.assertEqual([],r['truth']['ingredients'])
        self.assertEqual('IBUPROFEN',r['reviewed_suggestions']['ingredients'][0]['name'])
        self.assertTrue(all(v is None for v in score(r,{}).values()))

    def test_review_save_draft_and_resume_no_overwrite(self):
        q=self.batch();answers=iter(['','n','','','y','y'])
        with patch('builtins.print'):review(self.manifest,ask=lambda p:next(answers),open_image=False)
        row=load_manifest(self.manifest)['images'][0]
        self.assertFalse(row['truth']['confirmed']);self.assertTrue((self.base/row['evaluation_image']['path']).exists())
        payload,mime,metadata=prepare(row,self.base);self.assertEqual('image/jpeg',mime)
        with patch('builtins.print'):review(self.manifest,ask=lambda p:self.fail('Already saved'),open_image=False)
        self.assertEqual(1,len(load_manifest(self.manifest)['images']))

    def test_independent_source_required_for_confirmed(self):
        self.batch();answers=iter(['','y','AB','Physical package','y','bottle-a','test','y','y'])
        with patch('builtins.print'):review(self.manifest,ask=lambda p:next(answers),open_image=False)
        row=load_manifest(self.manifest)['images'][0];self.assertTrue(row['truth']['confirmed']);self.assertEqual('Physical package',row['truth']['confirmation_source'])

    def test_field_correction_and_interruption_persist(self):
        self.batch();answers=iter(['set brand "Corrected"'])
        def ask(p):
            try:return next(answers)
            except StopIteration:raise KeyboardInterrupt()
        with patch('builtins.print'),self.assertRaises(KeyboardInterrupt):review(self.manifest,ask=ask,open_image=False)
        q=json.loads((self.base/'drafts.json').read_text());self.assertEqual('Corrected',q['items'][0]['reviewed_fields']['brand'])
        self.assertEqual([],load_manifest(self.manifest)['images'])

    def test_skip_and_resume(self):
        self.batch()
        with patch('builtins.print'):review(self.manifest,ask=lambda p:'skip',open_image=False)
        self.assertEqual('skipped',json.loads((self.base/'drafts.json').read_text())['items'][0]['state'])
        answers=iter(['','n','','','y','y'])
        with patch('builtins.print'):review(self.manifest,ask=lambda p:next(answers),open_image=False)
        self.assertEqual(1,len(load_manifest(self.manifest)['images']))

    def test_split_group_cannot_cross_train_test(self):
        item=self.batch()['items'][0];r=make_row(item,suggestions()['fields'],False);r['path']='a.jpg';r['physical_product_id']='same';r['split']='train'
        other=copy.deepcopy(r);other.update(id='other',photo_group_id='other',source_sha256='other',split='test')
        self.manifest.write_text(json.dumps({'schema_version':1,'images':[r,other]}))
        with self.assertRaisesRegex(ValueError,'crosses'):load_manifest(self.manifest)

    def test_changed_original_derivative_rejected(self):
        self.test_review_save_draft_and_resume_no_overwrite()
        row=load_manifest(self.manifest)['images'][0];(self.base/row['evaluation_image']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Derived image hash'):prepare(row,self.base)

    def test_suggestions_preserve_combination_unknown_strength(self):
        from types import SimpleNamespace as S
        observed={'ocr':[{'text':'IBUPROFEN 125 mg','confidence':.9},{'text':'ACETAMINOPHEN 250 mg','confidence':.9}],
                  'detections':[{'class_name':'bottle'}]}
        match=S(evidence=[S(kind='ingredient',value=n) for n in ['IBUPROFEN','ACETAMINOPHEN']])
        r=suggest(observed,match);self.assertEqual('combination',r['fields']['category'])
        self.assertEqual({'125','250'},{i['strength'] for i in r['fields']['ingredients']})
        self.assertIsNone(r['fields']['negative'])

    def test_confirmed_duplicate_is_not_another_evaluation_image(self):
        sub=self.images/'optimized';sub.mkdir();Image.open(self.image).resize((20,30)).save(sub/'a.png')
        q=self.batch();small=next(i for i in q['items'] if i['info']['width']==20)
        large=next(i for i in q['items'] if i['info']['width']==40)
        answers=iter(['distinct','','n','','','y','y',large['id']])
        with patch('builtins.print'):review(self.manifest,ask=lambda p:next(answers),open_image=False)
        q=json.loads((self.base/'drafts.json').read_text())
        self.assertEqual('duplicate',next(i for i in q['items'] if i['id']==small['id'])['state'])
        self.assertEqual(1,len(load_manifest(self.manifest)['images']))

    def test_no_privacy_approval_no_save(self):
        self.batch();answers=iter(['','n','','','n'])
        with patch('builtins.print'):review(self.manifest,ask=lambda p:next(answers),open_image=False)
        self.assertEqual([],load_manifest(self.manifest)['images'])

    def test_offline_processing_limit_configurable(self):
        with patch.dict('os.environ',{'VA_RESEARCH_MAX_PIXELS':'100'}):
            with self.assertRaisesRegex(ValueError,'safety limit'):read(self.image)

    def test_existing_confirmed_annotation_never_overwritten(self):
        from research.annotate_medicine import save_annotation
        from tests.test_annotate_medicine import annotation
        save_annotation(self.manifest,self.image,annotation('legacy-confirmed'))
        before=self.manifest.read_bytes()
        with patch('builtins.print'):batch(self.manifest,self.images,reader=lambda i:self.fail('Known image should not infer'))
        self.assertEqual(before,self.manifest.read_bytes())
        q=json.loads((self.base/'drafts.json').read_text());self.assertEqual('saved',q['items'][0]['state'])
        self.assertFalse(q['items'][0]['truth']['confirmed'])
