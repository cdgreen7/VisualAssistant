import copy
import json
from pathlib import Path
import tempfile
import unittest
from research.medicine_eval import score,summary,aggregate,load_manifest,evaluate_image
from tests.test_medicine_api import catalog,Recognizer
from web.services import VisionServices
from PIL import Image


def row(combo=False):
    ingredients=[{'name':'IBUPROFEN','strength':'200','unit':'mg'}]
    if combo:ingredients=[{'name':'ACETAMINOPHEN','strength':'250','unit':'mg'},{'name':'IBUPROFEN','strength':'125','unit':'mg'}]
    return {'id':'fixture','path':'test.png','category':'combination' if combo else 'single_ingredient',
            'provenance':{'source':'synthetic unit test','local_evaluation_permitted':True},
            'truth':{'confirmed':True,'confirmed_by':'unit test author','confirmation_source':'synthetic test specification',
                     'negative':False,'readable':True,'object_category':'bottle','brand':'Equate','ingredients':ingredients,'exact_product':None}}


def body(combo=False,status='VERIFIED'):
    ingredients=row(combo)['truth']['ingredients']
    return {'ocr':[{'text':i['name']+' '+i['strength']+' '+i['unit'],'confidence':.99} for i in ingredients]+[{'text':'Equate','confidence':.9}],
            'match':{'status':status,'level':'FORMULATION','package':None,
                     'candidates':[{'ingredients':[{'name':i['name'],'strength':i['strength']+' '+i['unit']+'/1'} for i in ingredients]}],
                     'evidence':[{'kind':'ingredient','value':i['name']} for i in ingredients]+[{'kind':'strength','value':i['strength']+i['unit']} for i in ingredients]}}


class MedicineEvalTests(unittest.TestCase):
    def test_uncertain_not_verified(self):
        r=score(row(),body(status='UNCERTAIN'))
        self.assertTrue(r['ingredients']);self.assertFalse(r['formulation_verified']);self.assertIsNone(r['exact_product'])

    def test_unconfirmed_not_scored(self):
        r=row();r['truth']['confirmed']=False
        self.assertTrue(all(v is None for v in score(r,body()).values()))

    def test_combination_and_false_verified(self):
        self.assertTrue(score(row(True),body(True))['combination_verified'])
        wrong=body(True);wrong['match']['candidates'][0]['ingredients'][0]['strength']='125 mg/1'
        result=score(row(True),wrong)
        self.assertTrue(result['false_verified']);self.assertFalse(result['combination_verified'])

    def test_missing_second_ingredient_is_wrong(self):
        self.assertFalse(score(row(True),body())['ingredients'])
        self.assertTrue(score(row(True),body())['false_verified'])

    def test_package_claim_without_truth_is_not_assumed_safe(self):
        b=body();b['match']['level']='PACKAGE';b['match']['package']={'package_ndc':'11111-111-01'}
        self.assertIsNone(score(row(),b)['false_verified'])
        r=row();r['truth']['exact_product']={'package_ndc':'11111-111-02','verification_source':'physical package'}
        self.assertTrue(score(r,b)['false_verified']);self.assertFalse(score(r,b)['exact_product'])

    def test_negative_and_unreadable(self):
        r=row();r['truth']['negative']=True
        self.assertTrue(score(r,body())['false_verified'])
        self.assertTrue(score(r,body(status='NO_MATCH'))['negative_rejected'])
        r=row();r['truth']['readable']=False
        self.assertTrue(score(r,body(status='UNCERTAIN'))['appropriate_uncertainty'])

    def test_sample_thresholds(self):
        self.assertIsNone(summary([1,2])['median'])
        self.assertIsNone(summary(list(range(9)))['p90'])
        self.assertIsNone(summary(list(range(19)))['p95'])
        self.assertEqual(18.05,summary(list(range(20)))['p95'])

    def test_repeated_trials_do_not_multiply_accuracy_denominator(self):
        runs=[{'events':[{'event':'trial','success':True,'phase':'warm','scores':score(row(),body()),'ocr_ms':1} for _ in range(20)]}]
        counts=aggregate(runs)
        self.assertEqual(1,counts['ingredients']['evaluated']);self.assertEqual(20,runs[0]['warm']['ocr_ms']['n'])

    def test_manifest_requires_provenance_and_ground_truth_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'manifest.json';r=row()
            for invalid in ('source','confirmation_source'):
                bad=copy.deepcopy(r)
                (bad['provenance'] if invalid=='source' else bad['truth']).pop(invalid)
                p.write_text(json.dumps({'schema_version':1,'images':[bad]}))
                with self.assertRaises(ValueError):load_manifest(p)
            p.write_text(json.dumps({'schema_version':1,'images':[r,r]}))
            with self.assertRaises(ValueError):load_manifest(p)

    def test_real_endpoint_path_with_mocked_ocr_and_real_matcher(self):
        with tempfile.TemporaryDirectory() as d:
            Image.new('RGB',(100,200),'white').save(Path(d)/'test.png')
            lines=[{'text':'Equate Ibuprofen 200 mg','confidence':.99,'box':(0,0,20,20)}]
            services=VisionServices(loaders={'catalog':catalog,'recognizer':lambda:Recognizer(lines)})
            events=[];evaluate_image(row(),Path(d),3,events.append,services)
            trials=[e for e in events if e['event']=='trial']
            self.assertEqual(['cold','warm','warm','warm'],[e['phase'] for e in trials])
            self.assertTrue(all(e['http_status']==200 for e in trials))
            self.assertTrue(all(e['decode_ms'] is not None and e['crop_ms']>=0 for e in trials))
            self.assertNotIn('detector',services.loaded())


def hang_worker(row,base,trials,connection):
    import time
    connection.send({'event':'input'})
    time.sleep(10)


def fail_worker(row,base,trials,connection):
    connection.send({'event':'failure','error':'TestFailure'})
    connection.close()


class ProcessControlTests(unittest.TestCase):
    def test_timeout_preserves_partial_records(self):
        from research.medicine_eval import run_worker
        r=run_worker(row(),Path('.'),1,1,target=hang_worker)
        self.assertTrue(r['timeout']);self.assertEqual('input',r['events'][0]['event'])

    def test_worker_failure_is_recorded(self):
        from research.medicine_eval import run_worker
        r=run_worker(row(),Path('.'),1,5,target=fail_worker)
        self.assertFalse(r['timeout']);self.assertEqual('failure',r['events'][0]['event'])

    def test_swapped_combination_strength_associations(self):
        b=body(True);b['ocr']=[{'text':'ACETAMINOPHEN 125 mg','confidence':1},{'text':'IBUPROFEN 250 mg','confidence':1}]
        self.assertFalse(score(row(True),b)['strengths'])


class FailureTests(unittest.TestCase):
    def test_valid_image_exceeding_endpoint_pixel_limit_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            Image.new('RGB',(2700,2700),'white').save(Path(d)/'test.png')
            services=VisionServices(loaders={'catalog':catalog,'recognizer':lambda:Recognizer([])})
            events=[];evaluate_image(row(),Path(d),1,events.append,services)
            trials=[e for e in events if e['event']=='trial']
            self.assertTrue(all(e['http_status']==413 and not e['success'] for e in trials))
            self.assertTrue(all(e['scores']['false_verified'] is None for e in trials))
