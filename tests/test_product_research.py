"""Official research is mocked; it never changes identification or the registry."""
import copy
import io
import json
import socket
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from web.product_research import ProductResearch, observations, query_for, amounts
from web.server import create_app
from web.services import ServiceError


def evidence(combo=False):
    names = [('IBUPROFEN','200 mg')] if not combo else [('ACETAMINOPHEN','250 mg'),('IBUPROFEN','125 mg')]
    return {'evidence':[{'kind':'ingredient','value':n} for n,s in names]+[{'kind':'strength','value':s} for n,s in names],
            'ocr':[{'text':n+' '+s,'confidence':.95} for n,s in names]}


def record(combo=False):
    p=evidence(combo)
    return {'id':'official-1','set_id':'12345678-1234-1234-1234-123456789abc','effective_time':'20260101',
            'openfda':{'brand_name':['EQUATE'], 'substance_name':[e['value'] for e in p['evidence'] if e['kind']=='ingredient'],
                       'package_ndc':['11111-111-01'], 'upc':['012345678905']},
            'active_ingredient':[r['text'] for r in p['ocr']], 'purpose':['Pain reliever'],
            'dosage_and_administration':['DO NOT RETURN DIRECTIONS']}


def service(records=None):
    records=[record()] if records is None else records
    return ProductResearch(fetch=lambda q:{'results':records,'meta':{'results':{'total':len(records)}}})


class ProductResearchTests(unittest.TestCase):
    def test_exact_ndc_is_label_association_only(self):
        p=evidence();p['evidence'].append({'kind':'ndc','value':'11111-111-01'})
        r=service().research(p)['records'][0]
        self.assertTrue(r['exact_ndc']);self.assertIn('Not verified',r['exact_package'])
        self.assertNotIn('DO NOT RETURN',json.dumps(r))

    def test_exact_upc(self):
        r=service().research({'upc':'012345678905'})['records'][0]
        self.assertTrue(r['exact_upc']);self.assertEqual('INCOMPLETE',r['comparison'])

    def test_ingredient_strength_and_combination(self):
        for combo in [False,True]:
            r=service([record(combo)]).research(evidence(combo))['records'][0]
            self.assertEqual('SUPPORTED',r['comparison'])
            self.assertEqual(2 if combo else 1,len(r['ingredients']))

    def test_combination_never_becomes_plain_ibuprofen(self):
        r=service().research(evidence(True))['records'][0]
        self.assertEqual('CONFLICTING',r['comparison'])

    def test_swapped_strengths_conflict(self):
        r=record(True);r['active_ingredient']=['ACETAMINOPHEN 125 mg','IBUPROFEN 250 mg']
        self.assertEqual('CONFLICTING',service([r]).research(evidence(True))['records'][0]['comparison'])

    def test_missing_or_unassociated_strength_is_incomplete(self):
        for lines in [['IBUPROFEN'], ['ACETAMINOPHEN 250 mg IBUPROFEN 125 mg']]:
            p=evidence(len(lines[0])>20);p['ocr']=[{'text':t,'confidence':.9} for t in lines]
            self.assertEqual('INCOMPLETE',service([record(len(lines[0])>20)]).research(p)['records'][0]['comparison'])

    def test_multiple_conflicting_records_are_preserved(self):
        bad=record();bad['active_ingredient']=['IBUPROFEN 400 mg']
        r=service([record(),bad]).research(evidence())
        self.assertEqual(['SUPPORTED','CONFLICTING'],[c['comparison'] for c in r['records']])
        self.assertEqual(2,r['total_records'])

    def test_identifier_miss_falls_back_without_upgrading_package(self):
        queries=[]
        def fetch(q):
            queries.append(q)
            return {'results':[] if 'openfda.upc' in q else [record()]}
        p=evidence();p['upc']='036000291452'
        r=ProductResearch(fetch=fetch).research(p)
        self.assertEqual(2,len(queries))
        self.assertEqual('label_evidence_after_identifier_not_found',r['search_basis'])
        self.assertFalse(r['records'][0]['exact_upc'])
        self.assertIn('Not verified',r['records'][0]['exact_package'])

    def test_no_results_not_cached(self):
        s=service([])
        self.assertEqual('NO_RESULTS',s.research(evidence())['status'])
        self.assertFalse(s.research(evidence())['cached'])

    def test_success_cache_rechecks_new_ocr(self):
        s=service();s.research(evidence());p=evidence();p['ocr'][0]['text']='IBUPROFEN 400 mg'
        r=s.research(p);self.assertTrue(r['cached']);self.assertEqual('CONFLICTING',r['records'][0]['comparison'])

    def test_low_confidence_strength_does_not_support(self):
        p=evidence();p['ocr'][0]['confidence']=.3
        self.assertEqual('INCOMPLETE',service().research(p)['records'][0]['comparison'])

    def test_normalized_strength_query_has_unit_spacing(self):
        p=evidence();p['evidence'][1]['value']='200mg'
        self.assertIn('200 mg',query_for(observations(p)))

    def test_cache_expiry(self):
        now=[0];s=service();s.clock=lambda:now[0];s.research(evidence());now[0]=3601
        self.assertFalse(s.research(evidence())['cached'])

    def test_unverified_listing_cannot_support_or_match_upc(self):
        with self.assertRaises(ServiceError):
            service([{'title':'Equate ibuprofen 200 mg','upc':'012345678905'}]).research({'upc':'012345678905'})

    def test_concentration_does_not_equal_tablet(self):
        self.assertNotEqual(amounts('200 mg/5 mL'),amounts('200 mg'))

    def test_query_escapes_and_uses_all_ingredients(self):
        q=query_for(observations(evidence(True)))
        self.assertIn('ACETAMINOPHEN',q);self.assertIn('IBUPROFEN',q)
        p=evidence();p['evidence'].append({'kind':'brand','value':'x" OR *'})
        self.assertNotIn('" OR *',query_for(observations(p)))

    def test_invalid_input(self):
        for p in [None,{}, {'upc':'123'}, {'ocr':'bad'}, {'evidence':[1]}, {'evidence':[{'kind':'ndc','value':'123'}]}]:
            with self.assertRaises(ServiceError): service().research(p)

    def test_invalid_model_or_provider_response(self):
        with self.assertRaises(ServiceError): ProductResearch(fetch=lambda q:{'results':'bad'}).research(evidence())

    def test_disabled(self):
        with patch.dict('os.environ',{'PRODUCT_LOOKUP_ENABLED':'0'}),self.assertRaises(ServiceError): service().research(evidence())

    def test_http_failures_timeouts_and_rate_limits(self):
        cases=[(HTTPError('url',429,'limited',{},None),429), (HTTPError('url',500,'fail',{},None),502),
               (socket.timeout(),504),(URLError(socket.timeout()),504),(URLError('offline'),502)]
        for error,status in cases:
            with patch('web.product_research.urlopen',side_effect=error),self.assertRaises(ServiceError) as caught:
                ProductResearch().research(evidence())
            self.assertEqual(status,caught.exception.http_status)

    def test_http_404(self):
        with patch('web.product_research.urlopen',side_effect=HTTPError('url',404,'none',{},None)):
            self.assertEqual('NO_RESULTS',ProductResearch().research(evidence())['status'])

    def test_invalid_json(self):
        with patch('web.product_research.urlopen',return_value=io.BytesIO(b'not json')),self.assertRaises(ServiceError):
            ProductResearch().research(evidence())

    def test_bounded_concurrency(self):
        s=service();s.slots.acquire();s.slots.acquire()
        with self.assertRaises(ServiceError) as caught:s.research(evidence())
        self.assertEqual(409,caught.exception.http_status)
        s.slots.release();s.slots.release()
        self.assertEqual('RECORDS_FOUND',s.research(evidence())['status'])

    def test_endpoint_and_existing_health(self):
        with create_app(research_service=service()).test_client() as c:
            r=c.post('/api/products/research',json=evidence())
            self.assertEqual(200,r.status_code);self.assertEqual('RECORDS_FOUND',r.json['status'])
            self.assertEqual(400,c.post('/api/products/research',json={}).status_code)
            self.assertEqual(413,c.post('/api/products/research',json={'x':'a'*33000}).status_code)
            self.assertEqual(200,c.get('/health').status_code)
