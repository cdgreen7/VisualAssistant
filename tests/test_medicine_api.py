"""Automatic label crops reuse real ProductMatcher with mocked OCR/model."""
import io
import threading
import unittest
from PIL import Image
from web.server import create_app
from web.services import VisionServices, ServiceError
from vision.products.catalog import ProductCatalog
from tests.test_barcode_lookup import raw


def label(*lines):
    return [{'text':t,'confidence':.95,'box':(0,0,20,20)} for t in lines]


def catalog():
    equate=raw('11111-111',['11111-111-01'],brand='EQUATE IBUPROFEN')
    equate['dosage_form']='TABLET'
    combo=raw('22222-222',['22222-222-01'],brand='KROGER DUAL ACTION')
    combo['active_ingredients']=[{'name':'ACETAMINOPHEN','strength':'250 mg/1'}, {'name':'IBUPROFEN','strength':'125 mg/1'}]
    combo['dosage_form']='TABLET'
    return ProductCatalog.from_openfda_results([equate,combo])


class Recognizer:
    def __init__(self,lines):self.lines,self.shape=lines,None
    def read(self,image):self.shape=image.shape;return self.lines


def image():
    out=io.BytesIO();Image.new('RGB',(477,894),'white').save(out,'JPEG');return out.getvalue()


class MedicineApiTests(unittest.TestCase):
    def services(self,lines):
        self.reader=Recognizer(lines)
        return VisionServices(loaders={'catalog':catalog,'recognizer':lambda:self.reader})

    def post(self,services,assist=False,reader=None,data=None):
        with create_app(services=services,label_reader=reader).test_client() as client:
            return client.post('/api/medicine/identify-crop',data={
                'session_id':'session_1234','capture_id':'capture_1234','assist':str(assist).lower(),
                'image':(io.BytesIO(image() if data is None else data),'label.jpg','image/jpeg')})

    def test_reported_combination_against_installed_catalog_preserves_both_ingredients(self):
        import os
        from web.services import FDA_CACHE, _load_catalog
        if not os.path.exists(FDA_CACHE):self.skipTest('Local FDA cache is not installed')
        services=self.services(label('Kroger','Acetaminophen 250 mg','Ibuprofen 125 mg'))
        services._loaders['catalog']=_load_catalog
        body=self.post(services).get_json()
        self.assertEqual(body['match']['status'],'UNCERTAIN')
        self.assertEqual(set(body['observed_ingredients']),{'ACETAMINOPHEN','IBUPROFEN'})
        self.assertIn('Acetaminophen 250 mg',body['label_recognized'])
        self.assertIn('Ibuprofen 125 mg',body['label_recognized'])
        self.assertEqual(body['exact_package'],'Not verified')

    def test_equate_evidence_and_native_crop(self):
        r=self.post(self.services(label('Equate Ibuprofen','200 mg')))
        self.assertEqual(r.status_code,200)
        body=r.get_json()
        self.assertIn('Equate',body['label_recognized'])
        self.assertIn('200',body['formulation'])
        self.assertEqual(body['exact_package'],'Not verified')
        self.assertEqual(self.reader.shape,(894,477,3))
        self.assertEqual(body['capture_id'],'capture_1234')
        self.assertEqual(body['timing']['qwen_ms'],0)
        self.assertFalse(body['qwen']['enabled'])

    def test_combination_never_becomes_plain_ibuprofen(self):
        body=self.post(self.services(label('Kroger dual action','Acetaminophen 250 mg','Ibuprofen 125 mg'))).get_json()
        self.assertEqual(set(body['observed_ingredients']),{'ACETAMINOPHEN','IBUPROFEN'})
        self.assertIn('ACETAMINOPHEN',body['formulation'])
        self.assertIn('250',body['formulation'])
        self.assertIn('125',body['formulation'])
        self.assertEqual(body['exact_package'],'Not verified')
        self.assertTrue(body['formulation_supported'])

    def test_missing_conflicting_and_unassociated_strengths_remain_uncertain(self):
        for lines in [('Kroger dual action','Acetaminophen','Ibuprofen 125 mg'),
                      ('Kroger dual action','Acetaminophen 200 mg','Ibuprofen 125 mg'),
                      ('Kroger dual action','Acetaminophen','Ibuprofen','250 mg','125 mg'),
                      ('Kroger dual action','Acetaminophen 125 mg','Ibuprofen 250 mg'),
                      ('Kroger dual action','Acetaminophen 125 mg Ibuprofen 250 mg')]:
            body=self.post(self.services(label(*lines))).get_json()
            self.assertEqual(body['match']['status'],'UNCERTAIN',body)
            self.assertFalse(body['formulation_supported'])
            self.assertIn('Acetaminophen',body['label_recognized'])
            self.assertIn('Ibuprofen',body['label_recognized'])

    def test_unreadable_no_match(self):
        for lines in ([],label('unreadable xyz')):
            body=self.post(self.services(lines)).get_json()
            self.assertEqual(body['match']['status'],'NO_MATCH')
            self.assertEqual(body['exact_package'],'Not verified')

    def test_verified_package_only_with_existing_matcher_evidence(self):
        body=self.post(self.services(label('EQUATE IBUPROFEN','200 mg','NDC 11111-111-01'))).get_json()
        self.assertEqual(body['match']['level'],'PACKAGE')
        self.assertIn('11111-111-01',body['exact_package'])

    def test_qwen_conflicts_excluded_from_matching_and_failure_is_optional(self):
        class Reader:
            def read(self,image,ocr):
                return {'lines':['Acetaminophen 500 mg'],'unconfirmed_or_conflicting':['Acetaminophen 500 mg'],
                    'used_for_matching':False}
        body=self.post(self.services(label('Equate Ibuprofen','200 mg')),True,Reader()).get_json()
        self.assertNotIn('ACETAMINOPHEN',body['observed_ingredients'])
        self.assertFalse(body['qwen']['used_for_matching'])
        class Failed:
            def read(self,*args):raise ServiceError(504,'model_timeout','Reader timed out')
        body=self.post(self.services(label('Equate Ibuprofen','200 mg')),True,Failed()).get_json()
        self.assertIn('error',body['qwen'])
        self.assertEqual(body['exact_package'],'Not verified')

    def test_invalid_missing_and_session_validation(self):
        self.assertEqual(self.post(self.services([]),data=b'broken').status_code,400)
        with create_app(services=self.services([])).test_client() as client:
            self.assertEqual(client.post('/api/medicine/identify-crop').status_code,400)
            response=client.post('/api/medicine/identify-crop',data={'session_id':'session_1234','capture_id':'capture_1234'})
            self.assertEqual(response.get_json()['error']['code'],'missing_image')

    def test_byte_and_pixel_limits_and_no_storage(self):
        from web.server import CameraRequest
        self.assertEqual(self.post(self.services([]),data=b'x'*(8*1024*1024+1)).status_code,413)
        out=io.BytesIO();Image.new('RGB',(3000,2400),'white').save(out,'JPEG')
        self.assertEqual(self.post(self.services([]),data=out.getvalue()).status_code,413)
        with create_app().test_request_context('/api/medicine/identify-crop'):
            request=CameraRequest.from_values('/api/medicine/identify-crop')
            stream=request._get_file_stream(900000,'image/jpeg','crop.jpg')
            self.assertIsInstance(stream,io.BytesIO)
            stream.close();request.close()

    def test_actual_yolo_detection_can_finish_while_ocr_is_running(self):
        class Detector:
            def detect(self,image):return []
            def get_detections(self,result):return result
        services=self.services(label('Ibuprofen 200 mg'))
        services._loaders['detector']=lambda:Detector()
        services._get('detector')
        entered,release=threading.Event(),threading.Event()
        original=self.reader.read
        self.reader.read=lambda image:(entered.set(),release.wait(3),original(image))[-1]
        result=[]
        thread=threading.Thread(target=lambda:result.append(self.post(services).status_code))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            import numpy as np
            self.assertEqual(services.detect(np.zeros((360,640,3),np.uint8))[0],[])
        finally:release.set();thread.join(5)
        self.assertEqual(result,[200])

    def test_concurrent_ocr_rejected_while_yolo_lock_is_free(self):
        services=self.services(label('Ibuprofen 200 mg'))
        entered,release=threading.Event(),threading.Event()
        original=self.reader.read
        def blocked(img):
            entered.set();release.wait(3);return original(img)
        self.reader.read=blocked
        results=[]
        thread=threading.Thread(target=lambda:results.append(self.post(services).status_code))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.post(services).status_code,409)
            self.assertTrue(services._work_lock.acquire(blocking=False))
            services._work_lock.release()
        finally:release.set();thread.join(5)
        self.assertEqual(results,[200])
