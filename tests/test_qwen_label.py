import json
import unittest
from unittest.mock import Mock
import numpy as np
from web.qwen_label import QwenLabelReader
from web.services import ServiceError


class QwenLabelTests(unittest.TestCase):
    def test_conflicts_and_agreements_are_separate_untrusted_observations(self):
        reader=QwenLabelReader()
        reader._post=Mock(side_effect=[{'capabilities':['vision']},{'done':True,'message':{'content':json.dumps(
            {'lines':['Ibuprofen 200 mg','Acetaminophen 500 mg']})}}])
        result=reader.read(np.zeros((894,477,3),dtype=np.uint8),[{'text':'Ibuprofen 200 mg'}])
        self.assertEqual(result['agrees_with_ocr'],['Ibuprofen 200 mg'])
        self.assertEqual(result['unconfirmed_or_conflicting'],['Acetaminophen 500 mg'])
        self.assertFalse(result['used_for_matching'])
        self.assertEqual(reader._post.call_args.args[0],'chat')

    def test_invalid_response_and_remote_models_rejected(self):
        for answer in ({'done':False}, {'done':True,'message':{'content':'{"lines":[123]}'}}):
            reader=QwenLabelReader()
            reader._post=Mock(side_effect=[{'capabilities':['vision']},answer])
            with self.assertRaises(ServiceError):reader.read(np.zeros((30,20,3),np.uint8),[])
        reader=QwenLabelReader()
        reader._post=Mock(return_value={'capabilities':['vision'],'remote_host':'https://example.test'})
        with self.assertRaises(ServiceError):reader.read(np.zeros((30,20,3),np.uint8),[])

    def test_timeout_releases_reader_lock(self):
        reader=QwenLabelReader()
        reader._post=Mock(side_effect=ServiceError(504,'model_timeout','timeout'))
        with self.assertRaises(ServiceError):reader.read(np.zeros((30,20,3),np.uint8),[])
        self.assertTrue(reader._lock.acquire(False));reader._lock.release()
