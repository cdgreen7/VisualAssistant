import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from research.annotate_medicine import save_annotation,coverage,validate_row,image_info,prompt_annotation
from research.medicine_eval import load_manifest


def annotation(ident='one',product='bottle-1'):
    return {'id':ident,'path':'pending','category':'combination','physical_product_id':product,
            'provenance':{'source':'Synthetic test image','local_evaluation_permitted':True},
            'annotation_review':{'privacy_checked':True,'ingredient_pairs_checked':True},
            'truth':{'confirmed':True,'confirmed_by':'tester','confirmation_source':'Synthetic test specification',
                     'object_category':'bottle','brand':'Test','ingredients':[
                         {'name':'ACETAMINOPHEN','strength':'250','unit':'mg'},
                         {'name':'IBUPROFEN','strength':'125','unit':'mg'}],
                     'exact_product':None,'negative':False,'readable':True}}


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'schema_version':1,'dataset_id':'test','images':[]}))
        self.source=self.root/'source.png';Image.new('RGB',(24,32),'red').save(self.source)

    def save(self,r=None):return save_annotation(self.manifest,self.source,r or annotation())

    def test_add_preserves_original_and_evaluator_compatibility(self):
        original=self.source.read_bytes();r=self.save()
        self.assertEqual(original,self.source.read_bytes())
        self.assertEqual(original,(self.root/r['path']).read_bytes())
        loaded=load_manifest(self.manifest)
        self.assertEqual('250',loaded['images'][0]['truth']['ingredients'][0]['strength'])
        self.assertEqual('125',loaded['images'][0]['truth']['ingredients'][1]['strength'])

    def test_duplicate_image_hash_rejected_even_with_other_name(self):
        self.save()
        other=self.root/'renamed.png';other.write_bytes(self.source.read_bytes())
        with self.assertRaisesRegex(ValueError,'Duplicate image bytes'):save_annotation(self.manifest,other,annotation('two'))
        self.assertEqual(1,len(load_manifest(self.manifest)['images']))

    def test_missing_and_unsupported_images(self):
        with self.assertRaises(FileNotFoundError):image_info(self.root/'absent.jpg')
        gif=self.root/'test.jpg';Image.new('RGB',(10,10)).save(gif,'GIF')
        with self.assertRaisesRegex(ValueError,'JPEG and PNG'):image_info(gif)
        gif.write_bytes(b'broken');self.assertRaises(OSError,image_info,gif)

    def test_private_or_unpermitted_annotation_not_saved(self):
        for part,key in [('annotation_review','privacy_checked'),('provenance','local_evaluation_permitted')]:
            r=annotation();r[part][key]=False
            with self.assertRaises(ValueError):self.save(r)
        self.assertEqual([],load_manifest(self.manifest)['images'])

    def test_unknown_strength_saved_only_as_unconfirmed_draft(self):
        r=annotation();r['truth']['ingredients'][0]['strength']=None
        with self.assertRaises(ValueError):self.save(r)
        r['truth']['confirmed']=False;self.save(r)
        report=coverage(self.manifest)
        self.assertEqual(0,report['confirmed_strength_labels']);self.assertEqual(['one'],report['missing_annotations'])

    def test_association_review_required_and_duplicate_ingredients_rejected(self):
        r=annotation();r['annotation_review']['ingredient_pairs_checked']=False
        with self.assertRaisesRegex(ValueError,'review'):self.save(r)
        r=annotation();r['truth']['ingredients'][1]=copy.deepcopy(r['truth']['ingredients'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate ingredient'):self.save(r)

    def test_invalid_strength_unit_and_category(self):
        for field,value in [('strength','250/125'),('strength','0'),('unit','guessed units')]:
            r=annotation();r['truth']['ingredients'][0][field]=value
            with self.assertRaises(ValueError):self.save(r)
        r=annotation();r['category']='single_ingredient'
        with self.assertRaises(ValueError):self.save(r)

    def test_exact_identity_needs_confirmation_and_valid_ndc(self):
        r=annotation();r['truth']['exact_product']={'package_ndc':'123','verification_source':'physical package'}
        with self.assertRaises(ValueError):self.save(r)
        r['truth']['exact_product']['package_ndc']='11111-111-01';r['truth']['confirmed']=False
        with self.assertRaises(ValueError):self.save(r)
        r['truth']['confirmed']=True;self.save(r)
        self.assertEqual(1,coverage(self.manifest)['confirmed_exact_product_identities'])

    def test_negative_and_unreadable_examples(self):
        r=annotation();r['category']='non_medicine_bottle';r['truth'].update(negative=True,ingredients=[],readable=False)
        self.save(r);report=coverage(self.manifest)
        self.assertEqual(1,report['total_labeled_images']);self.assertEqual(0,report['confirmed_ingredient_labels'])

    def test_edit_draft_without_duplicate_copy(self):
        r=annotation();r['truth']['confirmed']=False;saved=self.save(r)
        updated=annotation();save_annotation(self.manifest,self.root/saved['path'],updated,edit_id='one')
        self.assertEqual(1,len(list((self.root/'images').iterdir())))
        self.assertEqual(1,coverage(self.manifest)['total_labeled_images'])

    def test_product_grouping_does_not_count_views_as_products(self):
        self.save();Image.new('RGB',(24,32),'blue').save(self.source)
        self.save(annotation('two'))
        report=coverage(self.manifest)
        self.assertEqual(2,report['total_images']);self.assertEqual(1,report['unique_physical_products'])

    def test_conflicting_truth_on_same_physical_product_rejected(self):
        self.save();Image.new('RGB',(24,32),'blue').save(self.source)
        r=annotation('two');r['truth']['ingredients'][0]['strength']='100'
        with self.assertRaisesRegex(ValueError,'Conflicting truth'):self.save(r)

    def test_report_missing_files_duplicates_and_invalid_annotations(self):
        r=self.save();data=load_manifest(self.manifest)
        dup=copy.deepcopy(r);dup['id']='duplicate';data['images'].append(dup)
        missing=copy.deepcopy(r);missing.update(id='missing',path='missing.png');missing['truth']['confirmed']=False
        data['images'].append(missing);bad=copy.deepcopy(r);bad['id']='bad';bad['category']='invalid';data['images'].append(bad)
        self.manifest.write_text(json.dumps(data));report=coverage(self.manifest)
        self.assertEqual(['missing'],report['missing_files'])
        self.assertEqual(1,len(report['duplicate_images']));self.assertEqual('bad',report['invalid_annotations'][0]['id'])

    def test_changed_file_blocks_add(self):
        r=self.save();Image.new('RGB',(24,32),'blue').save(self.root/r['path'])
        with self.assertRaisesRegex(ValueError,'changed'):self.save(annotation('two'))
        self.assertEqual(['one'],coverage(self.manifest)['changed_files'])

    def test_manifest_write_failure_rolls_back_new_copy(self):
        with patch('research.annotate_medicine.atomic_json',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.save()
        self.assertEqual([],list((self.root/'images').iterdir()))
        self.assertEqual([],load_manifest(self.manifest)['images'])

    def test_interactive_negative_workflow_and_local_preview(self):
        answers=iter(['negative-1','','no_bottle','iphone-safari','full_frame','My project photo','y','y','n','none','','','AB','Manual inspection','y','y','y'])
        with patch('research.annotate_medicine.inspect_image') as viewer,patch('builtins.print'):
            r=prompt_annotation(self.source,ask=lambda _:next(answers))
        viewer.assert_called_once_with(self.source.resolve())
        self.assertTrue(r['truth']['negative']);self.save(r)

    def test_report_handles_malformed_rows_without_losing_other_findings(self):
        self.manifest.write_text(json.dumps({'schema_version':1,'images':[None,{'id':'bad','truth':None,'physical_product_id':{}}]}))
        report=coverage(self.manifest)
        self.assertEqual(2,len(report['invalid_annotations']))
        self.assertEqual(0,report['total_labeled_images'])

    def test_cancel_confirmation_writes_nothing(self):
        answers=iter(['negative-1','','no_bottle','iphone-safari','full_frame','My project photo','y','y','n','none','','','AB','Manual inspection','y','y','n'])
        with patch('builtins.print'),self.assertRaisesRegex(ValueError,'Cancelled'):
            prompt_annotation(self.source,ask=lambda _:next(answers),open_image=False)
        self.assertEqual([],load_manifest(self.manifest)['images'])

    def test_source_change_after_inspection_is_rejected(self):
        r=annotation();r['sha256']=image_info(self.source)[1]['sha256']
        Image.new('RGB',(24,32),'blue').save(self.source)
        with self.assertRaisesRegex(ValueError,'changed since inspection'):self.save(r)

    def test_edit_preserves_existing_evaluation_roi(self):
        r=annotation();r['roi']=[1,2,20,30];saved=self.save(r)
        updated=annotation();updated['roi']=None
        result=save_annotation(self.manifest,self.root/saved['path'],updated,edit_id='one')
        self.assertEqual([1,2,20,30],result['roi'])
