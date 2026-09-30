import os
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import local_baseline as baseline
import plan_metadata

class ExecutionMetadataTests(unittest.TestCase):
    def setUp(self):
        self.case=dict(name='tiny',b=1,m=1,n=1,k=8,dtype='fp16',ta=False,tb=False,pattern='random',seed=7)
        self.plan=dict(schema_version=3,family='small',requested_family='auto',variant='dot',
                       similarity_available=False,tasks=1,blocks=1,available_cores=24,ub_bytes=192*1024,
                       ub_used=1536,problem=plan_metadata.problem(self.case))
    def test_strict_schema_layout_resources_and_request_identity(self):
        self.assertEqual(plan_metadata.validate_execution(self.plan,self.case),self.plan)
        for fields in ({'family':'other'},{'requested_family':'gm'},{'similarity_available':True},
                       {'ub_used':1535},{'blocks':True},{'tasks':2},{'variant':'rows'},
                       {'problem':dict(self.plan['problem'],k=16)},{'ub_bytes':32768}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                plan_metadata.validate_execution(dict(self.plan,**fields),self.case)
    def test_absent_similarity_requires_valid_small_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fake_run(command,**kwargs):
                directory=Path(command[1]);(directory/'y-0.bin').write_bytes((directory/'golden_y.bin').read_bytes())
                (directory/'execution_plan.json').write_text(json.dumps(self.plan))
                return mock.Mock(returncode=0)
            with mock.patch.object(baseline.subprocess,'run',side_effect=fake_run):
                result=baseline.run_case(Path('/fake'),self.case,root/'valid',0,1,10,True)
                self.assertEqual(result['status'],'PASS',result)
                self.assertEqual(result['similarity_validation'],'unavailable_in_small_family')
                self.assertNotIn('matmul_precision',result)
                self.plan['similarity_available']=True
                result=baseline.run_case(Path('/fake'),self.case,root/'invalid',0,1,10,True)
                self.assertEqual(result['status'],'FAIL',result)
    def test_small_rules_are_deterministic_and_cover_boundaries(self):
        cases=baseline.cases_for_suite('small')
        self.assertEqual(cases,baseline.cases_for_suite('small'))
        self.assertEqual(len(cases),len({c['name'] for c in cases}))
        self.assertTrue({7,8,9,31,32,33}<={c['b'] for c in cases})
        self.assertTrue({63,64,65}<={c['n'] for c in cases})
        self.assertTrue({248,256,264}<={c['k'] for c in cases})
        self.assertEqual({(c['dtype'],c['ta'],c['tb'])for c in cases},{(d,a,b)for d in ('fp16','bf16')for a in (False,True)for b in (False,True)})

    def test_forced_family_rejects_old_binary_and_wrong_request_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def legacy_run(command, **kwargs):
                directory=Path(command[1]);(directory/'y-0.bin').write_bytes((directory/'golden_y.bin').read_bytes())
                if self.plan is not None:
                    (directory/'execution_plan.json').write_text(json.dumps(self.plan))
                return mock.Mock(returncode=0)
            for metadata in (None, self.plan):
                self.plan=metadata
                with mock.patch.dict(os.environ, {'CANN_EXECUTION_FAMILY':'small'}), mock.patch.object(baseline.subprocess,'run',side_effect=legacy_run):
                    result=baseline.run_case(Path('/legacy'),self.case,root/str(metadata is None),0,1,10,False)
                    self.assertEqual(result['status'],'FAIL',result)
