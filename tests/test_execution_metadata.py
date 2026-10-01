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

    def test_expanded_candidates_preserve_resource_and_task_bounds(self):
        # Byte counts independently calculated from physical rows and 32B padding.
        for fields, used in ((dict(b=33),1536), (dict(m=128),13728),
                             (dict(n=5,k=256,ta=True),50496),
                             (dict(n=64,k=128),51264),
                             (dict(m=200,n=64,tb=True),26688)):
            case=dict(self.case,**fields)
            plan=dict(self.plan, problem=plan_metadata.problem(case),ub_used=used,
                      variant='dot' if case['n']==1 or case['tb'] else 'rows',
                      tasks=(case['b']+7)//8,blocks=min(24,(case['b']+7)//8))
            self.assertEqual(plan_metadata.validate_execution(plan,case),plan)
        for b,accepted in (((2**32-1)*8,True),((2**32-1)*8+1,False)):
            case=dict(self.case,b=b)
            plan=dict(self.plan,problem=plan_metadata.problem(case),tasks=(b+7)//8,blocks=24)
            if accepted:self.assertEqual(plan_metadata.validate_execution(plan,case),plan)
            else:
                with self.assertRaises(ValueError):plan_metadata.validate_execution(plan,case)

    def test_batched_dot_metadata_and_legacy_compatibility(self):
        case=dict(self.case,n=5,tb=True)
        plan=dict(self.plan,problem=plan_metadata.problem(case),dot_columns=5,ub_used=2176)
        self.assertEqual(plan_metadata.validate_execution(plan,case),plan)
        for bad in (dict(dot_columns=1),dict(dot_columns=4),dict(dot_columns=6),dict(dot_columns=True),
                    dict(dot_columns=None),dict(ub_used=2175),dict(ub_bytes=32768+2175)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                plan_metadata.validate_execution(dict(plan,**bad),case)
        # Missing field continues to mean the original Dot implementation.
        legacy=dict(plan,ub_used=1920);del legacy['dot_columns']
        self.assertEqual(plan_metadata.validate_execution(legacy,case),legacy)
        case=dict(case,n=64,k=128)
        plan.update(problem=plan_metadata.problem(case),dot_columns=32,ub_used=58528)
        self.assertEqual(plan_metadata.validate_execution(plan,case),plan)

    def test_stream_metadata_storage_and_missing_similarity(self):
        case=dict(self.case,m=2,n=32,k=2)
        stream=dict(planner_version=1,tile_m=16,tile_n=16,splits=2,row_pitch=32,
                    c_slot_elements=32,maxima_offset=256,partial_offset=384,scratch_bytes=640,
                    max_tiles_per_core=1,ub_used=47584,ub_budget=128*1024,
                    inner_tile=dict(m=16,n=16,k=16))
        plan=dict(self.plan,family='stream',variant='mix_stream',tasks=2,blocks=2,
                  problem=plan_metadata.problem(case),stream=stream)
        del plan['ub_used']
        self.assertEqual(plan_metadata.validate_execution(plan,case),plan)
        for key in stream:
            if isinstance(stream[key],int):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    plan_metadata.validate_execution(dict(plan,stream=dict(stream,**{key:stream[key]+1})),case)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fake_run(command,**kwargs):
                directory=Path(command[1])
                (directory/'y-0.bin').write_bytes((directory/'golden_y.bin').read_bytes())
                (directory/'execution_plan.json').write_text(json.dumps(plan))
                return mock.Mock(returncode=0)
            with mock.patch.object(baseline.subprocess,'run',side_effect=fake_run):
                result=baseline.run_case(Path('/fake'),case,root/'stream',0,1,10,True)
                self.assertEqual(result['status'],'PASS',result)
                self.assertEqual(result['similarity_validation'],'unavailable_in_stream_family')
                self.assertNotIn('matmul_precision',result)

    def test_pipeline_metadata_double_slots_and_validated_report(self):
        case=dict(self.case,m=2,n=32,k=2)
        stream=dict(planner_version=2,buffers=2,tile_m=16,tile_n=16,splits=1,row_pitch=32,
                    c_slot_elements=32,maxima_offset=256,partial_offset=256,scratch_bytes=384,
                    max_tiles_per_core=2,ub_used=47328,ub_budget=128*1024,
                    inner_tile=dict(m=16,n=16,k=16))
        selection=dict(planner_version=2,cost_model='joint-work-v1',score=100.25)
        plan=dict(self.plan,family='pipeline',requested_family='pipeline',variant='mix_pipeline',
                  problem=plan_metadata.problem(case),stream=stream,selection=selection)
        del plan['ub_used']
        self.assertEqual(plan_metadata.validate_execution(plan,case),plan)
        for bad in (dict(buffers=1),dict(buffers=True),dict(planner_version=1),dict(maxima_offset=128),dict(scratch_bytes=256)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                plan_metadata.validate_execution(dict(plan,stream=dict(stream,**bad)),case)
        for bad in (dict(score=float('nan')),dict(score=-1),dict(planner_version=1),dict(cost_model='unknown')):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                plan_metadata.validate_execution(dict(plan,selection=dict(selection,**bad)),case)
        missing=dict(plan);del missing['selection']
        with self.assertRaises(ValueError):plan_metadata.validate_execution(missing,case)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fake_run(command,**kwargs):
                directory=Path(command[1]);(directory/'y-0.bin').write_bytes((directory/'golden_y.bin').read_bytes())
                (directory/'execution_plan.json').write_text(json.dumps(plan))
                return mock.Mock(returncode=0)
            with mock.patch.dict(os.environ,{'CANN_EXECUTION_FAMILY':'pipeline'}),mock.patch.object(baseline.subprocess,'run',side_effect=fake_run):
                result=baseline.run_case(Path('/fake'),case,root/'pipeline',0,1,10,True)
                self.assertEqual(result['status'],'PASS',result)
                self.assertEqual(result['similarity_validation'],'unavailable_in_pipeline_family')

    def test_partial_reduction_schema_storage_and_forced_policy(self):
        case=dict(self.case,m=33,n=32,k=2)
        stream=dict(planner_version=3,buffers=2,tile_m=16,tile_n=16,splits=2,row_pitch=64,
                    c_slot_elements=256,maxima_offset=12288,partial_offset=12416,scratch_bytes=12928,
                    max_tiles_per_core=1,ub_used=48096,ub_budget=128*1024,inner_tile=dict(m=16,n=16,k=16))
        reduction=dict(version=1,mode='partials',segment_rows=32,segments=2,record_floats=16,
                       bytes=128,fold_ub_bytes=512,ub_used=48096)
        plan=dict(self.plan,family='pipeline',variant='mix_pipeline',requested_sum='partials',
                  tasks=6,blocks=6,problem=plan_metadata.problem(case),stream=stream,reduction=reduction,
                  selection=dict(planner_version=3,cost_model='joint-work-v2',score=100.25))
        del plan['ub_used']
        self.assertEqual(plan_metadata.validate_execution(plan,case),plan)
        for key in reduction:
            if type(reduction[key]) is int:
                with self.subTest(key=key),self.assertRaises(ValueError):
                    plan_metadata.validate_execution(dict(plan,reduction=dict(reduction,**{key:reduction[key]+1})),case)
        for fields in (dict(requested_sum='rows'),dict(requested_sum='unknown'),dict(reduction=None),
                       dict(stream=dict(stream,partial_offset=12544)),dict(stream=dict(stream,planner_version=2))):
            with self.subTest(fields=fields),self.assertRaises(ValueError):plan_metadata.validate_execution(dict(plan,**fields),case)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def fake_run(command,**kwargs):
                directory=Path(command[1]);(directory/'y-0.bin').write_bytes((directory/'golden_y.bin').read_bytes())
                (directory/'execution_plan.json').write_text(json.dumps(plan));return mock.Mock(returncode=0)
            with mock.patch.dict(os.environ,{'CANN_SUM_MODE':'partials'}),mock.patch.object(baseline.subprocess,'run',side_effect=fake_run):
                result=baseline.run_case(Path('/fake'),case,root/'valid',0,1,10,True)
                self.assertEqual(result['status'],'PASS',result)
                plan['requested_sum']='auto'
                result=baseline.run_case(Path('/fake'),case,root/'mismatch',0,1,10,True)
                self.assertEqual(result['status'],'FAIL',result)
