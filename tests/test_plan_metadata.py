import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_baseline as baseline

class PlanMetadataTests(unittest.TestCase):
    def test_versioned_plan_checks_input_identity_resources_and_request(self):
        case = dict(b=2, m=65, n=129, k=24, dtype='bf16', ta=True, tb=False)
        plan = dict(schema_version=2, planner_version=1, plan_id='gm-v1-80x144', policy='80x144',
                    tile_m=80, tile_n=144, tasks=2, blocks=2, available_cores=24,
                    ub_bytes=192*1024, ub_budget=128*1024, inner_tile=dict(m=16,n=16,k=16), heuristic_score=42.0, problem=case)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory / 'matmul_plan.json'
            path.write_text(json.dumps(plan))
            self.assertEqual(baseline.read_matmul_plan(directory, case), plan)
            mutations = [dict(policy='32x64'), dict(tile_m=81), dict(tasks=3), dict(ub_budget=192*1024),
                         dict(schema_version=3), dict(plan_id='wrong'), dict(heuristic_score=float('nan')),
                         dict(problem={**case, 'ta':False}), dict(problem={**case, 'k':32}),
                         dict(available_cores=True)]
            for changes in mutations:
                with self.subTest(changes=changes):
                    path.write_text(json.dumps({**copy.deepcopy(plan), **changes}))
                    with self.assertRaises(ValueError): baseline.read_matmul_plan(directory, case)

    def test_discovery_rejects_duplicates_hardware_drift_and_missing_auto(self):
        from plan_metadata import validate_discovery
        case = dict(b=1, m=65, n=129, k=24, dtype='fp16', ta=False, tb=False)
        plan = dict(schema_version=2, planner_version=1, plan_id='gm-v1-80x144', policy='80x144',
                    tile_m=80, tile_n=144, tasks=1, blocks=1, available_cores=24,
                    ub_bytes=192*1024, ub_budget=128*1024, heuristic_score=42.0,
                    inner_tile=dict(m=16,n=16,k=16), problem=case)
        data = dict(schema_version=2, auto={**copy.deepcopy(plan),'policy':'auto'},
                    candidates=[dict(policy='80x144',accepted=True,plan=plan)])
        self.assertEqual(list(validate_discovery(data,case)), ['80x144'])
        for change in ('duplicate','hardware','missing_auto','inner','bad_policy'):
            bad = copy.deepcopy(data)
            entry = bad['candidates'][0]
            if change == 'duplicate': bad['candidates'].append(entry)
            elif change == 'hardware': entry['plan']['available_cores'] = 16
            elif change == 'missing_auto': entry['accepted'] = False
            elif change == 'inner': entry['plan']['inner_tile']['k'] = 32
            elif change == 'bad_policy': entry['policy'] = '../80x144'
            with self.subTest(change=change), self.assertRaises(ValueError): validate_discovery(bad,case)
