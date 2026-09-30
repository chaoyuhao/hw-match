from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import case_rules

class CaseRuleTests(unittest.TestCase):
    def test_reproducible_bounded_unique_and_all_layout_families(self):
        cases = case_rules.generate_cases(count=64, seed=19, cores=24)
        self.assertEqual(cases, case_rules.generate_cases(count=64, seed=19, cores=24))
        self.assertNotEqual(cases, case_rules.generate_cases(count=64, seed=20, cores=24))
        self.assertEqual(len({c['case_id'] for c in cases}), 64)
        for family in {c['generation']['family'] for c in cases}:
            self.assertEqual({(c['dtype'], c['ta'], c['tb']) for c in cases if c['generation']['family']==family},
                             {(d,a,b) for d in ('fp16','bf16') for a in (False,True) for b in (False,True)})
        self.assertTrue(any(c['m'] % 16 or c['n'] % 16 for c in cases))
        self.assertTrue(any(c['k'] >= 512 for c in cases))
        for c in cases:
            self.assertEqual(c['k'] % 8, 0)
            self.assertLessEqual(c['b']*c['m']*c['n']*c['k'], 32*1024*1024)
            self.assertLessEqual(case_rules.memory_bound(c), 128*1024*1024)
            self.assertEqual(case_rules.identity(c), case_rules.identity({**c, 'name':'new label'}))
            self.assertNotEqual(case_rules.identity(c), case_rules.identity({**c, 'seed':c['seed']+1}))
    def test_invalid_requests_fail_without_unbounded_search(self):
        for kwargs in (dict(count=0), dict(count=4097), dict(cores=0), dict(seed=-1)):
            with self.assertRaises(ValueError): case_rules.generate_cases(**kwargs)
