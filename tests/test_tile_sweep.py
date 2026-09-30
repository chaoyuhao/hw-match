"""CPU tests of experiment orchestration and ranking, not NPU simulation."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


class TileSweepTests(unittest.TestCase):
    def command(self, output, *extra):
        return [sys.executable, str(SCRIPTS / "perf_local.py"), "--output-dir", str(output),
                "--tile-sweep", "--case", "layout_fp16_00", "--iterations", "3",
                "--warmup", "1", "--sweep-rounds", "2", *extra]

    def fixture(self, root):
        runner = root / "fixture runner $(literal)"
        runner.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
discover = sys.argv[1] == '--list-plans'
p = pathlib.Path(sys.argv[2] if discover else sys.argv[1])
case = json.loads((p / 'case.json').read_text())
tiles = {'32x64': (32,64), '32x128': (32,128), '64x128': (64,128),
         '80x144': (80,144), 'auto': (32,64)}
def make_plan(policy):
    tm, tn = tiles[policy]
    tasks = case['b'] * ((case['m']+tm-1)//tm) * ((case['n']+tn-1)//tn)
    return dict(schema_version=2, planner_version=1, plan_id=f'gm-v1-{tm}x{tn}',
                policy=policy, tile_m=tm, tile_n=tn, tasks=tasks, blocks=min(24,tasks), available_cores=24,
                ub_bytes=192*1024, ub_budget=128*1024, inner_tile=dict(m=16,n=16,k=16), heuristic_score=float(tm*tn),
                problem={k:case[k] for k in ('b','m','n','k','dtype','ta','tb')})
if discover:
    if os.environ.get('BAD_DISCOVERY'): sys.exit(3)
    data = dict(schema_version=2, auto=make_plan('auto'), candidates=[
        dict(policy=key, accepted=True, plan=make_plan(key)) for key in tiles if key != 'auto'])
    pathlib.Path(sys.argv[4]).write_text(json.dumps(data))
    sys.exit(0)
policy = os.environ['CANN_MATMUL_TILE']
plan = make_plan(policy)
if os.environ.get('BAD_PLAN') and policy == '32x64':
    plan['policy'] = 'auto'
if not os.environ.get('MISSING_PLAN'):
    (p / 'matmul_plan.json').write_text(json.dumps(plan))
for i in range(int(sys.argv[3])):
    data = (p / 'golden_y.bin').read_bytes()
    if os.environ.get('BAD_OUTPUT') and policy == '80x144':
        data = bytes(len(data))
    (p / f'y-{i}.bin').write_bytes(data)
if len(sys.argv) == 7:
    w, n = map(int, sys.argv[5:])
    t = {'32x64':40, '32x128':20, '64x128':30, '80x144':50, 'auto':40}[policy]
    (p / 'host_timings.csv').write_text('iteration,warmup,host_call_us\\n' +
        ''.join(f'{i},{int(i<w)},{1000 if i<w else t}\\n' for i in range(w+n)))
''')
        runner.chmod(0o755)
        profiler = root / "msprof"
        profiler.write_text(f"#!{sys.executable}\n" + '''
import os, pathlib, subprocess, sys
args = sys.argv[1:]
out = pathlib.Path(next(a.split('=',1)[1] for a in args if a.startswith('--output=')))
out.mkdir(parents=True)
app = next(i for i,a in enumerate(args) if not a.startswith('--'))
subprocess.run(' '.join(args[app:]), shell=True, check=True)
t = {'32x64':10, '32x128':20, '64x128':5, '80x144':30, 'auto':10}[os.environ['CANN_MATMUL_TILE']]
name = 'unrelated_kernel' if os.environ.get('BAD_PROFILE') else 'MatmulMaxSum_fixture'
count = 1 if os.environ.get('BAD_COUNT') else 2
(out / 'op_summary.csv').write_text('Op Name,Task Type,Task Duration(us)\\n' +
    ''.join(f'{name},MIX_AIC,{t}\\n' for _ in range(count)) +
    ('Other,AI_CORE,100\\n' if os.environ.get('EXTRA_GROUP') else ''))
''')
        profiler.chmod(0o755)
        env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                   CANN_MATMUL_TILE='invalid_inherited_value')
        for name in ('BAD_PLAN', 'BAD_OUTPUT', 'MISSING_PLAN', 'BAD_PROFILE', 'BAD_COUNT', 'EXTRA_GROUP'):
            env.pop(name, None)
        return runner, env

    def test_sweep_measures_each_policy_rotates_order_and_keeps_host_and_device_distinct(self):
        # Wrong environment propagation, warmup inclusion or ranking host as device must fail.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, env = self.fixture(root)
            out = root / 'matrix with spaces'
            command = self.command(out, '--binary', str(runner), '--profile', 'timeline', '--profile-repeat', '2')
            done = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['online_evaluation'], 'NOT_RUN')
            self.assertEqual(len(report['results']), 10)
            first = [x['requested_policy'] for x in report['results'] if x['round'] == 1]
            second = [x['requested_policy'] for x in report['results'] if x['round'] == 2]
            self.assertEqual(set(first), {'32x64','32x128','64x128','80x144','auto'})
            self.assertNotEqual(first, second)
            for result in report['results']:
                self.assertEqual(result['requested_policy'], result['matmul_plan']['policy'])
                self.assertEqual(result['matmul_plan'], result['profile']['matmul_plan'])
            first_directory = out / report['results'][0]['directory']
            for filename in ('x1.bin', 'x2.bin'):
                original = (first_directory / filename).read_bytes()
                for result in report['results']:
                    directory = out / result['directory']
                    self.assertEqual((directory / filename).read_bytes(), original)
                    self.assertEqual((directory / 'profile_run' / filename).read_bytes(), original)
            summary = report['sweep_summary'][0]
            self.assertEqual(summary['status'], 'COMPLETE')
            self.assertEqual(summary['host']['best_policy'], '32x128')
            self.assertEqual(summary['device']['best_policy'], '64x128')
            self.assertEqual(summary['device']['speedup_vs_auto'], 2)
            self.assertEqual(summary['candidates']['32x64']['features']['tasks'], 18)
            self.assertEqual(summary['candidates']['32x64']['features']['waves'], 1)
            self.assertEqual(summary['candidates']['32x64']['features']['active_core_fraction'], 0.75)
            with (out / 'sweep.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertEqual(float(next(r for r in rows if r['policy']=='64x128')['device_median_us']), 5)
            self.assertTrue((out / 'report.md').is_file())

    def test_missing_mismatched_plan_bad_precision_or_unrelated_profile_prevents_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, env = self.fixture(root)
            for fault in ('BAD_PLAN', 'MISSING_PLAN', 'BAD_OUTPUT', 'BAD_PROFILE', 'BAD_COUNT', 'EXTRA_GROUP'):
                with self.subTest(fault=fault):
                    out = root / fault
                    command = self.command(out, '--binary', str(runner), '--profile', 'timeline', '--profile-repeat', '2')
                    done = subprocess.run(command, env=dict(env, **{fault:'1'}), capture_output=True, text=True, timeout=90)
                    self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
                    report = json.loads((out / 'report.json').read_text())
                    self.assertTrue(any(r['status']=='FAIL' for r in report['results']))
                    summary = report['sweep_summary'][0]
                    self.assertEqual(summary['status'], 'INCOMPLETE')
                    self.assertIsNone(summary['host'])
                    self.assertIsNone(summary['device'])

    def test_generation_is_never_ranked_and_requires_no_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'generated'
            done = subprocess.run(self.command(out, '--generate-only'), capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(len(report['results']), 1)
            self.assertTrue(report['sweep']['discovery_pending'])
            self.assertIsNone(report['sweep']['expected_runs'])
            self.assertTrue(all(r['status']=='GENERATED' for r in report['results']))
            self.assertIsNone(report['sweep_summary'][0]['host'])
            self.assertEqual(len(list(out.rglob('golden_y.bin'))), 1)
            # A second run must not combine old and new measurements.
            again = subprocess.run(self.command(out, '--generate-only'), capture_output=True, text=True, timeout=60)
            self.assertNotEqual(again.returncode, 0)

    def test_discovery_failure_and_candidate_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, env = self.fixture(root)
            for fail in (False, True):
                out = root / str(fail)
                done = subprocess.run(self.command(out, '--binary', str(runner), '--sweep-candidates','1'),
                                      env=dict(env, **({'BAD_DISCOVERY':'1'} if fail else {})),
                                      capture_output=True, text=True, timeout=60)
                self.assertEqual(done.returncode, int(fail), done.stdout + done.stderr)
                report = json.loads((out / 'report.json').read_text())
                self.assertEqual(len(report['results']), 1 if fail else 4)
                if fail:
                    self.assertEqual(report['sweep_summary'][0]['status'],'INCOMPLETE')
                    self.assertIsNone(report['sweep_summary'][0]['host'])
                else:
                    self.assertEqual(report['sweep']['policies_by_case']['layout_fp16_00'], ['32x64','auto'])

    def test_default_host_sweep_never_invents_device_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner, env = self.fixture(root)
            out = root / 'host-only'
            done = subprocess.run(self.command(out, '--binary', str(runner)), env=env,
                                  capture_output=True, text=True, timeout=60)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['sweep']['ranking_metric'], 'host')
            self.assertEqual(report['sweep_summary'][0]['host']['best_policy'], '32x128')
            self.assertIsNone(report['sweep_summary'][0]['device'])
            self.assertTrue(all('profile' not in r for r in report['results']))
            with (out / 'sweep.csv').open() as f:
                self.assertTrue(all(not row['device_median_us'] for row in csv.DictReader(f)))

    def test_noisy_rounds_remain_near_best_and_incomplete_rounds_are_not_ranked(self):
        path = SCRIPTS / 'tile_sweep.py'
        self.assertTrue(path.exists(), 'tile sweep aggregation is missing')
        spec = importlib.util.spec_from_file_location('tile_sweep', path)
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        case = dict(name='sample', b=1, m=32, n=64, k=8, dtype='fp16', ta=False, tb=False,
                    pattern='random', seed=1)
        records = []
        for policy, times in [('32x64', [10, 20]), ('auto', [16, 16])]:
            for i, t in enumerate(times, 1):
                records.append(dict(case=case, round=i, requested_policy=policy, status='PASS',
                    host_call=dict(samples=3, median_us=t, p95_us=t),
                    matmul_plan=dict(policy=policy, tile_m=32, tile_n=64, tasks=1, blocks=1, available_cores=24)))
        summary = api.summarize([case], records, ['32x64', 'auto'], 2)[0]
        self.assertEqual(summary['host']['best_policy'], '32x64')
        self.assertEqual(set(summary['host']['near_best']), {'32x64', 'auto'})
        self.assertEqual(summary['host']['speedup_vs_auto'], 16/15)
        self.assertIsNone(api.summarize([case], records[:-1], ['32x64', 'auto'], 2)[0]['host'])
        # Changing the effective plan between rounds invalidates the comparison.
        records[-1]['matmul_plan']['available_cores'] = 16
        self.assertIsNone(api.summarize([case], records, ['32x64', 'auto'], 2)[0]['host'])


if __name__ == '__main__':
    unittest.main()
