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


class PerformanceTests(unittest.TestCase):
    def setUp(self):
        path = SCRIPTS / "perf_local.py"
        self.assertTrue(path.is_file(), "performance driver is missing")
        spec = importlib.util.spec_from_file_location("perf_local", path)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)

    def test_host_samples_exclude_warmup_and_validate_complete_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "host_timings.csv"
            path.write_text("iteration,warmup,host_call_us\n0,1,1000\n1,0,10\n2,0,20\n3,0,30\n")
            stats = self.api.read_host_timings(path, 1, 3)
            self.assertEqual(stats["median_us"], 20)
            self.assertEqual(stats["samples"], 3)
            for data in ("0,1,1\n1,0,nan\n2,0,20\n3,0,30\n",
                         "0,1,1\n1,0,10\n2,0,20\n",
                         "0,1,1\n1,0,10\n1,0,20\n3,0,30\n"):
                path.write_text("iteration,warmup,host_call_us\n" + data)
                with self.assertRaises(ValueError):
                    self.api.read_host_timings(path, 1, 3)

    def test_profile_records_are_grouped_and_keep_metrics_and_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "op_summary_0.csv").write_text(
                "Op Name,Task Type,Task Duration(us),aic_mac_ratio\n"
                "MatmulMaxSum,MIX_AIC,10,0.2\nMatmulMaxSum,MIX_AIC,20,0.4\n"
                "Other,AI_CORE,100,0.9\n")
            (path / "op_statistic_0.csv").write_text("do not count aggregate rows again")
            groups = self.api.read_profile(path)["groups"]
            match = next(g for g in groups if g["op_name"] == "MatmulMaxSum")
            self.assertEqual(match["task_duration"]["median_us"], 15)
            self.assertEqual(match["task_duration"]["samples"], 2)
            self.assertAlmostEqual(match["metrics_mean"]["aic_mac_ratio"], 0.3)
            self.assertEqual(len(groups), 2)

    def test_profile_missing_or_invalid_data_is_never_a_zero_time_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(ValueError):
                self.api.read_profile(path)
            (path / "op_summary_0.csv").write_text("Op Name,Task Type,Task Duration(us)\nfoo,AI_CORE,nan\n")
            with self.assertRaises(ValueError):
                self.api.read_profile(path)

    def test_stress_suite_covers_axes_layouts_and_remains_bounded(self):
        cases = self.api.performance_cases("stress")
        self.assertEqual(len(cases), len({c["name"] for c in cases}))
        self.assertEqual({(c["dtype"], c["ta"], c["tb"]) for c in cases},
                         {(d, a, b) for d in ("fp16", "bf16") for a in (False, True) for b in (False, True)})
        for axis in ("m", "n", "k"):
            self.assertTrue(any(c[axis] == 8192 for c in cases))
        self.assertTrue(any(c["b"] == 257 for c in cases))
        self.assertTrue(all(c["k"] % 8 == 0 and c["b"] * c["m"] * c["n"] <= 2**20 for c in cases))

    def test_comparison_matches_full_case_identity_and_never_failed_results(self):
        case = self.api.performance_cases("quick")[0]
        old = {"results": [{"case": case, "status": "PASS", "host_call": {"median_us": 40}}]}
        new = {"results": [{"case": case, "status": "PASS", "host_call": {"median_us": 20}}]}
        self.assertEqual(self.api.compare_reports(new, old)[0]["host_speedup"], 2)
        new["results"][0]["case"] = dict(case, seed=99)
        self.assertEqual(self.api.compare_reports(new, old), [])
        new["results"][0]["case"] = case
        new["results"][0]["status"] = "FAIL"
        self.assertEqual(self.api.compare_reports(new, old), [])

    def test_profile_command_passes_paths_as_arguments_without_shell(self):
        command = self.api.profile_command("/tmp/msprof", Path("/tmp/output dir"),
                                           ["/tmp/my runner", "/tmp/case $(bad)", "0", "5", "0"], "PipeUtilization")
        self.assertIn("--output=/tmp/output dir", command)
        self.assertEqual(command[-5:], ["/tmp/my runner", "/tmp/case $(bad)", "0", "5", "0"])

    def test_cli_benchmark_and_profile_validate_results_and_fail_on_bad_profile(self):
        # A fixture tests orchestration only. It is deliberately not a kernel simulation.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "fixture runner $(not_a_command)"
            runner.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
p = pathlib.Path(sys.argv[1])
profile_override = os.environ.get('DIFFERENT_PROFILE_TILE') and len(sys.argv) == 5
plan = dict(policy='32x128' if profile_override else 'auto', tile_m=32,
            tile_n=128 if profile_override else 64, tasks=12 if profile_override else 18,
            blocks=12 if profile_override else 18, available_cores=24)
(p / 'matmul_plan.json').write_text(json.dumps(plan))
for i in range(int(sys.argv[3])):
    (p / f'y-{i}.bin').write_bytes((p / 'golden_y.bin').read_bytes())
if len(sys.argv) == 7:
    w, n = map(int, sys.argv[5:])
    (p / 'host_timings.csv').write_text('iteration,warmup,host_call_us\\n' +
        ''.join(f'{i},{int(i < w)},10\\n' for i in range(w + n)))
''')
            runner.chmod(0o755)
            profiler = root / "msprof"
            profiler.write_text(f"#!{sys.executable}\n" + '''
import os, pathlib, subprocess, sys
args = sys.argv[1:]
out = pathlib.Path(next(a.split('=', 1)[1] for a in args if a.startswith('--output=')))
out.mkdir(parents=True)
app = next(i for i, a in enumerate(args) if not a.startswith('--'))
# Emulate msprof's documented application argument limitation: argv is joined.
subprocess.run(' '.join(args[app:]), shell=True, check=True)
duration = 'nan' if os.environ.get('BAD_PROFILE') else '3'
(out / 'op_summary_0.csv').write_text('Op Name,Task Type,Task Duration(us),aic_mac_ratio\\n'
                                    f'fixture,MIX_AIC,{duration},0.25\\n')
''')
            profiler.chmod(0o755)
            for fault in (None, "BAD_PROFILE", "DIFFERENT_PROFILE_TILE"):
                bad = fault is not None
                out = root / (fault or "good")
                env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ["PATH"])
                for variable in ("BAD_PROFILE", "DIFFERENT_PROFILE_TILE"):
                    env.pop(variable, None)
                if fault:
                    env[fault] = "1"
                command = [sys.executable, str(SCRIPTS / "perf_local.py"), "--binary", str(runner),
                           "--output-dir", str(out), "--case", "layout_fp16_00", "--iterations", "3",
                           "--warmup", "1", "--profile", "timeline", "--profile-repeat", "2"]
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, int(bad), result.stdout + result.stderr)
                report = json.loads((out / "report.json").read_text())
                self.assertEqual(report["online_evaluation"], "NOT_RUN")
                case_result = report["results"][0]
                self.assertEqual(case_result["status"], "FAIL" if bad else "PASS")
                if not bad:
                    self.assertEqual(case_result["host_call"]["samples"], 3)
                    self.assertEqual(case_result["profile"]["groups"][0]["task_duration"]["median_us"], 3)
                    self.assertEqual(case_result["matmul_plan"], case_result["profile"]["matmul_plan"])
                    self.assertEqual(case_result["matmul_plan"]["tasks"], 18)
                    self.assertTrue((out / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
