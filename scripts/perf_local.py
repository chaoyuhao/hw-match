#!/usr/bin/env python3
"""Local stress tests and msprof collection; never an official score estimator."""
import case_rules
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import local_baseline as baseline
import tile_sweep

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("PipeUtilization", "ArithmeticUtilization", "Memory", "MemoryL0", "MemoryUB",
           "ResourceConflictRatio", "L2Cache")


def performance_cases(suite):
    if suite == "regression":
        return baseline.cases_for_suite("full")
    cases = []

    def add(name, b, m, n, k, dtype, ta=False, tb=False):
        cases.append(dict(name=f"{name}_{dtype}_{int(ta)}{int(tb)}", b=b, m=m, n=n, k=k,
                          dtype=dtype, ta=ta, tb=tb, pattern="random", seed=20260929))

    for dtype in ("fp16", "bf16"):
        for ta in (False, True):
            for tb in (False, True):
                add("layout", 2, 65, 129, 128, dtype, ta, tb)
        if suite == "stress":
            for axis, values in (("b", (1, 2, 8, 32, 257)), ("m", (1, 32, 128, 1024, 8192)),
                                 ("n", (1, 64, 256, 2048, 8192)), ("k", (8, 24, 256, 2048, 8192))):
                for value in values:
                    shape = dict(b=1, m=33, n=65, k=32)
                    shape[axis] = value
                    add(f"sweep_{axis}{value}", dtype=dtype, **shape)
    return cases


def duration_stats(values):
    values = sorted(values)
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("timings must be finite, positive and nonempty")
    return dict(samples=len(values), min_us=values[0], median_us=statistics.median(values),
                p95_us=values[math.ceil(0.95 * len(values)) - 1], max_us=values[-1],
                mean_us=statistics.mean(values))


def read_host_timings(path, warmup, iterations):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != warmup + iterations:
        raise ValueError("incomplete host timing samples")
    samples = []
    for index, row in enumerate(rows):
        if row.get("iteration") != str(index) or row.get("warmup") != str(int(index < warmup)):
            raise ValueError("invalid host timing sequence or warmup flags")
        value = float(row.get("host_call_us", "nan"))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid host timing value")
        if index >= warmup:
            samples.append(value)
    return duration_stats(samples)


def profile_command(msprof, output, app_argv, metrics):
    return [str(msprof), f"--output={output}", "--ascendcl=on", "--runtime-api=on",
            "--task-time=on", "--ai-core=on", f"--aic-metrics={metrics}", *map(str, app_argv)]


def read_profile(directory):
    # Do not mix op_statistic aggregates or timeline events into per-task statistics.
    paths = sorted(directory.rglob("op_summary*.csv"))
    if not paths:
        raise ValueError("no op_summary*.csv exported; inspect msprof.log and raw PROF_* files")
    groups = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            if not {"Op Name", "Task Type", "Task Duration(us)"}.issubset(reader.fieldnames or []):
                raise ValueError(f"unsupported op_summary columns: {path}")
            for row in reader:
                value = float(row["Task Duration(us)"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"invalid Task Duration(us): {path}")
                key = (row.get("Device_id", "unknown"), row["Op Name"], row["Task Type"])
                group = groups.setdefault(key, dict(durations=[], metrics={}))
                group["durations"].append(value)
                for name, raw in row.items():
                    if name and name.lower().startswith(("aic_", "aiv_")):
                        try:
                            metric = float(raw)
                        except (ValueError, TypeError):
                            continue  # N/A means unavailable, never zero.
                        if math.isfinite(metric):
                            group["metrics"].setdefault(name, []).append(metric)
    if not groups:
        raise ValueError("op_summary has no task records")
    return dict(files=[str(p.relative_to(directory)) for p in paths], groups=[
        dict(device=key[0], op_name=key[1], task_type=key[2],
             task_duration=duration_stats(group["durations"]),
             metrics_mean={k: statistics.mean(v) for k, v in group["metrics"].items()},
             metrics_samples={k: len(v) for k, v in group["metrics"].items()})
        for key, group in sorted(groups.items())],
        scope="All profiled task records, including first call; not steady-state kernel launch counts")


def case_identity(case):
    return case_rules.identity(case)


def compare_reports(new, old):
    previous = {case_identity(r["case"]): r for r in old["results"]
                if r["status"] == "PASS" and "host_call" in r}
    result = []
    for current in new["results"]:
        prior = previous.get(case_identity(current["case"]))
        if current["status"] != "PASS" or not prior or "host_call" not in current:
            continue
        old_us, new_us = prior["host_call"]["median_us"], current["host_call"]["median_us"]
        duration_stats([old_us, new_us])
        result.append(dict(case=current["case"]["name"], old_host_median_us=old_us,
                           new_host_median_us=new_us, host_speedup=old_us / new_us))
    return result


def capture(command, timeout=15):
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return dict(command=command, exit_code=process.returncode,
                    output=(process.stdout + process.stderr)[-24000:])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, error=str(error))


def inspect_tools():
    results = {}
    for name, arguments in (("msprof", ["--help"]), ("mssanitizer", ["--help"]),
                            ("bisheng", ["--version"]), ("cmake", ["--version"]),
                            ("npu-smi", ["info"])):
        path = shutil.which(name)
        results[name] = dict(path=path, available=path is not None)
        if path:
            results[name]["probe"] = capture([path, *arguments])
    if results["msprof"]["available"]:
        results["msprof_op"] = capture([results["msprof"]["path"], "op", "--help"])
    return results


def sha256(path):
    with path.open("rb") as file:
        digest = hashlib.sha256()
        for data in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(data)
        return digest.hexdigest()


def provenance(binary):
    environment = {key: os.environ.get(key) for key in (
        "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES", "NPU_ARCH", "CANN_MATMUL_TILE", "CANN_EXECUTION_FAMILY", "CANN_SUM_MODE", "CANN_MATMUL_MAX")}
    result = dict(git=capture(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                  git_status=capture(["git", "-C", str(ROOT), "status", "--short"]),
                  kernel_sha256=sha256(ROOT / "kernel.asc"), environment=environment,
                  platform=platform.platform(), python=sys.version, numpy=baseline.np.__version__,
                  compiler=capture(["bisheng", "--version"]), npu=capture(["npu-smi", "info"]),
                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (
                      ROOT / "local/runner.asc", ROOT / "local/CMakeLists.txt",
                      ROOT / "scripts/local_baseline.py", ROOT / "scripts/perf_local.py",
                      ROOT / "scripts/tile_sweep.py", ROOT / "matmul_plan.h",
                      ROOT / "small_plan.h", ROOT / "small_vector.h",
                      ROOT / "stream_plan.h", ROOT / "stream_matmul.asc", ROOT / "joint_plan.h", ROOT / "reduction_plan.h", ROOT / "partial_sum.asc", ROOT / "iterate_plan.h", ROOT / "iterate_matmul.asc",
                      ROOT / "scripts/case_rules.py", ROOT / "scripts/plan_metadata.py")})
    if binary:
        result["binary"] = str(binary)
        result["binary_sha256"] = sha256(binary)
        result["binary_provenance"] = "Use run_perf.sh for a fresh build of these sources; direct Python invocation cannot verify this association"
    toolkit = environment["ASCEND_HOME_PATH"]
    if toolkit:
        result["toolkit_metadata"] = {}
        for relative in ("version.cfg", "x86_64-linux/ascend_toolkit_install.info",
                         "share/info/asc-devkit/version.info", "compiler/version.info"):
            path = Path(toolkit) / relative
            if path.is_file():
                result["toolkit_metadata"][relative] = path.read_text(errors="replace")[:12000]
    return result


def execute_profile(command, logfile, timeout):
    # msprof launches child processes; on timeout stop the entire local test process group.
    with logfile.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(1)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
    if code:
        raise ValueError(f"msprof exit={code}; see {logfile}")


def collect_profile(args, case, directory):
    _, golden = baseline.make_case(case, directory)
    app = [str(args.binary), str(directory), str(args.device), str(args.profile_repeat), "0"]
    output = directory / "profiling"
    # msprof may join app argv internally. Its documented workaround for quoted
    # arguments is a launcher script. Only simple paths reach that internal join;
    # shlex quotes every actual application argument in the saved launcher.
    launcher = directory / "profile_app.sh"
    launcher.write_text("#!/bin/bash\nset -euo pipefail\nexec " + shlex.join(app) + "\n")
    with tempfile.TemporaryDirectory(prefix="cann-msprof-", dir="/tmp") as temporary:
        entry = Path(temporary) / "launch.sh"
        shutil.copyfile(launcher, entry)
        command = profile_command(shutil.which("msprof"), output, ["/bin/bash", str(entry)], args.metrics)
        (directory / "command.json").write_text(json.dumps(dict(
            msprof=command, application=app, saved_launcher=str(launcher)), indent=2) + "\n")
        execute_profile(command, directory / "msprof.log", args.timeout)
    first = None
    precision = []
    for index in range(args.profile_repeat):
        actual = baseline.read_f32(directory / f"y-{index}.bin", case["b"])
        precision.append(baseline.compare(actual, golden))
        if first is not None and not baseline.np.array_equal(actual.view("<u4"), first.view("<u4")):
            raise ValueError("profiled outputs differ between calls")
        first = actual
    result = read_profile(output)
    result.update(command=command, application_command=app, saved_launcher=str(launcher),
                  precision=precision, metrics_set=args.metrics)
    plan = baseline.read_matmul_plan(directory, case)
    if plan is not None:
        result["matmul_plan"] = plan
    execution = baseline.read_execution_plan(directory, case, os.environ.get("CANN_EXECUTION_FAMILY", "auto"), os.environ.get("CANN_SUM_MODE", "auto"))
    if execution is not None:
        result["execution_plan"] = execution
    return result


def write_reports(report, output):
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Local performance report", "", "ONLINE_EVALUATION=NOT_RUN", "",
             "Host call = run_kernel + stream completion; includes tiling/alloc/free. "
             "Input/output copies, validation and file I/O are outside the interval. "
             "Separate profiler task durations include instrumentation and first calls; do not subtract the two.", "",
             "| Case | Status | Tile; tasks/blocks | Host median μs | Host p95 μs | Samples |",
             "|---|---|---|---:|---:|---:|"]
    for result in report["results"]:
        stats = result.get("host_call", {})
        plan = result.get("matmul_plan")
        tile = f"{plan['tile_m']}×{plan['tile_n']}; {plan['tasks']}/{plan['blocks']}" if plan else "—"
        execution = result.get("execution_plan", {})
        if execution.get("family") == "small":
            tile = f"small/{execution['variant']}; {execution['tasks']}/{execution['blocks']}"
        if execution.get("family") in ("stream", "pipeline", "iterate"):
            s = execution['stream']
            tile = f"{execution['family']} {s['tile_m']}×{s['tile_n']}, N/{s['splits']}; {execution['tasks']}/{execution['blocks']}"
        if execution.get("reduction"):
            reduction = execution["reduction"]
            tile += f"; sum={reduction['mode']} (M/{reduction['segment_rows']})"
        if execution.get("reduction_expansion", {}).get("applied"):
            tile += "; expanded rows→partials"
        if execution.get("upstream_control"):
            tile += "; S2 upstream=" + execution["upstream_control"]["status"]
        if execution.get("matmul_max_fusion"):
            tile += "; Matmul→Max=" + execution["matmul_max_fusion"]["status"]
        lines.append(f"| {result['case']['name']} | {result['status']} | {tile} | {stats.get('median_us', '—')} | "
                     f"{stats.get('p95_us', '—')} | {stats.get('samples', '—')} |")
    for result in report["results"]:
        if "error" in result:
            lines += ["", f"{result['case']['name']}: {result['error']}"]
        for group in result.get("profile", {}).get("groups", []):
            lines += ["", f"Profile {result['case']['name']} / {group['op_name']} ({group['task_type']}): "
                      f"Task Duration median {group['task_duration']['median_us']:.3f} μs; "
                      f"{group['task_duration']['samples']} records."]
    if "comparison" in report:
        lines += ["", "## Host comparison", "", "Only identical shape/dtype/layout/pattern/seed and PASS results. "
                  "Check both manifests for matching machine, SDK, timing options and machine load. These are not online scores.", "",
                  "| Case | Old/new host median ratio |", "|---|---:|"]
        lines += [f"| {r['case']} | {r['host_speedup']:.3f} |" for r in report["comparison"]]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def measure_case(args, case, directory):
    result = baseline.run_case(args.binary, case, directory, args.device, 2, args.timeout,
                               False, benchmark=(args.warmup, args.iterations))
    if result["status"] == "PASS":
        try:
            result["host_call"] = read_host_timings(directory / "host_timings.csv", args.warmup, args.iterations)
            if args.profile != "none":
                result["profile"] = collect_profile(args, case, directory / "profile_run")
                if result.get("execution_plan") != result["profile"].get("execution_plan"):
                    raise ValueError("profile and host measurement used different execution plans")
                if result.get("matmul_plan") != result["profile"].get("matmul_plan"):
                    raise ValueError("profile and host measurement used different Matmul plans")
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result.update(status="FAIL", error=str(error))
    (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite", choices=("quick", "stress", "regression", "generated"), default="quick")
    parser.add_argument("--case")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=120, help="per runner/profiler process, seconds")
    parser.add_argument("--profile", choices=("none", "timeline"), default="none")
    parser.add_argument("--metrics", choices=METRICS, default="PipeUtilization")
    parser.add_argument("--profile-repeat", type=int, default=5)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--tile-sweep", action="store_true", help="compare SDK-discovered tiles plus auto, sequentially")
    parser.add_argument("--sweep-rounds", type=int, default=2, help="tile sweep repetitions with rotated order (default: 2)")
    parser.add_argument("--sweep-candidates", type=int, default=6, help="top accepted candidates (plus reference and auto), 1..64")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--inspect-tools", action="store_true")
    modes.add_argument("--list-cases", action="store_true")
    modes.add_argument("--generate-only", action="store_true")
    case_rules.add_arguments(parser)
    args = parser.parse_args()
    if not 1 <= args.sweep_candidates <= 64:
        parser.error("--sweep-candidates must be between 1 and 64")
    if not 1 <= args.sweep_rounds <= 20:
        parser.error("--sweep-rounds must be between 1 and 20")
    if args.tile_sweep and (args.compare or args.inspect_tools):
        parser.error("--tile-sweep cannot be combined with --compare or --inspect-tools")
    if args.device < 0 or not 0 <= args.warmup <= 1000 or not 1 <= args.iterations <= 10000 or not 1 <= args.profile_repeat <= 1000 or args.timeout < 1:
        parser.error("invalid device/warmup/iterations/profile-repeat/timeout")
    cases = case_rules.from_arguments(args, parser) if args.suite == "generated" else performance_cases(args.suite)
    if args.case:
        cases = [c for c in cases if c["name"] == args.case]
        if not cases:
            parser.error("case name not found in selected suite; use --list-cases")
    if args.list_cases:
        for case in cases:
            print(f"{case['name']}: B/M/N/K={case['b']}/{case['m']}/{case['n']}/{case['k']}")
        return 0
    if not args.output_dir:
        parser.error("--output-dir is required")
    if args.inspect_tools:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = inspect_tools()
        (args.output_dir / "tools.json").write_text(json.dumps(report, indent=2) + "\n")
        for name, result in report.items():
            if "available" in result:
                detail = result["path"] if result["available"] else "not found on PATH"
            else:
                detail = result.get("error", f"help exit={result.get('exit_code')}; see tools.json")
            print(f"{name}: {detail}")
        print(f"Tool inventory (not an NPU test): {args.output_dir / 'tools.json'}")
        return 0
    if not args.generate_only:
        if not args.binary or not args.binary.is_file():
            parser.error("provide an existing --binary, or use run_perf.sh")
        args.binary = args.binary.resolve()
        if args.profile != "none" and not shutil.which("msprof"):
            parser.error("msprof is not on PATH; source the CANN environment and inspect installed tools")
    if args.generate_only and args.profile != "none":
        parser.error("--generate-only cannot collect a profile")
    previous = json.loads(args.compare.read_text()) if args.compare else None
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(online_evaluation="NOT_RUN", mode="GENERATED" if args.generate_only else "LOCAL_PERFORMANCE",
                  suite=args.suite, options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  provenance=provenance(args.binary), results=[])
    shutil.copyfile(ROOT / "kernel.asc", args.output_dir / "kernel_snapshot.asc")
    for header in ("matmul_plan.h", "small_plan.h", "small_vector.h", "stream_plan.h", "stream_matmul.asc", "joint_plan.h", "reduction_plan.h", "partial_sum.asc", "iterate_plan.h", "iterate_matmul.asc"):
        shutil.copyfile(ROOT / header, args.output_dir / header)
    print("LOCAL_PERFORMANCE; ONLINE_EVALUATION=NOT_RUN", flush=True)
    if args.tile_sweep:
        return tile_sweep.run(args, cases, report, measure_case)
    for case in cases:
        directory = args.output_dir / case["name"]
        print(f"[RUN] {case['name']}", flush=True)
        if args.generate_only:
            baseline.make_case(case, directory)
            result = dict(case=case, status="GENERATED")
        else:
            result = measure_case(args, case, directory)
        # This describes the unchanged reference kernel, not a hardware measurement.
        result["reference_work"] = dict(matmul_flops=2 * case["b"] * case["m"] * case["n"] * case["k"],
                                       full_similarity_bytes=4 * case["b"] * case["m"] * ((case["n"] + 15) // 16 * 16))
        report["results"].append(result)
        if previous is not None:
            report["comparison"] = compare_reports(report, previous)
        write_reports(report, args.output_dir)
        duration = result.get("host_call", {}).get("median_us")
        suffix = f"host median={duration:.3f} μs" if duration is not None else ""
        print(f"[{result['status']}] {case['name']} {suffix} {result.get('error', '')}", flush=True)
    failed = sum(r["status"] == "FAIL" for r in report["results"])
    print(f"cases={len(cases)} failures={failed}; report={args.output_dir / 'report.md'}")
    return int(failed != 0)


if __name__ == "__main__":
    sys.exit(main())
