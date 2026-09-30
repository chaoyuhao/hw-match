"""Sequential local tile experiments and conservative summaries; no online tuning."""
import csv
import json
import math
import os
import statistics
import subprocess
import plan_metadata

import local_baseline as baseline

def jobs(cases, rounds, policies):
    for repeat in range(rounds):
        for index, case in enumerate(cases):
            choices = policies[case['name']]
            if not choices:
                continue
            offset = (index + repeat) % len(choices)
            for policy in choices[offset:] + choices[:offset]:
                yield case, repeat + 1, policy


def discover(args, case, directory):
    directory.mkdir(parents=True)
    fields = [case[k] for k in ('b','m','n','k')] + [1 if case['dtype']=='fp16' else 2, int(case['ta']), int(case['tb'])]
    (directory / 'case.txt').write_text(' '.join(map(str, fields)) + '\n')
    (directory / 'case.json').write_text(json.dumps(case) + '\n')
    path = directory / 'candidates.json'
    with (directory / 'discovery.log').open('w') as log:
        done = subprocess.run([str(args.binary), '--list-plans', str(directory), str(args.device), str(path)],
                              stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
    if done.returncode:
        raise ValueError(f'candidate discovery exit={done.returncode}; see discovery.log')
    data = json.loads(path.read_text())
    accepted = plan_metadata.validate_discovery(data, case)
    # SDK output is already ordered by C++ heuristic. Limit experiments, not planning.
    policies = list(accepted)[:args.sweep_candidates]
    if '32x64' in accepted and '32x64' not in policies:
        policies.append('32x64')
    policies.append('auto')
    return policies, {**accepted, 'auto':data['auto']}, data


def device_stats(result, expected_samples=None):
    groups = result.get("profile", {}).get("groups", [])
    if len(groups) != 1:
        raise ValueError("tile sweep requires exactly one profiled Matmul task group")
    group = groups[0]
    if group["task_type"] != "MIX_AIC" or "MatmulMaxSum" not in group["op_name"]:
        raise ValueError("unexpected profiled kernel/type; inspect raw op_summary before comparing")
    stats = group["task_duration"]
    if expected_samples is not None and stats["samples"] != expected_samples:
        raise ValueError("unexpected number of profiled tasks; inspect raw op_summary")
    return stats


def features(case, plan):
    tasks, blocks = plan["tasks"], plan["blocks"]
    waves = (tasks + blocks - 1) // blocks
    return dict(tile_m=plan["tile_m"], tile_n=plan["tile_n"], tasks=tasks, blocks=blocks,
                available_cores=plan["available_cores"], waves=waves,
                active_core_fraction=blocks / plan["available_cores"],
                wave_slot_fraction=tasks / (waves * blocks),
                outer_tile_useful_fraction=case["b"] * case["m"] * case["n"] /
                    (tasks * plan["tile_m"] * plan["tile_n"]),
                matmul_flops=2 * case["b"] * case["m"] * case["n"] * case["k"],
                full_similarity_bytes=4 * case["b"] * case["m"] * ((case["n"] + 15) // 16 * 16))


def aggregate(stats):
    medians = [s["median_us"] for s in stats]
    if not medians or any(not math.isfinite(v) or v <= 0 for v in medians):
        raise ValueError("invalid round medians")
    return dict(median_us=statistics.median(medians), round_medians_us=medians,
                min_round_median_us=min(medians), max_round_median_us=max(medians),
                max_round_p95_us=max(s["p95_us"] for s in stats),
                samples=sum(s["samples"] for s in stats))


def rank(candidates, metric):
    if any(metric not in value for value in candidates.values()):
        return None
    best = min(candidates, key=lambda policy: candidates[policy][metric]["median_us"])
    stats = candidates[best][metric]
    near = []
    for policy, value in candidates.items():
        other = value[metric]
        # A heuristic shortlist, NOT a statistical confidence interval.
        overlap = (len(stats["round_medians_us"]) > 1 and
                   other["min_round_median_us"] <= stats["max_round_median_us"] and
                   stats["min_round_median_us"] <= other["max_round_median_us"])
        if other["median_us"] <= stats["median_us"] * 1.03 or overlap:
            near.append(policy)
    return dict(best_policy=best, best_median_us=stats["median_us"], near_best=near,
                speedup_vs_auto=candidates["auto"][metric]["median_us"] / stats["median_us"]
                    if "auto" in candidates else None,
                speedup_vs_32x64=candidates["32x64"][metric]["median_us"] / stats["median_us"]
                    if "32x64" in candidates else None)


def summarize(cases, results, policies, rounds):
    summaries = []
    for case in cases:
        records = [r for r in results if r["case"] == case]
        candidates = {}
        complete = True
        cores = set()
        selected_policies = policies[case["name"]] if isinstance(policies, dict) else policies
        for policy in selected_policies:
            rows = sorted((r for r in records if r["requested_policy"] == policy), key=lambda r: r["round"])
            candidate = dict(completed_rounds=len(rows))
            candidates[policy] = candidate
            try:
                if ([r["round"] for r in rows] != list(range(1, rounds + 1)) or
                        any(r["status"] != "PASS" for r in rows)):
                    raise ValueError("missing or unsuccessful rounds")
                plan = rows[0]["matmul_plan"]
                if plan["policy"] != policy or any(r["matmul_plan"] != plan for r in rows):
                    raise ValueError("Matmul plan changed between rounds or does not match request")
                cores.add(plan["available_cores"])
                candidate["features"] = features(case, plan)
                candidate["host"] = aggregate([r["host_call"] for r in rows])
                if any("profile" in r for r in rows):
                    candidate["device"] = aggregate([device_stats(r) for r in rows])
                candidate["status"] = "COMPLETE"
            except (KeyError, ValueError, ZeroDivisionError) as error:
                candidate.update(status="INCOMPLETE", error=str(error))
                complete = False
        if len(cores) != 1:
            complete = False
        generated = bool(records) and all(r["status"] == "GENERATED" for r in records)
        summaries.append(dict(case=case, status="COMPLETE" if complete else "GENERATED" if generated else "INCOMPLETE",
                              candidates=candidates, host=rank(candidates, "host") if complete else None,
                              device=rank(candidates, "device") if complete else None))
    return summaries


def write_reports(report, cases, output):
    summaries = summarize(cases, report["results"], report["sweep"]["policies_by_case"], report["sweep"]["rounds"])
    report["sweep_summary"] = summaries
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    metric = report["sweep"]["ranking_metric"]
    lines = ["# Local Matmul tile sweep", "", "ONLINE_EVALUATION=NOT_RUN", "",
             f"Primary comparison: {metric}; each value is the median of round medians (μs).",
             "Host includes tiling/allocation/synchronization/free; device uses separate profiled Task Duration, "
             "including first calls. Never mix or subtract these timing scopes.", "",
             "SDK-discovered candidates plus auto; sequential runs with rotated order. "
             "near_best: within 3% of best OR overlapping round-median ranges; not statistical significance. "
             "Incomplete/failed comparisons have no winner. Geometry features are not hardware utilization.", "",
             "| Case | Status | Candidates | Best observed | auto/best | near_best |",
             "|---|---|---:|---|---:|---|"]
    csv_rows = []
    for summary in summaries:
        ranking = summary[metric]
        for policy, candidate in summary["candidates"].items():
            row = dict(case=summary["case"]["name"], **{k: summary["case"][k] for k in
                       ("b", "m", "n", "k", "dtype", "ta", "tb", "pattern", "seed")},
                       policy=policy, status=candidate["status"], comparison_status=summary["status"],
                       completed_rounds=candidate["completed_rounds"], **candidate.get("features", {}))
            for name in ("host", "device"):
                measurements = candidate.get(name, {})
                for key in ("median_us", "min_round_median_us", "max_round_median_us", "max_round_p95_us", "samples"):
                    row[f"{name}_{key}"] = measurements.get(key)
                selected = summary[name]
                row[f"{name}_relative_to_best"] = (measurements["median_us"] / selected["best_median_us"]
                                                    if selected else None)
            csv_rows.append(row)
        winner = ranking["best_policy"] if ranking else "—"
        speedup = f"{ranking['speedup_vs_auto']:.3f}" if ranking else "—"
        near = ", ".join(ranking["near_best"]) if ranking else "—"
        lines.append(f"| {summary['case']['name']} | {summary['status']} | {len(summary['candidates'])} "
                     f"| {winner} | {speedup} | {near} |")
    for result in report["results"]:
        if result.get("error"):
            lines += ["", f"{result['case']['name']} / round {result['round']} / "
                      f"{result['requested_policy']}: {result['error']}"]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    with (output / "sweep.csv").open("w", newline="") as file:
        columns = list(dict.fromkeys(key for row in csv_rows for key in row))
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)


def run(args, cases, report, measure):
    policies, registry = {}, {}
    report['sweep'] = dict(policies_by_case=policies, rounds=args.sweep_rounds, expected_runs=None,
                           discovery_pending=args.generate_only, discovery={},
                           ranking_metric='device' if args.profile != 'none' else 'host',
                           overrides_inherited_CANN_MATMUL_TILE=True, execution_family="gm",
                           overrides_inherited_CANN_EXECUTION_FAMILY=True)
    for case in cases:
        name = case['name']
        policies[name] = []
        if args.generate_only:
            inputs = args.output_dir / name / 'inputs'
            baseline.make_case(case, inputs)
            report['results'].append(dict(case=case, status='GENERATED', discovery_pending=True,
                                         input_directory=str(inputs.relative_to(args.output_dir))))
        else:
            try:
                choices, plans, data = discover(args, case, args.output_dir / name / 'discovery')
                policies[name], registry[name] = choices, plans
                report['sweep']['discovery'][name] = data
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                report['results'].append(dict(case=case, status='FAIL', round=0,
                                             requested_policy='discovery', error=str(error)))
    if not args.generate_only:
        report['sweep']['expected_runs'] = sum(map(len, policies.values())) * args.sweep_rounds
    write_reports(report, cases, args.output_dir)
    old_policy = os.environ.get('CANN_MATMUL_TILE')
    old_family = os.environ.get('CANN_EXECUTION_FAMILY')
    try:
        os.environ['CANN_EXECUTION_FAMILY'] = 'gm'
        for case, repeat, policy in jobs(cases, args.sweep_rounds, policies):
            os.environ['CANN_MATMUL_TILE'] = policy
            directory = args.output_dir / case['name'] / f'round-{repeat}' / policy
            print(f"[RUN] {case['name']} round={repeat}/{args.sweep_rounds} tile={policy}", flush=True)
            result = measure(args, case, directory)
            if result['status'] == 'PASS':
                try:
                    if result.get('matmul_plan') != registry[case['name']][policy]:
                        raise ValueError('actual Matmul plan differs from SDK discovery')
                    if args.profile != 'none':
                        device_stats(result, args.profile_repeat)
                except (KeyError, ValueError) as error:
                    result.update(status='FAIL', error=str(error))
            result.update(round=repeat, requested_policy=policy, directory=str(directory.relative_to(args.output_dir)))
            if directory.is_dir():
                (directory / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            report['results'].append(result)
            write_reports(report, cases, args.output_dir)
            print(f"[{result['status']}] {case['name']} round={repeat} tile={policy} {result.get('error', '')}", flush=True)
    finally:
        if old_family is None:
            os.environ.pop('CANN_EXECUTION_FAMILY', None)
        else:
            os.environ['CANN_EXECUTION_FAMILY'] = old_family
        if old_policy is None:
            os.environ.pop('CANN_MATMUL_TILE', None)
        else:
            os.environ['CANN_MATMUL_TILE'] = old_policy
    failed = sum(r['status'] == 'FAIL' for r in report['results'])
    incomplete = sum(s['status'] == 'INCOMPLETE' for s in report['sweep_summary'])
    print(f"runs={len(report['results'])} failures={failed}; report={args.output_dir / 'report.md'}")
    return int(failed != 0 or incomplete != 0)
