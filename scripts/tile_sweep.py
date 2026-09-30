"""Sequential local tile experiments and conservative summaries; no online tuning."""
import csv
import json
import math
import os
import statistics

import local_baseline as baseline

POLICIES = ("32x64", "32x128", "64x128", "128x128", "auto")


def jobs(cases, rounds):
    for repeat in range(rounds):
        for index, case in enumerate(cases):
            offset = (index + repeat) % len(POLICIES)
            for policy in POLICIES[offset:] + POLICIES[:offset]:
                yield case, repeat + 1, policy


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
        for policy in policies:
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
        generated = len(records) == len(policies) * rounds and all(r["status"] == "GENERATED" for r in records)
        summaries.append(dict(case=case, status="COMPLETE" if complete else "GENERATED" if generated else "INCOMPLETE",
                              candidates=candidates, host=rank(candidates, "host") if complete else None,
                              device=rank(candidates, "device") if complete else None))
    return summaries


def write_reports(report, cases, output):
    summaries = summarize(cases, report["results"], POLICIES, report["sweep"]["rounds"])
    report["sweep_summary"] = summaries
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    metric = report["sweep"]["ranking_metric"]
    lines = ["# Local Matmul tile sweep", "", "ONLINE_EVALUATION=NOT_RUN", "",
             f"Primary comparison: {metric}; each value is the median of round medians (μs).",
             "Host includes tiling/allocation/synchronization/free; device uses separate profiled Task Duration, "
             "including first calls. Never mix or subtract these timing scopes.", "",
             "Four fixed tiles plus auto, sequential runs with rotated order. "
             "near_best: within 3% of best OR overlapping round-median ranges; not statistical significance. "
             "Incomplete/failed comparisons have no winner. Geometry features are not hardware utilization.", "",
             "| Case | Status | " + " | ".join(POLICIES) + " | Best observed | auto/best | near_best |",
             "|---|---|" + "---:|" * len(POLICIES) + "---|---:|---|"]
    csv_rows = []
    for summary in summaries:
        ranking = summary[metric]
        cells = []
        for policy in POLICIES:
            candidate = summary["candidates"][policy]
            stats = candidate.get(metric)
            cells.append(f"{stats['median_us']:.3f}" if stats else "—")
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
        lines.append(f"| {summary['case']['name']} | {summary['status']} | " + " | ".join(cells) +
                     f" | {winner} | {speedup} | {near} |")
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
    report["sweep"] = dict(policies=list(POLICIES), rounds=args.sweep_rounds,
                           expected_runs=len(cases) * len(POLICIES) * args.sweep_rounds,
                           ranking_metric="device" if args.profile != "none" else "host",
                           overrides_inherited_CANN_MATMUL_TILE=True)
    write_reports(report, cases, args.output_dir)
    old_policy = os.environ.get("CANN_MATMUL_TILE")
    try:
        for case, repeat, policy in jobs(cases, args.sweep_rounds):
            os.environ["CANN_MATMUL_TILE"] = policy
            directory = args.output_dir / case["name"] / f"round-{repeat}" / policy
            print(f"[RUN] {case['name']} round={repeat}/{args.sweep_rounds} tile={policy}", flush=True)
            if args.generate_only:
                inputs = args.output_dir / case["name"] / "inputs"
                if not inputs.exists():
                    baseline.make_case(case, inputs)
                result = dict(case=case, status="GENERATED", input_directory=str(inputs.relative_to(args.output_dir)))
            else:
                result = measure(args, case, directory)
                if result["status"] == "PASS":
                    try:
                        plan = result.get("matmul_plan")
                        if plan is None or plan["policy"] != policy:
                            raise ValueError("missing Matmul plan or actual policy differs from requested tile")
                        if args.profile != "none":
                            device_stats(result, args.profile_repeat)
                    except (KeyError, ValueError) as error:
                        result.update(status="FAIL", error=str(error))
            result.update(round=repeat, requested_policy=policy, directory=str(directory.relative_to(args.output_dir)))
            if directory.is_dir():
                (directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            report["results"].append(result)
            write_reports(report, cases, args.output_dir)
            print(f"[{result['status']}] {case['name']} round={repeat} tile={policy} {result.get('error', '')}", flush=True)
    finally:
        if old_policy is None:
            os.environ.pop("CANN_MATMUL_TILE", None)
        else:
            os.environ["CANN_MATMUL_TILE"] = old_policy
    failed = sum(r["status"] == "FAIL" for r in report["results"])
    incomplete = sum(s["status"] == "INCOMPLETE" for s in report["sweep_summary"])
    print(f"runs={len(report['results'])} failures={failed}; report={args.output_dir / 'report.md'}")
    return int(failed != 0 or incomplete != 0)
