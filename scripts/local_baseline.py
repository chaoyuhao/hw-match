#!/usr/bin/env python3
"""Generate and verify local NPU cases. Never an official scoring tool."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def quantize(values, dtype):
    values = np.asarray(values, dtype="<f4")
    if not np.all(np.isfinite(values)):
        raise ValueError("inputs must be finite")
    if dtype == "fp16":
        packed = values.astype("<f2")
        decoded = packed.astype(np.float64)
    elif dtype == "bf16":
        # IEEE round-to-nearest, ties-to-even, on finite FP32 inputs.
        bits = values.view("<u4")
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
        packed = (rounded >> 16).astype("<u2")
        decoded = (packed.astype("<u4") << 16).view("<f4").astype(np.float64)
    else:
        raise ValueError(f"unsupported dtype: {dtype}")
    if not np.all(np.isfinite(decoded)):
        raise ValueError("quantization overflow")
    return packed, decoded


def prepare_inputs(a, b, dtype, ta, tb):
    pa, qa = quantize(a, dtype)
    pb, qb = quantize(b, dtype)
    if qa.ndim != 3 or qb.ndim != 3 or qa.shape[0] != qb.shape[0] or qa.shape[2] != qb.shape[1]:
        raise ValueError("rank-3 inputs must have equal B and logical K; broadcasting is not allowed")
    similarity = qa @ qb
    y = similarity.max(axis=-1).sum(axis=-1).astype("<f4")
    if ta:
        pa = pa.swapaxes(-1, -2)
    if tb:
        pb = pb.swapaxes(-1, -2)
    return np.ascontiguousarray(pa), np.ascontiguousarray(pb), similarity, y


def compare(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or expected.size == 0:
        raise ValueError(f"shape mismatch: actual={actual.shape}, expected={expected.shape}")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(expected)):
        raise ValueError("nonfinite value in output or golden")
    diff = np.abs(actual - expected)
    tolerance = 1e-4 + 1e-4 * np.abs(expected)
    stats = {"max_abs_error": float(diff.max()),
             "max_tolerance_ratio": float((diff / tolerance).max())}
    if np.any(diff > tolerance):
        index = np.unravel_index(int(np.argmax(diff / tolerance)), diff.shape)
        raise ValueError(f"precision mismatch at {index}: actual={actual[index]}, "
                         f"golden={expected[index]}, stats={stats}")
    return stats


def cases_for_suite(suite):
    cases = []

    def add(name, shape, dtype, ta=False, tb=False, pattern="random"):
        b, m, n, k = shape
        cases.append(dict(name=f"{name}_{dtype}_{int(ta)}{int(tb)}", b=b, m=m, n=n, k=k,
                          dtype=dtype, ta=ta, tb=tb, pattern=pattern, seed=20260929))

    if suite == "reduction":
        for dtype in ("fp16", "bf16"):
            for ta in (False, True):
                for tb in (False, True):
                    for m, n in ((31, 63), (32, 64), (33, 65), (65, 255), (65, 256), (65, 257)):
                        add(f"reduce_m{m}_n{n}", (2, m, n, 24), dtype, ta, tb)
            add("reduce_negative_large", (1, 8192, 65, 32), dtype, False, False, "negative")
            for m in (1023, 1024, 1025, 8192):
                add(f"reduce_cancel_m{m}", (2, m, 65, 8), dtype, True, True, "row_cancellation")
            add("reduce_sum_carry", (2, 1026, 65, 8), dtype, True, True, "sum_carry")
            add("reduce_batch_groups", (257, 33, 257, 8), dtype, True, False, "negative")
        return cases

    add("template", (1, 1, 1, 32), "fp16")
    for dtype in ("fp16", "bf16"):
        for ta in (False, True):
            for tb in (False, True):
                add("tail", (2, 17, 33, 24), dtype, ta, tb)
                add("negative", (3, 3, 5, 8), dtype, ta, tb, "negative")
                if suite == "full":
                    add("aligned", (2, 128, 192, 256), dtype, ta, tb)
                    add("long_k", (2, 3, 5, 8192), dtype, ta, tb)
                    add("n1", (3, 17, 1, 40), dtype, ta, tb)
        add("known", (1, 2, 3, 8), dtype, True, True, "known")
        if suite == "full":
            add("long_n", (2, 3, 8192, 24), dtype, True, False)
            add("long_m", (2, 8192, 3, 8), dtype, False, True)
            add("zero", (2, 31, 17, 24), dtype, True, True, "zero")
            add("cancellation", (2, 3, 1, 8192), dtype, True, True, "cancellation")
            add("task_loop", (3, 257, 321, 24), dtype, True, False)
            add("batch_groups", (257, 3, 5, 8), dtype, True, True, "negative")
    return cases


def make_case(case, directory):
    directory.mkdir(parents=True, exist_ok=False)
    b, m, n, k = (case[key] for key in ("b", "m", "n", "k"))
    rng = np.random.default_rng(case["seed"])
    a = rng.uniform(-0.5, 0.5, (b, m, k)).astype(np.float32)
    x = rng.uniform(-0.5, 0.5, (b, k, n)).astype(np.float32)
    if case["pattern"] == "negative":
        a = np.abs(a) + 0.125
        x = -np.abs(x) - 0.125
    elif case["pattern"] == "zero":
        a.fill(0)
    elif case["pattern"] == "known":
        a.fill(0)
        x.fill(0)
        a[0, 0, 0] = a[0, 1, 1] = 1
        x[0, :2, :] = [[1, 0, -1], [0, 1, 0]]
    elif case["pattern"] == "cancellation":
        a[:, :, 1::2] = a[:, :, ::2]
        x[:, 1::2, :] = -x[:, ::2, :]
    elif case["pattern"] == "row_cancellation":
        # Exactly representable in both input formats. Each row's entire dot
        # product is one value, so this isolates cancellation in the M sum.
        a.fill(0)
        x.fill(0)
        cycle = np.array([4096, 0.03125, -4096, -0.015625], dtype=np.float32)
        a[:, :, 0] = cycle[np.arange(m) % 4]
        a[1::2, :, 0] *= -1
        x[:, 0, :] = 1
    elif case["pattern"] == "sum_carry":
        # A half-ulp after 4096 must be carried from rows 0..1023 into the
        # next chunk. Resetting compensation at row 1024 changes y to zero.
        a.fill(0)
        x.fill(0)
        a[:, 1022:1026, 0] = [4096, 2**-12, 2**-12, -4096]
        a[1::2, :, 0] *= -1
        x[:, 0, :] = 1
    pa, pb, similarity, golden = prepare_inputs(a, x, case["dtype"], case["ta"], case["tb"])
    pa.tofile(directory / "x1.bin")
    pb.tofile(directory / "x2.bin")
    golden.tofile(directory / "golden_y.bin")
    # FP64 intermediate reference keeps near-zero precision diagnostics honest.
    similarity.astype("<f8").tofile(directory / "golden_similarity.bin")
    fields = [b, m, n, k, 1 if case["dtype"] == "fp16" else 2, int(case["ta"]), int(case["tb"])]
    (directory / "case.txt").write_text(" ".join(map(str, fields)) + "\n")
    (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    return similarity, golden


def read_f32(path, count):
    if path.stat().st_size != count * 4:
        raise ValueError(f"{path.name}: expected exactly {count * 4} bytes, got {path.stat().st_size}")
    return np.fromfile(path, dtype="<f4")


def run_case(binary, case, directory, device, repeat, timeout, dump_similarity, benchmark=None):
    result = dict(case=case, status="FAIL", online_evaluation="NOT_RUN")
    start = time.monotonic()
    try:
        similarity, golden = make_case(case, directory)
        command = [str(binary), str(directory), str(device), str(repeat), str(int(dump_similarity))]
        if benchmark is not None:
            command.extend(str(value) for value in benchmark)
        result["command"] = command
        with (directory / "runtime.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if process.returncode:
            raise ValueError(f"runner exit={process.returncode}; see runtime.log")
        outputs = []
        result["precision"] = []
        for index in range(repeat):
            actual = read_f32(directory / f"y-{index}.bin", case["b"])
            result["precision"].append(compare(actual, golden))
            if outputs and not np.array_equal(actual.view("<u4"), outputs[0].view("<u4")):
                raise ValueError(f"repeat {index} differs bitwise from repeat 0")
            outputs.append(actual)
        if dump_similarity:
            pitch = ((case["n"] + 15) // 16) * 16
            actual = read_f32(directory / "similarity.bin", case["b"] * case["m"] * pitch)
            actual = actual.reshape(case["b"], case["m"], pitch)[..., :case["n"]]
            result["matmul_precision"] = compare(actual, similarity)
            # Independent reduction check isolates Max/Sum from Cube precision.
            reduction_golden = actual.astype(np.float64).max(axis=-1).sum(axis=-1).astype(np.float32)
            result["reduction_precision"] = compare(outputs[0], reduction_golden)
        result["status"] = "PASS"
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        result["error"] = str(exc)
    result["wall_seconds_including_startup_io"] = time.monotonic() - start
    if directory.is_dir():
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "full", "reduction"), default="smoke")
    parser.add_argument("--case", help="run exactly one named case from the selected suite")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--no-dump-similarity", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if args.device < 0 or args.repeat < 1 or args.timeout < 1:
        parser.error("device must be nonnegative; repeat and timeout must be positive")
    if not args.generate_only and (args.binary is None or not args.binary.is_file()):
        parser.error("provide an existing --binary or use --generate-only")
    cases = cases_for_suite(args.suite)
    if args.case:
        cases = [case for case in cases if case["name"] == args.case]
        if not cases:
            parser.error("case name not found in this suite")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(online_evaluation="NOT_RUN", mode="generate" if args.generate_only else "NPU",
                  suite=args.suite, results=[])
    kernel = Path(__file__).resolve().parents[1] / "kernel.asc"
    report["kernel_sha256"] = hashlib.sha256(kernel.read_bytes()).hexdigest()
    print("LOCAL_BASELINE; ONLINE_EVALUATION=NOT_RUN", flush=True)
    for case in cases:
        directory = args.output_dir / case["name"]
        if args.generate_only:
            make_case(case, directory)
            result = dict(case=case, status="GENERATED", online_evaluation="NOT_RUN")
        else:
            result = run_case(args.binary.resolve(), case, directory, args.device, args.repeat,
                              args.timeout, not args.no_dump_similarity)
        report["results"].append(result)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"[{result['status']}] {case['name']} {result.get('error', '')}", flush=True)
    failed = sum(result["status"] == "FAIL" for result in report["results"])
    print(f"cases={len(cases)} failures={failed}; report={args.output_dir / 'report.json'}")
    return int(failed != 0)


if __name__ == "__main__":
    sys.exit(main())
