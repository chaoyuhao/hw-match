# Rule-generated Matmul planning, phase 1

Goal: replace duplicated tile whitelists with one bounded C++ planner and SDK-validated discovery; generate reproducible test shapes by rules.
Architecture: ProblemDesc + HardwareCaps -> ranked geometry -> CANN tiling validation -> actual ExecutionPlan -> existing single-launch device computation. Local tools consume versioned actual plans and discover candidates from the runner.
Tech stack: C++14/Ascend C, Python standard library + NumPy/ml_dtypes, unittest.
Spec: docs/ARCHITECTURE_AUDIT.md (approved; user requested implementation).
Global constraints: one online kernel launch; unchanged public ABI; no online environment access/file I/O; FP32 accumulation; preserve existing device arithmetic; online CANN 9.0 compatibility is not established by host tests. No shape-to-winner table. Do not require profiling for routine iteration.
Review focus: overflow, metadata identity, SDK rejection fallback, actual-vs-requested plans, dynamic discovery, reproducible bounded generation, legacy report compatibility.

- [x] Task 1: Add matmul_plan.h (CANN-independent). ProblemDesc contains B/M/N/K/dtype/transpose; HardwareCaps contains cores/UB. Generate <=64 unique tiles with 16-aligned axes <=256 from powers of two, shape tails and core balancing. Rank by explicit dimensionless work/traffic/call heuristic. Preserve 32x64 fallback. UB ceiling stays 128 KiB with a conservative 64 KiB reserve. Test determinism, overflow, resources, new tiles, full coverage and physical offsets in CPU C++ tests.
- [x] Task 2: Integrate PrepareMatmul using GetTiling (auto: up to four ranked attempts plus fallback; explicit: one exact request). Query UB from PlatformAscendC. Return actual plan to runner. Add --list-plans CASE_DIR DEVICE OUTPUT_JSON (no launches), schema 2 plan metadata and arbitrary legal MxN local overrides. Test consumer rejection of mismatched identity/geometry/requests; confirm local-only I/O stays outside timed intervals.
- [x] Task 3: Add case_rules.py with seeded rule families, full-metadata stable identities, memory/work bounds and generated suites for correctness/performance. Refactor tile_sweep.py to discover each case's accepted tiles, bounded candidate count, per-case rotated ordering and conservative ranking. Generate-only records discovery pending, never fabricates SDK acceptance. Exercise nonlegacy dimensions and failure paths with fake SDK runner end-to-end tests.
- [x] Task 4: Document source-copy contract, routine generated checks, optional discovery sweeps and phase-2 extension points. Hash header/generator in provenance. Run full host suite and generated-data smoke, request one independent review, resolve defects, commit, fast-forward main and push for remote synchronization.

Validation boundary: CPU tests check pure planning/geometry and local orchestration only. Remote CANN build, NPU precision and timings must be reported separately. Previous online accepted candidate remains available in history.

Final verification: 51 host tests passed; 64 rule-generated datasets and FP64 goldens generated; shell syntax and diff checks passed. Independent review found no Critical/Important issue; the wrapper help omission was corrected and checked with --help. Device code and public entry remain identical to the base; one launch site. CANN/NPU and online evaluation remain NOT_RUN.
