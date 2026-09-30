# Small Vector Path Implementation Plan

Goal: add a bounded AIV-only BatchMatmulMaxSum path plus truthful execution metadata.
Architecture: shared metadata validation dispatches exactly one of the existing GM MIX kernel or SmallVector. SmallVector owns groups of eight outputs, loads one batch into UB, converts inputs to FP32, performs dot/row accumulation and ordered compensated M sum. Local code records schema 3 execution metadata alongside schema 2 Matmul metadata when applicable.
Tech stack: C++14/Ascend C, Python/NumPy, unittest and bounded CPU instruction doubles.
Spec: docs/superpowers/specs/2026-09-30-small-vector-fast-path-design.md (user requested implementation).

Global constraints: one launch; no online I/O or env reads; original ABI; existing GM math/planner unchanged. Only kernel.asc and new .h dependencies online. No NPU/compiler installed here; hardware performance must remain explicitly unverified.
Review focus: FP32 conversion before multiplication; UB padding/alignment and tail max; unique 32-byte output owner; forced-path failure without fallback launches; local metadata/profile matching and absent similarity handling.

- [x] Task 1: small_plan.h pure geometry/resource gate: SmallVariant None/Dot/Rows, SmallPlan(pitches/counts/UB/tasks/blocks). Exploration B<=32 M<=16 N<=64 K<=256, K%8==0, MNK<=32768; Rows additionally MK<=64. K-contiguous layouts first, TB=false Rows second; unsupported -> GM, forced small -> error. Reserve 32 KiB and cap small buffers 64 KiB. Add failing CPU tests for bounds, layouts, overflow, exact resource allocation and batch ownership; implement and run.
- [x] Task 2: small_vector.h device class sharing loading and FP32 sum. TQue input depth1, explicit V->S / S->V / V->MTE2 / S->MTE3 dependencies; padded physical rows with K/N-stride support. Dot uses Mul then WholeReduceSum over <=64-lane chunks and fixed scalar merge; Rows uses Muls/Add over actual N then WholeReduceMax. Use bounded CPU doubles to run actual helper against double golden, negative, tails, layouts, grouping and guards. Integrate exclusive dispatch returning optional scratch/ExecutionInfo; add mocked host launch test where practical.
- [x] Task 3: local runner family auto/gm/small, schema 3 execution_plan.json, legacy gm plan retained; similarity skipped only for actual small family. Python validates new schema, actual requested family and profile consistency; generated small suite includes dtype/layout pairs, boundaries and fallback. No new family sweep or NPU test matrix: user explicitly deferred these until online and development reach a bottleneck. Existing GM tile sweep forces gm so auto does not contaminate it. Test fake runner failure modes and one end-to-end real C++ helper contract.
- [x] Task 4: full host suite, source/ABI checks, docs and R8 log/hypothesis update, one fresh independent review; merge/push for remote testing. No claim of CANN build, NPU correctness or measured acceleration until remote evidence exists.

User override (2026-09-30): skip local NPU tests/performance matrices as a prerequisite. Enable a conservative experimental region directly for online evaluation: one group wave; Dot min(B,8)*M*N <= 16; Rows min(B,8)*M*K <= 32. Thresholds are hypotheses, not measured break-even points. CPU logic and interface checks remain necessary.

Verification: 58 host tests passed (72.194s); bash syntax and git diff whitespace checks passed. Independent review found a forced-family metadata identity gap; reproduced and fixed, including profiler collection. No CANN build or NPU test performed.
