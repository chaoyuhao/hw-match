# Parallel row-max reduction implementation plan

> **For agentic workers:** Execute inline using the executing-plans workflow; finish with a fresh code review. The user has authorized implementation.

**Goal:** Reduce the measured 3.41 ms large-M kernel latency by distributing row maxima across AIV blocks and eliminating per-row Vector-to-Scalar waits.

**Architecture:** Keep the existing Matmul loop, full FP32 similarity matrix, one MIX launch and original FP32 compensated summation order. Each block computes 32-row tiles using 256-column DMA tiles and WholeReduceMax repeats, writes row maxima into aligned GM scratch, and participates in a second AIV barrier. Final output owners read maxima in chunks and perform the original ordered sum.

**Tech stack:** Ascend C / CANN 9.0 target; user validation on 910B2C / CANN 9.1.0; local C++/Python checks without NPU.

**Spec:** User request to start the reduction optimization following `build-perf/run-20260929T2053/analysis.md`; reference implementation `d01d01b`.

## Constraints and review focus

- Official entry/signature unchanged; exactly one kernel launch. Submission remains self-contained in kernel.asc.
- Do not add debugging I/O, host tensor computation, atomic sums or persistent cross-call scratch to the submission.
- Keep diagnostic similarity at the beginning of the returned DeviceBuffer; append row-max scratch with checked size addition.
- GM row-max tiles each own 32 floats, avoiding shared 32-byte output regions. All blocks participate in both barriers, including blocks without row-max or final-output work.
- Mask N tails; initialize invalid row slots; handle negative values, M tails and N=1 correctly.
- Preserve Kahan order across all M, including chunk boundaries and cancellation; ensure copy-to-scalar and scalar-to-copy dependencies.
- UB: 32 KiB input tile + row-max buffers + 4 KiB sum input, in addition to the unchanged 128 KiB Matmul budget.

## Steps

- [x] Add a CPU behavioral harness that compiles the actual reduction helpers with bounded tensor/DMA stubs. Verify unique writes, padded-GM exclusion, row-max values, fixed-order sum, input immutability and task reuse. It must fail before the new helpers exist.
- [x] Implement ComputeRowMaxima and SumRowMaxima in kernel.asc; add aligned scratch and the second barrier. Keep Matmul and local diagnostic format intact.
- [x] Add a targeted NPU regression suite for new 32-row, 64-lane, 256-column and 1024-sum boundaries, long negative rows and cancellation between rows. Preserve existing 55-case suite for direct comparison.
- [x] Run CPU tests, generation and source/API checks; independently review indexing, synchronization, precision and launch constraints. These checks do not prove CANN compilation or hardware correctness.
- [x] Document measured reference and candidate limitations, with commands for the user's NPU full correctness, targeted cases and matched profiling run. Commit and push after verification.

## Development verification

- Fresh `python3 -m unittest discover -s tests -v`: 36 tests passed, including three families executing the actual reduction helpers on CPU.
- `bash scripts/run_local.sh --suite reduction --generate-only`: 62 cases generated, zero generation failures. Report: `build-baseline/run-20260929T133807Z-9ADsH9/cases/report.json`.
- Independent review found no critical or important defects; reviewed DMA strides, masks, scratch ownership, barriers, resource budget, compensated sum and launch count.
- No CANN compiler or NPU is available on this development host. Hardware correctness, synchronization, SDK compatibility and speedup remain unverified.

## Verification on the NPU

First run the original full suite and the new reduction suite, then stress and `sweep_m8192_fp16_00 --profile timeline`. Compare Task Duration with 3412.408 μs and host median with 3448.2545 μs, without mixing the two metrics. Preserve 1-kernel launch and verify final acceptance/score on the official platform.
