# R14 implementation plan

Spec: ../specs/2026-10-01-joint-pipeline-design.md
Execution: inline, already authorized by the user.

- [x] Add bounded joint planner and double-slot allocation; verify actual C++ planner with independent task enumeration and SDK rejection probes.
- [x] Implement explicit asynchronous producer/consumer loop; test actual helpers with delayed Matmul completion and FP64 reference.
- [x] Integrate host dispatch, family controls, snapshots and validated report metadata; preserve legacy controls and one-launch behavior.
- [x] Run complete host suite, get fresh whole-change review, fix material findings.
- [x] Update iteration/case records and copy instructions.

Integration: commit the verified tree, fast-forward main, and push the already authorized origin/main.

Baseline: 66 host tests passed (56.969 s), no local NPU run.

Progress: task 1 RED (missing joint planner) -> GREEN; task 2 RED (wrong double-slot ownership) -> GREEN; task 3 RED (missing joint dispatch / unsupported pipeline report) -> GREEN. Full host suite: 68 tests passed, 58.583 s. Fresh whole-change review complete: no actionable correctness findings. No NPU test. R12 Sum and Small device arithmetic verified byte-for-byte unchanged. Shell syntax and diff whitespace checks passed.
