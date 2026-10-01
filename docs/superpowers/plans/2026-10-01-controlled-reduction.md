# R16 Controlled Reduction Coverage Implementation Plan

> Execute inline with superpowers:executing-plans. User approved the in-chat bounded design and continued GitHub sync; do not request approval again.

**Goal:** Expand Partials coverage while preserving R15's accepted upstream execution plan.
**Architecture:** One post-selection host transformation, existing device helpers unchanged; local r15 control and truthful expansion metadata.
**Tech Stack:** C++14 host planner / Ascend C host entry / Python metadata and tests.
**Spec:** ../specs/2026-10-01-controlled-reduction-design.md
**Base:** 95d3aad; source unchanged since R15's 72/72 host checks.

## Global Constraints

One kernel; online only kernel.asc and own .h/.asc; no online debug or environment reads; no local NPU; no shape-by-case-ID rules; user copies files manually.

## Tasks

- [x] 1. RED/GREEN host planner test: `ExpandPartialReduction(const ProblemDesc&, const HardwareCaps&, const ExecutionPlan&) -> ExecutionPlan`; preserve every upstream field, gate on >=2 segments and real data compression, resource bounds, leave old Partials. Existing candidate generator/search stays byte-identical.
- [x] 2. RED/GREEN actual PrepareExecution/RunKernel test: apply after accepted SDK tiling only for unrestricted Auto, bool disable preserves R15, all existing forced controls unchanged, rejection sequences and scratch/one launch checked.
- [x] 3. Local control `CANN_SUM_MODE=r15`, expansion metadata, historical parsing and request checks. Test real writer output and malformed trace rejection; report actual changed mode and baseline score without re-running selector in Python.
- [x] 4. Full host suite, one fresh whole-change reviewer, pinned R15 CPU comparison and coverage summary. No device tests. Fix material findings before release.
- [x] 5. Update iteration log, H8 pending-S12 predictions and copy list/hashes; commit, fast-forward main and push authorized remote.

## Review Focus

SDK rejection must not change the next attempted tile; original Partials and Small must stay selected; padding alone cannot qualify a tiny M; resource fallback cannot corrupt frozen geometry; legacy metadata and forced controls must remain attributable. Tasks 1–3 explicitly check these boundaries.

Progress: Task 1 RED missing transformation, GREEN boundary/grid tests and all 5 planner tests. Tasks 2/3 RED missing bool and ignored trace, GREEN actual dispatch (72 JSON outputs + SDK rejection paths) and 10 metadata tests. Candidate generation/cost/search and every device helper remain unchanged.

Verification: 74/74 host tests passed in 69.142 s. Pinned R15 comparison: 3960/3960 upstream plans identical, 1470 Rows→Partials, existing 46 Small/1142 Partials preserved; device helpers and public entry text unchanged. Independent whole-change review found no substantive issues. CANN/NPU/online tests NOT_RUN for R16.

Release: implementation `2d7a9b9` fast-forwarded to main and pushed to origin/main. Documentation, pending-S12 hypotheses and all nine submission hashes checked. Replace only kernel.asc and joint_plan.h over R15.
