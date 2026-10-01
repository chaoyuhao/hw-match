# R15 Partial Sum Implementation Plan

> Use superpowers:executing-plans inline. User has already authorized implementation and ongoing GitHub sync.

**Goal:** Transfer compensated M-segment partials directly from the final Max owner to a short final merge.
**Architecture:** Shared partial_sum.asc writer/finalizer; reduction_plan.h owns geometry. GM and both Stream modes participate. Joint planner chooses rows/partials with actual resource and ownership costs; local controls preserve R14 comparison.
**Tech stack:** Ascend C / C++14 host planning; Python reports; checked CPU operations without NPU runs.
**Spec:** ../specs/2026-10-01-partial-sum-design.md
**Base:** 0a55bcc; R14 code unchanged since 68/68 host checks.

- [x] Task 1: add failing host tests, implement reduction layout and joint rows/partials candidates. Validate overflow, exact offsets/UB and fixed controls.
- [x] Task 2: add failing checked device-helper tests; implement compensated record writer/finalizer, integrate GM and Stream owners. Verify strong cancellation, record uniqueness, no producer scalar reads and existing async slot contract.
- [x] Task 3: integrate host launch args/allocation, local CANN_SUM_MODE, schema-3 extension and source snapshots. Test actual dispatch/writer, historical/new report validation and explicit request failures.
- [x] Task 4: complete host suite and fresh independent whole-change review; fix material findings.
- [x] Task 5: archive R15 mechanism, pending S11 hypotheses and submission hashes; commit, integrate and push authorized main.

Verification: 72/72 host tests passed in 68.063 s; 640-problem pinned R14 Rows comparison identical; shell syntax, whitespace and nine source hashes verified. Independent review found no substantive issues. CANN/NPU/online validation NOT_RUN for R15.

Release: implementation 5e27c5a fast-forwarded to main and pushed to origin/main. Submission hashes and pending S11 are archived in ITERATION_LOG.md.
