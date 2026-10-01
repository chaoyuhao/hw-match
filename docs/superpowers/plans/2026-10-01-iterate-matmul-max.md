# R18 Matmul → Max implementation plan

User approved implementing this structural optimization; proceed inline, host tests only. Baseline is R16/S12 (R17 fixed S2 remains an explicit source control, default off).

## Design and boundaries

Replace per-outer-N-tile Configure/IterateAll/End with one synchronous Iterate/GetTensorC session per `(batch, row block, N shard)`. C is VECIN/ND/FP32 on both host and device. FIRSTM iteration has a checked baseM/baseN footprint up to 32 KiB; consume only valid M/N, never padding, after full K. Reuse original input strides and all transpose/dtype dispatches. Store only shard row maxima or compensated records in application scratch; GetTensorC can still use internal GM on A2. This is not a claim of physical on-chip C delivery or established speedup.

Retain R16 selected outer geometry and reduction mode. For existing Stream/Pipeline retain N ownership. For GM derive the minimum N-shard count that fills available cores, capped by the outer N grid; no hidden-case identification. Keep original Small admission and explicit local controls. Request an independent VECIN SDK tiling with fixed bounded base geometry; check returned geometry/resources. On rejection retain the full accepted R16 plan before the sole launch. Never retry after launch.

The new source switch is on by default. A trailing host argument can disable it for comparisons; the local runner exposes it as a diagnostic switch. Reports distinguish new Iterate execution from the frozen reference and do not label old cost estimates as new-path scores. Existing Stream merge and compensated sum helpers are reused; where GM becomes row-owned, segment sizes are rebuilt explicitly.

## Tasks

1. Add `iterate_plan.h`: checked row/shard ownership, actual base-C footprint, scratch offsets and UB accounting. Add host tests for tails, large counts, rejection and ownership.
2. Add `iterate_matmul.asc`: local-C consumer and session producer, one Matmul configuration/End per task, FIRSTM block coordinates, exact valid masks, existing shard Max and final compensation. Add CPU tests against independent FP64 reference with poisoned padding and session/UB reuse checks.
3. Integrate new SDK preparation, single launch dispatch and full R16 rollback in `kernel.asc`; disable default S2. Verify actual preparation/dispatch with mocked SDK boundaries including rejection and failure, both dtypes/four layouts.
4. Update actual report writer/validator and source snapshot lists. Keep explicit controls and historical reports usable. Add corruption checks for new resources/identity.
5. Run full host regression, fresh code review, record pending online status and exact copy list in README/iteration log/hypotheses/fusion docs. Commit, fast-forward main and push.

## Acceptance

No extra launch or changes to official frozen files. No local NPU run. CPU tests verify mathematical semantics, ownership, lifecycle and resource accounting; only online can validate CANN compilation, Cube numerical behavior and performance. Compare future S14 to S12, with S13 retained as the fixed-upstream experiment.


## Hardware clarification

The user explicitly corrected the hardware assumption: only the local machine is known to be 910B2C. Official template CMake defaults to dav-2201 but allows NPU_ARCH override; README gives CANN9.0.0 without online SKU. Query runtime core/UB capacity. API legality and actual performance still require online validation; SDK fallback is not a guarantee against unsupported compile targets.

## Verification completed

Baseline 74/74; feature full host suite 76/76 (75.483 s). Actual device-helper doubles cover 320 cases, dispatch/report checks cover 159 JSON records in each S2 configuration; a final targeted run covers >10^12 batches and overflow without enumerating tasks. Fresh review found no actionable issue. No CANN/NPU/online validation.
