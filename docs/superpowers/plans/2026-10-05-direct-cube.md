# R22 Direct Cube implementation plan

> Use executing-plans to implement inline in `/tmp/cann-r22-direct-cube`.

Spec: `docs/superpowers/specs/2026-10-05-direct-cube-design.md`

Global constraints: one launch; immutable judge/main/CMake; no local NPU; FP32 C/Max and compensated Sum; platform/resource/alignment fallback; retain upstream tile selection.

## Task 1: Replace the eligible Cube execution layer

Produces: `DirectCaps`, `DirectPlan`, `MakeDirectPlan`; direct tile executor and GM/Stream device entry; additive execution metadata.
Consumes: current ProblemDesc, MatmulPlan, StreamPlan, reduction helpers, SDK9 APIs documented in the study.

1. Add CPU tests for resource rejection/admission and real physical-layout helper behavior; run them before implementation.
   Command: `python3 -m unittest discover -s tests -p 'test_direct_cube.py' -v`
   Expected: failure because the direct executor is not implemented.
2. Implement direct plan and tile executor; verify four transpose modes, K panels, A residency, output pitch and tails that remain 16-aligned.
   Same command; expected: PASS.
3. Add host and slot protocol checks, implement one-launch dispatch and local metadata; preserve fallback semantics.
   Command: `python3 -m unittest discover -s tests -v`
   Expected: all existing and new CPU tests PASS; no NPU claims.
4. Review SDK signatures and resource/flag ownership; archive R22 as pending online, add copy instructions and rollback switch; commit.

## Review Focus

CPU substitutes cannot prove compiler support or hardware ordering. Review classic LoadData fields (not V2), FP32 Fixpipe ND strides, Mmad accumulation initialization, local event directions, C/V flag identity in MIX1:1, final FREE drain, all-AIV barrier participation, platform gating, workspace/UB capacity, and unchanged numerical order. Check small and non-aligned fallback and all submitted include dependencies. No source may assume hidden shapes or hardcode 24 cores.
