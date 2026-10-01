#ifndef CANN_MATCH_ITERATE_PLAN_H
#define CANN_MATCH_ITERATE_PLAN_H
#include "joint_plan.h"
namespace local_baseline {
// Application scratch contains only row maxima/records, not C tiles.
// GetTensorC may still use the SDK's system workspace on the target hardware.
struct IteratePlan {
    StreamPlan stream{};
    uint32_t baseM = 0, baseN = 0, maxColumns = 0;
};
inline bool AcceptIterateTile(IteratePlan& plan, const ProblemDesc& p, const HardwareCaps& h,
                              const MatmulPlan& mm, uint32_t bm, uint32_t bn)
{
    if (!bm || !bn || bm % 16 || bn % 16 || bm > plan.stream.tileM ||
        bn > plan.stream.tileN || uint64_t(bm) * bn > 8192) return false;
    const auto& s = plan.stream;
    const uint64_t used = uint64_t(bm) * bn * 4 + s.tileM * 4 + 128 +
        (s.splits > 1 ? 256 : 0) + 14368 + s.reduction.foldUbBytes;
    if (used > 64 * 1024 || used + mm.ubBudget > h.ubBytes) return false;
    // p is validated by MakePlan; keeping it in the contract prevents accepting
    // a plan whose row span does not cover the supplied problem metadata.
    if (!p.m || !p.n || !plan.maxColumns || plan.maxColumns > p.n) return false;
    plan.baseM = bm; plan.baseN = bn; plan.stream.ubBytes = static_cast<uint32_t>(used);
    return true;
}
inline IteratePlan MakeIteratePlan(const ProblemDesc& p, const HardwareCaps& h, const ExecutionPlan& reference)
{
    const auto& mm = reference.matmul;
    MakePlan(p, h, {mm.tileM, mm.tileN});
    const uint32_t gridN = CeilDiv(p.n, mm.tileN);
    const uint64_t rows = PlanProduct(p.batches, CeilDiv(p.m, mm.tileM));
    const uint32_t needed = rows >= h.cores ? 1 : CeilDiv(h.cores, static_cast<uint32_t>(rows));
    const uint32_t splits = reference.family == ExecutionFamily::Gm ? std::min(gridN, needed) : reference.stream.splits;
    IteratePlan result;
    auto& s = result.stream;
    s = SplitStreamPlan(p, h, mm, splits, 1, reference.reduction.mode ? ReductionPolicy::Partials : ReductionPolicy::Rows);
    // Remove the old per-core GM C slots. Offsets remain byte offsets.
    s.scratchBytes -= s.maximaOffset;
    s.partialOffset -= s.maximaOffset;
    s.maximaOffset = 0; s.cSlotElements = 0; s.plannerVersion = 4;
    for (uint32_t shard = 0; shard < splits; ++shard) {
        const uint32_t first = gridN * shard / splits * mm.tileN;
        const uint32_t last = std::min(p.n, gridN * (shard + 1) / splits * mm.tileN);
        result.maxColumns = std::max(result.maxColumns, last - first);
    }
    const uint32_t bm = mm.tileM;
    const uint32_t bn = std::min(mm.tileN, (8192 / bm) / 16 * 16);
    if (!AcceptIterateTile(result, p, h, mm, bm, bn)) throw std::runtime_error("invalid local C budget");
    return result;
}
} // namespace local_baseline
#endif
