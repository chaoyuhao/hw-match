#ifndef CANN_MATCH_STREAM_PLAN_H
#define CANN_MATCH_STREAM_PLAN_H
#include "matmul_plan.h"

namespace local_baseline {
// POD copied to the device. Offsets and scratchBytes are bytes; C capacity is floats.
struct StreamPlan {
    uint32_t tileM, tileN, splits, blocks, rowPitch, ubBytes;
    uint64_t rowTasks, tasks, cSlotElements, maximaOffset, partialOffset, scratchBytes, maxTilesPerCore;
};
inline uint64_t StreamAdd(uint64_t a, uint64_t b)
{
    if (a > std::numeric_limits<uint64_t>::max() - b) throw std::runtime_error("stream size overflow");
    return a + b;
}
inline StreamPlan SplitStreamPlan(const ProblemDesc& p, const HardwareCaps& h,
                                  const MatmulPlan& mm, uint32_t splits)
{
    // Revalidate external geometry before divisions, multiplication or allocation.
    MakePlan(p, h, {mm.tileM, mm.tileN});
    const uint32_t columns = CeilDiv(p.n, mm.tileN);
    if (!splits || splits > columns) throw std::runtime_error("empty stream N shard");
    StreamPlan s{};
    s.tileM = mm.tileM; s.tileN = mm.tileN; s.splits = splits;
    s.rowPitch = CeilDiv(p.m, 32) * 32;
    s.rowTasks = PlanProduct(p.batches, CeilDiv(p.m, s.tileM));
    s.tasks = PlanProduct(s.rowTasks, splits);
    s.blocks = static_cast<uint32_t>(std::min<uint64_t>(s.tasks, h.cores));
    s.cSlotElements = uint64_t(std::min(p.m, s.tileM)) * s.tileN;
    s.maximaOffset = PlanProduct(s.blocks, s.cSlotElements * 4);
    const uint64_t finalBytes = PlanProduct(PlanProduct(p.batches, s.rowPitch), 4);
    s.partialOffset = StreamAdd(s.maximaOffset, splits > 1 ? finalBytes : 0);
    s.scratchBytes = StreamAdd(s.partialOffset, PlanProduct(finalBytes, splits));
    if (s.scratchBytes > std::numeric_limits<size_t>::max()) throw std::runtime_error("stream exceeds address space");
    // Producer: 32x256 + tileM + 32 floats. Merge: two 32-float queues.
    // Sum: unchanged R12 14368 bytes. Matmul retains a separate 64 KiB reserve.
    s.ubBytes = 32 * 256 * 4 + s.tileM * 4 + 32 * 4 + (splits > 1 ? 256 : 0) + 14368;
    if (s.ubBytes > 64 * 1024 || uint64_t(s.ubBytes) + mm.ubBudget > h.ubBytes)
        throw std::runtime_error("stream UB budget exceeded");
    // Exact grid-stride load using a shard-period, independent of batch count.
    uint32_t a = s.blocks, b = splits;
    while (b) { const uint32_t r = a % b; a = b; b = r; }
    const uint32_t period = splits / a;
    for (uint32_t core = 0; core < s.blocks; ++core) {
        const uint64_t count = (s.tasks - 1 - core) / s.blocks + 1;
        const uint64_t cycles = count / period, remainder = count % period;
        uint64_t cycleLoad = 0, tailLoad = 0;
        for (uint32_t i = 0; i < std::min<uint64_t>(count, period); ++i) {
            const uint32_t shard = (uint64_t(core) + uint64_t(i) * s.blocks) % splits;
            const uint32_t weight = columns * (shard + 1) / splits - columns * shard / splits;
            cycleLoad += weight;
            if (i < remainder) tailLoad += weight;
        }
        s.maxTilesPerCore = std::max(s.maxTilesPerCore, cycles * cycleLoad + tailLoad);
    }
    return s;
}
inline StreamPlan MakeStreamPlan(const ProblemDesc& p, const HardwareCaps& h, const MatmulPlan& mm)
{
    StreamPlan best = SplitStreamPlan(p, h, mm, 1);
    if (best.rowTasks >= h.cores) return best;
    const uint32_t needed = CeilDiv(h.cores, static_cast<uint32_t>(best.rowTasks));
    const uint32_t limit = CeilDiv(p.n, mm.tileN);
    std::array<uint32_t, 16> candidates{};
    size_t count = 0;
    auto add = [&](uint32_t splits) {
        splits = std::min(limit, std::max(1U, splits));
        if (std::find(candidates.begin(), candidates.begin() + count, splits) == candidates.begin() + count)
            candidates[count++] = splits;
    };
    add(needed - 1); add(needed); add(needed + 1);
    for (uint32_t splits = 2; splits <= std::min(needed, limit); splits *= 2) add(splits);
    for (size_t i = 0; i < count; ++i) {
        const auto next = SplitStreamPlan(p, h, mm, candidates[i]);
        if (next.maxTilesPerCore < best.maxTilesPerCore ||
            (next.maxTilesPerCore == best.maxTilesPerCore && next.splits < best.splits)) best = next;
    }
    return best;
}
} // namespace local_baseline
#endif
