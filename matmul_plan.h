#ifndef CANN_MATCH_MATMUL_PLAN_H
#define CANN_MATCH_MATMUL_PLAN_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace local_baseline {
// Host-only geometry: no runtime, SDK, I/O or device dependencies.
constexpr uint32_t PLANNER_VERSION = 1;
constexpr uint32_t MAX_OUTER_TILE = 256; // search bound, not a hardware limit
struct ProblemDesc { uint64_t batches; uint32_t m, n, k, dtype; bool ta, tb; };
struct HardwareCaps { uint32_t cores; uint64_t ubBytes; };
struct TileRequest { uint32_t m = 0, n = 0; }; // {0,0}: automatic
struct MatmulPlan {
    uint32_t tileM, tileN;
    uint64_t tasks;
    uint32_t blocks, ubBudget;
    double score;
};
struct CandidateSet { std::array<MatmulPlan, 64> plans{}; size_t count = 0; };

inline uint64_t PlanProduct(uint64_t a, uint64_t b)
{
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw std::runtime_error("plan size overflow");
    return a * b;
}
inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return a / b + (a % b != 0); }
inline void ValidateProblem(const ProblemDesc& p, const HardwareCaps& h)
{
    if (!p.batches || !p.m || p.m > 8192 || !p.n || p.n > 8192 || !p.k || p.k > 8192 ||
        (p.dtype != 1 && p.dtype != 2) || !h.cores || h.cores > 65535 || h.ubBytes <= 64 * 1024)
        throw std::runtime_error("invalid problem or hardware capabilities");
    PlanProduct(PlanProduct(p.batches, p.m), uint64_t(CeilDiv(p.n, 16)) * 16 * 4);
    PlanProduct(PlanProduct(p.batches, p.m), uint64_t(p.k) * 2);
    PlanProduct(PlanProduct(p.batches, p.n), uint64_t(p.k) * 2);
}
inline MatmulPlan MakePlan(const ProblemDesc& p, const HardwareCaps& h, TileRequest t)
{
    ValidateProblem(p, h);
    if (!t.m || !t.n || t.m % 16 || t.n % 16 || t.m > MAX_OUTER_TILE || t.n > MAX_OUTER_TILE)
        throw std::runtime_error("outer tile must be 16-aligned and in [16,256]");
    const uint64_t tasks = PlanProduct(p.batches, uint64_t(CeilDiv(p.m, t.m)) * CeilDiv(p.n, t.n));
    const uint32_t blocks = static_cast<uint32_t>(std::min<uint64_t>(h.cores, tasks));
    const uint64_t waves = tasks / blocks + (tasks % blocks != 0);
    // Dimensionless hypothesis: padded work + layout-sensitive operand traffic
    // + one per-call overhead proxy. These weights are not measured latency.
    const double work = double(std::min(p.m, t.m)) * std::min(p.n, t.n) * p.k;
    const double traffic = 2.0 * p.k * (std::min(p.m, t.m) * (p.ta ? 1.25 : 1.0) +
                                       std::min(p.n, t.n) * (p.tb ? 1.0 : 1.25));
    const double score = waves * (work + 16.0 * traffic + 65536.0);
    // Current reduction buffers use 47392 bytes. Reserve 64 KiB including
    // headroom; retain the previous 128 KiB Matmul ceiling. SDK validates inner tiles.
    const uint32_t budget = static_cast<uint32_t>(std::min<uint64_t>(128 * 1024, h.ubBytes - 64 * 1024));
    return {t.m, t.n, tasks, blocks, budget, score};
}
inline CandidateSet GenerateCandidates(const ProblemDesc& p, const HardwareCaps& h)
{
    ValidateProblem(p, h);
    std::array<uint32_t, 7> ms{}, ns{};
    size_t mc = 0, nc = 0;
    auto axes = [](uint32_t extent, uint32_t cores, std::array<uint32_t, 7>& values, size_t& count) {
        auto add = [&](uint32_t value) {
            value = std::min(MAX_OUTER_TILE, CeilDiv(std::max(1U, value), 16) * 16);
            if (std::find(values.begin(), values.begin() + count, value) == values.begin() + count)
                values[count++] = value;
        };
        for (uint32_t tile = 16; tile <= MAX_OUTER_TILE; tile *= 2) add(tile);
        add(extent);
        add(CeilDiv(extent, cores));
    };
    // Batch parallelism already supplies independent tasks.
    const uint32_t coresPerBatch = static_cast<uint32_t>(h.cores / std::min<uint64_t>(p.batches, h.cores));
    axes(p.m, coresPerBatch, ms, mc);
    axes(p.n, coresPerBatch, ns, nc);
    CandidateSet result;
    for (size_t i = 0; i < mc; ++i)
        for (size_t j = 0; j < nc; ++j)
            result.plans[result.count++] = MakePlan(p, h, {ms[i], ns[j]});
    std::sort(result.plans.begin(), result.plans.begin() + result.count, [](const MatmulPlan& a, const MatmulPlan& b) {
        if (a.score != b.score) return a.score < b.score;
        if (a.tileM != b.tileM) return a.tileM < b.tileM;
        return a.tileN < b.tileN;
    });
    return result;
}
// accept() must only prepare/validate; it must not launch device work.
template <typename Accept>
inline MatmulPlan FindSupportedPlan(const ProblemDesc& p, const HardwareCaps& caps,
                                   TileRequest request, Accept accept)
{
    if (request.m || request.n) {
        const auto plan = MakePlan(p, caps, request);
        if (accept(plan)) return plan;
        throw std::runtime_error("requested Matmul tile rejected by SDK");
    }
    const auto candidates = GenerateCandidates(p, caps);
    for (size_t i = 0; i < std::min<size_t>(4, candidates.count); ++i)
        if (accept(candidates.plans[i])) return candidates.plans[i];
    const auto fallback = MakePlan(p, caps, {32,64});
    if (accept(fallback)) return fallback;
    throw std::runtime_error("Matmul candidates and conservative fallback rejected by SDK");
}
} // namespace local_baseline
#endif
