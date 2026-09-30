#ifndef CANN_MATCH_SMALL_PLAN_H
#define CANN_MATCH_SMALL_PLAN_H
#include "matmul_plan.h"
namespace local_baseline {
enum class ExecutionFamily : uint32_t { Auto = 0, Gm = 1, Small = 2 };
enum class SmallVariant : uint32_t { None = 0, Dot = 1, Rows = 2 };
struct SmallPlan {
    SmallVariant variant = SmallVariant::None;
    uint32_t aRows = 0, aWidth = 0, aPitch = 0, aElements = 0;
    uint32_t bRows = 0, bWidth = 0, bPitch = 0, bElements = 0;
    uint32_t ubBytes = 0, tasks = 0, blocks = 0;
    // Zero keeps the scalar-per-dot implementation; otherwise columns per repeat group.
    uint32_t dotColumns = 0, productElements = 256, partialElements = 8;
};
inline SmallPlan MakeSmallPlan(const ProblemDesc& p, const HardwareCaps& caps)
{
    // The generic path owns invalid-input diagnostics and tensor overflow checks.
    if (!p.batches || !p.m || p.m>8192 || !p.n || p.n>8192 || !p.k || p.k>8192 ||
        (p.dtype!=1 && p.dtype!=2) || !caps.cores || caps.cores>65535)
        throw std::runtime_error("invalid small-path problem");
    PlanProduct(PlanProduct(p.batches,p.m), uint64_t(p.n)*4);
    PlanProduct(PlanProduct(p.batches,p.m), uint64_t(p.k)*2);
    PlanProduct(PlanProduct(p.batches,p.n), uint64_t(p.k)*2);
    // These limits match the current N-vector and K-product scratch capacities.
    // Batch count does not increase per-batch storage. Account for M and the
    // physical layout below, then let the automatic work budget decide.
    if (p.n>64 || p.k>256 || p.k%8) return {};
    const uint64_t groups = p.batches/8 + (p.batches%8 != 0);
    if (groups > std::numeric_limits<uint32_t>::max()) return {};
    SmallPlan s;
    if ((!p.ta || p.m==1) && (p.tb || p.n==1)) {
        s.variant=SmallVariant::Dot;
        s.aRows=p.m; s.aWidth=p.k; s.bRows=p.n; s.bWidth=p.k;
    } else if (!p.tb) {
        s.variant=SmallVariant::Rows;
        s.aRows=p.ta?p.k:p.m; s.aWidth=p.ta?p.m:p.k;
        s.bRows=p.k; s.bWidth=p.n;
    } else return {};
    s.aPitch=CeilDiv(s.aWidth,16)*16; s.bPitch=CeilDiv(s.bWidth,16)*16;
    s.aElements=s.aRows*s.aPitch; s.bElements=s.bRows*s.bPitch;
    // Input queues + decoded FP32 inputs + 256/64 float work + 2x32-byte buffers.
    s.ubBytes=6*(s.aElements+s.bElements)+1344;
    if (caps.ubBytes<=32768 || s.ubBytes>std::min<uint64_t>(65536,caps.ubBytes-32768)) return {};
    // Compare legal column tiles by vector issue count, then scratch size. This
    // is a structural estimate, not a calibrated latency or cross-family model.
    // Keep legacy admission above: extra scratch must never eject an old small case.
    if (s.variant == SmallVariant::Dot && p.n > 1) {
        const uint64_t limit = std::min<uint64_t>(65536,caps.ubBytes-32768);
        const uint32_t chunks = CeilDiv(p.k,64);
        uint32_t bestCalls = std::numeric_limits<uint32_t>::max();
        for (uint32_t width=8; width<=64; width*=2) {
            const uint32_t columns = std::min(p.n,width);
            const uint32_t products = columns*64;
            const uint32_t partials = CeilDiv(columns,8)*8;
            const uint32_t used = 6*(s.aElements+s.bElements)+4*(products+partials)+288;
            const uint32_t calls = CeilDiv(p.n,columns)*(3*chunks-1)+1;
            if (used<=limit && (calls<bestCalls || (calls==bestCalls && used<s.ubBytes))) {
                bestCalls=calls; s.dotColumns=columns;
                s.productElements=products; s.partialElements=partials; s.ubBytes=used;
            }
            if (columns==p.n) break;
        }
    }
    s.tasks=static_cast<uint32_t>(groups);
    s.blocks=std::min(s.tasks,caps.cores);
    return s;
}
// R9 work budgets retained in R10; candidate eligibility now follows storage capacity.
constexpr uint32_t SMALL_AUTO_DOT_LIMIT = 128;  // complete dot products per group
constexpr uint32_t SMALL_AUTO_ROWS_LIMIT = 256; // N-vector updates per group
inline bool UseSmallAutomatically(const ProblemDesc& p, const SmallPlan& s)
{
    if(s.variant==SmallVariant::None || s.tasks>s.blocks) return false;
    const uint64_t owned=std::min<uint64_t>(p.batches,8);
    // Experimental coverage limits; no measured performance break-even claimed.
    return s.variant==SmallVariant::Dot ? owned*p.m*p.n<=SMALL_AUTO_DOT_LIMIT
                                       : owned*p.m*p.k<=SMALL_AUTO_ROWS_LIMIT;
}
inline bool SelectSmall(const ProblemDesc& p, const SmallPlan& s, TileRequest tile, ExecutionFamily family)
{
    if (family == ExecutionFamily::Small) {
        if (tile.m || tile.n) throw std::runtime_error("small family conflicts with fixed Matmul tile");
        if (s.variant == SmallVariant::None) throw std::runtime_error("unsupported forced small problem");
        return true;
    }
    if (family == ExecutionFamily::Gm) return false;
    if (family != ExecutionFamily::Auto) throw std::runtime_error("invalid execution family");
    return !tile.m && !tile.n && UseSmallAutomatically(p, s);
}
} // namespace local_baseline
#endif
