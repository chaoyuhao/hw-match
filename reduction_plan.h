#ifndef CANN_MATCH_REDUCTION_PLAN_H
#define CANN_MATCH_REDUCTION_PLAN_H
#include "matmul_plan.h"
namespace local_baseline {
// GM: 32x256 input + 32 output + Max temporary + 14368-byte final Sum.
// For N>256 the temporary is 32x64 lanes instead of 32 compact maxima.
inline uint32_t GmReductionUbBase(uint32_t n) { return n > 256 ? 55456 : 47392; }
enum class ReductionPolicy : uint32_t { Auto = 0, Rows = 1, Partials = 2 };
// mode=0: row maxima; mode=1: 64B record containing eight high/low pairs.
struct ReductionPlan {
    uint32_t mode = 0, segmentRows = 0, segments = 0, foldUbBytes = 0;
    uint64_t bytes = 0;
};
inline void ValidateReductionPolicy(ReductionPolicy p)
{
    if(p!=ReductionPolicy::Auto && p!=ReductionPolicy::Rows && p!=ReductionPolicy::Partials)
        throw std::runtime_error("invalid reduction policy");
}
inline ReductionPlan MakeReductionPlan(const ProblemDesc& p,uint32_t span,ReductionPolicy policy)
{
    ValidateReductionPolicy(policy);
    if(!p.batches || !p.m || p.m>8192 || !span || span>256 || span%16 || policy==ReductionPolicy::Auto)
        throw std::runtime_error("invalid reduction geometry or unresolved policy");
    ReductionPlan r;r.mode=policy==ReductionPolicy::Partials;r.segmentRows=span;r.segments=CeilDiv(p.m,span);
    r.bytes=PlanProduct(p.batches,r.mode?uint64_t(r.segments)*64:uint64_t(CeilDiv(p.m,32))*128);
    if(r.bytes>std::numeric_limits<size_t>::max())throw std::runtime_error("reduction exceeds address space");
    if(r.mode){uint32_t capacity=8;while(capacity<span)capacity*=2;r.foldUbBytes=14*capacity+64;}
    return r;
}
inline double PartialFoldCost(uint32_t span)
{
    uint32_t width=8;while(width<span)width*=2;
    double work=2*CeilDiv(width,64)+2; // initialize/copy and pack the record
    for(uint32_t half=width/2;half>=8;half/=2)work+=9*CeilDiv(half,64);
    return work;
}
} // namespace local_baseline
#endif
