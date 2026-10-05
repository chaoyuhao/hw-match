#ifndef CANN_MATCH_PIPELINE_PLAN_H
#define CANN_MATCH_PIPELINE_PLAN_H
#include "direct_plan.h"
#include "stream_plan.h"
namespace local_baseline {
constexpr bool PIPELINE_CORE_ENABLED = true;
// Physical allocations, not SDK tiling hints. Byte fields include both slots.
struct PipelinePlan {
    bool enabled = false, residentA = false;
    uint32_t blockK = 0, paddedK = 0;
    uint32_t a1Bytes = 0, b1Bytes = 0, a0Bytes = 0, b0Bytes = 0, c0Bytes = 0;
    StreamPlan stream{};
    double score = 0;
};
inline PipelinePlan PipelineCandidate(const ProblemDesc& p, const HardwareCaps& h,
    const DirectCaps& d, uint32_t bm, uint32_t bn, uint32_t splits, ReductionPolicy policy)
{
    PipelinePlan q{};
    if (!d.supported || !h.cores || !splits ||
        (bm!=16 && bm!=32 && bm!=64 && bm!=128) || (bn!=32 && bn!=64 && bn!=128) ||
        splits > CeilDiv(p.n, bn)) return q;
    const uint64_t l1=std::min<uint64_t>(d.l1,512*1024);
    const uint64_t la=std::min<uint64_t>(d.l0a,64*1024), lb=std::min<uint64_t>(d.l0b,64*1024);
    if (uint64_t(bm)*bn*8 > std::min<uint64_t>(d.l0c,128*1024)) return q;
    q.paddedK=CeilDiv(p.k,16)*16;
    q.blockK=std::min(128u,q.paddedK);
    while (q.blockK>=16 && (uint64_t(bm)*q.blockK*4>la || uint64_t(bn)*q.blockK*4>lb ||
           uint64_t(bm+bn)*q.blockK*4>l1)) q.blockK=(q.blockK/2)/16*16;
    if (!q.blockK) return {};
    q.a0Bytes=bm*q.blockK*4; q.b0Bytes=bn*q.blockK*4; q.c0Bytes=bm*bn*8;
    q.b1Bytes=q.b0Bytes;
    q.residentA=uint64_t(bm)*q.paddedK*2+q.b1Bytes<=l1;
    q.a1Bytes=q.residentA?bm*q.paddedK*2:q.a0Bytes;
    // Reuse only the proven ownership/scratch arithmetic, with no SDK budget
    // and no old candidate selection. Physical C slots include padded M rows.
    MatmulPlan geometry{bm,bn,0,0,0,0};
    q.stream=SplitStreamPlan(p,h,geometry,splits,2,policy);
    auto& s=q.stream;
    s.plannerVersion=5;
    const uint64_t extra=PlanProduct(uint64_t(s.blocks)*2,(uint64_t(bm)-std::min(p.m,bm))*bn*4);
    s.cSlotElements=uint64_t(bm)*bn;
    s.maximaOffset=StreamAdd(s.maximaOffset,extra);
    s.partialOffset=StreamAdd(s.partialOffset,extra);
    s.scratchBytes=StreamAdd(s.scratchBytes,extra);
    // Two 32x256 UB input stripes; no Matmul server reserve.
    s.ubBytes+=32*256*4;
    if (s.ubBytes>h.ubBytes || s.scratchBytes>std::numeric_limits<size_t>::max()) return {};
    const double ownerWaves=double(s.tasks/s.blocks+(s.tasks%s.blocks!=0));
    const double tiles=double(s.maxTilesPerCore);
    const double macs=tiles*bm*bn*q.paddedK;
    const double input=2.0*q.paddedK*(tiles*bn+(q.residentA?ownerWaves:tiles)*bm);
    const double cTraffic=tiles*bm*bn*8.0;
    // Dimensionless throughput proxies, not measured device cycles. Compare
    // concurrent engines by max; count ownership and panel overhead separately.
    q.score=std::max(macs/4096.0,std::max(input/64.0,cTraffic/64.0))+
        tiles*(12.0+4.0*CeilDiv(p.k,q.blockK))+ownerWaves*32.0+
        double(s.reduction.bytes)/(128.0*s.blocks)+(splits>1?double(p.batches)*s.rowPitch*splits*4/(64.0*s.blocks):0);
    q.enabled=true;
    return q;
}
inline PipelinePlan MakePipelinePlan(const ProblemDesc& p, const HardwareCaps& h,
                                     const DirectCaps& d, ReductionPolicy policy=ReductionPolicy::Auto)
{
    ValidateProblem(p,h); ValidateReductionPolicy(policy);
    PipelinePlan best{};
    if (!d.supported) return best;
    for(uint32_t bm=16;bm<=128;bm*=2) for(uint32_t bn=32;bn<=128;bn*=2) {
        const uint64_t rowTasks=PlanProduct(p.batches,CeilDiv(p.m,bm));
        const uint32_t columns=CeilDiv(p.n,bn);
        const uint32_t needed=rowTasks>=h.cores?1:CeilDiv(h.cores,static_cast<uint32_t>(rowTasks));
        // N splitting exists to fill idle cores; never creates empty shards.
        const uint32_t candidates[]={1,std::min(columns,needed),std::min(columns,needed+1)};
        for(uint32_t splits:candidates) {
            const auto sum=policy==ReductionPolicy::Auto?ReductionPolicy::Partials:policy;
            auto q=PipelineCandidate(p,h,d,bm,bn,splits,sum);
            if(q.enabled && (!best.enabled || q.score<best.score ||
               (q.score==best.score && q.stream.splits<best.stream.splits))) best=q;
        }
    }
    return best;
}
}
#endif
