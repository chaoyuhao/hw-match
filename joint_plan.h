#ifndef CANN_MATCH_JOINT_PLAN_H
#define CANN_MATCH_JOINT_PLAN_H
#include "small_plan.h"
#include "stream_plan.h"

namespace local_baseline {
// Host-only joint planning. Scores are abstract work, NOT cycles or microseconds.
constexpr uint32_t EXECUTION_PLANNER_VERSION = 2;
struct ExecutionPlan {
    ExecutionFamily family = ExecutionFamily::Gm;
    MatmulPlan matmul{};
    StreamPlan stream{};
    double score = 0;
};
struct ExecutionCandidates { std::array<ExecutionPlan,192> plans{}; size_t count = 0; };
struct SplitCandidates { std::array<uint32_t,16> values{}; size_t count = 0; };
inline uint64_t ExecutionWaves(uint64_t tasks, uint32_t cores)
{ return tasks / cores + (tasks % cores != 0); }
inline SplitCandidates StreamSplitCandidates(const ProblemDesc& p, const HardwareCaps& h, const MatmulPlan& mm)
{
    const uint32_t limit = CeilDiv(p.n, mm.tileN);
    const uint64_t rows = PlanProduct(p.batches, CeilDiv(p.m, mm.tileM));
    const uint32_t needed = static_cast<uint32_t>(ExecutionWaves(h.cores, static_cast<uint32_t>(std::min<uint64_t>(rows,h.cores))));
    SplitCandidates result;
    auto add = [&](uint32_t value) {
        value = std::min(limit,std::max(1U,value));
        if (std::find(result.values.begin(),result.values.begin()+result.count,value)==result.values.begin()+result.count)
            result.values[result.count++] = value;
    };
    add(1); add(limit); add(needed-1); add(needed); add(needed+1);
    for(uint32_t s=2;s<=limit;s*=2) add(s);
    return result;
}
inline double VectorMaxCost(uint32_t rows, uint32_t columns)
{
    // 64-float reduction groups, DMA/queue setup per 32x256 chunk, C bytes / 256.
    return double(rows)*CeilDiv(columns,64) +
           double(CeilDiv(rows,32))*(8*CeilDiv(columns,256)+2*CeilDiv(columns,64)) +
           double(rows)*columns/64;
}
inline double ExecutionCost(const ProblemDesc& p, const ExecutionPlan& e)
{
    const auto& mm=e.matmul;
    const uint32_t rows=std::min(p.m,mm.tileM), cols=std::min(p.n,mm.tileN);
    // Padded Cube work + input traffic/layout + per-call setup. Symbolic weights
    // are deliberately explicit: ranking remains an uncalibrated hypothesis.
    const double cube=double(CeilDiv(rows,16))*CeilDiv(cols,16)*CeilDiv(p.k,16) +
        2.0*p.k*(rows*(p.ta?1.25:1.0)+cols*(p.tb?1.0:1.25))/256 + 64;
    const uint32_t blocks=e.family==ExecutionFamily::Gm?mm.blocks:e.stream.blocks;
    const uint64_t sumTasks=p.batches/8+(p.batches%8!=0);
    const double sum=double(ExecutionWaves(sumTasks,blocks))*std::min<uint64_t>(8,p.batches)*
                     (8.0*CeilDiv(p.m,64)+16);
    constexpr double barrier=64; // fixed structural synchronization proxy
    if(e.family==ExecutionFamily::Gm) {
        const uint64_t maxTasks=PlanProduct(p.batches,CeilDiv(p.m,32));
        return double(ExecutionWaves(mm.tasks,blocks))*cube +
            double(ExecutionWaves(maxTasks,blocks))*VectorMaxCost(std::min(p.m,32U),p.n) + 2*barrier + sum;
    }
    const auto& s=e.stream;
    const double vector=VectorMaxCost(rows,cols);
    const double taskWaves=double(ExecutionWaves(s.tasks,s.blocks));
    // Upper load estimate: max call count and max task count may belong to
    // different cores. Pipeline pays warmup/drain once per row/shard task.
    double cost=s.buffers==2 ? double(s.maxTilesPerCore)*std::max(cube,vector)+taskWaves*std::min(cube,vector)
                            : double(s.maxTilesPerCore)*(cube+vector);
    cost+=taskWaves*(16+double(rows)/8)+barrier+sum;
    if(s.splits>1) {
        const uint64_t mergeTasks=PlanProduct(p.batches,CeilDiv(p.m,32));
        cost+=double(ExecutionWaves(mergeTasks,s.blocks))*(s.splits*(8+double(std::min(p.m,32U))/8)+8)+barrier;
    }
    return cost;
}
inline bool ExecutionLess(const ExecutionPlan& a,const ExecutionPlan& b)
{
    if(a.score!=b.score) return a.score<b.score;
    if(a.family!=b.family) return static_cast<uint32_t>(a.family)<static_cast<uint32_t>(b.family);
    if(a.matmul.tileM!=b.matmul.tileM) return a.matmul.tileM<b.matmul.tileM;
    if(a.matmul.tileN!=b.matmul.tileN) return a.matmul.tileN<b.matmul.tileN;
    return a.stream.splits<b.stream.splits;
}
inline void AddExecutionGeometry(ExecutionCandidates& out,const ProblemDesc& p,const HardwareCaps& h,
                                 const MatmulPlan& mm,ExecutionFamily policy)
{
    if(policy==ExecutionFamily::Auto || policy==ExecutionFamily::Gm) {
        ExecutionPlan e; e.matmul=mm; e.score=ExecutionCost(p,e); out.plans[out.count++]=e;
    }
    const auto splits=StreamSplitCandidates(p,h,mm);
    for(uint32_t buffers=1;buffers<=2;++buffers) {
        const auto family=buffers==1?ExecutionFamily::Stream:ExecutionFamily::Pipeline;
        if(policy!=ExecutionFamily::Auto && policy!=family) continue;
        bool found=false; ExecutionPlan best;
        for(size_t i=0;i<splits.count;++i) {
            // Automatic double buffering requires overlap in EVERY shard.
            if(buffers==2 && policy==ExecutionFamily::Auto && CeilDiv(p.n,mm.tileN)/splits.values[i]<2) continue;
            ExecutionPlan e; e.family=family;e.matmul=mm;
            e.stream=SplitStreamPlan(p,h,mm,splits.values[i],buffers);
            e.stream.plannerVersion=EXECUTION_PLANNER_VERSION;
            e.score=ExecutionCost(p,e);
            if(!found || ExecutionLess(e,best)){best=e;found=true;}
        }
        if(found) out.plans[out.count++]=best;
    }
}
inline ExecutionCandidates GenerateExecutionCandidates(const ProblemDesc& p,const HardwareCaps& h,
                                                        TileRequest tile={},ExecutionFamily policy=ExecutionFamily::Auto)
{
    if(policy!=ExecutionFamily::Auto && policy!=ExecutionFamily::Gm &&
       policy!=ExecutionFamily::Stream && policy!=ExecutionFamily::Pipeline)
        throw std::runtime_error("invalid joint execution policy");
    ExecutionCandidates out;
    if(tile.m || tile.n) AddExecutionGeometry(out,p,h,MakePlan(p,h,tile),policy);
    else {
        const auto geometries=GenerateCandidates(p,h);
        for(size_t i=0;i<geometries.count;++i) AddExecutionGeometry(out,p,h,geometries.plans[i],policy);
    }
    std::sort(out.plans.begin(),out.plans.begin()+out.count,ExecutionLess);
    return out;
}
// accept prepares SDK tiling only. Never launch/retry device work here.
template<typename Accept>
inline ExecutionPlan FindSupportedExecutionPlan(const ProblemDesc& p,const HardwareCaps& h,
                                                TileRequest tile,ExecutionFamily policy,Accept accept)
{
    if(policy==ExecutionFamily::Gm || policy==ExecutionFamily::Stream ||
       (policy==ExecutionFamily::Auto && (tile.m || tile.n))) {
        ExecutionPlan e;
        e.matmul=FindSupportedPlan(p,h,tile,accept);
        if(policy==ExecutionFamily::Stream){e.family=policy;e.stream=MakeStreamPlan(p,h,e.matmul);}
        e.score=ExecutionCost(p,e); return e;
    }
    const auto candidates=GenerateExecutionCandidates(p,h,tile,policy);
    std::array<TileRequest,4> tried{};size_t count=0;
    for(size_t i=0;i<candidates.count && count<tried.size();++i) {
        const auto& e=candidates.plans[i];
        bool duplicate=false;
        for(size_t j=0;j<count;++j) duplicate|=tried[j].m==e.matmul.tileM && tried[j].n==e.matmul.tileN;
        if(duplicate) continue;
        tried[count++]={e.matmul.tileM,e.matmul.tileN};
        if(accept(e.matmul)) return e;
    }
    if(tile.m || tile.n) throw std::runtime_error("requested execution tile rejected by SDK");
    bool triedFallback=false;
    for(size_t i=0;i<count;++i) triedFallback|=tried[i].m==32 && tried[i].n==64;
    if(!triedFallback) {
        const auto fallback=GenerateExecutionCandidates(p,h,{32,64},policy).plans[0];
        if(accept(fallback.matmul)) return fallback;
    }
    throw std::runtime_error("joint execution candidates rejected by SDK");
}
} // namespace local_baseline
#endif
