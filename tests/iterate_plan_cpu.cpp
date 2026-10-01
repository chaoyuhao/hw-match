#include "iterate_plan.h"
#include <cassert>
int main() {
 using namespace local_baseline;
 for(unsigned cores:{1u,7u,24u,48u})for(unsigned ub:{96u,192u,256u})
 for(unsigned b:{1u,9u,257u})for(unsigned m:{1u,17u,255u,8192u})
 for(unsigned n:{1u,31u,129u,8192u})for(bool partial:{false,true}) {
  ProblemDesc p{b,m,n,33,2,true,true};HardwareCaps h{cores,ub*1024};
  ExecutionPlan ref;ref.matmul=MakePlan(p,h,{64,128});
  ref.reduction=MakeReductionPlan(p,32,partial?ReductionPolicy::Partials:ReductionPolicy::Rows);
  auto plan=MakeIteratePlan(p,h,ref);const auto& s=plan.stream;
  assert(s.reduction.mode==ref.reduction.mode && !s.maximaOffset && !s.cSlotElements);
  assert(s.buffers==1 && s.plannerVersion==4);
  assert(s.blocks==std::min<uint64_t>(h.cores,s.tasks));
  assert(s.ubBytes+ref.matmul.ubBudget<=h.ubBytes);
  unsigned next=0,maxWidth=0;
  for(unsigned shard=0;shard<s.splits;++shard){
   auto first=CeilDiv(n,s.tileN)*shard/s.splits*s.tileN;
   auto last=std::min(n,CeilDiv(n,s.tileN)*(shard+1)/s.splits*s.tileN);
   assert(first==next && last>first);next=last;maxWidth=std::max(maxWidth,last-first);
  }
  assert(next==n && maxWidth==plan.maxColumns);
  assert(AcceptIterateTile(plan,p,h,ref.matmul,plan.baseM,plan.baseN));
  const auto before=plan;
  for(auto bad: {TileRequest{0,64},TileRequest{17,64},TileRequest{256,256},TileRequest{8192,8192}})
   assert(!AcceptIterateTile(plan,p,h,ref.matmul,bad.m,bad.n));
  assert(plan.stream.scratchBytes==before.stream.scratchBytes && plan.baseM==before.baseM);
  if(s.blocks<h.cores)assert(s.splits==CeilDiv(n,s.tileN));
  ref.family=ExecutionFamily::Pipeline;
  ref.stream=SplitStreamPlan(p,h,ref.matmul,CeilDiv(n,128),2,partial?ReductionPolicy::Partials:ReductionPolicy::Rows);
  auto keep=MakeIteratePlan(p,h,ref);
  assert(keep.stream.splits==ref.stream.splits && keep.stream.blocks==ref.stream.blocks && keep.stream.tasks==ref.stream.tasks);
 }
 // Application scratch no longer depends on N except for the bounded shard count.
 ProblemDesc p{2,8192,8192,256,1,false,false};HardwareCaps h{24,192*1024};
 ExecutionPlan r;r.matmul=MakePlan(p,h,{128,128});r.reduction=MakeReductionPlan(p,32,ReductionPolicy::Partials);
 auto a=MakeIteratePlan(p,h,r);assert(a.stream.splits==1);
 assert(a.stream.scratchBytes==2*64*64 && a.stream.reduction.segmentRows==128);
 assert(a.baseM*a.baseN*4<=32768 && a.maxColumns==8192);
 // Very large batch counts must remain bounded host work and checked arithmetic.
 p={uint64_t(1)<<40,16,16,16,1,false,false};h={48,192*1024};
 r.matmul=MakePlan(p,h,{16,16});r.reduction=MakeReductionPlan(p,32,ReductionPolicy::Rows);
 auto large=MakeIteratePlan(p,h,r);
 assert(large.stream.tasks==(uint64_t(1)<<40) && large.stream.blocks==48);
 assert(large.stream.scratchBytes==(uint64_t(1)<<47));
 assert(large.stream.maxTilesPerCore==((uint64_t(1)<<40)+47)/48);
 bool caught=false;p.batches=std::numeric_limits<uint64_t>::max();
 try{MakeIteratePlan(p,h,r);}catch(const std::runtime_error&){caught=true;}
 assert(caught);
 p.batches=1;r.matmul.tileN=0;caught=false;
 try{MakeIteratePlan(p,h,r);}catch(const std::runtime_error&){caught=true;}
 assert(caught);

}
