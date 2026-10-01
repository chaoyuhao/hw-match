"""Execute actual host dispatch/metadata with mocked launches; no CANN emulation."""
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import plan_metadata

class SmallDispatchTests(unittest.TestCase):
    def test_actual_dispatch_one_launch_no_small_scratch_and_writer(self):
        kernel=(ROOT/'kernel.asc').read_text()
        self.assertTrue('inline PreparedExecution PrepareExecution(' in kernel, 'joint planner not connected to launch')
        prepare=kernel[kernel.index('inline bool TryPrepareIterate('):kernel.index('// The returned owner')]
        host=kernel[kernel.index('inline DeviceBuffer RunKernel('):kernel.index('} // namespace local_baseline',kernel.index('inline DeviceBuffer RunKernel('))]
        host=re.sub(r'SmallVectorKernel<(half|bfloat16_t)><<<small.blocks, nullptr, stream>>>\(',r'MockSmallLaunch<\1>(small.blocks, ',host)
        runner=(ROOT/'local/runner.asc').read_text()
        writers=runner[runner.index('void WritePlan('):runner.index('} // namespace')]
        prefix=r'''
#include "small_plan.h"
#include "joint_plan.h"
#include "iterate_plan.h"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>
using half=int16_t;using bfloat16_t=uint16_t;using aclrtStream=void*;
struct TensorInfo{const int64_t* shape;int64_t numDims;int32_t dtype;};
struct TensorGroupInfo{const TensorInfo*tensors;int64_t numTensors;};
static int rejectFusion=0,fusionTilings=0;static bool rejectS2;static int rejectFirst;static std::vector<std::pair<unsigned,unsigned>> attempts;
static int launches,allocs,tilers;static uint32_t launchReduction;static size_t scratchBytes;static bool failSync,failTiler;
int aclrtSynchronizeStreamWithTimeout(aclrtStream,int){return failSync?1:0;}
namespace platform_ascendc {
struct PlatformAscendCManager{static PlatformAscendCManager*GetInstance(){static PlatformAscendCManager p;return &p;}size_t GetLibApiWorkSpaceSize(){return 32;}};
}
namespace AscendC {namespace tiling {struct TCubeTiling{int baseM=16,baseN=16,baseK=16,iterateOrder=0;};}}
namespace matmul_tiling {
enum class TPosition {GM,VECIN}; enum class CubeFormat {ND};enum class DataType{DT_FLOAT,DT_FLOAT16,DT_BF16};enum class MatrixTraverse{FIRSTM};
struct MatmulApiTiling {
 unsigned m=0,n=0,k=0,sm=0,sn=0,bm=0,bn=0;bool c=false,traverse=false;size_t budget=0;
 explicit MatmulApiTiling(platform_ascendc::PlatformAscendCManager&){}
 void SetAType(TPosition p,CubeFormat,DataType dtype,bool){assert(p==TPosition::GM && dtype!=DataType::DT_FLOAT);}
 void SetBType(TPosition p,CubeFormat f,DataType dtype,bool t){SetAType(p,f,dtype,t);}
 void SetBiasType(TPosition,CubeFormat,DataType){}
 int SetCType(TPosition p,CubeFormat,DataType t){assert(p==TPosition::VECIN && t==DataType::DT_FLOAT);c=true;return rejectFusion==5?-1:0;}
 void SetOrgShape(unsigned M,unsigned N,unsigned K){m=M;n=N;k=K;}
 void SetShape(unsigned M,unsigned N,unsigned K){assert(m && M<=m && N<=n && K==k);sm=M;sn=N;}
 int SetFixSplit(unsigned M,unsigned N,int K){assert(M%16==0 && N%16==0 && K==-1);bm=M;bn=N;return rejectFusion==6?-1:0;}
 int SetTraverse(MatrixTraverse){traverse=true;return rejectFusion==7?-1:0;}
 void SetBufferSpace(int a,int b,size_t bytes){assert(a==-1 && b==-1 && bytes<=128*1024);budget=bytes;}
 void EnableBias(bool yes){assert(!yes);}
 int GetTiling(AscendC::tiling::TCubeTiling& t){
   ++fusionTilings;assert(!launches && c && traverse && sm && sn && budget);
   t={int(bm),int(bn),16,0};
   if(rejectFusion==2)t.baseM=512;
   if(rejectFusion==3)t.iterateOrder=1;
   if(rejectFusion==4)t.baseK=0;
   return rejectFusion==1?-1:0;
 }
};}
namespace local_baseline {
constexpr uint32_t REDUCE_ROWS=32;
void Check(int e,const char*){if(e)throw std::runtime_error("sync failed");}
size_t Bytes(uint64_t x,uint64_t y,uint64_t z,size_t s){return PlanProduct(PlanProduct(PlanProduct(x,y),z),s);}
struct DeviceBuffer{void*data=nullptr;DeviceBuffer()=default;explicit DeviceBuffer(size_t bytes){if(!allocs)scratchBytes=bytes;++allocs;data=(void*)16;}};
using FakeTiling=AscendC::tiling::TCubeTiling;
struct PreparedMatmul{MatmulPlan plan{};HardwareCaps caps{};FakeTiling tiling{};};
PREPARED_STRUCTS
HardwareCaps QueryCaps(uint32_t cores){return {cores,192*1024};}
bool TryPrepare(ProblemDesc p,HardwareCaps c,TileRequest tile,PreparedMatmul& out){++tilers;attempts.push_back({tile.m,tile.n});out={MakePlan(p,c,tile),c,{int(tile.m),int(tile.n),16,0}};assert(launches==0);return !failTiler && tilers>rejectFirst && !(rejectS2&&tile.m==32&&tile.n==64);}
struct StreamCompletion{bool pending=false;StreamCompletion(aclrtStream,DeviceBuffer&,DeviceBuffer&){}void Finish(){Check(failSync,"sync");}};
template<typename T,typename...Args> void MockSmallLaunch(uint32_t blocks,Args...){assert(blocks>0);++launches;}
template<typename T,typename...Args> void Dispatch(Args...args){++launches;launchReduction=std::get<sizeof...(args)-1>(std::make_tuple(args...)).mode;}
template<typename T,typename...Args> void DispatchIterate(Args...args){++launches;launchReduction=std::get<sizeof...(args)-1>(std::make_tuple(args...)).stream.reduction.mode;}
template<typename T,typename...Args> void DispatchStream(Args...args){++launches;launchReduction=std::get<sizeof...(args)-1>(std::make_tuple(args...)).reduction.mode;}
'''
        structs=kernel[kernel.index('struct PreparedExecution :'):kernel.index('inline HardwareCaps QueryCaps')]
        config='\n'.join(line for line in kernel.splitlines() if line.startswith(('constexpr bool S2_UPSTREAM_CONTROL','constexpr bool ITERATE_MATMUL_MAX')))
        prefix=prefix.replace('PREPARED_STRUCTS',config+'\n'+structs)
        suffix=r'''
} // namespace local_baseline
void Require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
'''
        main=r'''
void run(uint64_t batches,uint32_t m,uint32_t n,uint32_t k,bool ta,bool tb,local_baseline::ExecutionFamily family,bool expectedSmall,bool fail=false,local_baseline::TileRequest tile={},int dtype=1,local_baseline::ReductionPolicy sumPolicy=local_baseline::ReductionPolicy::Auto,bool expand=true,bool s2=false){
 using namespace local_baseline;
 int64_t as[]={static_cast<int64_t>(batches),ta?k:m,ta?m:k},bs[]={static_cast<int64_t>(batches),tb?n:k,tb?k:n},ys[]={static_cast<int64_t>(batches)};
 TensorInfo ai{as,3,dtype},bi{bs,3,dtype},yi{ys,1,0};TensorGroupInfo ag{&ai,1},bg{&bi,1},yg{&yi,1};
 uint8_t a=0,b=0,y=0;launches=allocs=tilers=0;ExecutionInfo actual;bool caught=false;
 try{auto scratch=RunKernel(&a,ag,&b,bg,&y,yg,24,(void*)1,ta,tb,tile,&actual,family,sumPolicy,expand,s2,false);
     assert(!fail);assert(bool(scratch.data)==!expectedSmall);assert(actual.isSmall==expectedSmall);assert(!actual.s2Attempted);
     if(family==ExecutionFamily::Stream || family==ExecutionFamily::Pipeline){assert(actual.isStream);assert(actual.stream.buffers==(family==ExecutionFamily::Pipeline?2u:1u));}
     if(family==ExecutionFamily::Gm || (tile.m&&family==ExecutionFamily::Auto))assert(!actual.isStream);
     if(!expectedSmall && family==ExecutionFamily::Auto && !tile.m){
       ProblemDesc p{batches,m,n,k,static_cast<uint32_t>(dtype),ta,tb};const auto best=GenerateExecutionCandidates(p,actual.caps,{},ExecutionFamily::Auto,sumPolicy).plans[0];
       assert(actual.isStream==(best.family!=ExecutionFamily::Gm));assert(actual.plan.tileM==best.matmul.tileM && actual.plan.tileN==best.matmul.tileN);assert(actual.baselineExecutionScore==best.score || sumPolicy!=ReductionPolicy::Auto);
       if(sumPolicy==ReductionPolicy::Auto){assert(actual.expansionTracked && actual.expansionEnabled==expand);if(!expand)assert(actual.reduction.mode==best.reduction.mode && actual.executionScore==best.score);}
     }}
 catch(const std::runtime_error&){caught=true;assert(fail);}
 assert(caught==fail);
 if(fail){assert(launches==(failSync?1:0));if(failTiler)assert(tilers>=1&&tilers<=5);else assert(tilers==(!expectedSmall&&failSync?1:0));assert(allocs==(!expectedSmall&&failSync?2:0));return;}
 if(!expectedSmall){assert(actual.reduction.mode==launchReduction);
   if(sumPolicy!=ReductionPolicy::Auto)assert(actual.reduction.mode==(sumPolicy==ReductionPolicy::Partials));
   assert(scratchBytes==(actual.isStream?actual.stream.scratchBytes:uint64_t(batches)*m*((n+15)/16*16)*4+actual.reduction.bytes));
 }
 assert(launches==1);assert(allocs==(expectedSmall?0:2));assert(tilers==(expectedSmall?0:1));
 ProblemDesc p{batches,m,n,k,static_cast<uint32_t>(dtype),ta,tb};
 std::cout<<ExecutionJson(actual,p,family==ExecutionFamily::Auto?"auto":family==ExecutionFamily::Gm?"gm":family==ExecutionFamily::Stream?"stream":family==ExecutionFamily::Pipeline?"pipeline":"small","auto",sumPolicy==ReductionPolicy::Auto?(expand?"auto":"r15"):sumPolicy==ReductionPolicy::Rows?"rows":"partials")<<'\n';
}

void frozenSdkSelection(){
 using namespace local_baseline;HardwareCaps h{24,192*1024};launches=0;
 for(unsigned batch:{1u,192u})for(unsigned m:{33u,1024u})for(int skip=0;skip<=4;++skip){
  ProblemDesc p{batch,m,1024,256,1,false,false};rejectFirst=skip;
  PreparedExecution old,current;bool failedOld=false,failedNew=false;
  attempts.clear();tilers=0;try{old=PrepareExecution(p,h,{},ExecutionFamily::Auto,ReductionPolicy::Auto,false);}catch(const std::runtime_error&){failedOld=true;}
  auto oldAttempts=attempts;attempts.clear();tilers=0;
  try{current=PrepareExecution(p,h,{},ExecutionFamily::Auto,ReductionPolicy::Auto);}catch(const std::runtime_error&){failedNew=true;}
  assert(failedOld==failedNew && oldAttempts==attempts);
  if(failedOld)continue;
  assert(current.plan.tileM==old.plan.tileM && current.plan.tileN==old.plan.tileN && current.plan.blocks==old.plan.blocks && current.plan.tasks==old.plan.tasks);
  assert(current.plan.ubBudget==old.plan.ubBudget && current.isStream==old.isStream);
  assert(current.tiling.baseM==old.tiling.baseM && current.tiling.baseN==old.tiling.baseN && current.tiling.baseK==old.tiling.baseK);
  assert(current.stream.splits==old.stream.splits && current.stream.buffers==old.stream.buffers && current.stream.blocks==old.stream.blocks && current.stream.tasks==old.stream.tasks);
  assert(current.baselineReductionMode==old.reduction.mode && current.baselineExecutionScore==old.executionScore);
  if(batch==192 && m==1024 && skip==0)assert(!old.reduction.mode && current.reduction.mode);
 }
 rejectFirst=0;
}

local_baseline::ExecutionInfo invokeS2(const local_baseline::ProblemDesc& p,bool use,bool defaults=false){
 using namespace local_baseline;
 int64_t as[]={int64_t(p.batches),p.ta?p.k:p.m,p.ta?p.m:p.k},bs[]={int64_t(p.batches),p.tb?p.n:p.k,p.tb?p.k:p.n},ys[]={int64_t(p.batches)};
 TensorInfo ai{as,3,int32_t(p.dtype)},bi{bs,3,int32_t(p.dtype)},yi{ys,1,0};TensorGroupInfo ag{&ai,1},bg{&bi,1},yg{&yi,1};
 uint8_t a=0,b=0,y=0;ExecutionInfo actual;launches=allocs=tilers=0;attempts.clear();
 if(defaults)RunKernel(&a,ag,&b,bg,&y,yg,24,(void*)1,p.ta,p.tb,{},&actual);
 else RunKernel(&a,ag,&b,bg,&y,yg,24,(void*)1,p.ta,p.tb,{},&actual,ExecutionFamily::Auto,ReductionPolicy::Auto,true,use,false);
 assert(launches==1 && allocs==(actual.isSmall?0:2));
 if(!actual.isSmall)assert(scratchBytes==(actual.isStream?actual.stream.scratchBytes:p.batches*p.m*((p.n+15)/16*16)*4+actual.reduction.bytes));
 return actual;
}
void s2Control(){
 using namespace local_baseline;
 for(unsigned dtype:{1u,2u})for(bool ta:{false,true})for(bool tb:{false,true})
 for(auto dims:{std::array<unsigned,4>{1,17,65,32},std::array<unsigned,4>{3,33,129,65},std::array<unsigned,4>{192,1024,1024,256},std::array<unsigned,4>{1,8192,16,32},std::array<unsigned,4>{32,32,64,256}}){
   ProblemDesc p{dims[0],dims[1],dims[2],dims[3],dtype,ta,tb};
   const auto old=invokeS2(p,false);auto oldAttempts=attempts;
   const auto fixed=invokeS2(p,true);
   assert(!old.isSmall && !fixed.isSmall && !fixed.isStream);
   assert(!old.s2Attempted && fixed.s2Attempted && fixed.s2Selected && !fixed.expansionTracked);
   assert(fixed.plan.tileM==32 && fixed.plan.tileN==64);
   const uint64_t tasks=p.batches*((p.m+31)/32)*((p.n+63)/64);
   assert(fixed.plan.tasks==tasks && fixed.plan.blocks==std::min<uint64_t>(24,tasks));
   assert(fixed.tiling.baseM==32 && fixed.tiling.baseN==64);
   assert(fixed.reduction.mode==old.reduction.mode && fixed.reduction.segmentRows==32);
   assert(fixed.s2Reference.reduction.mode==old.reduction.mode && fixed.s2Reference.score==old.executionScore);
   assert(fixed.s2Reference.matmul.tileM==old.plan.tileM && fixed.s2Reference.matmul.tileN==old.plan.tileN);
   assert(fixed.s2Reference.family==(old.isStream?(old.stream.buffers==2?ExecutionFamily::Pipeline:ExecutionFamily::Stream):ExecutionFamily::Gm));
   assert(fixed.executionScore>0 && fixed.reduction.mode==launchReduction);
   assert(attempts.size()==oldAttempts.size()+((old.plan.tileM==32 && old.plan.tileN==64)?0:1));
   assert(std::equal(oldAttempts.begin(),oldAttempts.end(),attempts.begin()));
   std::cout<<ExecutionJson(fixed,p,"auto","auto")<<'\n';
 }
 // Rejection must preserve the accepted reference plan, inner tile and scratch.
 ProblemDesc p{192,1024,1024,256,1,false,false};rejectS2=true;
 auto old=invokeS2(p,false);const auto expectedScratch=scratchBytes;
 auto rejected=invokeS2(p,true);
 assert(rejected.s2Attempted && !rejected.s2Selected);
 assert(rejected.isStream==old.isStream && rejected.plan.tileM==old.plan.tileM && rejected.plan.tileN==old.plan.tileN);
 assert(rejected.tiling.baseM==old.tiling.baseM && rejected.tiling.baseN==old.tiling.baseN && scratchBytes==expectedScratch);
 assert(rejected.executionScore==old.executionScore && rejected.reduction.mode==old.reduction.mode && rejected.expansionTracked);
 std::cout<<ExecutionJson(rejected,p,"auto","auto")<<'\n';rejectS2=false;
 // Exercise the same default argument used by the online entry and local runner.
 auto configured=invokeS2(p,false,true);
 assert(configured.s2Attempted==S2_UPSTREAM_CONTROL);
 if(S2_UPSTREAM_CONTROL)assert(configured.s2Selected && !configured.isStream && configured.plan.tileM==32 && configured.plan.tileN==64);
 std::cout<<ExecutionJson(configured,p,"auto","auto")<<'\n';
 run(1,1,1,32,false,false,ExecutionFamily::Auto,true,false,{},1,ReductionPolicy::Auto,true,true);
 run(1,17,65,32,false,false,ExecutionFamily::Auto,false,false,{},1,ReductionPolicy::Auto,false,true);
 run(3,17,129,33,false,false,ExecutionFamily::Pipeline,false,false,{16,32},2,ReductionPolicy::Auto,true,true);
 run(3,17,129,33,false,false,ExecutionFamily::Auto,false,false,{32,64},1,ReductionPolicy::Auto,true,true);
 for(auto mode:{ReductionPolicy::Rows,ReductionPolicy::Partials})
  run(1,8192,16,32,false,false,ExecutionFamily::Auto,false,false,{},1,mode,true,true);
 // Tiling or execution failure must not trigger a second launch.
 failTiler=true;bool caught=false;
 try{invokeS2(p,true);}catch(const std::runtime_error&){caught=true;}
 assert(caught && launches==0 && allocs==0);failTiler=false;
 failSync=true;caught=false;
 try{invokeS2(p,true);}catch(const std::runtime_error&){caught=true;}
 assert(caught && launches==1);failSync=false;
}


local_baseline::ExecutionInfo invokeFusion(const local_baseline::ProblemDesc& p){
 using namespace local_baseline;
 int64_t as[]={int64_t(p.batches),p.ta?p.k:p.m,p.ta?p.m:p.k},bs[]={int64_t(p.batches),p.tb?p.n:p.k,p.tb?p.k:p.n},ys[]={int64_t(p.batches)};
 TensorInfo ai{as,3,int32_t(p.dtype)},bi{bs,3,int32_t(p.dtype)},yi{ys,1,0};TensorGroupInfo ag{&ai,1},bg{&bi,1},yg{&yi,1};
 uint8_t a=0,b=0,y=0;ExecutionInfo actual;launches=allocs=tilers=fusionTilings=0;
 RunKernel(&a,ag,&b,bg,&y,yg,24,(void*)1,p.ta,p.tb,{},&actual,ExecutionFamily::Auto,ReductionPolicy::Auto,true,false,true);
 assert(launches==1 && allocs==(actual.isSmall?0:2));
 return actual;
}
void fusionControl(){
 using namespace local_baseline;
 for(unsigned dtype:{1u,2u})for(bool ta:{false,true})for(bool tb:{false,true})
 for(auto dims:{std::array<unsigned,4>{1,17,65,32},std::array<unsigned,4>{3,33,129,65},std::array<unsigned,4>{192,1024,1024,256},std::array<unsigned,4>{1,8192,16,32}}){
   ProblemDesc p{dims[0],dims[1],dims[2],dims[3],dtype,ta,tb};
   const auto old=invokeS2(p,false);const auto fused=invokeFusion(p);
   assert(fused.isIterate && fused.isStream && fused.fusionAttempted && !fused.expansionTracked && !fused.executionPlanner);
   assert(fused.reduction.mode==old.reduction.mode && launchReduction==old.reduction.mode);
   assert(fused.fusionReference.matmul.tileM==old.plan.tileM && fused.fusionReference.matmul.tileN==old.plan.tileN);
   assert(scratchBytes==fused.stream.scratchBytes && !fused.stream.cSlotElements && !fused.stream.maximaOffset);
   assert(fusionTilings==1 && fused.tiling.baseM==int(fused.iterative.baseM) && fused.tiling.baseN==int(fused.iterative.baseN));
   std::cout<<ExecutionJson(fused,p,"auto","auto")<<'\n';
 }
 ProblemDesc p{192,1024,1024,256,1,false,false};const auto old=invokeS2(p,false);const auto oldScratch=scratchBytes;
 for(rejectFusion=1;rejectFusion<=7;++rejectFusion){
   const auto rejected=invokeFusion(p);
   assert(rejected.fusionAttempted && !rejected.isIterate && rejected.isStream==old.isStream && rejected.expansionTracked==old.expansionTracked);
   assert(rejected.tiling.baseM==old.tiling.baseM && rejected.tiling.baseN==old.tiling.baseN && scratchBytes==oldScratch);
   assert(rejected.reduction.mode==old.reduction.mode && rejected.executionScore==old.executionScore);
   assert(rejected.stream.splits==old.stream.splits && rejected.stream.buffers==old.stream.buffers);
   std::cout<<ExecutionJson(rejected,p,"auto","auto")<<'\n';
 }
 rejectFusion=0;
 for(auto family:{ExecutionFamily::Gm,ExecutionFamily::Stream,ExecutionFamily::Pipeline}){
  launches=tilers=fusionTilings=0;
  auto forced=PrepareExecution(p,{24,192*1024},{},family,ReductionPolicy::Auto,true,false,true);
  assert(!forced.fusionAttempted && !forced.isIterate && fusionTilings==0);
 }
 for(auto mode:{ReductionPolicy::Rows,ReductionPolicy::Partials}){
  launches=tilers=fusionTilings=0;
  auto explicitSum=PrepareExecution(p,{24,192*1024},{},ExecutionFamily::Auto,mode,true,false,true);
  assert(!explicitSum.fusionAttempted && fusionTilings==0);
 }
 launches=tilers=fusionTilings=0;
 auto fixed=PrepareExecution(p,{24,192*1024},{32,64},ExecutionFamily::Auto,ReductionPolicy::Auto,true,false,true);
 auto r15=PrepareExecution(p,{24,192*1024},{},ExecutionFamily::Auto,ReductionPolicy::Auto,false,false,true);
 assert(!fixed.fusionAttempted && !r15.fusionAttempted && !fusionTilings);
 auto tiny=invokeFusion({1,1,1,32,1,false,false});assert(tiny.isSmall && !tiny.fusionAttempted && fusionTilings==0);
 failSync=true;bool caught=false;try{invokeFusion(p);}catch(const std::runtime_error&){caught=true;}
 assert(caught && launches==1);failSync=false;
 failTiler=true;caught=false;try{invokeFusion(p);}catch(const std::runtime_error&){caught=true;}
 assert(caught && launches==0 && fusionTilings==0);failTiler=false;
}

int main(){using local_baseline::ExecutionFamily;using local_baseline::ReductionPolicy;
 s2Control();
 fusionControl();
 assert(SumPolicy("r15")==ReductionPolicy::Auto);
 frozenSdkSelection();
 for(int dtype:{1,2})for(bool expand:{false,true}) {
  run(192,1024,1024,256,false,false,ExecutionFamily::Auto,false,false,{},dtype,ReductionPolicy::Auto,expand);
  run(1,1,1,32,false,false,ExecutionFamily::Auto,true,false,{},dtype,ReductionPolicy::Auto,expand);
 }

 run(1,1,1,32,false,false,ExecutionFamily::Auto,true);
 run(1,1,5,8,true,false,ExecutionFamily::Auto,true);
 run(1,17,65,32,false,false,ExecutionFamily::Auto,false);
 run(1,3,5,8,true,true,ExecutionFamily::Auto,false);
 run(1,1,1,32,false,false,ExecutionFamily::Gm,false);
 run(1,1,1,32,false,false,ExecutionFamily::Auto,false,false,{32,64});
 run(1,3,5,8,true,true,ExecutionFamily::Small,false,true);
 run(1,1,1,32,false,false,ExecutionFamily::Small,false,true,{32,64});
 run(33,1,16,8,false,true,ExecutionFamily::Auto,true);
 run(1,2,16,64,false,true,ExecutionFamily::Auto,true);
 run(1,1,64,128,false,true,ExecutionFamily::Auto,true);
 run(192,1,1,8,false,false,ExecutionFamily::Auto,true);
 run(193,1,1,8,false,false,ExecutionFamily::Auto,false);
 run(1,17,1,8,false,false,ExecutionFamily::Auto,true);
 run(1,32,5,8,true,false,ExecutionFamily::Auto,true);
 run(1,1,5,256,true,false,ExecutionFamily::Auto,true);
 run(1,1,64,128,false,false,ExecutionFamily::Auto,true);
 run(1,1,64,256,false,false,ExecutionFamily::Auto,false);
 for(int dtype:{1,2})for(bool ta:{false,true})for(bool tb:{false,true})
  run(3,17,129,33,ta,tb,ExecutionFamily::Stream,false,false,{16,32},dtype);
 for(int dtype:{1,2})for(bool ta:{false,true})for(bool tb:{false,true})
  run(3,17,129,33,ta,tb,ExecutionFamily::Pipeline,false,false,{16,32},dtype);
 for(int dtype:{1,2})for(auto f:{ExecutionFamily::Auto,ExecutionFamily::Gm,ExecutionFamily::Stream,ExecutionFamily::Pipeline})
 for(auto policy:{ReductionPolicy::Rows,ReductionPolicy::Partials})for(unsigned m:{1u,8192u})
  run(1,m,1,32,false,false,f,false,false,{},dtype,policy);
 for(auto policy:{ReductionPolicy::Rows,ReductionPolicy::Partials})run(1,1,1,32,false,false,ExecutionFamily::Small,false,true,{},1,policy);
 failTiler=true;run(3,17,129,33,false,false,ExecutionFamily::Stream,false,true);failTiler=false;
 failSync=true;run(3,17,129,33,false,false,ExecutionFamily::Stream,false,true);failSync=false;
 failTiler=true;run(3,17,129,33,false,false,ExecutionFamily::Pipeline,false,true);failTiler=false;
 failSync=true;run(3,17,129,33,false,false,ExecutionFamily::Pipeline,false,true);failSync=false;
 failSync=true;run(1,1,1,32,false,false,ExecutionFamily::Auto,true,true);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'test.cpp';binary=tmp/'test'
            source.write_text((prefix+prepare+host+suffix+writers+main).replace('constexpr bool S2_UPSTREAM_CONTROL = false;', 'constexpr bool S2_UPSTREAM_CONTROL = true;'))
            done=subprocess.run(['/usr/bin/g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            rows=[json.loads(line) for line in done.stdout.splitlines()]
            self.assertEqual(len(rows),159)
            self.assertTrue(any(row.get('upstream_control',{}).get('status')=='selected' and
                                (row['upstream_control']['reference_tile_m'],row['upstream_control']['reference_tile_n'])==(32,64)
                                for row in rows))
            self.assertTrue({16,32} <= {row.get('dot_columns') for row in rows})
            for row in rows:
                self.assertEqual(plan_metadata.validate_execution(row,row['problem']),row)

            fused=next(row for row in rows if row['family']=='iterate')
            for section, fields in [('stream',dict(c_slot_elements=32)),('stream',dict(maxima_offset=4)),
                                    ('stream',dict(ub_used=fused['stream']['ub_used']+32)),
                                    ('stream',dict(partial_offset=fused['stream']['partial_offset']+32)),
                                    ('iterate',dict(version=True)),('iterate',dict(c_position='gm')),
                                    ('iterate',dict(max_columns=0)),('iterate',dict(sessions=1)),
                                    ('matmul_max_fusion',dict(status='sdk_rejected')),
                                    ('matmul_max_fusion',dict(reference_score=0)),
                                    ('matmul_max_fusion',dict(reference_splits=-1))]:
                bad=json.loads(json.dumps(fused));bad[section].update(fields)
                with self.assertRaises(ValueError):plan_metadata.validate_execution(bad,bad['problem'])
            for section in ('iterate','matmul_max_fusion','reduction'):
                bad=json.loads(json.dumps(fused));del bad[section]
                with self.assertRaises(ValueError):plan_metadata.validate_execution(bad,bad['problem'])

            selected=next(row for row in rows if row.get('upstream_control',{}).get('status')=='selected')
            rejected=next(row for row in rows if row.get('upstream_control',{}).get('status')=='sdk_rejected')
            for original,changes in [(selected,dict(version=True)),(selected,dict(status='guess')),
                                     (selected,dict(reference_tile_m=17)),(selected,dict(reference_score=-1)),
                                     (selected,dict(reference_reduction='invalid')),
                                     (rejected,dict(reference_score=rejected['selection']['score']+1))]:
                bad=json.loads(json.dumps(original));bad['upstream_control'].update(changes)
                with self.assertRaises(ValueError):plan_metadata.validate_execution(bad,bad['problem'])
            bad=json.loads(json.dumps(selected));bad['upstream_control']=None
            with self.assertRaises(ValueError):plan_metadata.validate_execution(bad,bad['problem'])
            bad=json.loads(json.dumps(selected));bad['requested_family']='gm'
            with self.assertRaises(ValueError):plan_metadata.validate_execution(bad,bad['problem'])
            bad=json.loads(json.dumps(selected));bad['upstream_control']['reference_reduction']='rows' if bad['reduction']['mode']=='partials' else 'partials'
            with self.assertRaises(ValueError):plan_metadata.validate_execution(bad,bad['problem'])

            source.write_text((prefix+prepare+host+suffix+writers+main).replace(
                'constexpr bool S2_UPSTREAM_CONTROL = true;', 'constexpr bool S2_UPSTREAM_CONTROL = false;'))
            done=subprocess.run(['/usr/bin/g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            restored=[json.loads(line) for line in done.stdout.splitlines()]
            self.assertEqual(len(restored),len(rows))
            changes=[(a,b) for a,b in zip(rows,restored) if a!=b]
            self.assertEqual(len(changes),1)  # Only the default-argument invocation changes.
            self.assertNotIn('upstream_control',changes[0][1])
            for row in restored:self.assertEqual(plan_metadata.validate_execution(row,row['problem']),row)
