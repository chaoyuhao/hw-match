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
        prepare=kernel[kernel.index('inline PreparedExecution PrepareExecution('):kernel.index('// The returned owner')]
        host=kernel[kernel.index('inline DeviceBuffer RunKernel('):kernel.index('} // namespace local_baseline',kernel.index('inline DeviceBuffer RunKernel('))]
        host=re.sub(r'SmallVectorKernel<(half|bfloat16_t)><<<small.blocks, nullptr, stream>>>\(',r'MockSmallLaunch<\1>(small.blocks, ',host)
        runner=(ROOT/'local/runner.asc').read_text()
        writers=runner[runner.index('void WritePlan('):runner.index('} // namespace')]
        prefix=r'''
#include "small_plan.h"
#include "joint_plan.h"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <vector>
using half=int16_t;using bfloat16_t=uint16_t;using aclrtStream=void*;
struct TensorInfo{const int64_t* shape;int64_t numDims;int32_t dtype;};
struct TensorGroupInfo{const TensorInfo*tensors;int64_t numTensors;};
static int rejectFirst;static std::vector<std::pair<unsigned,unsigned>> attempts;
static int launches,allocs,tilers;static uint32_t launchReduction;static size_t scratchBytes;static bool failSync,failTiler;
int aclrtSynchronizeStreamWithTimeout(aclrtStream,int){return failSync?1:0;}
namespace platform_ascendc {
struct PlatformAscendCManager{static PlatformAscendCManager*GetInstance(){static PlatformAscendCManager p;return &p;}size_t GetLibApiWorkSpaceSize(){return 32;}};
}
namespace local_baseline {
constexpr uint32_t REDUCE_ROWS=32;
void Check(int e,const char*){if(e)throw std::runtime_error("sync failed");}
size_t Bytes(uint64_t x,uint64_t y,uint64_t z,size_t s){return PlanProduct(PlanProduct(PlanProduct(x,y),z),s);}
struct DeviceBuffer{void*data=nullptr;DeviceBuffer()=default;explicit DeviceBuffer(size_t bytes){if(!allocs)scratchBytes=bytes;++allocs;data=(void*)16;}};
struct FakeTiling{uint32_t baseM=16,baseN=16,baseK=16;};
struct PreparedMatmul{MatmulPlan plan{};HardwareCaps caps{};FakeTiling tiling{};};
struct PreparedExecution:PreparedMatmul{bool isStream=false;StreamPlan stream{};ReductionPlan reduction{};uint32_t executionPlanner=0;double executionScore=0;bool expansionTracked=false,expansionEnabled=false;uint32_t baselineReductionMode=0;double baselineExecutionScore=0;};
struct ExecutionInfo:PreparedExecution{bool isSmall=false;SmallPlan small{};};
HardwareCaps QueryCaps(uint32_t cores){return {cores,192*1024};}
bool TryPrepare(ProblemDesc p,HardwareCaps c,TileRequest tile,PreparedMatmul& out){++tilers;attempts.push_back({tile.m,tile.n});out={MakePlan(p,c,tile),c,{tile.m,tile.n,16}};return !failTiler && tilers>rejectFirst;}
struct StreamCompletion{bool pending=false;StreamCompletion(aclrtStream,DeviceBuffer&,DeviceBuffer&){}void Finish(){Check(failSync,"sync");}};
template<typename T,typename...Args> void MockSmallLaunch(uint32_t blocks,Args...){assert(blocks>0);++launches;}
template<typename T,typename...Args> void Dispatch(Args...args){++launches;launchReduction=std::get<sizeof...(args)-1>(std::make_tuple(args...)).mode;}
template<typename T,typename...Args> void DispatchStream(Args...args){++launches;launchReduction=std::get<sizeof...(args)-1>(std::make_tuple(args...)).reduction.mode;}
'''
        suffix=r'''
} // namespace local_baseline
void Require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
'''
        main=r'''
void run(uint64_t batches,uint32_t m,uint32_t n,uint32_t k,bool ta,bool tb,local_baseline::ExecutionFamily family,bool expectedSmall,bool fail=false,local_baseline::TileRequest tile={},int dtype=1,local_baseline::ReductionPolicy sumPolicy=local_baseline::ReductionPolicy::Auto,bool expand=true){
 using namespace local_baseline;
 int64_t as[]={static_cast<int64_t>(batches),ta?k:m,ta?m:k},bs[]={static_cast<int64_t>(batches),tb?n:k,tb?k:n},ys[]={static_cast<int64_t>(batches)};
 TensorInfo ai{as,3,dtype},bi{bs,3,dtype},yi{ys,1,0};TensorGroupInfo ag{&ai,1},bg{&bi,1},yg{&yi,1};
 uint8_t a=0,b=0,y=0;launches=allocs=tilers=0;ExecutionInfo actual;bool caught=false;
 try{auto scratch=RunKernel(&a,ag,&b,bg,&y,yg,24,(void*)1,ta,tb,tile,&actual,family,sumPolicy,expand);
     assert(!fail);assert(bool(scratch.data)==!expectedSmall);assert(actual.isSmall==expectedSmall);
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
 using namespace local_baseline;HardwareCaps h{24,192*1024};
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
int main(){using local_baseline::ExecutionFamily;using local_baseline::ReductionPolicy;
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
            source.write_text(prefix+prepare+host+suffix+writers+main)
            done=subprocess.run(['/usr/bin/g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            rows=[json.loads(line) for line in done.stdout.splitlines()]
            self.assertEqual(len(rows),72)
            self.assertTrue({16,32} <= {row.get('dot_columns') for row in rows})
            for row in rows:
                self.assertEqual(plan_metadata.validate_execution(row,row['problem']),row)
