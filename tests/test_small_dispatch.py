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
        host=kernel[kernel.index('inline DeviceBuffer RunKernel('):kernel.index('} // namespace local_baseline',kernel.index('inline DeviceBuffer RunKernel('))]
        host=re.sub(r'SmallVectorKernel<(half|bfloat16_t)><<<small.blocks, nullptr, stream>>>\(',r'MockSmallLaunch<\1>(small.blocks, ',host)
        runner=(ROOT/'local/runner.asc').read_text()
        writers=runner[runner.index('void WritePlan('):runner.index('} // namespace')]
        prefix=r'''
#include "small_plan.h"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <sstream>
using half=int16_t;using bfloat16_t=uint16_t;using aclrtStream=void*;
struct TensorInfo{const int64_t* shape;int64_t numDims;int32_t dtype;};
struct TensorGroupInfo{const TensorInfo*tensors;int64_t numTensors;};
static int launches,allocs,tilers;static bool failSync;
int aclrtSynchronizeStreamWithTimeout(aclrtStream,int){return failSync?1:0;}
namespace platform_ascendc {
struct PlatformAscendCManager{static PlatformAscendCManager*GetInstance(){static PlatformAscendCManager p;return &p;}size_t GetLibApiWorkSpaceSize(){return 32;}};
}
namespace local_baseline {
constexpr uint32_t REDUCE_ROWS=32;
void Check(int e,const char*){if(e)throw std::runtime_error("sync failed");}
size_t Bytes(uint64_t x,uint64_t y,uint64_t z,size_t s){return PlanProduct(PlanProduct(PlanProduct(x,y),z),s);}
struct DeviceBuffer{void*data=nullptr;DeviceBuffer()=default;explicit DeviceBuffer(size_t){++allocs;data=(void*)16;}};
struct FakeTiling{uint32_t baseM=16,baseN=16,baseK=16;};
struct PreparedMatmul{MatmulPlan plan{};HardwareCaps caps{};FakeTiling tiling{};};
struct ExecutionInfo:PreparedMatmul{bool isSmall=false;SmallPlan small{};};
HardwareCaps QueryCaps(uint32_t cores){return {cores,192*1024};}
PreparedMatmul PrepareMatmul(ProblemDesc p,HardwareCaps c,TileRequest tile){++tilers;return {MakePlan(p,c,tile.m?tile:TileRequest{32,64}),c,{}};}
struct StreamCompletion{bool pending=false;StreamCompletion(aclrtStream,DeviceBuffer&,DeviceBuffer&){}void Finish(){Check(failSync,"sync");}};
template<typename T,typename...Args> void MockSmallLaunch(uint32_t blocks,Args...){assert(blocks>0);++launches;}
template<typename T,typename...Args> void Dispatch(Args...){++launches;}
'''
        suffix=r'''
} // namespace local_baseline
void Require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
'''
        main=r'''
void run(uint32_t m,uint32_t n,uint32_t k,bool ta,bool tb,local_baseline::ExecutionFamily family,bool expectedSmall,bool fail=false,local_baseline::TileRequest tile={}){
 using namespace local_baseline;
 int64_t as[]={1,ta?k:m,ta?m:k},bs[]={1,tb?n:k,tb?k:n},ys[]={1};
 TensorInfo ai{as,3,1},bi{bs,3,1},yi{ys,1,0};TensorGroupInfo ag{&ai,1},bg{&bi,1},yg{&yi,1};
 uint8_t a=0,b=0,y=0;launches=allocs=tilers=0;ExecutionInfo actual;bool caught=false;
 try{auto scratch=RunKernel(&a,ag,&b,bg,&y,yg,24,(void*)1,ta,tb,tile,&actual,family);
     assert(!fail);assert(bool(scratch.data)==!expectedSmall);assert(actual.isSmall==expectedSmall);}
 catch(const std::runtime_error&){caught=true;assert(fail);}
 assert(caught==fail);
 if(fail){assert(launches==(failSync?1:0));assert(tilers==0&&allocs==0);return;}
 assert(launches==1);assert(allocs==(expectedSmall?0:2));assert(tilers==(expectedSmall?0:1));
 ProblemDesc p{1,m,n,k,1,ta,tb};
 std::cout<<ExecutionJson(actual,p,family==ExecutionFamily::Auto?"auto":family==ExecutionFamily::Gm?"gm":"small","auto")<<'\n';
}
int main(){using local_baseline::ExecutionFamily;
 run(1,1,32,false,false,ExecutionFamily::Auto,true);
 run(1,5,8,true,false,ExecutionFamily::Auto,true);
 run(17,65,32,false,false,ExecutionFamily::Auto,false);
 run(3,5,8,true,true,ExecutionFamily::Auto,false);
 run(1,1,32,false,false,ExecutionFamily::Gm,false);
 run(1,1,32,false,false,ExecutionFamily::Auto,false,false,{32,64});
 run(3,5,8,true,true,ExecutionFamily::Small,false,true);
 run(1,1,32,false,false,ExecutionFamily::Small,false,true,{32,64});
 failSync=true;run(1,1,32,false,false,ExecutionFamily::Auto,true,true);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);source=tmp/'test.cpp';binary=tmp/'test'
            source.write_text(prefix+host+suffix+writers+main)
            done=subprocess.run(['/usr/bin/g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(source),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            rows=[json.loads(line) for line in done.stdout.splitlines()]
            self.assertEqual(len(rows),6)
            for row in rows:
                self.assertEqual(plan_metadata.validate_execution(row,row['problem']),row)
