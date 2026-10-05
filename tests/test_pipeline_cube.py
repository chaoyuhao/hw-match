"""R23 CPU checks: physical resources/layouts; no NPU performance claims."""
from pathlib import Path
import subprocess
import tempfile
import re
import unittest
ROOT=Path(__file__).resolve().parents[1]
class PipelineCubeTests(unittest.TestCase):
    def run_cpp(self, code):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'test.cpp';exe=Path(tmp)/'test';src.write_text(code)
            r=subprocess.run(['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-pthread','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            r=subprocess.run([str(exe)],capture_output=True,text=True,timeout=90)
            self.assertEqual(r.returncode,0,r.stderr)
    def test_generated_physical_plans_and_capacity(self):
        self.assertTrue((ROOT/'pipeline_plan.h').exists(),'independent physical planner missing')
        self.run_cpp(r'''
#include "pipeline_plan.h"
#include <cassert>
using namespace local_baseline;
int main(){
 DirectCaps d{true,524288,65536,65536,131072};HardwareCaps h{24,196608};
 for(unsigned m:{1u,15u,17u,65u,256u,8192u})for(unsigned n:{1u,31u,129u,8192u})
 for(unsigned k:{1u,8u,24u,137u,256u,8192u})for(unsigned cores:{1u,7u,24u,32u}){
  h.cores=cores;ProblemDesc p{3,m,n,k,2,true,true};auto q=MakePipelinePlan(p,h,d);
  assert(q.enabled && q.stream.plannerVersion==5 && q.stream.buffers==2);
  assert(q.stream.blocks<=cores && q.stream.blocks<=q.stream.tasks);
  assert(q.a1Bytes+q.b1Bytes<=d.l1 && q.a0Bytes<=d.l0a && q.b0Bytes<=d.l0b && q.c0Bytes<=d.l0c);
  assert(q.stream.ubBytes<=h.ubBytes && q.stream.cSlotElements==uint64_t(q.stream.tileM)*q.stream.tileN);
  assert(q.stream.maximaOffset==q.stream.blocks*2*q.stream.cSlotElements*4);
  assert(q.stream.splits<=CeilDiv(n,q.stream.tileN) && q.stream.reduction.segmentRows>=16);
  assert(q.c0Bytes==2*q.stream.tileM*q.stream.tileN*4);
  assert(q.a0Bytes==2*q.stream.tileM*q.blockK*2 && q.b0Bytes==2*q.stream.tileN*q.blockK*2);
  auto again=MakePipelinePlan(p,h,d);assert(q.stream.tileM==again.stream.tileM && q.score==again.score);
 }
 ProblemDesc p{1,129,257,8192,1,false,false};
 auto off=d;off.supported=false;assert(!MakePipelinePlan(p,h,off).enabled);
 off=d;off.l0c=1;assert(!MakePipelinePlan(p,h,off).enabled);
 off=d;off.l1=1;assert(!MakePipelinePlan(p,h,off).enabled);
 h.ubBytes=65537;assert(!MakePipelinePlan(p,h,d).enabled);
 h.ubBytes=196608;
 auto q=PipelineCandidate(p,h,d,128,128,1,ReductionPolicy::Partials);assert(q.enabled&&!q.residentA);
 p.k=24;q=PipelineCandidate(p,h,d,128,128,1,ReductionPolicy::Partials);assert(q.residentA && q.blockK==32);
}
''')
    def test_actual_padded_cube_and_event_lifecycle(self):
        self.assertTrue((ROOT/'pipeline_cube.asc').exists(),'pipeline executor missing')
        helper=(ROOT/'pipeline_cube.asc').read_text().split('// Pipeline C/V execution')[0]
        self.run_cpp((ROOT/'tests/pipeline_cube_stubs.h').read_text()+
                     '\n#include "pipeline_plan.h"\nnamespace local_baseline {\n'+helper+'\n}\n'+
                     (ROOT/'tests/pipeline_cube_main.cpp').read_text())

    def test_actual_cv_ring_prefetched_stripes_and_reduction(self):
        kernel=(ROOT/'kernel.asc').read_text();stream=(ROOT/'stream_matmul.asc').read_text()
        code=(ROOT/'pipeline_cube.asc').read_text()
        constants='\n'.join(re.findall(r'constexpr uint32_t REDUCE_\w+ = \d+;',kernel))
        fold=kernel[kernel.index('__aicore__ inline void FoldMaxColumns('):kernel.index('__aicore__ inline void ComputeRowMaxima(')]
        sums=kernel[kernel.index('__aicore__ inline void SumRowMaxima('):kernel.index('template <typename T, bool TA, bool TB>',kernel.index('__aicore__ inline void SumRowMaxima('))]
        merge=stream[stream.index('__aicore__ inline void MergeStreamMaxima('):stream.index('// Device entry point')]
        stubs=(ROOT/'tests/reduction_cpu_stubs.h').read_text().replace('#pragma once','')
        stubs=stubs.replace('enum { PIPE_V, PIPE_ALL };','enum { PIPE_V, PIPE_ALL, PIPE_FIX, PIPE_MTE2 };')
        stubs=stubs.replace('inline float readGm(float* p) {','static void (*readHook)(float*)=nullptr;\ninline float readGm(float* p) {\nif(readHook)readHook(p);')
        stubs='#include <deque>\n'+stubs
        stubs=stubs.replace('bytes <= 64 * 1024','bytes <= 192 * 1024')
        a=stubs.index('template <TPosition P, int N> struct TQue {');b=stubs.index('struct TPipe {',a)
        stubs=stubs[:a]+(ROOT/'tests/pipeline_queue_stub.h').read_text()+stubs[b:]
        a=stubs.index('    template <typename Buffer> void InitBuffer(Buffer& b, int depth, size_t n) {')
        b=stubs.index('    int FetchEventID',a)
        stubs=stubs[:a]+"    template<TPosition P,int N>void InitBuffer(TQue<P,N>& b,int depth,size_t n){bytes+=depth*n;require(bytes<=192*1024,\"UB overflow\");b.Init(depth,n); }\n"+stubs[b:]
        protocol=code.split('// Pipeline C/V execution')[1].split('// Pipeline device entry')[0]
        self.run_cpp(stubs+'\n#include "pipeline_plan.h"\n'+(ROOT/'tests/pipeline_protocol_stubs.h').read_text()+
                     '\nnamespace local_baseline {\n'+constants+'\n'+(ROOT/'partial_sum.asc').read_text()+sums+fold+merge+protocol+
                     '\n}\n'+(ROOT/'tests/pipeline_protocol_main.cpp').read_text())
