"""Post-selection expansion must never reselect an upstream execution plan."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class ReductionCoverageTests(unittest.TestCase):
    def test_expansion_preserves_geometry_and_only_compresses_real_rows(self):
        source = r'''
#include "joint_plan.h"
#include <cassert>
using namespace local_baseline;
void upstream(const ExecutionPlan& a,const ExecutionPlan& b){
 assert(a.family==b.family);
 assert(a.matmul.tileM==b.matmul.tileM && a.matmul.tileN==b.matmul.tileN);
 assert(a.matmul.tasks==b.matmul.tasks && a.matmul.blocks==b.matmul.blocks);
 assert(a.matmul.ubBudget==b.matmul.ubBudget && a.matmul.score==b.matmul.score);
 if(a.family!=ExecutionFamily::Gm){
  const auto& x=a.stream;const auto& y=b.stream;
  assert(x.tileM==y.tileM && x.tileN==y.tileN && x.splits==y.splits && x.blocks==y.blocks);
  assert(x.buffers==y.buffers && x.rowPitch==y.rowPitch && x.rowTasks==y.rowTasks && x.tasks==y.tasks);
  assert(x.cSlotElements==y.cSlotElements && x.maximaOffset==y.maximaOffset);
  assert(x.maxTilesPerCore==y.maxTilesPerCore && x.plannerVersion==y.plannerVersion);
 }
}
ExecutionPlan gm(ProblemDesc p,HardwareCaps h){
 ExecutionPlan e;e.matmul=MakePlan(p,h,{32,64});e.reduction=MakeReductionPlan(p,32,ReductionPolicy::Rows);e.score=ExecutionCost(p,e);return e;
}
int main(){
 HardwareCaps h{24,192*1024};ProblemDesc p{192,64,1024,256,1,false,false};
 auto a=gm(p,h),b=ExpandPartialReduction(p,h,a);upstream(a,b);
 assert(a.reduction.mode==0 && b.reduction.mode==1);
 assert(b.reduction.segmentRows==32 && b.reduction.segments==2 && b.reduction.bytes==24576);
 assert(b.reduction.foldUbBytes==512 && b.score==ExecutionCost(p,b));
 auto limited=h;limited.ubBytes=a.matmul.ubBudget+55967;
 assert(ExpandPartialReduction(p,limited,a).reduction.mode==0);
 limited.ubBytes++;assert(ExpandPartialReduction(p,limited,a).reduction.mode==1);
 for(unsigned m:{1u,16u,17u,32u}){p.m=m;assert(!ExpandPartialReduction(p,h,gm(p,h)).reduction.mode);}
 p.m=33;assert(ExpandPartialReduction(p,h,gm(p,h)).reduction.mode==1);
 for(unsigned tm:{16u,32u,48u,64u,128u,256u})for(unsigned m:{16u,32u,33u,65u,97u,8192u})
 for(unsigned splits:{1u,2u})for(unsigned buffers:{1u,2u}){
  p.m=m;ExecutionPlan x;x.family=buffers==1?ExecutionFamily::Stream:ExecutionFamily::Pipeline;
  x.matmul=MakePlan(p,h,{tm,64});x.stream=SplitStreamPlan(p,h,x.matmul,splits,buffers);
  x.stream.plannerVersion=3;x.reduction=x.stream.reduction;x.score=ExecutionCost(p,x);
  auto y=ExpandPartialReduction(p,h,x);upstream(x,y);
  assert(y.reduction.mode==y.stream.reduction.mode && y.stream.scratchBytes<=x.stream.scratchBytes);
  if(tm==16 && splits==1)assert(!y.reduction.mode); // 16 floats/record cannot compress a 16-row segment.
  if(tm==64 && m==65 && splits==1){assert(y.reduction.mode && y.reduction.segments==2 && y.reduction.bytes==24576);}
  if(y.reduction.mode){assert(y.stream.ubBytes<=65536);assert(y.reduction.bytes<uint64_t(p.batches)*p.m*4);}
  auto z=ExpandPartialReduction(p,h,y);upstream(y,z);assert(z.reduction.mode==y.reduction.mode && z.score==y.score);
 }
 unsigned changes=0,retained=0;
 for(unsigned batch:{1u,8u,192u})for(unsigned m:{1u,32u,64u,256u,1024u,8192u})
 for(unsigned n:{64u,1024u})for(bool ta:{false,true})for(bool tb:{false,true}){
  ProblemDesc q{batch,m,n,256,1,ta,tb};auto x=GenerateExecutionCandidates(q,h).plans[0];auto y=ExpandPartialReduction(q,h,x);upstream(x,y);
  changes+=!x.reduction.mode&&y.reduction.mode;retained+=x.reduction.mode;
  if(x.reduction.mode)assert(y.score==x.score && y.reduction.bytes==x.reduction.bytes);
 }
 assert(changes && retained);
}
'''
        with tempfile.TemporaryDirectory() as d:
            cpp=Path(d)/'coverage.cpp';binary=Path(d)/'coverage';cpp.write_text(source)
            built=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(built.returncode,0,built.stderr)
            ran=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
            self.assertEqual(ran.returncode,0,ran.stderr)
