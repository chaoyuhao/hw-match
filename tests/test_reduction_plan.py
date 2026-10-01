"""Actual reduction ownership/allocation and joint selection, no CANN dependency."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class ReductionPlanTests(unittest.TestCase):
 def test_layout_and_joint_reduction_selection(self):
  self.assertTrue((ROOT/'reduction_plan.h').exists(),'partial reduction planner missing')
  source=r'''
#include "joint_plan.h"
#include <cassert>
#include <set>
using namespace local_baseline;
int main(){
 HardwareCaps h{24,192*1024};std::set<unsigned> modes;
 for(unsigned m:{1u,17u,32u,65u,256u,1025u,8192u})for(unsigned span:{16u,32u,48u,128u,256u}){
  ProblemDesc p{3,m,1024,256,1,false,false};auto mm=MakePlan(p,h,{span,64});
  auto r=MakeReductionPlan(p,span,ReductionPolicy::Partials);
  assert(r.mode==1 && r.segments==(m+span-1)/span && r.bytes==3ull*r.segments*64);
  unsigned cap=8;while(cap<span)cap*=2;assert(r.foldUbBytes==14*cap+64);
  auto rows=MakeReductionPlan(p,span,ReductionPolicy::Rows);
  assert(rows.mode==0 && rows.bytes==3ull*((m+31)/32)*32*4 && rows.foldUbBytes==0);
  for(unsigned splits:{1u,2u})for(unsigned buffers:{1u,2u}){
   auto s=SplitStreamPlan(p,h,mm,splits,buffers,ReductionPolicy::Partials);
   assert(s.reduction.segmentRows==(splits==1?span:32));
   assert(s.maximaOffset==uint64_t(s.blocks)*buffers*s.cSlotElements*4);
   assert(s.partialOffset==s.maximaOffset+(splits>1?s.reduction.bytes:0));
   assert(s.scratchBytes==s.maximaOffset+s.reduction.bytes+(splits>1?3ull*s.rowPitch*splits*4:0));
   auto old=SplitStreamPlan(p,h,mm,splits,buffers);
   assert(s.ubBytes==old.ubBytes+s.reduction.foldUbBytes && s.ubBytes<=65536);
  }
 }
 for(uint64_t b:{1u,8u,192u})for(unsigned m:{16u,256u,1024u,8192u}){
  ProblemDesc p{b,m,1024,256,1,false,false};
  auto cs=GenerateExecutionCandidates(p,h);assert(cs.count<=147);
  for(size_t i=0;i<cs.count;++i){auto e=cs.plans[i];assert(e.score==ExecutionCost(p,e));
   if(e.family!=ExecutionFamily::Gm)assert(e.reduction.mode==e.stream.reduction.mode);
  }
  modes.insert(cs.plans[0].reduction.mode);
  for(auto mode:{ReductionPolicy::Rows,ReductionPolicy::Partials}) {
   auto all=GenerateExecutionCandidates(p,h,{},ExecutionFamily::Auto,mode);
   for(size_t i=0;i<all.count;++i)assert(all.plans[i].reduction.mode==(mode==ReductionPolicy::Partials));
  }
 }
 assert(modes.count(0)&&modes.count(1));
 ProblemDesc p{1,8192,64,32,1,false,false};
 for(auto f:{ExecutionFamily::Auto,ExecutionFamily::Gm,ExecutionFamily::Stream,ExecutionFamily::Pipeline}){
  unsigned calls=0;auto e=FindSupportedExecutionPlan(p,h,{32,64},f,[&](const MatmulPlan&){++calls;return true;},ReductionPolicy::Partials);
  assert(calls==1&&e.reduction.mode==1);
 }
 auto legacy=FindSupportedExecutionPlan(p,h,{32,64},ExecutionFamily::Gm,[](const MatmulPlan&){return true;});assert(!legacy.reduction.mode);
 bool bad=false;try{MakeReductionPlan(p,0,ReductionPolicy::Partials);}catch(const std::runtime_error&){bad=true;}assert(bad);
 p.batches=UINT64_MAX;bad=false;try{MakeReductionPlan(p,32,ReductionPolicy::Partials);}catch(const std::runtime_error&){bad=true;}assert(bad);
}
'''
  with tempfile.TemporaryDirectory() as d:
   cpp=Path(d)/'plan.cpp';out=Path(d)/'plan';cpp.write_text(source)
   done=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(cpp),'-o',str(out)],capture_output=True,text=True)
   self.assertEqual(done.returncode,0,done.stderr)
   done=subprocess.run([str(out)],capture_output=True,text=True,timeout=20)
   self.assertEqual(done.returncode,0,done.stderr)
