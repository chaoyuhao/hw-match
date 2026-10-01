"""End-to-end geometry/family selection; actual host C++, no CANN dependency."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class JointPlanTests(unittest.TestCase):
    def test_joint_candidates_load_storage_and_sdk_rejection(self):
        self.assertTrue((ROOT/'joint_plan.h').exists(), 'joint execution planner is missing')
        source = r'''
#include "joint_plan.h"
#include <cassert>
#include <set>
#include <vector>
using namespace local_baseline;
int main() {
 HardwareCaps h{24,192*1024};
 std::set<ExecutionFamily> selected;
 for (uint32_t m : {1u,17u,128u,400u,1024u})
 for (uint32_t n : {1u,65u,256u,1024u}) for (uint32_t k : {8u,256u,1024u}) {
  ProblemDesc p{1,m,n,k,1,false,false}; auto cs=GenerateExecutionCandidates(p,h);
  assert(cs.count && cs.count<=147);
  assert(cs.plans[0].score>0);
  bool gm=false,stream=false,pipeline=false;
  for(size_t i=0;i<cs.count;++i) {
   const auto& e=cs.plans[i]; if(i) assert(!ExecutionLess(e,cs.plans[i-1]));
   assert(e.score==ExecutionCost(p,e));
   if(e.family==ExecutionFamily::Gm){gm=true;continue;}
   auto s=e.stream; stream|=e.family==ExecutionFamily::Stream; pipeline|=e.family==ExecutionFamily::Pipeline;
   assert(s.buffers==(e.family==ExecutionFamily::Pipeline?2u:1u));
   assert(s.plannerVersion==2 && s.maximaOffset==uint64_t(s.blocks)*s.buffers*s.cSlotElements*4);
   assert(s.scratchBytes==s.partialOffset+p.batches*s.splits*s.rowPitch*4);
   assert(s.ubBytes<=65536 && s.ubBytes+e.matmul.ubBudget<=h.ubBytes);
   std::vector<uint64_t> loads(s.blocks);
   uint32_t columns=CeilDiv(n,s.tileN);
   for(uint64_t t=0;t<s.tasks;++t) {
    auto shard=t%s.splits;
    loads[t%s.blocks]+=columns*(shard+1)/s.splits-columns*shard/s.splits;
   }
   assert(*std::max_element(loads.begin(),loads.end())==s.maxTilesPerCore);
   if(s.buffers==2) assert(columns/s.splits>=2);
  }
  assert(gm && stream); if(n>16) assert(pipeline);
  unsigned calls=0;
  auto e=FindSupportedExecutionPlan(p,h,{},ExecutionFamily::Auto,[&](const MatmulPlan&){++calls;return true;});
  assert(calls==1 && e.score==cs.plans[0].score && e.family==cs.plans[0].family);
  selected.insert(e.family);
 }
 assert(selected.count(ExecutionFamily::Gm) && selected.count(ExecutionFamily::Pipeline));
 ProblemDesc p{1,400,1024,256,1,false,false};auto mm=MakePlan(p,h,{16,64});
 auto choices=StreamSplitCandidates(p,h,mm);
 assert(std::find(choices.values.begin(),choices.values.begin()+choices.count,2)!=choices.values.begin()+choices.count);
 auto old=MakeStreamPlan(p,h,mm),two=SplitStreamPlan(p,h,mm,2,2);
 assert(old.rowTasks>=h.cores && old.maxTilesPerCore==32 && two.maxTilesPerCore==24);
 assert(two.maximaOffset==2*SplitStreamPlan(p,h,mm,2).maximaOffset);
 assert(two.ubBytes==SplitStreamPlan(p,h,mm,2).ubBytes);
 std::set<std::pair<unsigned,unsigned>> attempts;unsigned calls=0;bool bad=false;
 try{FindSupportedExecutionPlan(p,h,{},ExecutionFamily::Auto,[&](const MatmulPlan& c){++calls;assert(attempts.insert({c.tileM,c.tileN}).second);return false;});}
 catch(const std::runtime_error&){bad=true;}assert(bad && calls<=5);
 calls=0;auto fixed=FindSupportedExecutionPlan(p,h,{16,64},ExecutionFamily::Auto,[&](const MatmulPlan& c){++calls;return c.tileM==16&&c.tileN==64;});
 assert(calls==1 && fixed.family==ExecutionFamily::Gm);
 calls=0;bad=false;try{FindSupportedExecutionPlan(p,h,{16,64},ExecutionFamily::Pipeline,[&](const MatmulPlan&){++calls;return false;});}catch(const std::runtime_error&){bad=true;}assert(bad&&calls==1);
 auto forced=FindSupportedExecutionPlan(p,h,{16,64},ExecutionFamily::Pipeline,[](const MatmulPlan&){return true;});assert(forced.stream.buffers==2);
 auto legacy=FindSupportedExecutionPlan(p,h,{16,64},ExecutionFamily::Stream,[](const MatmulPlan&){return true;});assert(legacy.stream.plannerVersion==1&&legacy.stream.splits==1);
 for(unsigned buffers:{0u,3u}) {bad=false;try{SplitStreamPlan(p,h,mm,1,buffers);}catch(const std::runtime_error&){bad=true;}assert(bad);}
 p.batches=UINT64_MAX;bad=false;try{GenerateExecutionCandidates(p,h);}catch(const std::runtime_error&){bad=true;}assert(bad);
 // Search work is independent of batch count, including the exact load computation.
 p.batches=uint64_t(1)<<32;assert(GenerateExecutionCandidates(p,h).count);
}
'''
        with tempfile.TemporaryDirectory() as d:
            cpp=Path(d)/'joint.cpp'; binary=Path(d)/'joint';cpp.write_text(source)
            done=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True,timeout=20)
            self.assertEqual(done.returncode,0,done.stderr)
