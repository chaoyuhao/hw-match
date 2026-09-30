"""Ownership and allocation checks, independent of CANN."""
from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class StreamPlanTests(unittest.TestCase):
    def test_rule_generated_coverage_load_storage_and_overflow(self):
        source = r'''
#include "stream_plan.h"
#include <cassert>
#include <vector>
using namespace local_baseline;
int main() {
 for (uint32_t cores : {1u, 3u, 24u}) for (uint32_t m : {1u,15u,16u,17u,31u,33u,65u})
 for (uint32_t n : {1u,15u,17u,63u,65u,257u}) for (uint64_t b : {1u,3u,25u}) {
  ProblemDesc p{b,m,n,32,1,false,false}; HardwareCaps h{cores,192*1024};
  auto mm=MakePlan(p,h,{16,32}); auto s=MakeStreamPlan(p,h,mm);
  auto one=SplitStreamPlan(p,h,mm,1);
  assert(s.maxTilesPerCore<=one.maxTilesPerCore);
  if(s.rowTasks>=cores) assert(s.splits==1);
  assert(s.ubBytes<=64*1024 && s.cSlotElements==uint64_t(std::min(m,16u))*32);
  assert(s.maximaOffset==uint64_t(s.blocks)*s.cSlotElements*4);
  assert(s.partialOffset==s.maximaOffset+(s.splits>1?b*s.rowPitch*4:0));
  assert(s.scratchBytes==s.partialOffset+b*s.splits*s.rowPitch*4);
  std::vector<unsigned> coverage(b*((m+15)/16)*((n+31)/32));
  std::vector<uint64_t> loads(s.blocks);
  for(uint64_t task=0;task<s.tasks;++task) {
   auto shard=task%s.splits; auto row=task/s.splits;
   auto first=((n+31)/32)*shard/s.splits, last=((n+31)/32)*(shard+1)/s.splits;
   assert(last>first);
   for(auto col=first;col<last;++col) {++coverage[row*((n+31)/32)+col];++loads[task%s.blocks];}
  }
  for(auto count:coverage) assert(count==1);
  assert(*std::max_element(loads.begin(),loads.end())==s.maxTilesPerCore);
 }
 ProblemDesc p{uint64_t(1)<<32,16,8192,32,2,true,true}; HardwareCaps h{24,192*1024};
 auto huge=MakeStreamPlan(p,h,MakePlan(p,h,{16,16})); assert(huge.splits==1);
 bool bad=false;try {p.batches=UINT64_MAX;MakeStreamPlan(p,h,{});}catch(const std::runtime_error&){bad=true;}assert(bad);
 p={1,2,32,16,1,false,false}; auto mm=MakePlan(p,h,{16,16});
 bad=false;try{SplitStreamPlan(p,h,mm,3);}catch(const std::runtime_error&){bad=true;}assert(bad);
 assert(SplitStreamPlan(p,h,mm,2).splits==2);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp=Path(directory)/'plan.cpp'; binary=Path(directory)/'plan'
            cpp.write_text(source)
            done=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
            self.assertEqual(done.returncode,0,done.stderr)
