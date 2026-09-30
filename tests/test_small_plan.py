from pathlib import Path
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
class SmallPlanTests(unittest.TestCase):
    def test_geometry_layout_resource_and_fallback_contract(self):
        source = r'''
#include "small_plan.h"
#include <cassert>
using namespace local_baseline;
int main() {
 HardwareCaps h{24,192*1024};
 for (uint32_t dtype : {1U,2U}) for(bool ta : {false,true}) for(bool tb : {false,true}) {
  auto s = MakeSmallPlan({9,3,5,8,dtype,ta,tb},h);
  assert(s.variant == (ta && tb ? SmallVariant::None : tb ? SmallVariant::Dot : SmallVariant::Rows));
  if(s.variant != SmallVariant::None) {
   assert(s.tasks==2 && s.blocks==2 && s.ubBytes==6*(s.aElements+s.bElements)+1344);
   assert(s.aPitch%16==0 && s.bPitch%16==0);
  }
  assert(MakeSmallPlan({1,1,1,8,dtype,ta,tb},h).variant==SmallVariant::Dot);
 }
 for (uint64_t b : {1,7,8,9,16,17,32}) {
  auto s=MakeSmallPlan({b,1,1,8,1,false,false},h);
  assert(s.tasks==(b+7)/8 && s.blocks==s.tasks);
 }
 for(auto p : {ProblemDesc{33,1,1,8,1,false,false}, ProblemDesc{1,17,1,8,1,false,false},
  ProblemDesc{1,1,65,8,1,false,false}, ProblemDesc{1,1,1,264,1,false,false},
  ProblemDesc{1,1,1,7,1,false,false}, ProblemDesc{1,16,64,256,1,false,true},
  ProblemDesc{1,3,5,24,1,false,false}})
  assert(MakeSmallPlan(p,h).variant==SmallVariant::None);
 assert(MakeSmallPlan({1,1,1,8,1,false,false},{24,32768}).variant==SmallVariant::None);
 // Coverage and fallback at both new per-owner budget boundaries.
 struct Scenario {uint64_t b;uint32_t m,n,k;bool ta,tb,automatic;};
 for (const auto& t : {
   Scenario{1,1,17,8,false,true,true},
   Scenario{1,2,64,8,false,true,true},
   Scenario{1,3,43,8,false,true,false},
   Scenario{8,1,16,8,false,true,true},
   Scenario{8,1,17,8,false,true,false},
   Scenario{32,1,16,8,false,true,true},
   Scenario{1,1,5,64,false,false,true},
   Scenario{4,1,5,64,true,false,true},
   Scenario{5,1,5,64,false,false,false},
   Scenario{8,1,5,32,true,false,true},
   Scenario{8,1,5,40,false,false,false}}) {
  for(uint32_t dtype : {1U,2U}) {
   ProblemDesc p{t.b,t.m,t.n,t.k,dtype,t.ta,t.tb};auto s=MakeSmallPlan(p,h);
   assert(s.variant!=SmallVariant::None);
   assert(UseSmallAutomatically(p,s)==t.automatic);
   assert(SelectSmall(p,s,{},ExecutionFamily::Auto)==t.automatic);
   assert(!SelectSmall(p,s,{},ExecutionFamily::Gm));
   assert(!SelectSmall(p,s,{32,64},ExecutionFamily::Auto));
   assert(SelectSmall(p,s,{},ExecutionFamily::Small));
  }
 }
 auto waves=MakeSmallPlan({32,1,1,8,1,false,false},{1,192*1024});
 assert(!UseSmallAutomatically({32,1,1,8,1,false,false},waves));
 bool conflict=false;try{SelectSmall({1,1,1,8,1,false,false},waves,{16,16},ExecutionFamily::Small);}
 catch(const std::runtime_error&){conflict=true;}assert(conflict);
 bool unsupported=false;try{SelectSmall({1,1,1,8,1,false,false},{},{},ExecutionFamily::Small);}
 catch(const std::runtime_error&){unsupported=true;}assert(unsupported);
 bool caught=false; try {MakeSmallPlan({UINT64_MAX,8192,8192,8192,1,false,false},h);}
 catch(const std::runtime_error&) {caught=true;} assert(caught);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'main.cpp'; f.write_text(source); exe=Path(tmp)/'test'
            done=subprocess.run(['/usr/bin/g++','-std=c++14','-Wall','-Wextra','-Werror','-I',str(ROOT),str(f),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
