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
   assert(s.tasks==2 && s.blocks==2 && s.ubBytes==6*(s.aElements+s.bElements)+4*(s.productElements+s.partialElements)+288);
   assert(s.aPitch%16==0 && s.bPitch%16==0);
  }
  assert(MakeSmallPlan({1,1,1,8,dtype,ta,tb},h).variant==SmallVariant::Dot);
 }
 for (uint64_t b : {1,7,8,9,16,17,32,33,191,192}) {
  auto s=MakeSmallPlan({b,1,1,8,1,false,false},h);
  assert(s.tasks==(b+7)/8 && s.blocks==s.tasks);
 }
 for(auto p : {ProblemDesc{1,1,65,8,1,false,false}, ProblemDesc{1,1,1,264,1,false,false},
  ProblemDesc{1,1,1,7,1,false,false}, ProblemDesc{1,16,64,256,1,false,true},
  ProblemDesc{1,8192,1,8,1,false,false}})
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
   Scenario{33,1,16,8,false,true,true},
   Scenario{192,1,1,8,false,false,true},
   Scenario{193,1,1,8,false,false,false},
   Scenario{1,17,1,8,false,false,true},
   Scenario{1,128,1,8,false,false,true},
   Scenario{1,129,1,8,false,false,false},
   Scenario{1,32,5,8,true,false,true},
   Scenario{1,33,5,8,true,false,false},
   Scenario{1,1,5,256,true,false,true},
   Scenario{1,1,64,128,false,false,true},
   Scenario{2,1,5,128,false,false,true},
   Scenario{3,1,5,128,false,false,false},
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
 // Per-batch memory is independent of B; reject task narrowing overflow.
 const uint64_t largestB=uint64_t(UINT32_MAX)*8;
 auto largest=MakeSmallPlan({largestB,1,1,8,1,false,false},h);
 assert(largest.tasks==UINT32_MAX && largest.blocks==24);
 assert(!UseSmallAutomatically({largestB,1,1,8,1,false,false},largest));
 assert(MakeSmallPlan({largestB+1,1,1,8,1,false,false},h).variant==SmallVariant::None);
 auto largeWork=MakeSmallPlan({1,200,64,8,1,false,true},h);
 assert(largeWork.variant==SmallVariant::Dot && largeWork.ubBytes==42272);
 assert(!UseSmallAutomatically({1,200,64,8,1,false,true},largeWork));
 auto fits=MakeSmallPlan({1,1,64,128,1,false,false},h);
 assert(fits.variant==SmallVariant::Rows && fits.ubBytes==51264);
 assert(MakeSmallPlan({1,1,64,256,1,false,false},h).variant==SmallVariant::None);
 assert(MakeSmallPlan({1,1,64,128,1,false,false},{24,32768+fits.ubBytes}).variant==SmallVariant::Rows);
 assert(MakeSmallPlan({1,1,64,128,1,false,false},{24,32767+fits.ubBytes}).variant==SmallVariant::None);
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

    def test_batched_dot_adapts_columns_to_real_resources(self):
        source = r'''
#include "small_plan.h"
#include <cassert>
using namespace local_baseline;
int main() {
 HardwareCaps h{24,192*1024};
 auto tiny=MakeSmallPlan({1,1,5,8,1,false,true},h);
 assert(tiny.dotColumns==5 && tiny.productElements==320 && tiny.partialElements==8);
 assert(tiny.ubBytes==2176);
 ProblemDesc p{1,1,64,128,1,false,true};
 auto normal=MakeSmallPlan(p,h);
 assert(normal.dotColumns==32 && normal.ubBytes==58528);
 auto reduced=MakeSmallPlan(p,{24,32768+54368});
 assert(reduced.dotColumns==16 && reduced.ubBytes==54368);
 auto minimum=MakeSmallPlan(p,{24,32768+52288});
 assert(minimum.dotColumns==8 && minimum.ubBytes==52288);
 auto old=MakeSmallPlan(p,{24,32768+52287});
 assert(old.dotColumns==0 && old.ubBytes==51264);
 assert(old.productElements==256 && old.partialElements==8);
 assert(MakeSmallPlan(p,{24,32768+51263}).variant==SmallVariant::None);
 for(uint32_t n=2;n<=64;++n)for(uint32_t k=8;k<=256;k+=8) {
   ProblemDesc q{1,1,n,k,1,false,true};
   auto s=MakeSmallPlan(q,h);if(s.variant==SmallVariant::None)continue;
   assert(s.dotColumns==0 || (s.dotColumns<=n && (s.dotColumns==n || s.dotColumns%8==0)));
   assert(s.ubBytes==6*(s.aElements+s.bElements)+4*(s.productElements+s.partialElements)+288);
   assert(s.ubBytes<=65536 && s.partialElements>=8);
   if(s.dotColumns)assert(s.productElements==s.dotColumns*64 && s.partialElements>=s.dotColumns);
   // R11 does not widen automatic admission.
   assert(UseSmallAutomatically(q,s)==(n<=128));
 }
 auto single=MakeSmallPlan({1,1,1,128,1,false,true},h);
 auto rows=MakeSmallPlan({1,1,5,128,1,false,false},h);
 assert(single.dotColumns==0 && rows.dotColumns==0);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'main.cpp';f.write_text(source);exe=Path(tmp)/'test'
            done=subprocess.run(['/usr/bin/g++','-std=c++14','-Wall','-Wextra','-Werror','-I',str(ROOT),str(f),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(exe)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
