"""Actual compensated record writer/finalizer, FP64 fixtures and ownership checks."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
class PartialSumTests(unittest.TestCase):
 def test_compensation_records_and_final_merge(self):
  self.assertTrue((ROOT/'partial_sum.asc').exists(),'compensated record helpers missing')
  kernel=(ROOT/'kernel.asc').read_text()
  constants='\n'.join(re.findall(r'constexpr uint32_t REDUCE_\w+ = \d+;',kernel))
  fixture=(ROOT/'tests/reduction_cpu_main.cpp').read_text()
  aligned=fixture[fixture.index('struct Aligned'):fixture.index('void check(')]
  check=fixture[fixture.index('void checkSum('):fixture.index('\nint main(')]
  check=check.replace('void checkSum(uint32_t m, int pattern)', 'void checkSum(uint32_t m, int pattern, uint32_t span)')
  check=check.replace('using namespace AscendC;', 'using namespace AscendC;using namespace local_baseline;',1)
  check=check.replace('const size_t count = batches * rowPitch;', 'const size_t count = batches * rowPitch;\n    auto plan=MakeReductionPlan({batches,m,1,1,1,false,false},span,ReductionPolicy::Partials);\n    Aligned records(plan.bytes/4+32);float* record=records.data+16;')
  check=check.replace('std::vector<float> expected(batches);', 'regions.push_back({record,plan.bytes/4,false,std::vector<unsigned>(plan.bytes/4),std::vector<bool>(plan.bytes/4)});\n    std::vector<float> expected(batches);')
  a=check.index('        for (blockIdx = 0; blockIdx < blockNum; ++blockIdx) {');b=check.index('        for (uint32_t b = 0; b < batches; ++b) {',a)
  check=check[:a]+r'''
        regions[2].writes.assign(plan.bytes/4,0);regions[2].readable.assign(plan.bytes/4,false);
        std::vector<TPipe> pipes(blockNum);
        for(blockIdx=0;blockIdx<blockNum;++blockIdx){
            PartialSumWriter writer;writer.Init(pipes[blockIdx],span);
            auto data=std::make_shared<std::vector<float>>(span, -3.402823466e38F);
            LocalTensor<float> tile{data,0};GlobalTensor<float> dst;dst.SetGlobalBuffer(record);
            for(uint64_t task=blockIdx;task<uint64_t(batches)*plan.segments;task+=blockNum){
                uint32_t batch=task/plan.segments,row=(task%plan.segments)*span;
                uint32_t real=std::min(span,m-row);
                std::fill(data->begin(),data->end(),-3.402823466e38F);
                for(uint32_t i=0;i<real;++i)tile.SetValue(i,rp[batch*rowPitch+row+i]);
                writer.Write(tile,real,dst[task*16]);
            }
            require(pipes[blockIdx].bytes==plan.foldUbBytes,"writer UB mismatch");
        }
        require(scalarReads==0,"producer pulled partials through Scalar");
        for(auto count:regions[2].writes)require(count==1,"record ownership/write incomplete");
        for(blockIdx=0;blockIdx<blockNum;++blockIdx){
            FinalizePartialSums(pipes[blockIdx],(GM_ADDR)record,(GM_ADDR)yp,batches,plan);
            require(pipes[blockIdx].bytes==plan.foldUbBytes+14368,"final UB mismatch");
        }
''' +check[b:]
  check=check.replace('if (m >= 1024) require(scalarReads <= batches * 32ull * ((m + 1023) / 1024),','if (m >= 1024) require(scalarReads <= batches * 16ull * ((plan.segments + 127) / 128),')
  check=check.replace('"sum wrote output padding");','"sum wrote output padding");\n        for(unsigned i=0;i<16;++i)require(records.data[i]==123456.f && records.data[16+plan.bytes/4+i]==123456.f,"record guard");')
  main='''
int main(){for(unsigned span:{16u,32u,48u,64u,128u,256u})
 for(unsigned m:{1u,7u,8u,9u,17u,31u,32u,33u,47u,48u,49u,255u,256u,257u,1023u,1024u,1025u,1026u,2047u,2048u,2049u,4097u,8191u,8192u})
 for(int pattern=0;pattern<8;++pattern)checkSum(m,pattern,span);}
'''
  source='#include "tests/reduction_cpu_stubs.h"\n#include "reduction_plan.h"\n#include <cstdlib>\n#define __gm__\nnamespace local_baseline {\n'+constants+'\n'+(ROOT/'partial_sum.asc').read_text()+'\n}\n'+aligned+check+main
  with tempfile.TemporaryDirectory() as d:
   cpp=Path(d)/'partial.cpp';binary=Path(d)/'partial';cpp.write_text(source)
   done=subprocess.run(['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(ROOT),str(cpp),'-o',str(binary)],capture_output=True,text=True)
   self.assertEqual(done.returncode,0,done.stderr)
   done=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
   self.assertEqual(done.returncode,0,done.stderr)
