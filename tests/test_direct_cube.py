"""Execute direct planner and physical Cube helpers; no CANN/NPU simulation claim."""
from pathlib import Path
import subprocess
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DirectCubeTests(unittest.TestCase):
    def compile_run(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp) / 'direct.cpp'
            binary = Path(tmp) / 'direct'
            cpp.write_text(source)
            result = subprocess.run(['g++', '-std=c++14', '-O2', '-ffp-contract=off',
                                     '-Wall', '-Wextra', '-Werror', '-pthread', '-I', str(ROOT),
                                     str(cpp), '-o', str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_resource_admission_residency_and_fallback(self):
        self.assertTrue((ROOT / 'direct_plan.h').exists(), 'direct planner not implemented')
        self.compile_run(r'''
#include "direct_plan.h"
#include <cassert>
using namespace local_baseline;
int main() {
 DirectCaps caps{true, 512*1024, 64*1024, 64*1024, 128*1024};
 ProblemDesc p{3,128,256,256,1,false,false};
 auto d=MakeDirectPlan(p,{128,256},caps);
 assert(d.enabled && d.residentA && d.blockK==128);
 assert(d.a1Bytes==65536 && d.b1Bytes==65536 && d.c0Bytes==131072);
 assert(d.a0Bytes==32768 && d.b0Bytes==65536);
 for(bool ta:{false,true})for(bool tb:{false,true})for(unsigned dtype:{1u,2u}) {
   p.ta=ta;p.tb=tb;p.dtype=dtype;assert(MakeDirectPlan(p,{128,256},caps).enabled);
 }
 p.k=8192;d=MakeDirectPlan(p,{128,256},caps);
 assert(d.enabled && !d.residentA && d.a1Bytes==32768);
 p.k=256;
 for(auto bad:{ProblemDesc{3,127,256,256,1,0,0},ProblemDesc{3,128,255,256,1,0,0},
               ProblemDesc{3,128,256,248,1,0,0}}) assert(!MakeDirectPlan(bad,{128,256},caps).enabled);
 auto wrong=caps;wrong.supported=false;assert(!MakeDirectPlan(p,{128,256},wrong).enabled);
 wrong=caps;wrong.l0c=131071;assert(!MakeDirectPlan(p,{128,256},wrong).enabled);
 wrong=caps;wrong.l0a=4095;assert(!MakeDirectPlan(p,{128,256},wrong).enabled);
 wrong=caps;wrong.l0b=32768;d=MakeDirectPlan(p,{128,256},wrong);assert(d.enabled && d.blockK==64);
 wrong=caps;wrong.l1=12287;assert(!MakeDirectPlan(p,{128,256},wrong).enabled);
 // Outer tiles larger than the real dimensions use only the actual capacity.
 p={1,16,32,16,1,0,0};d=MakeDirectPlan(p,{256,256},caps);
 assert(d.enabled && d.c0Bytes==2048 && d.a1Bytes==512);
 assert(!MakeDirectPlan(p,{0,32},caps).enabled);
 assert(!MakeDirectPlan(p,{17,32},caps).enabled);
}
''')

    def test_actual_cube_physical_layouts_full_k_and_pitched_fp32(self):
        self.assertTrue((ROOT / 'direct_cube.asc').exists(), 'direct Cube executor not implemented')
        helper = (ROOT / 'direct_cube.asc').read_text().split('// Direct C/V execution')[0]
        self.compile_run((ROOT / 'tests/direct_cube_stubs.h').read_text() +
                         '\n#include "direct_plan.h"\nnamespace local_baseline {\n' + helper +
                         '\n}\n' + (ROOT / 'tests/direct_cube_main.cpp').read_text())

    def test_actual_cv_protocol_reuse_drain_and_reductions(self):
        code = (ROOT / 'direct_cube.asc').read_text()
        self.assertIn('DirectStreamCube(', code, 'independent Cube producer not implemented')
        kernel = (ROOT / 'kernel.asc').read_text()
        stream = (ROOT / 'stream_matmul.asc').read_text()
        constants = '\n'.join(re.findall(r'constexpr uint32_t REDUCE_\w+ = \d+;', kernel))
        fold = kernel[kernel.index('__aicore__ inline void FoldMaxColumns('):kernel.index('__aicore__ inline void ComputeRowMaxima(')]
        sums = kernel[kernel.index('__aicore__ inline void SumRowMaxima('):kernel.index('template <typename T, bool TA, bool TB>', kernel.index('__aicore__ inline void SumRowMaxima('))]
        consume = stream[:stream.index('template <typename T, bool TA, bool TB, bool Async')]
        merge = stream[stream.index('__aicore__ inline void MergeStreamMaxima('):stream.index('// Device entry point')]
        stubs = (ROOT / 'tests/reduction_cpu_stubs.h').read_text().replace('#pragma once', '')
        stubs = stubs.replace('enum { PIPE_V, PIPE_ALL };', 'enum { PIPE_V, PIPE_ALL, PIPE_FIX, PIPE_MTE2 };')
        stubs = stubs.replace('inline float readGm(float* p) {', 'static void (*readHook)(float*)=nullptr;\ninline float readGm(float* p) {\nif(readHook)readHook(p);')
        protocol = code.split('// Direct C/V execution')[1].split('// Direct device entry')[0]
        source = stubs + '\n#include "stream_plan.h"\n#include "direct_plan.h"\n'
        source += (ROOT / 'tests/direct_protocol_stubs.h').read_text()
        source += '\nnamespace local_baseline {\n' + constants + '\n'
        source += (ROOT / 'partial_sum.asc').read_text() + sums + fold + consume + merge + protocol
        source += '\n}\n' + (ROOT / 'tests/direct_protocol_main.cpp').read_text()
        self.compile_run(source)
