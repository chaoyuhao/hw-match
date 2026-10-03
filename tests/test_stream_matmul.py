"""Actual streaming helpers with checked CPU operations, not an NPU simulator."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class StreamMatmulTests(unittest.TestCase):
    def test_actual_helpers_reuse_slots_merge_shards_and_match_fp64(self):
        self.assertTrue((ROOT/'stream_matmul.asc').exists(), 'stream helpers not implemented')
        kernel=(ROOT/'kernel.asc').read_text()
        start=kernel.index('__aicore__ inline void SumRowMaxima(')
        end=kernel.index('template <typename T, bool TA, bool TB>',start)
        constants='\n'.join(re.findall(r'constexpr uint32_t REDUCE_\w+ = \d+;',kernel))
        fold=kernel[kernel.index('__aicore__ inline void FoldMaxColumns('):kernel.index('__aicore__ inline void ComputeRowMaxima(')]
        helpers=(ROOT/'stream_matmul.asc').read_text().split('// Device entry point')[0]
        stubs=(ROOT/'tests/reduction_cpu_stubs.h').read_text().replace('#pragma once','')
        # Observe explicit Scalar fences and each copied C element. This checks
        # emitted helper sequencing; it cannot emulate asynchronous hardware.
        stubs=stubs.replace('template <HardEvent E> void WaitFlag(int) {}',
            'static unsigned mte2Fences=0;\ntemplate <HardEvent E> void WaitFlag(int) { if(E==HardEvent::MTE2_S) ++mte2Fences; }')
        stubs=stubs.replace('inline float readGm(float* p) {','static void (*readHook)(float*)=nullptr;\ninline float readGm(float* p) {\n    if(readHook) readHook(p);')
        stubs=stubs.replace('inline void writeGm(float* p, float v) {','static void (*writeHook)(float*)=nullptr;\ninline void writeGm(float* p, float v) {\n    if(writeHook) writeHook(p);')
        source=stubs+'\n#include "stream_plan.h"\n#define __gm__\nnamespace local_baseline {\n'+constants+'\n'+(ROOT/'partial_sum.asc').read_text()+kernel[start:end]+fold+helpers+'\n}\n'+(ROOT/'tests/stream_cpu_main.cpp').read_text()
        with tempfile.TemporaryDirectory() as directory:
            cpp=Path(directory)/'stream.cpp'; binary=Path(directory)/'stream';cpp.write_text(source)
            done=subprocess.run(['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(ROOT),str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True,timeout=60)
            self.assertEqual(done.returncode,0,done.stderr)
