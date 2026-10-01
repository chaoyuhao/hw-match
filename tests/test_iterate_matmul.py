"""Real Iterate producer/planner with checked CPU operations; not an NPU simulator."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import re
ROOT = Path(__file__).resolve().parents[1]

class IterateMatmulTests(unittest.TestCase):
    def test_session_ownership_padding_compensation_and_resources(self):
        self.assertTrue((ROOT/'iterate_matmul.asc').exists(), 'Iterate producer not implemented')
        kernel=(ROOT/'kernel.asc').read_text()
        start=kernel.index('__aicore__ inline void SumRowMaxima(')
        end=kernel.index('template <typename T, bool TA, bool TB>',start)
        constants='\n'.join(re.findall(r'constexpr uint32_t REDUCE_\w+ = \d+;',kernel))
        helpers=(ROOT/'stream_matmul.asc').read_text().split('// Device entry point')[0]
        iterate=(ROOT/'iterate_matmul.asc').read_text().split('// Device entry point')[0]
        source='#include "tests/reduction_cpu_stubs.h"\n#include "iterate_plan.h"\n#define __gm__\nnamespace local_baseline {\n'+constants+'\n'+(ROOT/'partial_sum.asc').read_text()+kernel[start:end]+helpers+iterate+'\n}\n'+(ROOT/'tests/iterate_cpu_main.cpp').read_text()
        with tempfile.TemporaryDirectory() as directory:
            cpp=Path(directory)/'iterate.cpp'; binary=Path(directory)/'iterate';cpp.write_text(source)
            done=subprocess.run(['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-I',str(ROOT),str(cpp),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)

    def test_planner_uses_resources_and_checked_sdk_footprint(self):
        self.assertTrue((ROOT/'iterate_plan.h').exists(), 'Iterate planner not implemented')
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/'plan'
            done=subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT),str(ROOT/'tests/iterate_plan_cpu.cpp'),'-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            done=subprocess.run([str(binary)],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
