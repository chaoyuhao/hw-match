"""Exercise the actual new device helpers on CPU; not an NPU/synchronization test."""
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ParallelReductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="cann-reduction-cpu-")
        cls.binary = Path(cls.tmp.name) / "reduction_cpu"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_family(self, family):
        kernel = (ROOT / "kernel.asc").read_text()
        self.assertTrue("__aicore__ inline void ComputeRowMaxima(" in kernel,
                      "parallel row-max helper has not been implemented")
        self.assertTrue("__aicore__ inline void SumRowMaxima(" in kernel)
        if not shutil.which("g++"):
            self.skipTest("host C++ compiler unavailable")
        if not self.binary.exists():
            start = kernel.index("__aicore__ inline void ComputeRowMaxima(")
            end = kernel.index("template <typename T, bool TA, bool TB>", start)
            constants = "\n".join(re.findall(r"constexpr uint32_t REDUCE_\w+ = \d+;", kernel))
            source = Path(self.tmp.name) / "test.cpp"
            source.write_text('#include "reduction_cpu_stubs.h"\n#define __gm__\nnamespace local_baseline {\n' +
                              constants + "\n" + kernel[start:end] + "\n}\n" +
                              (ROOT / "tests/reduction_cpu_main.cpp").read_text())
            compile_result = subprocess.run(["g++", "-std=c++14", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror",
                                             "-I", str(ROOT / "tests"), str(source), "-o", str(self.binary)],
                                            capture_output=True, text=True, timeout=60)
            self.assertEqual(compile_result.returncode, 0, compile_result.stdout + compile_result.stderr)
        result = subprocess.run([str(self.binary), str(family)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_negative_rows_and_dma_vector_tails_exclude_padding(self):
        self.run_family(0)

    def test_task_reuse_idle_cores_and_batch_groups_have_unique_output_owners(self):
        self.run_family(1)

    def test_sum_preserves_cancellation_across_chunk_boundaries(self):
        self.run_family(2)

    def test_sum_fp64_golden_padding_and_bounded_scalar_work(self):
        self.run_family(3)


if __name__ == "__main__":
    unittest.main()
