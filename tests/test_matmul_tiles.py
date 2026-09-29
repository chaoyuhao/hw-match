"""Execute actual integer tiling/index helpers; does not simulate Cube or CANN."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MatmulTileTests(unittest.TestCase):
    def run_family(self, family):
        source = (ROOT / "kernel.asc").read_text()
        self.assertTrue("struct MatmulPlan" in source, "shape-aware Matmul planning is not implemented")
        if not shutil.which("g++"):
            self.skipTest("host C++ compiler unavailable")
        helpers = source[source.index("struct MatmulPlan"):source.index("__aicore__ inline void ComputeRowMaxima")]
        with tempfile.TemporaryDirectory(prefix="cann-tile-cpu-") as tmp:
            file = Path(tmp) / "test.cpp"
            file.write_text("#include <algorithm>\n#include <cstdint>\n#include <limits>\n#include <stdexcept>\n"
                            "#define __aicore__\nnamespace local_baseline {\n" + helpers + "\n}\n" +
                            (ROOT / "tests/matmul_tiles_cpu.cpp").read_text())
            binary = Path(tmp) / "test"
            build = subprocess.run(["g++", "-std=c++14", "-O2", "-Wall", "-Wextra", "-Werror",
                                    str(file), "-o", str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(binary), str(family)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_auto_reduces_tasks_without_losing_parallelism(self):
        self.run_family(0)

    def test_every_output_element_has_one_owner_for_all_tiles_and_tails(self):
        self.run_family(1)

    def test_transposed_offsets_preserve_physical_strides_and_64_bit_addresses(self):
        self.run_family(2)


if __name__ == "__main__":
    unittest.main()
