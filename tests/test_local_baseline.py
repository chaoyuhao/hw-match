import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "scripts/local_baseline.py"


class LocalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.is_file(), "local baseline data/verification tool is missing")
        spec = importlib.util.spec_from_file_location("local_baseline", MODULE)
        self.api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)

    def test_known_values_and_all_storage_layouts(self):
        a = np.array([[[1, 0], [0, 1]]], dtype=np.float32)
        b = np.array([[[1, 0, -1], [0, 1, 0]]], dtype=np.float32)
        for dtype in ("fp16", "bf16"):
            for ta in (False, True):
                for tb in (False, True):
                    with self.subTest(dtype=dtype, ta=ta, tb=tb):
                        pa, pb, similarity, y = self.api.prepare_inputs(a, b, dtype, ta, tb)
                        self.assertEqual(pa.shape, (1, 2, 2))
                        self.assertEqual(pb.shape, (1, 3, 2) if tb else (1, 2, 3))
                        np.testing.assert_array_equal(y, [2.0])
                        np.testing.assert_array_equal(similarity, [[[1, 0, -1], [0, 1, 0]]])

    def test_all_negative_maximum_and_batch_pairing(self):
        a = np.array([[[1, 0]], [[2, 0]]], dtype=np.float32)
        b = np.array([[[-1, -2], [0, 0]], [[-3, -4], [0, 0]]], dtype=np.float32)
        _, _, _, y = self.api.prepare_inputs(a, b, "fp16", True, True)
        np.testing.assert_array_equal(y, [-1.0, -6.0])

    def test_nonsquare_storage_contains_transposed_data(self):
        a = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        b = np.array([[[1, 0, -1, 2], [0, 1, 2, -1], [1, 1, 0, 0]]], dtype=np.float32)
        pa, pb, similarity, y = self.api.prepare_inputs(a, b, "fp16", True, True)
        np.testing.assert_array_equal(pa, [[[1, 4], [2, 5], [3, 6]]])
        np.testing.assert_array_equal(pb, [[[1, 0, 1], [0, 1, 1], [-1, 2, 0], [2, -1, 0]]])
        np.testing.assert_array_equal(similarity, [[[4, 5, 3, 0], [10, 11, 6, 3]]])
        np.testing.assert_array_equal(y, [16])

    def test_bf16_rounds_ties_to_even_and_golden_uses_stored_values(self):
        values = np.array([0, 1, -2, 1 + 1 / 256, 1 + 3 / 256], dtype=np.float32)
        packed, decoded = self.api.quantize(values, "bf16")
        np.testing.assert_array_equal(packed, [0x0000, 0x3F80, 0xC000, 0x3F80, 0x3F82])
        np.testing.assert_array_equal(decoded, [0, 1, -2, 1, 1.015625])
        _, _, _, y = self.api.prepare_inputs(
            np.array([[[1.00390625]]]), np.array([[[1.00390625]]]), "bf16", False, False)
        np.testing.assert_array_equal(y, [1.0])

    def test_verifier_rejects_truncation_nonfinite_and_wrong_values(self):
        for value in ([1], [1, np.nan], [1, np.inf], [1, -2], [1, 2, 3]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.api.compare(np.asarray(value, dtype=np.float32), np.array([1, 2], dtype=np.float32))
        stats = self.api.compare(np.array([1, 2.00001]), np.array([1, 2]))
        self.assertLess(stats["max_abs_error"], 0.0001)

    def test_suite_covers_dtype_layout_tail_negative_and_long_k(self):
        cases = self.api.cases_for_suite("full")
        self.assertEqual({(c["dtype"], c["ta"], c["tb"]) for c in cases},
                         {(d, a, b) for d in ("fp16", "bf16") for a in (False, True) for b in (False, True)})
        self.assertTrue(any(c["k"] == 8192 for c in cases))
        self.assertTrue(any(c["m"] % 16 and c["n"] % 16 and c["k"] == 24 for c in cases))
        self.assertTrue(any(c["pattern"] == "negative" for c in cases))
        self.assertTrue(any(c["b"] > 1 for c in cases))
        self.assertTrue(any(c["n"] > 1024 for c in cases))
        self.assertTrue(any(c["b"] > 8 * 24 for c in cases), "exercise reduction group reuse on the 24-Cube-core host")

    def test_reduction_suite_covers_new_tile_and_sum_boundaries(self):
        cases = self.api.cases_for_suite("reduction")
        for n in (63, 64, 65, 255, 256, 257):
            layouts = {(c["dtype"], c["ta"], c["tb"]) for c in cases if c["n"] == n}
            self.assertEqual(layouts, {(d, a, b) for d in ("fp16", "bf16")
                                     for a in (False, True) for b in (False, True)})
        self.assertTrue({31, 32, 33, 1023, 1024, 1025, 8192}.issubset({c["m"] for c in cases}))
        self.assertTrue(any(c["b"] == 257 and c["m"] > 32 for c in cases))
        self.assertTrue(any(c["pattern"] == "negative" and c["m"] == 8192 for c in cases))
        self.assertEqual(len(self.api.cases_for_suite("full")), 55)

    def test_row_cancellation_pattern_has_known_positive_and_negative_totals(self):
        candidates = [c for c in self.api.cases_for_suite("reduction")
                      if c["pattern"] == "row_cancellation" and c["m"] == 1024]
        self.assertEqual(len(candidates), 2)
        with tempfile.TemporaryDirectory() as tmp:
            for case in candidates:
                similarity, y = self.api.make_case(case, Path(tmp) / case["name"])
                np.testing.assert_array_equal(y, [4.0, -4.0])
                np.testing.assert_array_equal(similarity[0, :4, 0], [4096, 0.03125, -4096, -0.015625])

    def test_sum_carry_fixture_requires_compensation_across_chunk_boundary(self):
        candidates = [c for c in self.api.cases_for_suite("reduction") if c["pattern"] == "sum_carry"]
        self.assertEqual(len(candidates), 2)
        with tempfile.TemporaryDirectory() as tmp:
            for case in candidates:
                _, y = self.api.make_case(case, Path(tmp) / case["name"])
                np.testing.assert_array_equal(y, [2**-11, -2**-11])

    def test_failed_runner_never_produces_a_success_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failing = root / "runner"
            failing.write_text("#!/bin/sh\nexit 7\n")
            failing.chmod(0o755)
            case = self.api.cases_for_suite("smoke")[0]
            result = self.api.run_case(failing, case, root / "case", 0, 2, 10, False)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("7", result["error"])
            self.assertEqual(result["online_evaluation"], "NOT_RUN")

    def test_missing_output_cannot_pass_even_if_runner_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "runner"
            runner.write_text("#!/bin/sh\nexit 0\n")
            runner.chmod(0o755)
            case = self.api.cases_for_suite("smoke")[0]
            result = self.api.run_case(runner, case, root / "case", 0, 2, 10, False)
            self.assertEqual(result["status"], "FAIL")

    def test_repeated_outputs_must_match_bitwise_even_within_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "runner"
            runner.write_text(f"#!{sys.executable}\n" + """
import pathlib, struct, sys
p = pathlib.Path(sys.argv[1])
data = (p / 'golden_y.bin').read_bytes()
(p / 'y-0.bin').write_bytes(data)
bits, = struct.unpack('<I', data)
(p / 'y-1.bin').write_bytes(struct.pack('<I', bits ^ 1))
""")
            runner.chmod(0o755)
            result = self.api.run_case(runner, self.api.cases_for_suite("smoke")[0], root / "case", 0, 2, 10, False)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("bitwise", result["error"])

    def test_invalid_intermediate_is_not_hidden_by_correct_final_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "runner"
            runner.write_text(f"#!{sys.executable}\n" + """
import pathlib, sys
p = pathlib.Path(sys.argv[1])
(p / 'y-0.bin').write_bytes((p / 'golden_y.bin').read_bytes())
(p / 'similarity.bin').write_bytes(bytes(16 * 4))
""")
            runner.chmod(0o755)
            result = self.api.run_case(runner, self.api.cases_for_suite("smoke")[0], root / "case", 0, 1, 10, True)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("precision mismatch", result["error"])

    def test_timeout_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "runner"
            runner.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(10)\n")
            runner.chmod(0o755)
            result = self.api.run_case(runner, self.api.cases_for_suite("smoke")[0], root / "case", 0, 1, 1, False)
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("timed out", result["error"])


if __name__ == "__main__":
    unittest.main()
