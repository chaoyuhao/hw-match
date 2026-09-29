"""Exercise the diagnostic driver; fake only the unavailable CANN build/runtime.

These tests do not compile Ascend C or establish NPU compatibility.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_env.sh"


class CheckEnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cann-check-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.toolkit = self.root / "toolkit"
        self.toolkit.mkdir()
        self.env = dict(os.environ, ASCEND_HOME_PATH=str(self.toolkit),
                        PATH=str(self.bin) + ":/usr/bin:/bin",
                        CHECK_CALLS=str(self.root / "calls"))
        self.write_tool("bisheng", 'echo "test compiler (not CANN)"\n')
        self.write_tool("npu-smi", 'echo "test device inventory"\n')
        self.write_tool("cmake", r'''
printf '%s\n' "$*" >> "$CHECK_CALLS"
if [ "$1" = "--version" ]; then
    echo "cmake version 3.25.1"
elif [ "$1" = "--build" ]; then
    [ "${CHECK_BUILD_FAIL:-0}" = 0 ] || exit 8
    printf '#!/bin/sh\nexit %s\n' "${CHECK_RUN_EXIT:-0}" > "$2/env_probe"
    chmod +x "$2/env_probe"
else
    [ "${CHECK_CONFIG_FAIL:-0}" = 0 ] || exit 7
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "-B" ]; then mkdir -p "$2"; break; fi
        shift
    done
fi
''')

    def write_tool(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def run_check(self, *args):
        result = subprocess.run(
            ["/bin/bash", str(SCRIPT), "--output-dir", str(self.root / "reports"), *args],
            env=self.env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=30,
        )
        reports = list((self.root / "reports").glob("*/report.log"))
        if reports:
            self.assertIn("ONLINE_EVALUATION=NOT_RUN", reports[0].read_text())
        return result

    def test_inspection_does_not_configure_build_or_run_kernel(self):
        result = self.run_check("--inspect-only")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[SKIP] configure", result.stdout)
        self.assertIn("[SKIP] runtime", result.stdout)
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["--version"])

    def test_configure_failure_cannot_be_reported_as_success(self):
        self.env["CHECK_CONFIG_FAIL"] = "1"
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("[FAIL] configure", result.stdout)
        self.assertIn("[SKIP] build", result.stdout)
        self.assertIn("[SKIP] runtime", result.stdout)
        self.assertNotIn("--build", (self.root / "calls").read_text())

    def test_build_failure_skips_runtime(self):
        self.env["CHECK_BUILD_FAIL"] = "1"
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("[FAIL] build", result.stdout)
        self.assertIn("[SKIP] runtime", result.stdout)

    def test_runtime_failure_is_preserved(self):
        self.env["CHECK_RUN_EXIT"] = "9"
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("[PASS] build", result.stdout)
        self.assertIn("[FAIL] runtime", result.stdout)

    def test_compile_only_does_not_execute_failing_runtime(self):
        self.env["CHECK_RUN_EXIT"] = "9"
        result = self.run_check("--no-run")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[PASS] build", result.stdout)
        self.assertIn("[SKIP] runtime", result.stdout)

    def test_broken_compiler_is_not_treated_as_available(self):
        self.write_tool("bisheng", 'echo "broken shared library" >&2\nexit 127\n')
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("[FAIL] compiler", result.stdout)
        self.assertIn("[SKIP] configure", result.stdout)

    def test_noop_build_cannot_pass_without_an_executable(self):
        self.write_tool("cmake", 'echo "no-op build command"\n')
        result = self.run_check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("[FAIL] build_artifact", result.stdout)
        self.assertIn("[SKIP] runtime", result.stdout)

    def test_parent_environment_script_is_used_with_latest_symlink(self):
        (self.root / "latest").symlink_to(self.toolkit, target_is_directory=True)
        self.env["ASCEND_HOME_PATH"] = str(self.root / "latest")
        (self.root / "set_env.sh").write_text('export CANN_CHECK_SOURCE_LOADED=1\n')
        self.write_tool("bisheng", '[ "${CANN_CHECK_SOURCE_LOADED:-}" = 1 ] || exit 6\necho "test compiler"\n')
        result = self.run_check("--no-run")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[PASS] compiler", result.stdout)
        self.assertIn("[PASS] build", result.stdout)

    def test_older_toolkit_warns_without_claiming_online_compatibility(self):
        (self.toolkit / "version.cfg").write_text("version=8.1.RC1\n")
        result = self.run_check("--no-run")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[WARN] toolkit_version: observed=8.1.RC1", result.stdout)
        self.assertNotIn("[PASS] toolkit_version", result.stdout)

    def test_environment_script_failure_preserves_remaining_diagnostics(self):
        env_script = self.root / "broken_env.sh"
        env_script.write_text("set -e\nfalse\necho unreachable_environment_command\n")
        result = self.run_check("--inspect-only", "--env-script", str(env_script))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("[FAIL] env_script", result.stdout)
        self.assertIn("[PASS] compiler", result.stdout)
        self.assertIn("Local diagnostic summary:", result.stdout)
        self.assertNotIn("unreachable_environment_command", result.stdout)

    def test_invalid_device_argument_is_rejected(self):
        result = self.run_check("--device", "7;echo injected")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertFalse((self.root / "calls").exists())

    def test_environment_script_exit_preserves_remaining_diagnostics(self):
        env_script = self.root / "exiting_env.sh"
        env_script.write_text("exit 3\n")
        result = self.run_check("--inspect-only", "--env-script", str(env_script))
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("[FAIL] env_script: source exited 3", result.stdout)
        self.assertIn("[PASS] compiler", result.stdout)
        self.assertIn("Local diagnostic summary:", result.stdout)


if __name__ == "__main__":
    unittest.main()
