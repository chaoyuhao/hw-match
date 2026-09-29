#!/usr/bin/env bash
# Local diagnostics only. The competition platform determines acceptance/score.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=full
DEVICE=0
ARCH="${NPU_ARCH:-dav-2201}"
OUTPUT_DIR="${PROJECT_ROOT}/build-envcheck"
ENV_SCRIPT=""

usage() {
    cat <<'EOF'
Usage: bash scripts/check_env.sh [options]
  --inspect-only      Collect diagnostics; do not configure, compile or run a kernel
  --no-run            Collect diagnostics and compile; do not run a kernel
  --device ID         ACL logical device ID (default: 0, not npu-smi physical ID)
  --arch ARCH         Compiler target (default: NPU_ARCH or dav-2201)
  --env-script PATH   Source this CANN environment script inside the checker
  --output-dir DIR    Parent of a new, unique report/build directory
  -h, --help          Show this message

No packages are installed. Existing build directories are not removed.
An environment check is NOT a competition correctness test or score.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --inspect-only) MODE=inspect; shift ;;
        --no-run) MODE=compile; shift ;;
        --device|--arch|--env-script|--output-dir)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                printf 'Missing value for %s\n' "$1" >&2; exit 2
            fi
            case "$1" in
                --device) DEVICE="$2" ;;
                --arch) ARCH="$2" ;;
                --env-script) ENV_SCRIPT="$2" ;;
                --output-dir) OUTPUT_DIR="$2" ;;
            esac
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
if ! [[ "$DEVICE" =~ ^[0-9]+$ ]] || ! [[ "$ARCH" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "Invalid device ID or architecture." >&2; exit 2
fi
if [ -n "$ENV_SCRIPT" ] && [ ! -r "$ENV_SCRIPT" ]; then
    printf 'Environment script is not readable: %s\n' "$ENV_SCRIPT" >&2; exit 2
fi
mkdir -p "$OUTPUT_DIR" || exit 1
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
REPORT_DIR="$(mktemp -d "${OUTPUT_DIR}/check-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")" || exit 1

FAILURES=0
WARNINGS=0
record() {
    local level="$1" component="$2"
    shift 2
    [ "$level" != FAIL ] || FAILURES=$((FAILURES + 1))
    [ "$level" != WARN ] || WARNINGS=$((WARNINGS + 1))
    printf '[%s] %s: %s\n' "$level" "$component" "$*"
}

# Keep individual tool logs even if a later stage fails. Missing/broken tools
# must not cause an early exit that hides all remaining diagnostics.
run_check() {
    local failure_level="$1" component="$2" limit="$3" rc
    shift 3
    printf '\nCommand:'
    printf ' %q' "$@"
    printf '\n'
    timeout --kill-after=5s "$limit" "$@" >"${REPORT_DIR}/${component}.log" 2>&1
    rc=$?
    cat "${REPORT_DIR}/${component}.log"
    if [ "$rc" -eq 0 ]; then
        record PASS "$component" "command completed"
        return 0
    fi
    record "$failure_level" "$component" "exit=${rc}; see ${component}.log (124/137 may indicate timeout)"
    return 1
}

check_tool() {
    local component="$1" command_name="$2" command_path
    command_path="$(command -v "$command_name" 2>/dev/null || true)"
    if [ -z "$command_path" ]; then
        record FAIL "$component" "${command_name} is not in PATH"
        return 1
    fi
    printf '%s=%s\n' "$command_name" "$command_path"
    run_check FAIL "$component" 20s "$command_path" --version
}

main() {
    echo "LOCAL_ENVIRONMENT_CHECK (not a benchmark)"
    echo "ONLINE_EVALUATION=NOT_RUN"
    echo "Competition target from README: CANN 9.0.0"
    printf 'mode=%s arch=%s ACL_logical_device=%s\n' "$MODE" "$ARCH" "$DEVICE"
    printf 'report_directory=%s\n' "$REPORT_DIR"
    date -u
    uname -a
    if [ -r /etc/os-release ]; then
        sed -n '/^PRETTY_NAME=/p' /etc/os-release
    fi
    if command -v git >/dev/null 2>&1; then
        git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true
        git -C "$PROJECT_ROOT" status --short 2>/dev/null || true
    fi
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "${PROJECT_ROOT}/kernel.asc"
    fi
    if ! command -v timeout >/dev/null 2>&1; then
        record FAIL timeout "GNU timeout is required to bound diagnostic/build/runtime commands"
        return 1
    fi

    local toolkit="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}" candidate compiler
    if [ -z "$toolkit" ]; then
        compiler="$(command -v bisheng 2>/dev/null || true)"
        case "$compiler" in
            */compiler/ccec_compiler/bin/bisheng)
                toolkit="${compiler%/compiler/ccec_compiler/bin/bisheng}" ;;
        esac
    fi
    if [ -z "$ENV_SCRIPT" ]; then
        local candidates=()
        if [ -n "$toolkit" ]; then
            candidates=("$toolkit/set_env.sh" "$toolkit/../set_env.sh")
        else
            candidates=(/usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/cann/set_env.sh)
            if [ -n "${HOME:-}" ]; then
                candidates+=("$HOME/Ascend/ascend-toolkit/set_env.sh" "$HOME/Ascend/cann/set_env.sh")
            fi
        fi
        for candidate in "${candidates[@]}"; do
            if [ -r "$candidate" ]; then ENV_SCRIPT="$candidate"; break; fi
        done
    fi
    if [ -n "$ENV_SCRIPT" ]; then
        printf 'Loading environment script: %s\n' "$ENV_SCRIPT"
        local source_rc=1 env_snapshot=""
        # A separate Bash preserves the vendor script's errexit/exit semantics.
        # Never put the exported environment (which may contain secrets) in logs.
        env_snapshot="$(mktemp "${TMPDIR:-/tmp}/cann-envcheck-XXXXXX")"
        if [ -n "$env_snapshot" ]; then
            timeout --kill-after=5s 20s bash -c '
                readonly CANN_ENV_SNAPSHOT_DEST="$2"
                source "$1"
                source_rc=$?
                [ "$source_rc" -eq 0 ] || exit "$source_rc"
                export -p > "$CANN_ENV_SNAPSHOT_DEST"
            ' bash "$ENV_SCRIPT" "$env_snapshot"
            source_rc=$?
            if [ "$source_rc" -eq 0 ]; then
                if [ -s "$env_snapshot" ]; then
                    source "$env_snapshot"
                    source_rc=$?
                else
                    source_rc=1
                fi
            fi
            rm -f "$env_snapshot"
        fi
        if [ "$source_rc" -eq 0 ]; then
            record PASS env_script "loaded"
        else
            record FAIL env_script "source exited ${source_rc}; continuing diagnostics"
        fi
    else
        record WARN env_script "not found; checking the existing environment"
    fi
    toolkit="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-$toolkit}}"
    printf 'ASCEND_HOME_PATH=%s\nASCEND_TOOLKIT_HOME=%s\n' "${ASCEND_HOME_PATH:-<unset>}" "${ASCEND_TOOLKIT_HOME:-<unset>}"
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s\nASCEND_VISIBLE_DEVICES=%s\n' "${ASCEND_RT_VISIBLE_DEVICES:-<unset>}" "${ASCEND_VISIBLE_DEVICES:-<unset>}"
    printf 'ASC_DIR=%s\nCMAKE_PREFIX_PATH=%s\n' "${ASC_DIR:-<unset>}" "${CMAKE_PREFIX_PATH:-<unset>}"

    local version="" version_file found_version
    if [ -n "$toolkit" ] && [ -d "$toolkit" ]; then
        printf 'Toolkit candidate: %s\n' "$toolkit"
        # Follow SDK directory symlinks, including a symlink named latest.
        run_check WARN sdk_files 30s find -L "$toolkit" -maxdepth 8 \
            \( -name ASCConfig.cmake -o -name ascendc.cmake -o -name version.info -o -name version.cfg \) -print || true
        for version_file in "$toolkit/version.cfg" "$toolkit/version.info" \
            "$toolkit/ascend_toolkit_install.info" "$toolkit/$(uname -m)-linux/ascend_toolkit_install.info"; do
            if [ -r "$version_file" ]; then
                printf 'Toolkit version metadata: %s\n' "$version_file"
                found_version="$(sed -nE 's/^[[:space:]]*(version|Version|CANN_VERSION)[[:space:]]*[=:][[:space:]]*([^[:space:]]+).*/\2/p' "$version_file" | sed -n '1p')"
                printf 'Observed toolkit version: %s\n' "${found_version:-<unrecognized metadata>}"
                if [ -z "$version" ]; then version="$found_version"; fi
            fi
        done
    else
        record WARN sdk_files "toolkit directory unknown; CMake will test actual package discovery"
    fi
    if [ "$version" = 9.0.0 ]; then
        record PASS toolkit_version "metadata reports 9.0.0; online compatibility is still untested"
    else
        record WARN toolkit_version "observed=${version:-UNKNOWN}; target=9.0.0; driver/Clang versions are not CANN versions"
    fi

    local build_ready=1 python_command=""
    check_tool compiler bisheng || build_ready=0
    check_tool cmake cmake || build_ready=0
    check_tool make make || build_ready=0
    if command -v npu-smi >/dev/null 2>&1; then
        run_check WARN device_inventory 20s npu-smi info || true
    else
        record WARN device_inventory "npu-smi unavailable; ACL runtime probe is independent of npu-smi"
    fi
    for candidate in python3 python3.10 python; do
        if command -v "$candidate" >/dev/null 2>&1; then python_command="$candidate"; break; fi
    done
    if [ -n "$python_command" ]; then
        run_check WARN python 20s "$python_command" - <<'PY'
import importlib
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)
missing = []
for name in ("numpy", "ml_dtypes"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "unknown"), module.__file__)
    except Exception as exc:
        missing.append(name)
        print(name, "unavailable:", exc)
sys.exit(bool(missing))
PY
        # Python only generates/verifies competition data; the standalone probe
        # needs neither Python packages nor PyTorch.
    else
        record WARN python "Python unavailable; competition data generation needs numpy and ml_dtypes"
    fi

    if [ "$MODE" = inspect ] || [ "$build_ready" -eq 0 ]; then
        record SKIP configure "inspection requested or required tools failed"
        record SKIP build "configure was not attempted"
        record SKIP runtime "no probe was built"
    elif run_check FAIL configure 120s cmake -S "${PROJECT_ROOT}/scripts/env_probe" \
        -B "${REPORT_DIR}/build" -G "Unix Makefiles" "-DNPU_ARCH=${ARCH}"; then
        if run_check FAIL build 180s cmake --build "${REPORT_DIR}/build" --parallel 2; then
            if [ ! -x "${REPORT_DIR}/build/env_probe" ]; then
                record FAIL build_artifact "build command produced no executable env_probe"
                record SKIP runtime "probe executable is missing"
            elif [ "$MODE" = compile ]; then
                record SKIP runtime "--no-run requested; NPU execution remains untested"
            else
                run_check FAIL runtime 30s "${REPORT_DIR}/build/env_probe" "$DEVICE" || true
            fi
        else
            record SKIP runtime "build failed"
        fi
    else
        record SKIP build "CMake configuration failed; inspect configure.log for ASC package/toolchain errors"
        record SKIP runtime "no probe was built"
    fi

    printf '\nLocal diagnostic summary: failures=%d warnings=%d\n' "$FAILURES" "$WARNINGS"
    echo "Scope: standalone vector-add probe; BatchMatmulMaxSum and Cube Matmul are NOT validated."
    echo "ONLINE_EVALUATION=NOT_RUN"
    echo "Acceptance, performance and score are determined by the official online platform."
    printf 'Report: %s/report.log\n' "$REPORT_DIR"
    [ "$FAILURES" -eq 0 ]
}

main 2>&1 | tee "${REPORT_DIR}/report.log"
CHECK_STATUSES=("${PIPESTATUS[@]}")
if [ "${CHECK_STATUSES[1]}" -ne 0 ]; then
    echo "Failed to save the complete diagnostic report." >&2
    exit 1
fi
exit "${CHECK_STATUSES[0]}"
