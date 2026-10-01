#!/usr/bin/env bash
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_REQUIRED=1
for argument in "$@"; do
    case "$argument" in
        --inspect-tools|--list-cases|--generate-only) BUILD_REQUIRED=0 ;;
        -h|--help) python3 "$PROJECT_ROOT/scripts/perf_local.py" --help; exit 0 ;;
        --output-dir|--binary|--output-dir=*|--binary=*)
            echo 'run_perf.sh manages fresh builds and unique output paths; use perf_local.py for custom paths.' >&2
            exit 2 ;;
    esac
done
if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    for candidate in "$ASCEND_HOME_PATH/set_env.sh" "$ASCEND_HOME_PATH/../set_env.sh"; do
        if [ -r "$candidate" ]; then
            echo "Loading $candidate"
            source "$candidate"
            break
        fi
    done
fi
set -euo pipefail
if [ "$BUILD_REQUIRED" -eq 1 ]; then
    for command_name in bisheng cmake make timeout; do
        command -v "$command_name" >/dev/null || { echo "Missing $command_name; source the CANN environment first." >&2; exit 1; }
    done
fi
mkdir -p "$PROJECT_ROOT/build-perf"
RUN_DIR="$(mktemp -d "$PROJECT_ROOT/build-perf/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
echo "Report directory: $RUN_DIR"
run_all() {
    echo 'LOCAL_PERFORMANCE; ONLINE_EVALUATION=NOT_RUN'
    git -C "$PROJECT_ROOT" rev-parse HEAD
    git -C "$PROJECT_ROOT" status --short
    sha256sum "$PROJECT_ROOT/kernel.asc" "$PROJECT_ROOT/matmul_plan.h" "$PROJECT_ROOT/small_plan.h" "$PROJECT_ROOT/small_vector.h" "$PROJECT_ROOT/stream_plan.h" "$PROJECT_ROOT/stream_matmul.asc" "$PROJECT_ROOT/joint_plan.h" "$PROJECT_ROOT/reduction_plan.h" "$PROJECT_ROOT/partial_sum.asc"
    if [ "$BUILD_REQUIRED" -eq 0 ]; then
        python3 "$PROJECT_ROOT/scripts/perf_local.py" --output-dir "$RUN_DIR/cases" "$@"
    else
        timeout --kill-after=5s 120s cmake -S "$PROJECT_ROOT/local" -B "$RUN_DIR/build" \
            -G 'Unix Makefiles' "-DNPU_ARCH=${NPU_ARCH:-dav-2201}"
        timeout --kill-after=5s 600s cmake --build "$RUN_DIR/build" --parallel 2
        python3 "$PROJECT_ROOT/scripts/perf_local.py" --binary "$RUN_DIR/build/baseline_runner" \
            --output-dir "$RUN_DIR/cases" "$@"
    fi
}
run_all "$@" 2>&1 | tee "$RUN_DIR/run.log"
