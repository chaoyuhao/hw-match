#!/usr/bin/env bash
set -eo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENERATE_ONLY=0
for argument in "$@"; do
    case "$argument" in
        --generate-only) GENERATE_ONLY=1 ;;
        -h|--help)
            echo 'Usage: bash scripts/run_local.sh [--suite smoke|full|reduction|tiling|generated|small] [--case NAME] [--device 0] [--repeat 2] [--timeout 120] [--generate-only] [--no-dump-similarity]'
            echo 'Generated suite: --case-count 64 --case-seed 20260930 --case-cores 24 (boundary hint only).'
            echo 'Local tile experiment: CANN_MATMUL_TILE=auto|MxN; M/N: 16-aligned, 16..256 (default auto).'
            echo 'Optional family override: CANN_EXECUTION_FAMILY=auto|gm|small|stream|pipeline; online always uses auto.'
            exit 0 ;;
        --output-dir|--binary|--output-dir=*|--binary=*)
            echo 'run_local.sh manages unique output/build paths; use local_baseline.py directly for custom paths.' >&2
            exit 2 ;;
    esac
done
if [ "$GENERATE_ONLY" -eq 0 ]; then
    CANN_ENV_SCRIPT=""
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        for candidate in "$ASCEND_HOME_PATH/set_env.sh" "$ASCEND_HOME_PATH/../set_env.sh"; do
            if [ -r "$candidate" ]; then CANN_ENV_SCRIPT="$candidate"; break; fi
        done
    fi
    if [ -n "$CANN_ENV_SCRIPT" ]; then
        echo "Loading $CANN_ENV_SCRIPT"
        source "$CANN_ENV_SCRIPT"
    fi
    set -euo pipefail
    for command_name in bisheng cmake make timeout; do
        command -v "$command_name" >/dev/null || { echo "Missing $command_name; source the CANN environment first." >&2; exit 1; }
    done
fi
set -euo pipefail
mkdir -p "$PROJECT_ROOT/build-baseline"
RUN_DIR="$(mktemp -d "$PROJECT_ROOT/build-baseline/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
echo "Report: $RUN_DIR/run.log"
run_all() {
    echo 'LOCAL_BASELINE; ONLINE_EVALUATION=NOT_RUN'
    git -C "$PROJECT_ROOT" rev-parse HEAD
    git -C "$PROJECT_ROOT" status --short
    sha256sum "$PROJECT_ROOT/kernel.asc" "$PROJECT_ROOT/matmul_plan.h" "$PROJECT_ROOT/small_plan.h" "$PROJECT_ROOT/small_vector.h" "$PROJECT_ROOT/stream_plan.h" "$PROJECT_ROOT/stream_matmul.asc" "$PROJECT_ROOT/joint_plan.h"
    printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-<unset>}"
    if [ "$GENERATE_ONLY" -eq 1 ]; then
        python3 "$PROJECT_ROOT/scripts/local_baseline.py" --output-dir "$RUN_DIR/cases" "$@"
    else
        timeout --kill-after=5s 120s cmake -S "$PROJECT_ROOT/local" -B "$RUN_DIR/build" \
            -G 'Unix Makefiles' "-DNPU_ARCH=${NPU_ARCH:-dav-2201}"
        timeout --kill-after=5s 600s cmake --build "$RUN_DIR/build" --parallel 2
        python3 "$PROJECT_ROOT/scripts/local_baseline.py" --binary "$RUN_DIR/build/baseline_runner" \
            --output-dir "$RUN_DIR/cases" "$@"
    fi
}
run_all "$@" 2>&1 | tee "$RUN_DIR/run.log"
