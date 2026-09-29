#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="batch_matmul_max_sum_custom"

echo "=== [1/4] Set CANN env ==="
CANN_ENV_SCRIPT=""
if [ -n "${ASCEND_HOME_PATH:-}" ]; then
    for candidate in \
        "${ASCEND_HOME_PATH}/set_env.sh" \
        "${ASCEND_HOME_PATH}/../set_env.sh"; do
        if [ -r "${candidate}" ]; then
            CANN_ENV_SCRIPT="${candidate}"
            break
        fi
    done
fi

if [ -n "${CANN_ENV_SCRIPT}" ]; then
    echo "Loading ${CANN_ENV_SCRIPT}"
    # Vendor environment scripts may reference variables that are not set yet.
    set +u
    source "${CANN_ENV_SCRIPT}"
    set -u
else
    echo "No set_env.sh found; checking the current compiler environment."
fi

if ! command -v bisheng >/dev/null 2>&1; then
    echo "ERROR: bisheng is not in PATH. Source your installed CANN set_env.sh first."
    echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}"
    exit 1
fi

echo "=== [2/4] Build ==="
rm -rf build
mkdir -p build
cd build
cmake ..
make -j4
cd ..

echo "=== [3/4] Gen test data ==="
cd build
python3 ../scripts/gen_data.py

echo "=== [4/4] Run + Verify ==="
rm -f input/*.bin
cp input/case0/* input/ 2>/dev/null || true
cp output/golden_case0/* output/ 2>/dev/null || true
find output -name '*.bin' ! -name 'golden_*' -delete 2>/dev/null || true
if timeout 120 "./${OP_NAME}"; then
    if python3 ../scripts/verify_result.py 0; then
        echo "=== PASSED ==="
    else
        echo "=== FAILED ==="
        exit 1
    fi
else
    echo "=== FAILED (kernel exited non-zero or timed out) ==="
    exit 1
fi
