#ifndef CANN_MATCH_DIRECT_PLAN_H
#define CANN_MATCH_DIRECT_PLAN_H
#include "matmul_plan.h"

namespace local_baseline {
// R22 changes the Cube engine, not the outer geometry or numerical reduction.
constexpr bool DIRECT_CUBE_ENABLED = true;
struct DirectCaps {
    bool supported = false;
    uint64_t l1 = 0, l0a = 0, l0b = 0, l0c = 0;
};
struct DirectPlan {
    bool enabled = false, residentA = false;
    uint32_t blockK = 0, a1Bytes = 0, b1Bytes = 0;
    uint32_t a0Bytes = 0, b0Bytes = 0, c0Bytes = 0;
};
inline DirectPlan MakeDirectPlan(const ProblemDesc& p, TileRequest tile, DirectCaps caps)
{
    DirectPlan d{};
    if (!caps.supported || !p.batches || !p.m || !p.n || !p.k ||
        p.m > 8192 || p.n > 8192 || p.k > 8192 ||
        p.m % 16 || p.n % 16 || p.k % 16 || (p.dtype != 1 && p.dtype != 2) ||
        !tile.m || !tile.n || tile.m % 16 || tile.n % 16 || tile.m > 256 || tile.n > 256)
        return d;
    const uint32_t m = std::min(p.m, tile.m), n = std::min(p.n, tile.n);
    // Limit to the documented A2 capacities even if a query returns more.
    caps.l1 = std::min<uint64_t>(caps.l1, 512 * 1024);
    caps.l0a = std::min<uint64_t>(caps.l0a, 64 * 1024);
    caps.l0b = std::min<uint64_t>(caps.l0b, 64 * 1024);
    caps.l0c = std::min<uint64_t>(caps.l0c, 128 * 1024);
    if (uint64_t(m) * n * 4 > caps.l0c) return d;
    for (uint32_t bk = std::min(128U, p.k); bk >= 16; bk = bk / 2 / 16 * 16) {
        const uint32_t a = m * bk * 2, b = n * bk * 2;
        if (a > caps.l0a || b > caps.l0b || uint64_t(a) + b > caps.l1) continue;
        d.enabled = true;
        d.blockK = bk;
        d.residentA = uint64_t(m) * p.k * 2 + b <= caps.l1;
        d.a1Bytes = d.residentA ? m * p.k * 2 : a;
        d.b1Bytes = b; d.a0Bytes = a; d.b0Bytes = b; d.c0Bytes = m * n * 4;
        return d;
    }
    return d;
}
} // namespace local_baseline
#endif
