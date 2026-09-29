#include <cassert>
#include <cstdlib>
#include <vector>
using namespace local_baseline;

void PlanTests()
{
    auto large = SelectMatmulPlan(1, 8192, 65, 24);
    assert(large.tileM == 128 && large.tileN == 128 && large.tasks == 64 && large.blocks == 24);
    auto medium = SelectMatmulPlan(1, 1024, 65, 24);
    assert(medium.tileM == 32 && medium.tileN == 128 && medium.tasks == 32);
    auto small = SelectMatmulPlan(2, 65, 129, 24);
    assert(small.tileM == 32 && small.tileN == 64 && small.blocks == 18);
    auto forced = SelectMatmulPlan(2, 65, 129, 24, 4);
    assert(forced.tileM == 128 && forced.tileN == 128 && forced.blocks == 4);
    for (uint64_t b : {1, 2, 8, 257})
        for (uint32_t m : {1, 33, 65, 127, 129, 1024, 8192})
            for (uint32_t n : {1, 65, 127, 129, 8192})
                for (uint32_t cores : {1, 3, 24, 32}) {
                    auto ref = SelectMatmulPlan(b, m, n, cores, 1);
                    auto plan = SelectMatmulPlan(b, m, n, cores);
                    assert(plan.tasks <= ref.tasks && plan.blocks == ref.blocks);
                    const uint64_t referenceTasks = b * ((m + 31) / 32) * ((n + 63) / 64);
                    assert(ref.tasks == referenceTasks && ref.blocks == std::min<uint64_t>(cores, referenceTasks));
                }
    for (uint32_t policy : {5, 99}) {
        bool rejected = false;
        try { SelectMatmulPlan(1, 1, 1, 24, policy); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    bool overflow = false;
    try { SelectMatmulPlan(UINT64_MAX, 8192, 8192, 24); }
    catch (const std::runtime_error&) { overflow = true; }
    assert(overflow);
}

void OwnershipTests()
{
    for (uint32_t policy = 0; policy <= 4; ++policy)
        for (uint32_t m : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257})
            for (uint32_t n : {1, 63, 64, 65, 127, 128, 129, 257})
                for (uint32_t cores : {1, 3, 24}) {
                    const uint64_t batches = 3;
                    const uint32_t pitch = (n + 15) / 16 * 16;
                    auto plan = SelectMatmulPlan(batches, m, n, cores, policy);
                    std::vector<unsigned char> writes(batches * m * pitch, 0);
                    for (uint32_t block = 0; block < plan.blocks; ++block)
                        for (uint64_t task = block; task < plan.tasks; task += plan.blocks) {
                            auto tile = GetMatmulBlock(task, m, n, 24, pitch, plan.tileM, plan.tileN, false, false);
                            assert(tile.batch < batches && tile.rows && tile.cols);
                            assert(tile.row + tile.rows <= m && tile.col + tile.cols <= n);
                            for (uint32_t i = 0; i < tile.rows; ++i)
                                for (uint32_t j = 0; j < tile.cols; ++j)
                                    assert(++writes.at(tile.cOffset + i * pitch + j) == 1);
                        }
                    for (uint64_t row = 0; row < batches * m; ++row)
                        for (uint32_t col = 0; col < pitch; ++col)
                            assert(writes[row * pitch + col] == (col < n ? 1 : 0));
                }
}

void OffsetTests()
{
    const uint32_t m = 129, n = 257, pitch = 272;
    for (uint32_t k : {24, 8192})
        for (bool ta : {false, true})
            for (bool tb : {false, true})
                for (uint32_t policy = 1; policy <= 4; ++policy) {
                    auto plan = SelectMatmulPlan(2, m, n, 24, policy);
                    for (uint64_t task = 0; task < plan.tasks; ++task) {
                        auto t = GetMatmulBlock(task, m, n, k, pitch, plan.tileM, plan.tileN, ta, tb);
                        for (uint32_t i : {0U, t.rows - 1})
                            for (uint32_t j : {0U, t.cols - 1})
                                for (uint32_t c : {0U, k - 1}) {
                                    const uint64_t ai = t.aOffset + (ta ? uint64_t(c) * m + i : uint64_t(i) * k + c);
                                    const uint64_t bi = t.bOffset + (tb ? uint64_t(j) * k + c : uint64_t(c) * n + j);
                                    const uint64_t wantA = ta ? (t.batch * k + c) * m + t.row + i : (t.batch * m + t.row + i) * k + c;
                                    const uint64_t wantB = tb ? (t.batch * n + t.col + j) * k + c : (t.batch * k + c) * n + t.col + j;
                                    assert(ai == wantA && bi == wantB);
                                    assert(t.cOffset + i * pitch + j == (t.batch * m + t.row + i) * pitch + t.col + j);
                                }
                    }
                    const uint64_t farBatch = uint64_t(1) << 25;
                    const uint64_t perBatch = plan.tasks / 2;
                    auto far = GetMatmulBlock(farBatch * perBatch, m, n, k, pitch, plan.tileM, plan.tileN, ta, tb);
                    assert(far.aOffset == farBatch * m * k && far.aOffset > UINT32_MAX);
                    assert(far.bOffset == farBatch * k * n && far.cOffset == farBatch * m * pitch);
                }
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    switch (std::atoi(argv[1])) {
        case 0: PlanTests(); break;
        case 1: OwnershipTests(); break;
        case 2: OffsetTests(); break;
        default: return 2;
    }
}
