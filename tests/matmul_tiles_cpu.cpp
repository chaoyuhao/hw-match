#include <cassert>
#include <cstdlib>
#include <vector>
using namespace local_baseline;

void PlanTests()
{
    const HardwareCaps caps{24, 192 * 1024};
    bool nonLegacy = false;
    for (uint64_t b : {1, 2, 257})
        for (uint32_t m : {1, 33, 65, 129, 1024, 8192})
            for (uint32_t n : {1, 65, 129, 8192})
                for (uint32_t k : {8, 24, 8192}) {
                    const ProblemDesc problem{b, m, n, k, 1, false, true};
                    const auto first = GenerateCandidates(problem, caps);
                    const auto again = GenerateCandidates(problem, caps);
                    assert(first.count > 4 && first.count <= 64 && first.count == again.count);
                    bool reference = false;
                    for (size_t i = 0; i < first.count; ++i) {
                        const auto &p = first.plans[i], &q = again.plans[i];
                        assert(p.tileM == q.tileM && p.tileN == q.tileN && p.score == q.score);
                        assert(p.tileM % 16 == 0 && p.tileN % 16 == 0);
                        assert(p.tileM <= 256 && p.tileN <= 256 && p.ubBudget == 128 * 1024);
                        assert(p.tasks == b * ((m + p.tileM - 1) / p.tileM) * ((n + p.tileN - 1) / p.tileN));
                        assert(p.blocks == std::min<uint64_t>(24, p.tasks));
                        if (i) assert(first.plans[i-1].score <= p.score);
                        for (size_t j = 0; j < i; ++j)
                            assert(p.tileM != first.plans[j].tileM || p.tileN != first.plans[j].tileN);
                        reference |= p.tileM == 32 && p.tileN == 64;
                        nonLegacy |= p.tileM == 80 || p.tileN == 80;
                    }
                    assert(reference);
                }
    assert(nonLegacy);
    const ProblemDesc fallbackProblem{1, 512, 512, 256, 1, false, false};
    int attempts = 0;
    auto supported = FindSupportedPlan(fallbackProblem, caps, {}, [&](const MatmulPlan&) { return ++attempts == 5; });
    assert(attempts == 5 && supported.tileM == 32 && supported.tileN == 64);
    attempts = 0;
    bool failed = false;
    try { FindSupportedPlan(fallbackProblem, caps, {80,144}, [&](const MatmulPlan&) { ++attempts; return false; }); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed && attempts == 1); // Explicit requests must never silently fall back.
    attempts = 0; failed = false;
    try { FindSupportedPlan(fallbackProblem, caps, {}, [&](const MatmulPlan&) { ++attempts; return false; }); }
    catch (const std::runtime_error&) { failed = true; }
    assert(failed && attempts == 5);

    const ProblemDesc valid{1, 65, 129, 24, 1, false, false};
    assert(MakePlan(valid, {24, 160 * 1024}, {80, 144}).ubBudget == 96 * 1024);
    for (auto tile : {TileRequest{0, 16}, TileRequest{17, 32}, TileRequest{272, 32}}) {
        bool rejected = false;
        try { MakePlan(valid, caps, tile); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    for (auto bad : {ProblemDesc{UINT64_MAX, 8192, 8192, 8192, 1, false, false},
                     ProblemDesc{1, 0, 1, 8, 1, false, false},
                     ProblemDesc{1, 1, 1, 8, 0, false, false}}) {
        bool rejected = false;
        try { GenerateCandidates(bad, caps); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    for (auto bad : {HardwareCaps{0, 192 * 1024}, HardwareCaps{24, 64 * 1024}}) {
        bool rejected = false;
        try { GenerateCandidates(valid, bad); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
}

void OwnershipTests()
{
    for (auto request : {TileRequest{16,16}, TileRequest{32,64}, TileRequest{80,144}, TileRequest{256,256}})
        for (uint32_t m : {1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257})
            for (uint32_t n : {1, 63, 64, 65, 127, 128, 129, 257})
                for (uint32_t cores : {1, 3, 24}) {
                    const uint64_t batches = 3;
                    const uint32_t pitch = (n + 15) / 16 * 16;
                    auto plan = MakePlan({batches, m, n, 24, 1, false, false}, {cores, 192 * 1024}, request);
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
                for (auto request : {TileRequest{16,16}, TileRequest{32,64}, TileRequest{80,144}, TileRequest{256,256}}) {
                    auto plan = MakePlan({2, m, n, k, 1, ta, tb}, {24, 192 * 1024}, request);
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
