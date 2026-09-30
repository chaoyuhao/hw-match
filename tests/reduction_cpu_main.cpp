#include <cstdlib>
#include <iostream>

struct Aligned {
    float* data = nullptr;
    explicit Aligned(size_t count) {
        require(posix_memalign(reinterpret_cast<void**>(&data), 64, count * 4) == 0, "allocation failed");
        std::fill(data, data + count, 123456.0f);
    }
    ~Aligned() { std::free(data); }
};

void check(uint32_t batches, uint32_t m, uint32_t n, uint32_t cores, int pattern) {
    using namespace AscendC;
    const uint32_t pitch = (n + 15) / 16 * 16, rowPitch = (m + 31) / 32 * 32;
    const size_t sCount = static_cast<size_t>(batches) * m * pitch;
    const size_t rCount = static_cast<size_t>(batches) * rowPitch;
    Aligned s(sCount + 32), maxima(rCount + 32), output(batches + 32);
    auto* sp = s.data + 16; auto* rp = maxima.data + 16; auto* yp = output.data + 16;
    regions = {{sp, sCount, true, std::vector<unsigned>(sCount), std::vector<bool>(sCount)},
               {rp, rCount, false, std::vector<unsigned>(rCount), std::vector<bool>(rCount)},
               {yp, batches, false, std::vector<unsigned>(batches), std::vector<bool>(batches)}};
    std::vector<float> expectedRows(static_cast<size_t>(batches) * m), expectedY(batches);
    for (uint32_t b = 0; b < batches; ++b) {
        double sum = 0;
        for (uint32_t row = 0; row < m; ++row) {
            float best = -std::numeric_limits<float>::infinity();
            for (uint32_t col = 0; col < n; ++col) {
                float v = static_cast<int>((b * 113 + row * 97 + col * 13) % 1021) / 256.0f - 2.0f;
                if (pattern == 1) v = -std::abs(v) - 0.25f;
                if (pattern == 2) {
                    const float cancellation[] = {4096.0f, 0.03125f, -4096.0f, -0.015625f};
                    v = cancellation[row % 4] - col * 0.125f;
                }
                if (pattern == 3) {
                    // The half-ulp at row 1023 must survive the next chunk.
                    v = row == 1022 ? 4096.0f : row == 1025 ? -4096.0f :
                        (row == 1023 || row == 1024) ? 0.000244140625f : 0.0f;
                    v -= col * 0.125f;
                }
                const size_t pos = (static_cast<size_t>(b) * m + row) * pitch + col;
                sp[pos] = v; regions[0].readable[pos] = true; best = std::max(best, v);
            }
            expectedRows[static_cast<size_t>(b) * m + row] = best;
            sum += static_cast<double>(best);
        }
        expectedY[b] = sum;
        if (pattern == 3 && m >= 1026) require(sum == 0.00048828125f, "bad cancellation fixture");
    }
    const std::vector<float> before(sp, sp + sCount);
    blockNum = cores;
    // Execute actual helpers in two phases; this does not emulate hardware SyncAll.
    std::vector<TPipe> pipes(cores);
    for (blockIdx = 0; blockIdx < cores; ++blockIdx)
        local_baseline::ComputeRowMaxima(pipes[blockIdx], reinterpret_cast<GM_ADDR>(sp),
                                        reinterpret_cast<GM_ADDR>(rp), batches, m, n, pitch, rowPitch);
    for (uint32_t b = 0; b < batches; ++b)
        for (uint32_t row = 0; row < m; ++row)
            require(rp[static_cast<size_t>(b) * rowPitch + row] == expectedRows[static_cast<size_t>(b) * m + row], "incorrect row maximum");
    for (blockIdx = 0; blockIdx < cores; ++blockIdx)
        local_baseline::SumRowMaxima(pipes[blockIdx], reinterpret_cast<GM_ADDR>(rp),
                                    reinterpret_cast<GM_ADDR>(yp), batches, m, rowPitch);
    require(std::memcmp(expectedY.data(), yp, batches * 4) == 0, "sum differs from exact fixture golden");
    require(std::memcmp(before.data(), sp, sCount * 4) == 0, "similarity was modified");
    for (auto i : regions[1].writes) require(i == 1, "missing row-max tile write");
    for (auto i : regions[2].writes) require(i == 1, "missing output write");
    for (size_t i = 0; i < 16; ++i) {
        require(maxima.data[i] == 123456.0f && maxima.data[16 + rCount + i] == 123456.0f, "row-max guard modified");
        require(output.data[i] == 123456.0f && output.data[16 + batches + i] == 123456.0f, "output guard modified");
    }
}

// Independent numerical fixtures for the final M reduction, without a Matmul
// model. In particular, a rounded chunk sum cannot recover all these residuals.
void checkSum(uint32_t m, int pattern) {
    using namespace AscendC;
    const uint32_t batches = 9, rowPitch = (m + 31) / 32 * 32;
    const size_t count = batches * rowPitch;
    Aligned maxima(count + 32), output(batches + 32);
    auto* rp = maxima.data + 16; auto* yp = output.data + 16;
    // ComputeRowMaxima initializes padding too; poison it to catch accidental
    // inclusion in Sum without forbidding a legal aligned DMA read.
    regions = {{rp, count, true, std::vector<unsigned>(count), std::vector<bool>(count, true)},
               {yp, batches, false, std::vector<unsigned>(batches), std::vector<bool>(batches)}};
    std::vector<float> expected(batches);
    for (uint32_t b = 0; b < batches; ++b) {
        long double exact = 0;
        uint32_t state = 1729 + b;
        for (uint32_t i = 0; i < m; ++i) {
            state = state * 1664525u + 1013904223u;
            float v = std::ldexp(float(int(state % 2049) - 1024), int((state >> 12) % 24) - 20);
            if (pattern == 0) v = 0;
            if (pattern == 1) v = std::abs(v);
            if (pattern == 2) v = -std::abs(v);
            if (pattern == 3) {
                const float cycle[] = {16777216.f, 1.f, -16777216.f};
                v = cycle[i % 3];
            }
            if (pattern == 4) {
                v = i == 0 ? 4096.f : i == m - 1 ? -4096.f : std::ldexp(1.f, -12);
            }
            if (pattern == 5) {
                // Cancellation occurs after hundreds of independently rounded
                // partial sums, not just at adjacent input positions.
                const uint32_t half = m / 2;
                v = i < half ? 4096.f : i < 2 * half ? -4096.f : std::ldexp(1.f, -11);
                if (m >= 3 && i == 1) v = 4096.00048828125f;
            }
            if (pattern == 6) {
                const float cycle[] = {4096.f, std::ldexp(1.f, -12), std::ldexp(1.f, -12), -4096.f};
                v = (i >= 1022 && i < 1026) ? cycle[i - 1022] : 0.f;
            }
            if (b % 2) v = -v;
            rp[b * rowPitch + i] = v;
            regions[0].readable[b * rowPitch + i] = true;
            exact += static_cast<long double>(v);
        }
        expected[b] = static_cast<float>(exact);
    }
    const std::vector<float> before(rp, rp + count);
    std::vector<float> firstResult;
    blockNum = 3;
    for (int repeat = 0; repeat < 2; ++repeat) {
        regions[1].writes.assign(batches, 0);
        scalarReads = 0;
        for (blockIdx = 0; blockIdx < blockNum; ++blockIdx) {
            TPipe pipe;
            local_baseline::SumRowMaxima(pipe, reinterpret_cast<GM_ADDR>(rp),
                                        reinterpret_cast<GM_ADDR>(yp), batches, m, rowPitch);
        }
        for (uint32_t b = 0; b < batches; ++b) {
            require(std::isfinite(yp[b]) && std::abs(double(yp[b]) - expected[b]) <=
                    1e-4 + 1e-4 * std::abs(double(expected[b])), "sum exceeds FP64-golden tolerance");
            require(regions[1].writes[b] == 1, "missing final sum output");
        }
        if (repeat == 0) firstResult.assign(yp, yp + batches);
        else require(std::memcmp(firstResult.data(), yp, batches * 4) == 0, "sum is not repeatable");
        // A structural cost check, not a wall-clock/NPU benchmark: at least
        // 32x fewer scalar loads for full 1024-row chunks.
        if (m >= 1024) require(scalarReads <= batches * 32ull * ((m + 1023) / 1024),
                               "final sum still reads individual rows through Scalar");
        require(std::memcmp(before.data(), rp, count * 4) == 0, "sum modified row maxima");
        for (uint32_t i = 0; i < 16; ++i)
            require(output.data[i] == 123456.f && output.data[16 + batches + i] == 123456.f,
                    "sum wrote output padding");
    }
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "need case family");
        const int family = std::atoi(argv[1]);
        if (family == 0) {
            for (auto n : {1u, 7u, 8u, 63u, 64u, 65u, 255u, 256u, 257u, 1025u, 8192u})
                for (auto m : {1u, 31u, 32u, 33u, 65u}) check(2, m, n, 24, 1);
        } else if (family == 1) {
            for (auto b : {1u, 8u, 9u, 32u, 257u})
                for (auto cores : {1u, 3u, 24u}) check(b, 65, 257, cores, 0);
            check(1, 8192, 65, 24, 0);
        } else if (family == 2) {
            for (auto m : {1023u, 1024u, 1025u, 8192u}) check(3, m, 65, 24, 2);
            check(2, 1026, 65, 24, 3);
        } else {
            checkSum(1024, 0);
            for (auto m : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u,
                           127u, 128u, 129u, 255u, 256u, 257u, 511u, 512u, 513u,
                           1023u, 1024u, 1025u, 1026u, 2047u, 2048u, 2049u, 8191u, 8192u})
                for (int pattern = 0; pattern < 8; ++pattern) checkSum(m, pattern);
        }
        std::cout << "CPU reduction helper behavior passed; hardware synchronization is not simulated.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
