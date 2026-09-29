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
        float sum = 0, compensation = 0;
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
            const float corrected = best - compensation, next = sum + corrected;
            compensation = (next - sum) - corrected; sum = next;
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
    require(std::memcmp(expectedY.data(), yp, batches * 4) == 0, "ordered compensated sum changed");
    require(std::memcmp(before.data(), sp, sCount * 4) == 0, "similarity was modified");
    for (auto i : regions[1].writes) require(i == 1, "missing row-max tile write");
    for (auto i : regions[2].writes) require(i == 1, "missing output write");
    for (size_t i = 0; i < 16; ++i) {
        require(maxima.data[i] == 123456.0f && maxima.data[16 + rCount + i] == 123456.0f, "row-max guard modified");
        require(output.data[i] == 123456.0f && output.data[16 + batches + i] == 123456.0f, "output guard modified");
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
        } else {
            for (auto m : {1023u, 1024u, 1025u, 8192u}) check(3, m, 65, 24, 2);
            check(2, 1026, 65, 24, 3);
        }
        std::cout << "CPU reduction helper behavior passed; hardware synchronization is not simulated.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
