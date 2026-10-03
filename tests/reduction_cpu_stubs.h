// Behavioral test doubles for the reduction helpers only, NOT a CANN simulator.
// These model bounds/strides/masks and synchronous float operations, not pipelines.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#define __aicore__
using GM_ADDR = uint8_t*;
enum { PIPE_V, PIPE_ALL };
inline void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
namespace AscendC {
static uint32_t blockIdx, blockNum;
static uint64_t scalarReads = 0;
static uint64_t wholeMaxCalls = 0;
inline uint32_t GetBlockIdx() { return blockIdx; }
inline uint32_t GetBlockNum() { return blockNum; }
enum class TPosition { VECIN, VECOUT, VECCALC };
enum class HardEvent { MTE2_S, S_MTE2, S_MTE3, MTE3_S, V_S, S_V };
enum class ReduceOrder { ORDER_ONLY_VALUE };
template <int P> void PipeBarrier() {}
template <HardEvent E> void SetFlag(int) {}
template <HardEvent E> void WaitFlag(int) {}
struct Region {
    float* data;
    size_t size;
    bool input;
    std::vector<unsigned> writes;
    std::vector<bool> readable;
};
static std::vector<Region> regions;
inline Region& locate(float* p, size_t& i) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    for (auto& r : regions) {
        const auto begin = reinterpret_cast<uintptr_t>(r.data);
        if (address >= begin && address < begin + r.size * sizeof(float)) {
            i = (address - begin) / sizeof(float);
            return r;
        }
    }
    throw std::runtime_error("GM access outside allocation (including guard)");
}
inline float readGm(float* p) {
    size_t i = 0; auto& r = locate(p, i);
    require(r.readable[i], "read of padding or unwritten GM");
    return *p;
}
inline void writeGm(float* p, float v) {
    size_t i = 0; auto& r = locate(p, i);
    require(!r.input, "input was written");
    require(r.writes[i]++ == 0, "multiple writes/owners for GM element");
    r.readable[i] = true;
    *p = v;
}
template <typename T> struct LocalTensor {
    std::shared_ptr<std::vector<float>> data;
    size_t offset = 0;
    LocalTensor operator[](size_t i) const { return {data, offset + i}; }
    float LaneValue(size_t i) const { return data->at(offset + i); }
    float GetValue(size_t i) const { ++scalarReads; return LaneValue(i); }
    void SetValue(size_t i, float v) const { data->at(offset + i) = v; }
};
template <typename T> struct GlobalTensor {
    float* data = nullptr;
    void SetGlobalBuffer(float* p) { data = p; }
    GlobalTensor operator[](size_t i) const { return {data + i}; }
};
template <TPosition P> struct TBuf {
    std::shared_ptr<std::vector<float>> data;
    template <typename T> LocalTensor<T> Get() { return {data, 0}; }
};
template <TPosition P, int N> struct TQue {
    std::shared_ptr<std::vector<float>> data;
    int state = 0;
    template <typename T> LocalTensor<T> AllocTensor() {
        require(state == 0, "queue allocated twice"); state = 1; return {data, 0};
    }
    template <typename T> void EnQue(LocalTensor<T>) { require(state == 1, "bad enqueue"); state = 2; }
    template <typename T> LocalTensor<T> DeQue() { require(state == 2, "bad dequeue"); state = 3; return {data, 0}; }
    template <typename T> void FreeTensor(LocalTensor<T>) { require(state == 3, "bad free"); state = 0; }
};
struct TPipe {
    size_t bytes = 0;
    template <typename Buffer> void InitBuffer(Buffer& b, size_t n) {
        bytes += n;
        require(bytes <= 64 * 1024, "reduction UB allocation exceeds reserved budget");
        b.data = std::make_shared<std::vector<float>>(n / 4, std::numeric_limits<float>::quiet_NaN());
    }
    template <typename Buffer> void InitBuffer(Buffer& b, int depth, size_t n) {
        require(depth == 1, "only single buffer modeled"); InitBuffer(b, n);
    }
    int FetchEventID(HardEvent) { return 0; }
};
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride; uint16_t rsv; };
template <typename T> struct DataCopyPadExtParams { bool isPad; uint8_t leftPadding, rightPadding; T paddingValue; };
inline void DataCopyPad(LocalTensor<float> dst, GlobalTensor<float> src,
                        DataCopyExtParams p, DataCopyPadExtParams<float> pad) {
    require(p.blockLen % 4 == 0 && p.srcStride % 4 == 0, "unaligned float copy");
    require(p.blockCount <= 4095 && pad.leftPadding == 0 && pad.rightPadding < 8, "invalid DMA parameters");
    const size_t count = p.blockLen / 4, rounded = (count + 7) / 8 * 8;
    for (size_t row = 0; row < p.blockCount; ++row) {
        const size_t from = row * (count + p.srcStride / 4);
        const size_t to = row * (rounded + p.dstStride * 8);
        for (size_t i = 0; i < count; ++i) dst.SetValue(to + i, readGm(src.data + from + i));
        for (size_t i = count; i < rounded; ++i)
            dst.SetValue(to + i, pad.isPad ? pad.paddingValue : std::numeric_limits<float>::quiet_NaN());
    }
}
inline void DataCopy(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t n) {
    require(n % 8 == 0, "unaligned DMA length");
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, readGm(src.data + i));
}
inline void DataCopy(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t n) {
    require(n % 8 == 0 && reinterpret_cast<uintptr_t>(dst.data) % 32 == 0, "unaligned tile write");
    for (uint32_t i = 0; i < n; ++i) writeGm(dst.data + i, src.LaneValue(i));
}
inline void DataCopyPad(GlobalTensor<float> dst, LocalTensor<float> src, DataCopyExtParams p) {
    require(p.blockCount == 1 && p.blockLen % 4 == 0, "unexpected output copy");
    for (uint32_t i = 0; i < p.blockLen / 4; ++i) writeGm(dst.data + i, src.LaneValue(i));
}
inline void Duplicate(LocalTensor<float> dst, float value, uint32_t n) {
    require(dst.offset % 8 == 0, "unaligned vector fill");
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, value);
}
template <typename Op>
inline void binary(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n, Op op) {
    require(dst.offset % 8 == 0 && a.offset % 8 == 0 && b.offset % 8 == 0, "unaligned vector arithmetic");
    for (uint32_t i = 0; i < n; ++i) {
        const float x = a.LaneValue(i), y = b.LaneValue(i);
        require(std::isfinite(x) && std::isfinite(y), "vector arithmetic read invalid lane");
        dst.SetValue(i, op(x, y));
    }
}
inline void Add(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    binary(dst, a, b, n, [](float x, float y) { return x + y; });
}
inline void Sub(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    binary(dst, a, b, n, [](float x, float y) { return x - y; });
}
inline void Muls(LocalTensor<float> dst, LocalTensor<float> a, float v, uint32_t n) {
    require(dst.offset % 8 == 0 && a.offset % 8 == 0, "unaligned vector scale");
    for (uint32_t i = 0; i < n; ++i) {
        require(std::isfinite(a.LaneValue(i)), "vector scale read invalid lane");
        dst.SetValue(i, a.LaneValue(i) * v);
    }
}
inline void WholeReduceMax(LocalTensor<float> dst, LocalTensor<float> src, int mask, int repeats,
                           int dstStride, int blockStride, int repeatStride, ReduceOrder) {
    ++wholeMaxCalls;
    require(mask > 0 && mask <= 64 && repeats > 0 && repeats <= 255, "invalid reduction mask/repeats");
    for (int r = 0; r < repeats; ++r) {
        float value = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < mask; ++i) {
            const float item = src.LaneValue(r * repeatStride * 8 + (i / 8) * blockStride * 8 + i % 8);
            require(std::isfinite(item), "masked reduction read invalid/uninitialized lane");
            value = std::max(value, item);
        }
        dst.SetValue(r * dstStride, value);
    }
}
inline void Max(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) dst.SetValue(i, std::max(a.LaneValue(i), b.LaneValue(i)));
}
struct BinaryRepeatParams {
    uint8_t dstBlkStride, src0BlkStride, src1BlkStride;
    uint8_t dstRepStride, src0RepStride, src1RepStride;
};
inline void Max(LocalTensor<float> dst, LocalTensor<float> a, LocalTensor<float> b,
                uint64_t mask, uint8_t repeats, const BinaryRepeatParams& p) {
    require(mask>0 && mask<=64 && repeats>0,"invalid repeated Max mask/count");
    require(dst.offset%8==0 && a.offset%8==0 && b.offset%8==0,"unaligned repeated Max");
    // This test surface uses contiguous blocks inside each repeat. Reject
    // partial same-repeat alias and every cross-repeat RAW/WAR dependency.
    require(p.dstBlkStride==1 && p.src0BlkStride==1 && p.src1BlkStride==1,
            "unmodeled Max block stride");
    auto aliases = [&](LocalTensor<float> src, uint8_t stride) {
        if (src.data != dst.data) return;
        if (src.offset==dst.offset && stride==p.dstRepStride &&
            (repeats==1 || uint64_t(stride)*8>=mask)) return;
        for(unsigned w=0;w<repeats;++w)for(unsigned r=0;r<repeats;++r){
            const size_t out=dst.offset+w*p.dstRepStride*8;
            const size_t in=src.offset+r*stride*8;
            require(out+mask<=in || in+mask<=out || (w==r && out==in),
                    "unsupported repeated Max address dependency");
        }
    };
    aliases(a,p.src0RepStride);aliases(b,p.src1RepStride);
    // Read the whole instruction before writing, as independent vector lanes.
    // Check bounds/poison for every active lane; this does not model V timing.
    std::vector<size_t> destinations;
    std::vector<float> values;
    for(unsigned r=0;r<repeats;++r)for(unsigned lane=0;lane<mask;++lane){
        const size_t di=r*p.dstRepStride*8+(lane/8)*p.dstBlkStride*8+lane%8;
        const size_t ai=r*p.src0RepStride*8+(lane/8)*p.src0BlkStride*8+lane%8;
        const size_t bi=r*p.src1RepStride*8+(lane/8)*p.src1BlkStride*8+lane%8;
        const float x=a.LaneValue(ai),y=b.LaneValue(bi);
        require(std::isfinite(x)&&std::isfinite(y),"repeated Max read invalid lane");
        destinations.push_back(di);values.push_back(std::max(x,y));
    }
    for(size_t i=0;i<values.size();++i)dst.SetValue(destinations[i],values[i]);
}
} // namespace AscendC
