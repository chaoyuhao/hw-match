#ifndef CANN_MATCH_SMALL_VECTOR_H
#define CANN_MATCH_SMALL_VECTOR_H
#include "small_plan.h"
#include "kernel_operator.h"

namespace local_baseline {
// One owner per 32-byte output group; no scratch GM or inter-core dependency.
template<typename T> class SmallVectorOp {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe, GM_ADDR a, GM_ADDR b, GM_ADDR y,
                               uint64_t batches, uint32_t m, uint32_t n, uint32_t k,
                               bool ta, SmallPlan plan)
    {
        pipe_ = &pipe; batches_ = batches; m_ = m; n_ = n; k_ = k; ta_ = ta; plan_ = plan;
        aGm_.SetGlobalBuffer((__gm__ T*)a, batches * m * k);
        bGm_.SetGlobalBuffer((__gm__ T*)b, batches * k * n);
        yGm_.SetGlobalBuffer((__gm__ float*)y, batches);
        pipe.InitBuffer(aQueue_, 1, plan.aElements * sizeof(T));
        pipe.InitBuffer(bQueue_, 1, plan.bElements * sizeof(T));
        pipe.InitBuffer(aFloat_, plan.aElements * sizeof(float));
        pipe.InitBuffer(bFloat_, plan.bElements * sizeof(float));
        pipe.InitBuffer(product_, 256 * sizeof(float));
        pipe.InitBuffer(temp_, 64 * sizeof(float));
        pipe.InitBuffer(partial_, 32);
        pipe.InitBuffer(output_, 32);
    }
    __aicore__ inline void Process()
    {
        using namespace AscendC;
        auto out = output_.template Get<float>();
        for (uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * 8; first < batches_;
             first += static_cast<uint64_t>(GetBlockNum()) * 8) {
            const uint32_t count = batches_ - first < 8 ? batches_ - first : 8;
            for (uint32_t bi = 0; bi < count; ++bi) {
                Load(first + bi);
                float sum = 0.0f, compensation = 0.0f;
                for (uint32_t row = 0; row < m_; ++row) {
                    const float maximum = plan_.variant == SmallVariant::Dot ? DotRow(row) : VectorRow(row);
                    const float corrected = maximum - compensation;
                    const float next = sum + corrected;
                    compensation = (next - sum) - corrected;
                    sum = next;
                }
                out.SetValue(bi, sum);
            }
            Fence<HardEvent::S_MTE3>();
            DataCopyExtParams copy{1, count * 4, 0, 0, 0};
            DataCopyPad(yGm_[first], out, copy);
            Fence<HardEvent::MTE3_S>();
        }
    }
private:
    template<AscendC::HardEvent event> __aicore__ inline void Fence()
    {
        const auto id = pipe_->FetchEventID(event);
        AscendC::SetFlag<event>(id);
        AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline void Load(uint64_t batch)
    {
        using namespace AscendC;
        auto a = aQueue_.template AllocTensor<T>();
        auto b = bQueue_.template AllocTensor<T>();
        DataCopyExtParams ca{static_cast<uint16_t>(plan_.aRows), plan_.aWidth * 2, 0, 0, 0};
        DataCopyExtParams cb{static_cast<uint16_t>(plan_.bRows), plan_.bWidth * 2, 0, 0, 0};
        DataCopyPadExtParams<T> pa{true, 0, static_cast<uint8_t>(plan_.aPitch - plan_.aWidth), T(0)};
        DataCopyPadExtParams<T> pb{true, 0, static_cast<uint8_t>(plan_.bPitch - plan_.bWidth), T(0)};
        DataCopyPad(a, aGm_[batch * m_ * k_], ca, pa);
        DataCopyPad(b, bGm_[batch * k_ * n_], cb, pb);
        aQueue_.EnQue(a); bQueue_.EnQue(b);
        a = aQueue_.template DeQue<T>(); b = bQueue_.template DeQue<T>();
        // Promote BEFORE multiplication, including BF16. No FP16 products.
        Cast(aFloat_.template Get<float>(), a, RoundMode::CAST_NONE, plan_.aElements);
        Cast(bFloat_.template Get<float>(), b, RoundMode::CAST_NONE, plan_.bElements);
        Fence<HardEvent::V_MTE2>();
        aQueue_.FreeTensor(a); bQueue_.FreeTensor(b);
        Fence<HardEvent::V_S>();
    }
    __aicore__ inline float DotRow(uint32_t row)
    {
        using namespace AscendC;
        auto a = aFloat_.template Get<float>();
        auto b = bFloat_.template Get<float>();
        auto product = product_.template Get<float>();
        auto partial = partial_.template Get<float>();
        float maximum = 0.0f;
        for (uint32_t col = 0; col < n_; ++col) {
            Mul(product, a[row * plan_.aPitch], b[col * plan_.bPitch], k_);
            PipeBarrier<PIPE_V>();
            uint32_t chunks = 0;
            for (uint32_t begin = 0; begin < k_; begin += 64) {
                const uint32_t mask = k_ - begin < 64 ? k_ - begin : 64;
                // WholeReduceSum permits float-aligned compact destinations.
                WholeReduceSum(partial[chunks++], product[begin], mask, 1, 1, 1, 8);
            }
            Fence<HardEvent::V_S>();
            float value = partial.GetValue(0);
            for (uint32_t i = 1; i < chunks; ++i) value += partial.GetValue(i);
            if (col == 0 || value > maximum) maximum = value;
            Fence<HardEvent::S_V>();
        }
        return maximum;
    }
    __aicore__ inline float VectorRow(uint32_t row)
    {
        using namespace AscendC;
        auto a = aFloat_.template Get<float>();
        auto b = bFloat_.template Get<float>();
        auto product = product_.template Get<float>();
        auto temp = temp_.template Get<float>();
        auto partial = partial_.template Get<float>();
        Duplicate(product, 0.0f, n_);
        PipeBarrier<PIPE_V>();
        for (uint32_t k = 0; k < k_; ++k) {
            const float value = a.GetValue(ta_ ? k * plan_.aPitch + row : row * plan_.aPitch + k);
            Muls(temp, b[k * plan_.bPitch], value, n_);
            PipeBarrier<PIPE_V>();
            Add(product, product, temp, n_);
            PipeBarrier<PIPE_V>();
        }
        WholeReduceMax(partial, product, n_, 1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
        Fence<HardEvent::V_S>();
        const float maximum = partial.GetValue(0);
        Fence<HardEvent::S_V>();
        return maximum;
    }
    AscendC::TPipe* pipe_;
    uint64_t batches_;
    uint32_t m_, n_, k_;
    bool ta_;
    SmallPlan plan_;
    AscendC::GlobalTensor<T> aGm_, bGm_;
    AscendC::GlobalTensor<float> yGm_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> aQueue_, bQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aFloat_, bFloat_, product_, temp_, partial_;
    AscendC::TBuf<AscendC::TPosition::VECOUT> output_;
};
} // namespace local_baseline
#endif
