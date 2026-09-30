# 并行行最大值与最终求和

## R12：向量补偿树替换逐行标量求和（2026-09-30）

本轮只修改 `kernel.asc` 中通用 MIX 路径的 `SumRowMaxima`。线上对照为当前 R11/S7；Matmul、行最大值、两次全核同步、small 路径、分派规则、GM scratch 和输出 owner 均保持原逻辑。`matmul_plan.h` 仅更新 UB 用量注释，选择行为不变。已有 R11 文件时，线上只需替换 **kernel.asc**，没有新增提交文件。

原来每个 batch 按 M 行逐项 `GetValue` 后串行 FP32 Kahan 相加。现在仍每次读取最多 1024 个行最大值，但在 Vector 上做固定的补偿树：

1. 只复制真实行，DMA 补齐到 8 个 float 时填零，再将有效 UB 区间补零到最近的 2 的幂，最少 8、最多 1024。`ComputeRowMaxima` 的尾行哨兵是负数，不能参与 Sum。
2. 每层将左右两半相加。用 TwoSum 的向量 Add/Sub 保留主值相加的舍入残差，并合并左右子树的低位部分；主值用 Muls(1) 写回。每个向量操作之间显式建立 PIPE_V 依赖，所有源偏移保持 32 字节对齐。
3. 归约到 8 个 lane 后停止：每块最多读回 8 个主值和 8 个残差。Scalar 用 Neumaier 补偿合并，补偿跨 lane、跨 1024 行块保留，最后输出 `sum + compensation`。不足 9 行直接合并真实行，无须读取残差。

以 M=8192 为例，原来每个 batch 有 **8192 次标量读**，新实现为 **128 次**，结构上减少 64 倍；同时增加向量计算与同步，**不能把读数变化当成端到端加速比**。它没有消除最终输出 owner 的集中，也没有融合 similarity 的生产与消费。

普通树形求和不能恢复子树中已经丢掉的小量。例如 `[4096, 2^-12, 2^-12, -4096]` 跨块时，单独对每块做普通 FP32 求和再补偿可能失去残差。本轮采用补偿树正是为保护这类输入；它仍是 FP32 算法，不声称任意输入下与 FP64 逐位相同。README 要求 Max(N) 在 Sum(M) 前完成和重复结果一致，没有要求串行 Kahan 顺序。

新增 UB：1024 个残差 float + 三组 512 个临时 float，共 10240 字节；归约阶段累计申请由 37152 增为 **47392 字节**，仍在现有 64 KiB 预留内，Matmul 预算不变。输入队列提供 MTE2→Vector 依赖；V_S 保护 Scalar 读取，S_V 和 S_MTE2 分别保护残差/输入缓冲后续复用；原输出写回依赖保持。

CPU 检查运行实际 device helper，新增固定边界与强抵消输入，使用独立 FP64 或更高精度 golden，检查尾部哨兵、输出 guard、唯一 writer、输入不变、重复输出逐位相同，以及每 1024 行最多 32 次标量读取的上限。新增工作量检查已确认旧实现失败、新实现通过；它不是 NPU 性能测试。完整主机回归 `python3 -m unittest discover -s tests -v`：**62 项通过，52.867 秒**。CPU 替身不模拟异步流水、真实指令舍入或 CANN 编译。

按用户要求，本轮未安排本地 NPU 验证：`LOCAL_CANN_BUILD=NOT_RUN`、`LOCAL_NPU_TEST=NOT_RUN`。现已收到 **S8：15/15 Pass，错误占比均为 0.00%**，按对话关联 R12 `b864c51`，平台源码哈希未核验。相对 S7，点 13 快 5.73 倍（214.07→37.38 μs），点 8–12 耗时下降 13.78%～37.65%，点 5/6 变慢 8.00%/7.25%；完整结果见 [迭代记录](ITERATION_LOG.md)。大幅响应支持旧版最终 Sum 的重要性，但没有实际分派或阶段计时，不将点号当成 shape。

接口核对：[CANN 9.0 对齐与地址重叠约束](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0004.html)。普通 WholeReduceSum 使用树形相加，但不会自动携带本实现需要的舍入残差，见 [WholeReduceSum](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0081.html)。

## R4 历史记录：并行行最大值

以下记录描述 R4，其本地设备数据不作为 R12 的验证证据；R12 的线上结果单独记录在上文。NPU 命令保留为可选历史入口，不作为当前迭代前置条件。

参照版本是 `d01d01b`：本地 `B=1,M=8192,N=65,K=32,FP16,transpose=00` 的 host median 为 **3448.2545 μs**，独立 profiler 的 Task Duration median 为 **3412.408 μs**。并行归约版本 `72abad1` 在相同 CANN 9.1.0 / 910B2C 环境下，该例 PASS，host median 为 **358.3775 μs**，Task Duration median 为 **324.048 μs**，分别快 **9.62× / 10.53×**；源码哈希和采样配置已核对，普通重复及 profiling 输出误差均为 0。报告位于 `build-perf/run-20260929T2225/`。

用户随后反馈线上再次 **15/15 通过**，13 个点变快，详见 [线上对比](ONLINE_BASELINE.md)。尚未收到本轮本地 full 55 项、reduction 62 项和完整 stress 48 项报告，不能沿用旧版记录充当这些回归的通过证据。

## 改动

保留原 Matmul 分块、任务循环、tiling、完整 FP32 similarity 矩阵和一次 MIX kernel launch。只替换后续归约：

1. Matmul 完成后，原有 `SyncAll<true>()` 保证 similarity 可读。
2. 将 `(batch, M 的 32 行块)` 分给所有 AIV block，按 block 编号步进遍历任务。每次通过二维 DataCopyPad 读取最多 `32×256` 个 FP32 元素；每条 WholeReduceMax 用 repeat 一次处理最多 32 行，每行最多 64 列，再用向量 Max 合并列段。
3. 行最大值写入独立 GM 区域。每个任务独占 32 个 float，batch 行跨度为 `round_up(M,32)`，尾行初始化但不参与最终求和。所有 AIV 经过第二次 `SyncAll<true>()`，包括没有计算任务的核。
4. 最终输出仍按每 8 个 batch 一个 owner 分组。每次加载最多 1024 个行最大值后同步到 Scalar，按照原行顺序执行 FP32 补偿求和，补偿值跨 chunk 保留。

对于 M=8192,N=65，原来单核执行的 8192 次行归约被拆成 256 个行块任务，24 核配置下每核约 10–11 个。最终求和仍有 8192 个标量值读取，但不再逐行等待 Vector 归约；加载行最大值只需要 8 个 chunk。这是结构变化，不是实测加速比。

没有采用浮点原子或乱序部分和。保留最终求和顺序，是为了在本轮减少数值变化，尤其是跨行正负抵消。整次调用仍只包含一次 kernel launch。

## 空间与同步

- scratch = 原 similarity 字节数 + `4*B*round_up(M,32)`。总和检查 size_t 溢出，仍只有一个用户 scratch 分配和原系统 workspace 分配。
- similarity 位于同一 allocation 的开头，原本地诊断回读继续有效。
- 归约 UB 共 37152 字节，Matmul 的 UB 预算继续为 128 KiB。
- DataCopyPad 的 GM 源 stride 使用字节，UB 目标 stride 使用 32 字节块；只复制真实 N 列，不读取 similarity 的未写 padding。
- 输入队列负责 DMA/Vector 依赖；行最大值通过输出队列写 GM。最终求和显式处理 MTE2→Scalar、Scalar→MTE2，以及输出 Scalar→MTE3、MTE3→Scalar 依赖。
- 第二次全核同步增加固定开销，小 shape 的收益需测量；本轮未消除完整 similarity 的 GM 流量，也未优化 Matmul tile/通信成本。

## 在 NPU 上验证

先拉取并验证正确性：

```bash
git pull --ff-only
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash scripts/run_local.sh --suite full &&
bash scripts/run_local.sh --suite reduction
```

`full` 保留原 55 项；`reduction` 新增 62 项，覆盖两种 dtype、四种转置布局、M=31/32/33、N=63/64/65/255/256/257、M=1023/1024/1025/8192、大 M 全负数及 B=257 跨核复用。`reduce_sum_carry` 的第 1022..1025 行最大值为 `[4096, 2^-12, 2^-12, -4096]`，正确和为 `2^-11`；两个 batch 一正一负。若在 1024 行分块处丢失补偿，结果会变成 0。

通过后，先复测原先的慢用例：

```bash
bash scripts/run_perf.sh --suite stress --case sweep_m8192_fp16_00 \
  --profile timeline \
  --compare build-perf/run-20260929T124242Z-FOOZQN/cases/report.json
```

上面的 compare 路径来自用户原 profiling 报告；如果旧 run 目录已移动，改为其当前路径。自动比较的是 host median；另看 Task Duration 是否低于原来的 3412.408 μs，不混用两种时间。

再运行 `bash scripts/run_perf.sh --suite stress` 检查其他 48 项的耗时变化。回传新 `cases/report.json` 和 `report.md`；编译或执行失败时保留 `run.log` / `runtime.log`。无需拷回生成的所有二进制输入。

最终线上只替换 `kernel.asc`，没有新增提交依赖，也不需要提交包；得分及实际 launch 规则仍由线上平台确认。

## 开发机验证的边界

`tests/test_parallel_reduction.py` 提取实际两个 device helper，用 C++14 和带边界检查的 AscendC 替身执行。它检查 DMA stride、mask、行尾、全负值、GM 写入唯一性、未写数据读取、input 不变和补偿求和顺序。测试同时覆盖无任务的核、单核及多核循环。

该检查不模拟异步流水线、核间同步、CANN 编译器或真实硬件。其他 Python 测试检查数据生成和报告失败处理。新增 NPU 用例仅在开发机生成，标记为 GENERATED。

接口依据：[WholeReduceMax](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0079.html)、[DataCopyPad](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/memory_vector_compute/data_move/DataCopyPad_GMToUB.md)、[SyncAll](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md)。
