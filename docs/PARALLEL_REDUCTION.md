# 第一轮优化：并行行最大值

参照版本是 `d01d01b`：此前线上 15/15 通过；本地 `B=1,M=8192,N=65,K=32,FP16,transpose=00` 的 host median 为 **3448.2545 μs**，独立 profiler 的 Task Duration median 为 **3412.408 μs**。当前版本是新候选，尚无真实 CANN 编译、NPU 回归或提速结果。

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
