# cann-learning-hub：与当前实现有关的新资料

调研日期：2026-09-30。资料来自用户本地克隆的 [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub)，固定版本 `738809a238ffd72c3b170f9c7a801293e6dcde1b`（2026-09-29）。`/cann-learning-hub/` 已加入本项目 `.gitignore`，不作为子模块或提交内容；下文相对链接需要本地存在该克隆。

对照实现为 `3fe1ef5` 的 `kernel.asc`，以及之前的 [baseline 调研](BASELINE_RESEARCH.md)、[性能工具说明](LOCAL_PERFORMANCE.md)和 [Matmul 分块候选](MATMUL_TILING.md)。本次只整理资料，没有修改 kernel，也没有新增 NPU 或线上测试结果。

## 1. 最有用的补充：完整的 Matmul → Vector 直调示例

优先读 [Kernel 直调课程](../cann-learning-hub/tutorials/ascendc_operator_development_light/README.md)，其配套表明确写了 Atlas A2/A3、CANN 9.0.0 及以上，比需要注册算子工程的课程更贴近我们的比赛入口。

具体源码是 [matmul_abs.asc](../cann-learning-hub/tutorials/ascendc_operator_development_light/03_simple_operator_practice/answer/03.05/matmul_abs.asc)，讲解在 [03.05 CV 融合课程](../cann-learning-hub/tutorials/ascendc_operator_development_light/03_simple_operator_practice/03.05_cv_fused_operator_development.ipynb)。重点位置：

| 源码位置 | 示例做什么 | 对我们的价值 |
| --- | --- | --- |
| 40、78 行 | Host tiling 和 Matmul 的 C 类型都设置为 `VECIN / ND / FP32` | 展示 Vector 消费 Matmul 结果的配套配置 |
| 115–134 行 | `Iterate<true>()` 每次获得一个结果块，`GetTensorC<true>()` 写入 LocalTensor | 可参考的完整调用顺序，不止是 API 定义 |
| 138–156 行 | 对当前结果块执行 Abs，经过队列后搬出、释放 | 可以研究把逐元素操作换成块内行最大值，并维护跨 N 块状态 |
| 173–181 行 | MIX kernel、`__kfc_workspace__`、`REGIST_MATMUL_OBJ` | 有完整的 Cube/Vector 协作和系统 workspace 示例 |

**哪些已经知道，哪些是新增：** 我们最初的调研已经列出 `Iterate + GetTensorC`，也提出过逐块 MaxSim。新收获是与 `.asc` 直调工程匹配的具体源码，可用于验证 C 类型、迭代顺序、队列生命周期及核配比，降低从 API 文档独立拼实现的成本。

当前实现先写完整的 FP32 `S[B,M,pitch]`，全核同步后重新读 S 做行最大值，再同步并求和。值得试验的方向是让每个 owner 持有一段 M 行，逐块处理所有 N：

```text
完成一个输出块的全部 K 点积
    → Vector 对该块沿 N 取最大值
    → 更新 owner 持有的行最大值
所有 N 块处理完 → 写出行最大值 → 同步 → 按固定顺序求和
```

若采用这类所有权划分，显式持久中间结果可从 O(BMN) 变为 O(BM)，另计 Matmul 系统 workspace 和块缓冲。这也为减少一次全局阶段屏障提供了设计空间；仅替换 Abs 并不会自动得到这些收益。

适配时有几个具体差异：

- 教程固定 M=1024、N=640、K=256、FP16、有 bias，搬出逻辑依赖对齐块；我们的 BF16、四种布局、M/N 尾块都需要另外处理。
- 教程使用 `__mix__(1,2)`、`SetDim(2)`、`numBlocks=1`；当前是 `__mix__(1,1)`。不能直接替换核配比并沿用核编号、任务划分和屏障参与者。
- 教程按 `FIRSTM` 遍历，并用 `computeRound` 恢复块坐标。归约版本必须按真实迭代顺序维护正确的行状态；可以研究改变遍历顺序或限制单次任务的 M 范围。
- N 尾部不得把零 padding 当有效候选；必须完成 K 累加后再对 N 取最大值。M 无效行不得进入最终和。
- 小 B、小 M、大 N 时只按 M 分工可能用不满核。若拆 N，则需先合并各 N 分片的行最大值，再对 M 求和。
- A2 课程明确画出了 Cube→GM、GM→UB 的通路。`GetTensorC(LocalTensor)` 不证明底层绕过 GM，也不证明 Cube 与 Vector 已充分重叠；需要实测流水和系统 workspace 行为。
- 需要重新核算 Matmul、C 块、行最大值和双缓冲的 UB 占用，不能直接照抄 `SetBufferSpace(-1,-1,-1)`。

## 2. 性能定位：从汇总比例走到具体指令和代码行

[04.04 仿真分析课程](../cann-learning-hub/tutorials/ascendc_operator_development_light/04_debug/04.04_simulation_analysis.ipynb) 给出了我们当前脚本尚未接入的完整流程：

1. 用 `CMAKE_ASC_RUN_MODE=sim` 和 `CMAKE_ASC_ARCHITECTURES=dav-2201` 配置仿真构建，并为 ASC 编译增加 `-g`。
2. 用 `msprof op simulator` 采集；型号由 `--soc-version` 或对应模拟器库选择。
3. 查看 `trace.json` / `visualize_data.bin`、各 Cube/Vector 核的指令流水，以及 `dump/object_dump.txt`。
4. 在 Chrome tracing 或 MindStudio Insight 中把长耗时区域对应回代码位置。

教程以 Ascend910B1 举例，我们的设备是 910B2C。接入前需检查已安装模拟器支持的型号及当前 CANN 工具参数。教程还展示 `--core-id=0`，但我们的 kernel 有跨核同步，不能未经验证就套用只采集一个核的简化示例。

之前的 `run_perf.sh` 已有设备 Task Duration、汇总流水指标和工具探测；新增价值是可以继续辨别时间花在 Matmul server 的等待、地址计算、搬运、Cube 运算还是末尾求和。各流水线可能重叠，某个 Scalar 比例高不等于该源代码函数占相同比例的端到端耗时。

[Scalar 优化文章](../cann-learning-hub/blogs/operator/scalar_npu_operator_performance_optimization/scalar_npu_operator_performance_optimization.md) 进一步提供了寄存器 spill、Load/Store、指令缓存和变量生命周期的诊断线索。它的实测平台是 **Ascend 950**，百分比和提速数值不能当作我们的 910B 结论。

可检查当前 `GetMatmulBlock` 的 64 位除法/取余、Matmul 每任务初始化、归约同步以及 `SumRowMaxima` 的标量循环。它们只是待检验的假设；例如返回一个结构体不必然导致 spill，编译器可能完全消除该开销。仿真用于定位，实际提速仍以上板 Task Duration 和线上结果验收。

## 3. 分块选择：不只比较块数和活跃核数

[MX 量化 Matmul 优化文章](../cann-learning-hub/blogs/operator/mx_quantized_matmul_optimization/mx_quantized_matmul_optimization.md) 有两种值得借鉴的调度思想：

- **尾轮负载均衡**：最后一轮任务不足时再细分，让更多核参与剩余工作。
- **SWAT / 滑动窗口任务排列**：改变多核领取 M/N 块的顺序，改善 A/B 在 L2 中的复用。即使 tile 尺寸相同，调度顺序也可能改变搬运开销。

当前 `SelectMatmulPlan` 只在活跃核数不减少时选择更少的任务，没有估计尾轮、K、dtype、布局、内部 Matmul tiling 或 L2 复用。

例如在“24 核、每任务同样耗时、每核同时执行一个任务”的简化模型中，25 个任务需要两轮，任务槽利用率只有 `25/(2*24)≈52.1%`，虽然第一轮 24 核全都活跃。实际大小尾块、调用开销和流水会改变结果；这个算式只说明当前筛选规则不能保证负载均衡。

对于此前提出的“树状选择”，这些资料支持更丰富的决策条件：公开 shape、K、dtype、转置、资源约束、尾轮成本，以及实测过的算法族。先用离线测量找有效候选，再形成 Host 分派规则；不应假设 tile 大小与性能单调，也不需要在评测调用中计时搜参。

MXFP4/MXFP8、Scale 缓冲和文章里的 UnitFlag 路径不能直接照搬到当前 FP16/BF16 实现。这里借鉴的是调度思路，硬件指令与 API 支持需另行核对。

## 4. 模板特化：Host 选路径，Device 保留常量

[TilingKey 模板化编程](../cann-learning-hub/blogs/operator/tilingkey_template_programming/TilingKey模板化编程.md) 解释了通过多版本编译和 Host 选择路径，消除 Device 侧无用分支、帮助常量折叠的机制。

我们已对 dtype 和两个 transpose 属性做 C++ 模板分派，但 `tileM/tileN` 仍是运行时参数。可以对少数高收益分块试验编译期特化，并把每次调用不变的网格、步长计算放在 Host；这可能减少地址计算，是否有效应看生成代码和设备耗时。

模板参数固定 tile 并不会自动消除由动态 shape 导致的全部除法。扩展组合还会增加编译时间和二进制体积，应按测量结果保留版本。

文章的 `gert::TilingContext`、`TILING_KEY_IS` 和算子工程注册流程不适合原样迁入冻结的线上工程；我们可以在现有 `run_kernel` 内借鉴 Host 分派 + C++ 模板特化，仍然每次只启动选中的一个 kernel。

## 5. 还有一个独立的性能样例库

learning-hub 的 [README“阶段五”](../cann-learning-hub/README.md) 指向另一个仓库 `cann-samples`：

- [matmul_story](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/matmul_story)：目录说明涉及 tiling、数据搬运、尾轮负载均衡和量化。
- [flash_attn_lite_story](https://gitcode.com/cann/cann-samples/tree/master/Samples/2_Performance/flash_attn_lite_story)：目录说明涉及 Cube/Vector 融合优化，值得继续检查逐块消费 Matmul 结果的调度。

这些性能样例不是当前本地 clone 中的源码。本次网页读取未成功，尚未核实其实际实现和支持架构；不能把目录介绍当成已经验证的 910B 实现。

## 建议的下一轮实验顺序

结合用户确认的开发方向，先推进通用规则与执行结构，详细 profiling 用于瓶颈或退化定位，不作为每轮的前置条件。完整模块审计见 [实现架构审计](ARCHITECTURE_AUDIT.md)。

1. **规则规划和自动验证**：生成合法分块候选，按实际输入选择，并让 runner/采样器消费同一份计划；本地用例按边界和规模规则生成。
2. **逐块 Matmul + 行最大值**：研究数据所有权和中间存储，逐步减少完整 similarity 的生命周期；保留已有实现作参照。
3. **流水和调度变体**：在资源约束内展开双缓冲、尾轮均衡、任务排列与少量专用路径，每种路径覆盖 dtype、布局、负值、尾块及抵消用例。
4. **按问题启用源码级仿真和性能矩阵**：出现线上瓶颈或本地退化时，针对少量代表性输入定位原因，再调整规则。正确性回归和线上验收持续保留。

线上约束继续保持：只改 `kernel.asc` 或新增 `.asc`/`.h`，每次调用恰好一个 kernel launch；最终分数由统一平台决定。教程中的 main、打印、独立算子注册工程和多次启动测试程序仅作本地学习参考。
