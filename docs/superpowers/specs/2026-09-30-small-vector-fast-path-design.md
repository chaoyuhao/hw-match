# 小规模 Vector 快速路径设计

日期：2026-09-30。状态：设计稿，尚未实现、编译或测量。用户要求先准备下一版设计；本轮只记录结果与设计，不修改计算代码。基准代码为 `1cc3cba`，线上关联反馈为 [迭代记录 S3](../../ITERATION_LOG.md)。

目标是验证较小实际输入能否通过简单的核内计算减少通用 Matmul 通信、中间矩阵和跨核同步成本。线上前几个点约 12–17 μs，而最优约 1–3 μs，是优先调查的线索，不能据此推断它们的 shape 或承诺达到 1 μs。小规模快路径提前于完整 Matmul/Max 融合、双缓冲与大规模树形求和。

## 方案选择

| 方案 | 可能收益 | 代价与适用范围 | 本轮选择 |
| --- | --- | --- | --- |
| 纯 Vector 核内点积与归约 | 省去 Matmul server、完整 similarity GM scratch 和全核屏障 | 算量上升、布局不连续或循环太多时可能更慢 | 优先实现一个受限快速路径 |
| MIX Matmul 单 owner 融合 | 保留 Cube 计算能力，减少部分阶段同步 | 仍有通信与 workspace，结果搬运与 owner 设计更复杂 | 留待后续中等规模路径 |
| 继续调通用分块 | 改动较小、覆盖广 | 通用路径的固定成本仍在 | 当前版本作为回退与对照；本轮冻结评分规则 |

已有规划、用例、runner 和采样工具继续使用，只增加快速路径必要的分派和验证。候选等价去重、复杂成本模型、完整 autotune 框架不与本轮混做，便于归因线上变化。

## 分派与单次启动

```text
校验输入元数据，恢复逻辑 B/M/N/K 与真实布局
  → 快路径资源与已验证性能范围是否满足
      → 满足：一个 AIV-only kernel → FP32 最终输出
      → 不满足：现有 MIX MatmulMaxSum → FP32 最终输出
```

分派发生在 host，依据尺寸、dtype、布局、UB 需求、每 owner 的工作量及任务轮次；不得读输入内容搜参，也不得按线上点号、缓存答案或 case 名分派。每次调用只进入一个 launch 分支。快路径使用单独的轻量准备入口，绕过 Matmul tiler 和两笔 GM scratch/workspace 分配，仍使用原 `run_kernel` ABI。

快路径返回前维持现有 stream 完成契约。若快路径启动后发生运行错误，直接向本地 runner/调用方报告错误，不能再启动通用 kernel“补算”，以免违反单次 launch。

## 核内工作与布局

一个 AIV 独占连续 8 个 FP32 输出对应的 batch 组，逐个 batch 计算；最后不足 8 个只写有效字节。这样延续现有 32 字节输出组所有权，避免相邻核写同一输出块。按 batch 组分配任务，不拆同一 batch 的 M/N/K 到多个核；资源按一个 batch 复用。首版核数保守限制在传入 `availableCoreNum`，不假设 AIV 核数等于 Cube 核数的固定倍数。

输入按物理布局搬到 UB，并先转换为 FP32，再乘法与累加。初始化和搬运 padding 时显式记录有效元素；任何无效 N 列都不得参与 Max。实现两种短循环，共享加载、输出和精度检查：

- **K 连续点积**：满足 `(!TA || M==1) && (TB || N==1)` 时，A 的逻辑行与 B 的逻辑列沿 K 连续。用 FP32 向量乘与 K 归约得到一个完整点积，更新该行最大值，再按 M 顺序补偿求和。单维为 1 的等价布局由实际跨度判定，不写特定 shape 白名单。
- **N 连续累加**：`TB==false` 时，B 的每个 K 行沿 N 连续；为一个 M 行保留 FP32 的 N 向量，按 K 更新 `c[:] += a[m,k] * b[k,:]`，完成全部 K 后才求 Max。A 从已转换的 UB 按真实跨度读取，覆盖 TA 的两种布局。该循环有每 K 的向量指令/依赖开销，只在较短 K 的验证范围启用。

两个条件都满足时优先选 K 连续点积，避免分派歧义。`TA=true, TB=true, M>1, N>1` 在首版走通用回退；暂不引入逐元素重排来强行扩大命中范围。算子的两 dtype/四布局仍全部支持，只是快速分支的覆盖范围不同。

行最大值从首个有效点积初始化。最终 M 维先沿用顺序补偿求和，因为目标 M 很小；本轮不同时改变 M 求和算法。K 连续路径归约次序与 Cube 可能不同，必须单独验证误差与重复一致性。快路径不创建完整 similarity GM 副本。

## 资源与启用边界

“UB 能放下”只说明资源合法，不代表快速。选择器同时限制每 batch 的乘加量、点积数量或 N 向量更新次数、batch 组任务轮次及对齐后 UB 字节数。布局不同的性能阈值分别验证。

首批本地探索以 `B<=32, M<=16, N<=64, K<=256, M*N*K<=32768` 为规模上界；N 连续循环另限制 `M*K<=64`。这些是降低开发实验成本的初始上界，**不是已证明更快的线上分派阈值**。最终 auto 只启用经过成对测试支持的区域；不能把全部探索范围默认视作快速。

资源检查统计实际对齐后的输入队列、FP32 A/B、点积/行累加临时缓冲、归约工作空间、行最大值和输出。每个 buffer 计入队列深度与 padding；首版只使用单缓冲，计划总额不超过 `min(64 KiB, 实际 UB 减保留余量)`。实现时固定并测试保留余量及归约 workspace 大小，未知或不足就选择通用路径。资源上界不依赖缓存或其他核的空闲 UB。

## 本地工具需要的最小变化

- 新增本地 family 控制：`auto`、`gm`、`small`。`small` 在不支持的布局/尺寸上明确拒绝，不静默回退；`auto` 记录命中路径或回退理由。已有固定 Matmul tile 控制只用于 `gm`，与强制 `small` 同时设置时报冲突。
- 提升执行计划 schema：公共字段为完整输入身份、算法族、变体、核数、任务数和 UB 预算；Matmul tile/内部 tiling 只在 `gm` 中有意义。旧 schema 1/2 报告继续可读，不能给 Vector 记录伪造 Matmul 几何。
- 同一构建、相同量化输入比较 `gm` 与 `small`，保持至少两轮并轮换顺序；沿用现有采样逻辑，不先建设全量 family×tile 矩阵。再运行 auto 验证选择正确。
- 快路径没有 similarity GM。默认正确性 runner 必须允许仅校验最终 y，并记录 `similarity_available=false`；不得为诊断强行多发一个 kernel。通用路径仍可回读 S。
- 现有 tile sweep 只接受 `MatmulMaxSum/MIX_AIC`。family 比较必须按实际算法校验 kernel 名、Task Type、记录数和普通运行/profile 计划一致性；先用真实 AIV-only profile 确认导出格式，再加入明确匹配。不能继续把单一 MIX 类型当作所有合法 kernel 的条件。
- 新增 header 如有则进入源码 hash 和快照。线上仅复制 `kernel.asc` 及其新增 `.h/.asc` 依赖。

## 验证与完成条件

规则自动生成探索范围内的尺寸、对齐边界与其相邻值，以及分派阈值前后样本；K 保持 8 的倍数。加入 B=8 分组边界前后和多组循环，确保尾输出不越界。使用同一几何覆盖 FP16/BF16 与四布局，使快速命中和回退都可检查。

复用随机、全负、零、强抵消、近零输出和已知答案；增加最大值近似相等的输入，捕获点积舍入改变 Max 选项的影响。检查输入不变、输出 guard、有限输出、重复一致及 FP64 golden。测试不比较两路径逐位相等；两者分别满足精度阈值，且各自重复结果一致。

先做主机分派/地址/资源/工具契约测试，再在本地 CANN 9.1 编译验证两条路径及通用回退，最后上 CANN 9.0 平台检查全部 15 点。CPU 替身只能检查逻辑，不能模拟真实指令舍入和异步行为。

性能只需要有针对性的成对测试覆盖命中区域、边界与回退。对连续两轮均显示收益、差距超过局部波动的区域启用 auto；其他区域保持 `gm`，不为了提高覆盖率接受可见退化。线上单轮结果保留全部点，尤其跟踪 1–5、13、15；若它们没有受益，不能擅自认定它们“不属于小输入”。全量 profiling 不是前置条件。

完成标志：单次 launch、精度和内存契约通过；实际分派可追溯；给出有设备证据的启用范围；更新统一迭代记录与逐点假设。达到 1 μs 不是预先承诺的验收值。

## 官方接口依据

官方 [Vector Add 示例](../../../cann-learning-hub/tutorials/ascendc_operator_development/02_AscendC_basic/src/add_custom.asc) 展示 `KERNEL_TYPE_AIV_ONLY`。该目录是本地忽略的参考克隆，线上不依赖它。

[CANN 9.0 Cast](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html) 说明 half/bfloat16 与 float 的转换支持；实现需以 Atlas A2 对应表及实际 SDK 编译为准。

[CANN 9.0 ReduceSum](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0078.html) 给出 FP32、对齐及临时空间约束，并提醒软件实现的 ReduceSum 在部分场景可能比基础归约指令慢。可据此比较短 K 下的 [WholeReduceSum](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0081.html)；不能仅因为函数名是归约就假设成本很低。
