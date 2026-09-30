# 小规模 Vector 快速路径设计

日期：2026-09-30。状态：R8 已收到 S4 线上 15/15 通过反馈，未做本地 CANN/NPU 测试，详见 [迭代记录](../../ITERATION_LOG.md)。本文保留 R8 设计；R9 只将 Dot/Rows 自动预算分别改为 128/256，其他合法性条件不变；S5 已 15/15 通过，未见明显新增加速，结果与前置候选限制分析见迭代记录。通用计算和分块规则保持 `1cc3cba` 行为。

用户最新决策覆盖原先“设备成对测试后再启用”门槛：在线上表现和开发思路双双陷入瓶颈前，不额外做本地 NPU 测试。R8 直接启用保守实验范围并提交线上验证；不把资源合法或 CPU 检查通过当作加速证据。

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
  → 快路径资源与保守实验范围是否满足
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

首批本地探索以 `B<=32, M<=16, N<=64, K<=256, M*N*K<=32768` 为规模上界；N 连续循环另限制 `M*K<=64`。这些是降低开发实验成本的初始上界，**不是已证明更快的线上分派阈值**。R8 auto 另外要求任务组一轮完成，Dot 每 owner 的 `min(B,8)*M*N<=16`，Rows 的 `min(B,8)*M*K<=32`。这些阈值未经测量；按用户决定先交线上验证，不扩大到全部探索范围。

资源检查统计实际对齐后的输入队列、FP32 A/B、点积/行累加临时缓冲、归约工作空间、行最大值和输出。每个 buffer 计入队列深度与 padding；首版只使用单缓冲，计划总额不超过 `min(64 KiB, 实际 UB 减保留余量)`。实现时固定并测试保留余量及归约 workspace 大小，未知或不足就选择通用路径。资源上界不依赖缓存或其他核的空闲 UB。

## 本地工具需要的最小变化

- 新增本地 family 控制：`auto`、`gm`、`small`。`small` 在不支持的布局/尺寸上明确拒绝，不静默回退；`auto` 记录实际路径。已有固定 Matmul tile 控制只用于 `gm`，与强制 `small` 同时设置时报冲突。
- 提升执行计划 schema：公共字段为完整输入身份、算法族、变体、核数、任务数和 UB 预算；Matmul tile/内部 tiling 只在 `gm` 中有意义。旧 schema 1/2 报告继续可读，不能给 Vector 记录伪造 Matmul 几何。
- 保留本地强制 family 的诊断能力，不新增 family sweep，不要求本轮运行 NPU 对照。只有后续线上与开发思路都遇到瓶颈时，才按需要做同输入成对采样。
- 快路径没有 similarity GM。默认正确性 runner 必须允许仅校验最终 y，并记录 `similarity_available=false`；不得为诊断强行多发一个 kernel。通用路径仍可回读 S。
- 现有 tile sweep 强制 GM，仍只接受 `MatmulMaxSum/MIX_AIC`。通用报告保留不同任务类型并对比实际执行计划，本轮不增加 AIV 排名器；没有真实 profile 时不猜测其导出类型。
- 新增 header 如有则进入源码 hash 和快照。线上仅复制 `kernel.asc` 及其新增 `.h/.asc` 依赖。

## 验证与完成条件

规则自动生成探索范围内的尺寸、对齐边界与其相邻值，以及分派阈值前后样本；K 保持 8 的倍数。加入 B=8 分组边界前后和多组循环，确保尾输出不越界。使用同一几何覆盖 FP16/BF16 与四布局，使快速命中和回退都可检查。

复用随机、全负、零、强抵消、近零输出和已知答案；增加最大值近似相等的输入，捕获点积舍入改变 Max 选项的影响。检查输入不变、输出 guard、有限输出、重复一致及 FP64 golden。测试不比较两路径逐位相等；两者分别满足精度阈值，且各自重复结果一致。

本轮做主机分派/地址/资源/工具契约检查，随后直接上 CANN 9.0 平台检查编译及全部 15 点；按用户决定不插入本地 NPU 环节。CPU 替身只能检查逻辑，不能模拟真实指令舍入和异步行为。

线上继续保留全部 15 点，尤其跟踪 1–5、13、15；若它们没有受益，不能认定它们“不属于小输入”，也可能只是未命中本轮保守规则。若性能退化，先结合改动与线上结果调整规则，避免默认扩展为大量本地采样。

本轮开发交付条件：主机逻辑/单次启动/元数据检查通过，给出可复制的线上候选，并更新迭代记录与逐点假设。实际 CANN 编译、NPU 正确性和加速需等待线上结果；达到 1 μs 不是预先承诺的验收值。

## R10 实施更新

R9 的 S5 未见明显新增收益，用户授权继续扩展前置候选。R10 移除 B≤32、M≤16、MNK≤32768 和 Rows MK≤64 的固定初筛，只保留通用输入范围、N≤64/K≤256 的现有缓冲容量、布局条件及精确 UB 容量检查。每批输入仍完整载入 UB，设备实现不变。

auto 沿用 Dot 每 owner 点积数≤128、Rows 更新次数≤256 及单波限制；所以 B 的有效上限随可用核心数变化，较大 M 也要经过工作量和 UB 限制。候选生成与自动启用分开，避免把能放下等同于更快。任务组数在窄化到 uint32 前检查溢出。

R10 已收到 S6：15/15 通过，点 2 耗时下降 10.31%，其余小幅变化。具体版本、原始测量及判读见统一迭代记录。

## R11 实施更新：批量 Dot

用户在 S6 后要求继续改变计算机制。保留 small/GM 分派范围，仅把多列 Dot 改为列块 repeat 计算。候选宽度由 `min(N,8/16/32/64)` 生成；UB 合法性检查后比较 Vector API 调用次数，同分选较小缓冲。列块未覆盖全部 N 时，列宽必须是 8 个 float 的倍数，确保后续 Add 起址 32B 对齐；单块可处理任意实际 N≤64。

输入 padding/转换及 output owner 沿用。FP32 product 缓冲是 `[列块,64]`，按 K 分段复用；每段 Mul 使用 A repeat stride 0、B repeat stride 为实际 FP32 行距。WholeReduceSum 以 element 单位的 dstRepStride=1 写连续列分数；跨 K 分段 Add 仍按旧顺序相加。全部 K 完成后沿真实 N 取最大值，最后每行一次 Scalar 读取，再按旧 M 顺序补偿求和。

实际 UB 为 `6×(A_padded+B_padded)+4×(64×C+align8(C))+288`：输入队列和 FP32 副本、列块 product、K 分段 partial、64-float 行分数及 32B 输出。先检查旧 Dot 基线可容纳，再选择新缓冲；N=1 或没有合法列块时沿用原 Dot，保持原 small 覆盖。Rows 不变。

提交前未获得设备编译或加速证据。现已收到 S7：15/15 通过，13 点变慢、2 点略快，未见性能收益；版本按对话关联。CPU 检查要求资源/步长/mask 与分派契约、FP64 golden、未初始化读取及边界保护；另对相同 Dot 的旧/新 CPU 实现验证原 K 分段相加顺序未变。该逐位对照不扩展为不同算法族或真实设备的逐位等价要求。线上只替换配套的 small_plan.h 与 small_vector.h。

## 官方接口依据

官方 [Vector Add 示例](../../../cann-learning-hub/tutorials/ascendc_operator_development/02_AscendC_basic/src/add_custom.asc) 展示 `KERNEL_TYPE_AIV_ONLY`。该目录是本地忽略的参考克隆，线上不依赖它。

[CANN 9.0 Cast](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html) 说明 half/bfloat16 与 float 的转换支持；实现需以 Atlas A2 对应表及实际 SDK 编译为准。

[CANN 9.0 ReduceSum](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0078.html) 给出 FP32、对齐及临时空间约束，并提醒软件实现的 ReduceSum 在部分场景可能比基础归约指令慢。可据此比较短 K 下的 [WholeReduceSum](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0081.html)；不能仅因为函数名是归约就假设成本很低。


R11 新用的 [CANN 9.0 Mul 高维切分接口](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0037.html) 支持 FP32 mask/repeat 和 BinaryRepeatParams；[WholeReduceSum](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0081.html) 的目的 repeat 步长单位是元素，FP32 目的地址要求 4B 对齐、源地址要求 32B 对齐。[Add](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0035.html) 操作数要求 32B 对齐；本实现仅同地址原位更新，遵守[重叠约束](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0004.html)。文档依据不替代真实 SDK 编译。
