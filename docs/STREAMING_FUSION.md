# 跨阶段融合：Matmul → Max → Sum

当前源码为 R17，最新机制与复制清单见文末。以下 R13/R14/R15/R16 段落保留为 S9/S10/S11/S12 的历史解释；各段“默认”指当时版本。

R13 的通用默认路径不再保存完整 similarity：每个活跃核组拥有一个 C 临时槽，一个任务拥有 `(batch, M 行块, N 分片)`，遍历分片内所有 N 块。每块沿完整 K 做 Matmul，立即在 Vector 上更新该任务的行最大值，再复用 C 槽。最终 Sum 沿用 R12 的补偿树，Small 的算法和自动门槛不变。

```text
各核组：完整 K Matmul → 每核 C 槽 → 更新行 Max → 下一个 N 块
                                   ↓
                          写出分片的行最大值
                                   ↓
         S=1：全核同步 ─────────→ R12 Sum
         S>1：全核同步 → 跨分片逐行 Max → 全核同步 → R12 Sum
```

这是同步 Matmul 加有界 GM 中转。仍有 O(BMN) 个 C 元素的逻辑写入和读取，尚未实现异步流水、L1 输入复用或 C 直达 UB。可能的收益来自更小的工作集、较短的生产消费间隔，以及 S=1 时少一次全核阶段屏障；逐块 Vector 消费也可能增加 Cube 等待。2026-10-01 收到 S9：15/15 通过，点 14 约快 2 倍，点 8–12 全部变慢；本机制没有取得全面收益，详见[迭代记录](ITERATION_LOG.md)。

## 所有权与规划

`stream_plan.h` 定义普通 POD `StreamPlan` 和纯主机规划函数；设备算法放在 `stream_matmul.asc`，由 `kernel.asc` 在已有公共 helper 之后 include。接口、Matmul SDK 的 inner tile 和外层任务几何分别记录。

外层 tile 继续由原有规则候选和 SDK 接受检查选定。令 `R=B*ceil(M/Tm)`、`C=ceil(N/Tn)`：

- `R≥可用核数` 时用 S=1，优先让 B/M 提供并行度。
- 否则生成补足并行度所需 `ceil(核数/R)` 的邻近值和不超过它的 2 的幂，始终保留 S=1，限制 `1≤S≤C`。
- 第 s 片负责 N 块 `[floor(C*s/S), floor(C*(s+1)/S))`。按实际 grid-stride 分配计算每核 Matmul 块数，最小化最大值；相同负载优先较少分片。
- 计算负载只遍历 shard 周期，不随 B 线性增长。它是 Matmul 块数模型，不是耗时模型：尾块实际工作量、内层 tiling、通信与归约成本尚未校准。B/M 够用时也可能有最后一轮不均衡，这属于后续选择器改进空间。

默认 Auto 先按原规则选择 Small，其余走 Stream。显式 `CANN_EXECUTION_FAMILY=gm` 保留 R12 完整 similarity 路径；`stream` 可用于本地强制验证。为兼容已有 tile sweep，Auto + 显式 tile 仍走 GM。所有 SDK 候选重试发生在 launch 前；运行失败不再发起第二个 kernel。

## 内存与同步

令 `P=min(R*S,可用核数)`、`rowPitch=align32(M)`、`slot=min(M,Tm)*Tn`（元素数）：

```text
[C 临时槽 P*slot*4 字节][最终行 Max B*rowPitch*4 字节][S>1 时各分片 Max]
```

S=1 时最终与分片 Max 为同一数组；S>1 时最后区域占 `B*S*rowPitch*4` 字节。对齐、乘加及 size_t 上界均检查；另分配 Matmul 的系统 workspace。C 用 `SetOrgShape(M,N,K,K,Tn)` 保持原始 A/B 跨度，并单独设置紧凑 C 行距。

每个行块起点 16 对齐，写回量向上对齐到 8 个 float；不同任务不共享 32 字节输出块。只读取真实列、真实行；Max 的 padding 初始化为负哨兵，Sum 的 padding 按 R12 填零。S>1 必须逐行合并 Max，不能先求各分片 Sum 再 Max。

同步 IterateAll 完成才消费 C；所有 MTE2 读取之后显式 `MTE2_S` 等待，Scalar 才能再次提交对同一 C 槽的 Matmul。输入队列约束 Vector 消费和 UB 复用，输出队列约束写回；全核屏障前完成本阶段管线。MIX 1:1 的 REGIST 负责 AIC server；后续屏障只由所有 AIV 参与。

归约显式 UB 为 `32768 + 4*Tm + 128 + (S>1 ? 256 : 0) + 14368` 字节，最大 48544，仍在原来保留的 64 KiB 内；Matmul 预算维持至多 128 KiB。没有同时引入双缓冲。

## 报告、验证与线上复制

执行报告 schema 3 增加 `family=stream` / `variant=mix_stream`，记录真实 task/block、N 分片、C 容量、各区域偏移、scratch 总量、UB 及最大每核 Matmul 块数。历史 GM/Small 报告继续可读。诊断的 pinned host 缓冲也只在实际 GM 回读时分配。Stream 的 `similarity_available=false`，本地 runner 不回读 C scratch 伪装成完整 similarity，报告明确写 `unavailable_in_stream_family`。

主机全量检查 66/66 通过。新增测试执行实际 producer、合并及 R12 Sum helper，使用检查地址/读写/队列的 CPU 替身；覆盖 160 组布局/边界/负数/零/抵消/长 K/槽复用场景，输入值均可由 FP16/BF16 精确表示，FP64 golden 独立计算。模拟 Matmul 用 float，不能证明真实 Cube 的精度、编译或异步时序。主机分派另外覆盖两 dtype×四布局、强制路径、SDK 拒绝和同步失败时不得重发 launch。

提交前状态为 `LOCAL_CANN_BUILD=NOT_RUN`、`LOCAL_NPU_TEST=NOT_RUN`、`ONLINE_EVALUATION=NOT_RUN`。现已收到用户提供的 S9 线上 15/15 Pass，按对话关联 R13；本地状态不变，平台源码哈希未核验。继续保留 S8/S9 对照，不安排额外本地 NPU 验证。

若线上已有 R12，替换 **kernel.asc、small_plan.h**，新增 **stream_plan.h、stream_matmul.asc**。同时保留同目录已有的 **matmul_plan.h、small_vector.h**，共六个源码文件。其余平台原始文件不变，不需要提交包。

官方依据与后续异步方向见[已批准设计](superpowers/specs/2026-09-30-next-wave-optimization-design.md)。


## R14：联合规划与双 GM 槽异步流水（2026-10-01）

R13 先按完整 GM 网格选 tile，然后把它交给不同任务网格的 Stream。本轮在 `joint_plan.h` 联合比较 `(family, tileM, tileN, N splits, buffers)`；SDK 接受后直接执行完整计划，不再二次修改分工。Auto 保留 Small 准入，其余在 GM、同步 Stream 和异步 Pipeline 中选择。

规则仍来自实际 B/M/N/K、布局和资源。每组外层 tile 的 N 分片候选包括 1、2 的幂、完整 N 网格和填满核数附近值，不因行任务数超过核数而提前停止。每 geometry/family 保留最低结构成本，最多 147 个最终候选。SDK 至多尝试 4 个不同 geometry，再尝试未试过的 32×64 回退；拒绝同 geometry 后不会再为另一 family 重复尝试相同 SDK tiling。显式 tile 拒绝时直接报错，不偷偷改 tile。

共同成本包含：16×16×16 Cube 工作单元、布局相关输入字节、每调用固定项、32×256 DMA/64 列 Max 工作、任务初始化、跨 N 分片合并、屏障和 R12 最终 Sum。同步 producer 累加 Cube 和 Vector；Pipeline 按 `max(Cube,Vector)` 的稳态及每个任务的预热/排空估计。任务最大数和块最大数可能属于不同核，因此是结构上界近似。固定权重没有测量校准，`selection.score` **不是周期、微秒、预测加速比或最优性证明**；SDK 内层 tiling 对真实代价的影响也尚未校准。

Pipeline 自动候选要求每个 N shard 至少有两块，才存在块间重叠；本地强制 pipeline 仍能处理单块以验证边界。其执行顺序为：

```text
预热：Cube 写槽0 → Wait → End
稳态：Cube 异步写槽1 ── 与 Vector 读槽0 / 更新 Max 重叠
      槽0读完的 MTE2_S fence → Wait 槽1 → End
      Cube 异步写槽0 ── 与 Vector 读槽1 / 更新 Max 重叠
收尾：消费最后一块 → 写行 Max → 原有跨片 Max（如需要）→ R12 Sum
```

每次只有一个 Matmul 调用在途；上一调用 Wait/End 后才更改 shape/A/B。每个旧槽在全部 DMA 读完后才可重写；任务结束和全核屏障前排空在途计算。接口使用 `IterateAll<false>(c, 0, false, true)` 与 `WaitIterateAll()`，依据 [CANN 9.0 异步说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10015.html) 和 [WaitIterateAll API](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0641.html)。

只将每核 GM C 槽扩为两个；UB 输入队列仍为一份 32×256 float，归约 UB 峰值仍为 48544 字节，未突破预留 64 KiB。两个槽的首地址为 `(core*buffers + slot)*cSlotElements`，行 Max/部分 Max 从全部 C 槽之后开始。C 的 GM 逻辑读写量仍是 O(BMN)，本轮没有实现 A 在 L1 跨调用常驻、C 直达 UB 或 Max 就地部分 Sum。

### 本地对照与报告

- `auto`：Small 或新的联合规划。
- `gm`：旧 GM 规划与完整 similarity，保留 R12 对照。
- `stream`：旧规划与同步单槽，保留 R13 对照。
- `pipeline`：只在异步双槽候选中联合选 tile/分片。
- Auto + 显式 `CANN_MATMUL_TILE` 保持 GM sweep 契约。线上入口固定使用 Auto；本地可通过 `CANN_EXECUTION_FAMILY` 指定族。

报告 schema 3 增加 `family=pipeline`、`variant=mix_pipeline`、Stream 的 `buffers`/planner v2 及 `selection` 成本模型版本/分数；实际 SDK `baseM/baseN/baseK` 继续记录。旧 v1 Stream 报告缺少 buffers 时按 1 验证。Stream/Pipeline 均明确无完整 similarity，源码快照与哈希纳入 `joint_plan.h`。

### 提交与验证

已有 R13 时，替换 **kernel.asc、small_plan.h、stream_plan.h、stream_matmul.asc**，新增 **joint_plan.h**；继续保留 **matmul_plan.h、small_vector.h**。共七个源码文件，放在同一目录。其余平台原始文件不改，不需要提交包。

完整主机回归 **68/68 通过（58.583 秒）**。真实 C++ planner 检查规则候选、独立任务枚举负载、双槽偏移、资源上界、溢出和有界 SDK 拒绝。真实设备 helper 由 CPU 操作替身执行 320 组组合，异步 C 只在 Wait 时生成，检查未完成读取、跨核槽覆盖、旧槽消费 fence、实际 pending 期间消费以及 FP64 参考；补充真实主机分派/JSON 和旧新报告验证。CPU 替身不是硬件模拟器，不能证明真实 Cube 精度、设备时序或 SDK 编译。

提交前状态为 `LOCAL_CANN_BUILD=NOT_RUN`、`LOCAL_NPU_TEST=NOT_RUN`、`ONLINE_EVALUATION=NOT_RUN`。现已收到 S10：15/15 Pass、全部错误占比 0.00%，按对话关联 R14 `65f7817`，平台源码哈希未核验；本地状态不变。较 S9 13 点变快、2 点变慢，8/9/11 基本恢复 S8，5/10/12 较 S8 进一步下降 18.98%/14.98%/12.69%，14 主要收益保留，15 仍在历史约 108–115 μs 区间。完整结果见[迭代记录](ITERATION_LOG.md)。没有实际 family/tiling 或阶段计时，不能把联合改动的全部收益归于双缓冲。


## R15：Max owner 生成补偿部分和（2026-10-01）

R14 最后阶段仍由少量 batch owner 读取整条 M 维行 Max，再执行 R12 补偿树。R15 把树的前几层放在已持有完整行 Max 的核上：GM 每 32 行、Stream/Pipeline 无 N 分片时每 tileM 行生成一条 64B 记录，包含 8 个主值和 8 个补偿值。producer 全程使用 Vector，不读回 Scalar，也不把一段提前舍入成单个 float。有 N 分片时，仍先逐行合并所有分片的 Max，再由合并 owner 生成记录；不能对各 N 片提前求和。

最终阶段仍按每组 8 batch 独占输出，用两次带 stride 的 DMA 把记录拆成连续的主值和补偿数组，再作向量 TwoSum 树和固定顺序 Neumaier 合并。每个读取块最多 128 条记录，跨块保留补偿；尾段只复制有效行，其余填零，避免将 Max 的负 sentinel 加入 Sum。原有全核屏障覆盖记录写回，R14 异步 Matmul 的 Wait/End 和槽复用 fence 保留。

以 M=8192 为结构例子，旧行 Max 为每 batch 32 KiB：GM 32 行一段时记录占 16 KiB；Stream tileM=128 时为 4 KiB，最终输入块从 8 个降到 1 个。这里描述数据量与工作分配，不是耗时倍率，也不代表已知官方测试 shape。短 M、窄 tile 或已有足够 batch 并行时，额外局部树可能抵消收益，因此保留 Rows 路径供模型选择。

### 规划、预算与观测

`reduction_plan.h` 定义普通几何/存储 POD；`partial_sum.asc` 实现设备 writer/finalizer。联合候选加入 sum mode，比较 producer 局部树、跨 N Max 和最终 high/low 读取与求和成本。每个 geometry/family 仍只保留最低成本，最多 147 个候选；SDK 尝试次数不增加。Rows 成本保留 R14，Partials 成本是未校准的结构估计，改变归约也可能改变最终 family/tile/分片选择。

局部 writer 占 `14*capacity+64` 字节，capacity 为不小于段长的 2 的幂；最终归约仍占 14368 字节。GM 归约总 UB 为 47904 字节；Stream/Pipeline 本轮最大 51936 字节，均在独立预留的 64 KiB 内。S=1 的最终区域直接保存记录；S>1 保留完整分片行 Max，额外最终区域保存合并后的记录，偏移及乘加均检查溢出。

本地 runner 增加 `CANN_SUM_MODE=auto|rows|partials`，显式模式跳过 Small；与强制 Small 冲突时报错。线上使用 Auto 且不读环境变量。执行 schema 3 增加 `requested_sum/reduction`，联合模型 v3 / `joint-work-v2`，旧报告可继续读取。本地报告展示实际 sum mode、段长，快照与哈希纳入两个新文件。完整语义见 [本地对照](LOCAL_BASELINE.md#当前路径与本地对照可选诊断不是提交前置条件)。

### 提交与验证

线上已有 R14 时，替换 **kernel.asc、stream_plan.h、stream_matmul.asc、joint_plan.h**，新增 **reduction_plan.h、partial_sum.asc**；继续保留 **matmul_plan.h、small_plan.h、small_vector.h**。共九个文件，同目录复制即可。

主机整套回归 **72/72 通过（68.063 秒）**。真实 helper CPU 检查覆盖 GM、Stream/Pipeline 的 S=1/S>1、尾块、多核所有权、延迟 Matmul 完成、强抵消/近零与跨段跨读取块的补偿、重复输出、producer 无 Scalar 读回及实际 UB；实际 host 分派和 metadata 检查覆盖单 launch、显式控制、失败及历史兼容。另将 640 组规则输入的新版强制 Rows 候选与固定 R14 `65f7817` 比较，全部 family/tile/排序/成本/布局/负载一致。

提交前状态为 `LOCAL_CANN_BUILD=NOT_RUN`、`LOCAL_NPU_TEST=NOT_RUN`、`ONLINE_EVALUATION=NOT_RUN`。现已收到 S11：按对话关联 `5e27c5a`，15/15 Pass、全部错误占比 0.00%，点 13 较 S10 -23.90%，其余变化在 -2.90%～+3.53%，点 15 无突破；完整结果见[迭代记录](ITERATION_LOG.md)。本地测试状态不变，平台源码哈希和实际计划未核验，通过不代表所有 Partials 分支均命中。依据 [CANN 9.0 DataCopyPad](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0265.html) 的 GM byte stride / UB 32B stride 规则，本轮记录读取使用最多 128 个 32B 块。设计见 [R15 spec](superpowers/specs/2026-10-01-partial-sum-design.md)。


## R16：冻结上游计划的受控覆盖（2026-10-01）

本轮检验“R15 成本模型是否过于保守”。`PrepareExecution` 先完成原 R15 候选排序、SDK 拒绝/接受，再调用 `ExpandPartialReduction`。只作用于默认 Auto family/tile/sum 的 Rows；原 Partials、Small 和所有显式控制保持原行为。转换要求至少两个 M 段、`16*segments < M`，并满足现有归约 64KiB 和 Matmul 合计 UB 预算。没有增加 batch 或估计时间门槛，避免未校准成本继续挡住实验。

只改变 reduction 及其依赖的 scratch 偏移/大小/UB；family/tile/N split/buffers/tasks/blocks/C slot/SDK tiling 全部来自原已接受计划。设备实现、向量补偿树、异步流水和跨核屏障不变。对于短 M，局部树和打包可能仍然得不偿失；本轮是有边界的覆盖实验，需通过线上反馈确定适用范围。

同一 3960 组 CPU 元数据网格，Partials 1142→2612（28.84%→65.96%），新增 1470 组；所有上游计划字段与固定 R15 一致。此统计未经 SDK 接受，不代表官方输入覆盖。主机整套 74/74 通过（69.142 秒），另包含连续 SDK 拒绝下的接受顺序/结果一致性。未做本地 CANN/NPU 编译执行。现收到 S12：15/15 Pass，较 S11 没有新增明显提速；点 13 +1.98% 保留 R15 主要收益，点 8–12 近似持平，点 15 -0.22% 仍无突破。实际执行计划未提供，CPU 覆盖扩展不证明官方点已切换。

本地 `CANN_SUM_MODE=r15` 可关闭扩围而保留 Small；auto 启用 R16，rows/partials 与 forced family/tile 继续作为旧控制。报告 `reduction_expansion` v1 保留 baseline mode/score 和实际切换；最终 `selection.score` 可能高于 baseline，因为扩围没有重新用原模型淘汰候选。线上不读取这些变量。

**线上已有 R15 只替换 kernel.asc、joint_plan.h，无新增文件。** 完整九文件哈希与 S12 观察项见[迭代记录](ITERATION_LOG.md)，规则与边界见[设计](superpowers/specs/2026-10-01-controlled-reduction-design.md)。

S12 后结束连续扩围试探，当前代码仍为 R16；下一轮优先研究逐块 Matmul 配置/Wait/End 与 C 的 GM 中转。完整反馈与边界见 [S12 档案](ITERATION_LOG.md#s12r16-受控扩围的线上反馈2026-10-01-收录)；本次仅更新记录，不新增设备实现，也不把候选结构成本写成已定位的线上瓶颈。


## R17：S2 上游受控对照（2026-10-01）

本轮检验点 15 从 S2 76.91 μs 到当前 109.31 μs 的历史退化是否涉及上游选择。`kernel.asc` 的 `S2_UPSTREAM_CONTROL` 默认 true，false 恢复 R16；只对默认 Auto、sum=auto 生效，Small 与显式 family/tile/sum/r15 控制保留。

先计算完整 R16 计划与归约模式，再尝试 GM32×64。已有相同 tile 时复用 SDK 结果，否则另请求一次；拒绝时原计划完整保留。选用时保留 Rows/Partials 模式，按 GM 的 32 行段重建存储、任务/核数与实际结构成本。仍使用当前向量补偿 Sum，设备 helper 不变，不是完整 S2 回滚。Stream 转 GM 会改变 C 生命周期与部分和分段，不声称隔离了单一参数；不能据此自动断言某种 dtype/shape 更适合旧块。

本地报告的 `upstream_control` 记录实际 selected/sdk_rejected 及 R16 参考，selected 不继续输出旧上游的扩围 trace。线上无调试输出和测时分派，始终只启动一个 kernel。仍运行 R16 host 规划作为参考，因此不是 host 规划耗时优化。提交时没有增加成本阈值；现 S13 支持点 15 对这一组合有收益，同时揭示全局固定旧方案的明显代价，尚不足以确定普适阈值。

已有 R16 只替换 **kernel.asc**，无新增文件；本轮 GM 完整 C 存储和旧几何可能使其他点变慢，属于受控实验。未做本地 CANN/NPU 测试；S13 已收到 15/15 Pass，完整结果、验证与源码哈希见[迭代记录](ITERATION_LOG.md)。

验证：74/74 主机检查通过（70.797 秒），独立审查未发现实质问题；两种源码开关配置各验证 120 条真实 writer JSON 及 SDK/启动失败路径。本轮设备 helper 未改；主机验证之外，现已收到下方 S13 线上反馈。


S13：点 15 109.31→76.16 μs（-30.33%），接近 S2 76.91 μs；点 8–14 明显退化，点 12 从 120.15 μs 增至原文 1.14 ms。支持上游组合适用性不同，不能全局固定旧方案，也未区分 family 与 tile/SDK 的各自贡献。R17 仍执行 R16 host 规划，不能归因为省掉匹配开销。当前只归档，开关仍为 true；R16/S12 和 R17/S13 保留为两份对照，下一步优先区分 GM 方式与几何选择。详见 [S13](ITERATION_LOG.md#s13r17-固定上游对照的线上反馈2026-10-01-收录) 与 [H10](CASE_HYPOTHESES.md#h10--s13-修订2026-10-01)。
