# 跨阶段融合：Matmul → Max → Sum

当前源码为 R21：先折叠列组、延后横向 Max。最新机制与复制清单见文末；R13–R20 段落保留为历史说明，各段“默认”指当时版本。最新线上结果 S16 对应 R20，15/15 Pass；R21 尚无线上结果，Iterate 连续实验仍关闭。

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


## R18：连续 Matmul→Max（2026-10-01）

**后续更正（S14/R19）：此节是提交前设计记录，旧代码误将紧凑 ND 尾块行距视为 baseN；仅屏蔽无效列不能保证寻址正确。下节记录修复与证据，R18 的 CPU 通过记录不能继续作为尾块正确性的依据。**

新入口保留 R16 自动选择作为参考，默认关闭 R17 的 S2 固定对照。Small 准入和设备计算不变；其余默认 Auto 在 SDK 接受后进入 `iterate`，显式 family/tile/sum、r15 和 S2 对照仍走原实现。

```text
每个 owner 配置整段 N 和完整 K（一次）
  Iterate → GetTensorC 到单块 UB → 按有效列更新行 Max
  Iterate → GetTensorC 到复用 UB → 更新同一行 Max
  ... → End（一次）→ 片内行 Max / 补偿记录
全核同步 → 必要时跨 N 分片 Max → 原补偿 Sum
```

原 Stream/Pipeline 的 M/N 所有权不变。GM 参考计划使用原 tileM/tileN，但把 N 划分为填满可用核数所需的最少分片，减少每个 owner 的启停和完整 C 存储。若该分工相较原 GM 不合适，可能发生性能退化，当前没有在线测时或隐藏点号分派。保留 Rows/Partials 模式，按新 owner 段长重建归约；不承诺保持旧部分和分段顺序。

Host 使用 VECIN C、FIRSTM 和 `SetFixSplit`，内部 C 至多 32 KiB，实际 SDK baseM/baseN 决定 UB 分配和坐标步长；总归约 UB 至多 64 KiB，Matmul 使用单独预算。按有效 rows/cols 发出 WholeReduceMax，不初始化或读取 C 的无效 padding。每块消费后归还 VECIN queue，下一块复用遵守队列依赖；End 后沿用 AIV-only 全核屏障，仍一个 MIX launch。

参考 [CANN 9.0 Iterate](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0638.html)、[GetTensorC](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0639.html)、[SetTraverse](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0685.html)，以及本地官方课程的 `matmul_abs.asc`。应用代码不再保存 C，但 SDK 底层仍可能借助 GM；这不是已验证的物理片上直达。同步 Iterate 消费也不同于此前异步双槽 Pipeline，真实收益等待线上。

线上型号未知；910B2C 仅为本地设备。原模板默认 `dav-2201` 编译目标允许覆盖，不能据此认定平台卡型。新计划读取可用核数/UB，SDK 拒绝时完整恢复 R16；这种回退不能处理编译失败，也不证明所有硬件均支持该 MIX 接口。

本地 `CANN_MATMUL_MAX=auto|r16` 用于可选诊断；线上不读取。新报告 `family=iterate` / `variant=mix_iterate`，stream planner v4 将 C scratch 容量/起点置零，增加实际 UB C 容量、最大会话宽度、会话数及 R16 参考。旧性能评分不冒充新路径预测。历史 schema 3 报告继续可读。

**已有 R16/R17：替换 kernel.asc，新增 iterate_plan.h、iterate_matmul.asc**，其他八个依赖保留。76/76 主机检查通过；无本地 CANN/NPU、无线上的本轮结果。完整哈希和比较基线见 [R18 归档](ITERATION_LOG.md#r18连续-matmulmax2026-10-01)。

## R19：ND 尾块契约修正与默认回退（2026-10-03）

S14 的点 3/6/8 均 WA100%，且通过点 10/12 相对 S12 大幅变慢。因此先将连续路径关闭，默认恢复 R16 自动方案，S2 对照保持关闭；修复后的连续路径保留为实验；后续 S15 已验证默认回退配置 15/15 Pass，连续实验仍未线上验证。既有 R18 的提交只需更新 `kernel.asc` 和 `iterate_matmul.asc`，其余九个依赖不变，不制作提交包。

### 已复现的代码缺陷

`GetTensorC<true>(LocalTensor, 0, true)` 的 sequential ND 输出按当前 tile 的实际列宽紧凑排列。旧实现按 `baseN` 定位下一行；当尾块宽度小于 `baseN` 时，从第二行起即可读到错误位置。旧 CPU 替身也按 `baseN` 写出，恰好隐藏了同一错误。这是我们对生产者/消费者布局契约及测试替身的共同误判。

以 baseN=16、尾块实际宽度=8 为例，第二行应从第 8 个 float 开始，旧代码却从第 16 个开始。把 CPU 替身改成紧凑布局并将未写尾部置为 NaN 后，旧生产代码触发 `masked reduction read invalid/uninitialized lane`；修正 `ConsumeLocalC` 的行距为 `validCols` 后通过。这证明了具体缺陷，不证明未知 shape 的三个线上点都由它单独导致。

WholeReduceMax 的行重复步长按 32 字节计，FP32 因而要求行宽为 8 的倍数。当前外层 tileN 与 SDK 接受的 baseN 均 16 对齐；准入 N%8==0 可保证所有分片及内部尾块的行距合法。不满足时在新的 SDK tiling 前返回完整 R16 计划。此筛选是指令合法性条件，不是性能阈值；后续若支持任意宽度，需要单独实现布局转换或其他消费方式，不能只去掉条件。

### 官方源码依据

核对官方 `cann/asc-devkit` 的 **9.0.0 tag**，只用公开接口，未把 SDK 内部文件加入提交依赖：

| 官方源文件 | 核对内容 | 下载内容 SHA-256 |
| --- | --- | --- |
| [n_loop_norm_base.h](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/impl/adv_api/detail/matmul/scheduler/iterator/n_loop/n_loop_norm_base.h) | `UpdateInnerParams` 尾轮使用 `tailBaseShape_`，不是固定 baseN | `2a398c917a8507a5c132ca4ddb31026dc4e760d4328691ef86199f03eb2fb1f0` |
| [copy_cube_out_fixpipe.h](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/impl/adv_api/detail/matmul/stage/copy_cube_out/copy_cube_out_fixpipe.h) | `CopyOutNZ2ND` 的 sequential 分支用 `baseWidth` 作目的行距 | `400344586a5a57bde7d80df8d2b74068d8df9f510958aed8aaa6aafa5b71b95d` |
| [matmul_client.h](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/include/adv_api/matmul/matmul_client.h) | Cube/Vector 分离实现的同步 LocalTensor `GetTensorC` 经 GM workspace；`CopyToUB` 普通 ND 分支平坦拷贝，不恢复 baseN 行填充 | `70b5067fe70e2ec42e7ac4f708ce6a0a991e91c1d605e539986123471f45b5bb` |

公开 tag 与线上实际 SDK 二进制未逐一核验，线上卡型仍未知。不把这里的分离实现推论为所有硬件的物理路径，也不把 VECIN API 视为已绕过 GM。ND_ALIGN 的搬运契约不同，未作为未经验证的替换方案。

### 验证范围与剩余跨模块机会

主机全套 76/76 通过（74.986 秒）。实际设备 helper 在 CPU 指令替身下检查 320 组：包含 8 列紧凑尾块、多 M/N 块、四转置、分片后 Max、Rows/Partials、负数/零/抵消及资源所有权。真实 host 分派在两种 S2 配置下各校验 191 条 JSON，其中新增 32 组非对齐宽度回退；检查单次 launch、保留原 scratch/计划、SDK 拒绝和默认开关。元数据 v2 校验对齐条件，v1 历史兼容。未做本地 CANN 编译/NPU 测试；提交后的 S15 默认回退配置 15/15 Pass，但连续路径仍关闭。

跨模块优化没有做完，R18 也不能算已成功完成 Matmul→Max：

| 剩余机会 | 当前源码边界 | 下一步优先级与约束 |
| --- | --- | --- |
| 连续产出与异步消费结合 | R18 同步 GetTensorC、单 UB 块，替代了旧 Pipeline 的双槽重叠；不能把少 End 等同于更快 | 正确性之后优先；保留 R16 Pipeline 对照，先证明真实 API 所有权及同步契约，再尝试重叠 |
| 按实际内部块选择整个执行方案 | R18 是先选 R16 再转换，没有把连续族的内块、交接次数、等待和 owner 并行度一起重新排序 | 高；SDK 接受仅说明配置可用，不是低延迟证明。避免所有通用输入无条件转换 |
| 完整 batch 的 owner 直接收尾 | 通用路径即使单 owner 拥有一个 batch，也仍经 GM 记录和全核屏障交给最后一级 Sum | 在有足够 batch 并行度时有价值；多 owner/跨 N 分片仍必须先正确合并 Max，不能各片先 Sum |
| 各阶段复用 UB 与扩大块 | 当前不同阶段的缓冲预算/分配未统一按生命周期复用 | 后续；需要证明旧 DMA/Vector/Matmul 引用全部结束后才复用，不能只把预算数字放宽 |

本轮只修复可复现缺陷并隔离失败实验，不同时引入上述性能改动。S14 未定位出哪项等待或布局对应哪一个隐藏点，也没有证明点 15 的具体输入或执行路径。

## R20：跨任务延续双槽 Pipeline（2026-10-03）

这是对现有 Pipeline 调度的有界改动。S15 全通过后，选择保留 R16 的分块、N 分片、SDK tiling 与 Rows/Partials，仅移除每个任务边界的重新预热。同步 Iterate 路径仍关闭；更换 GetTensorC 输出格式或重新设计内部 tiling 的方案留待独立实验。

### 改动原理

旧实现每个 `(batch, M 行块, N 分片)` 先提交首块、立即 Wait/End；中间 N 块使用两个 GM C 槽重叠生产和 Max；最后一块消费和部分和写出后才开始下一任务。新实现把最后一块的预提交目标延伸到同一核下一个 `task + blockNum` 的首块：

```text
旧：任务 A 最后块 Max → A 的记录写出 → 提交 B 首块 → 等待 → 消费 B
新：提交 B 首块 → A 最后块 Max → A 的记录写出 → 等待 B → 消费 B
```

当同一核拥有多个任务时，下一任务的 Cube 可以与当前任务的最后一次 Max、Rows 写回或补偿记录生成并行。只有本核第一个任务需要首块立即等待。Wait/End 总次数仍是每个 Matmul 块一次，Cube 计算量、块数和 A/B 地址不变；这里只增加可用于覆盖等待的独立工作。只有一个任务的核没有新增跨任务重叠机会；GM、单槽 Stream 和 Small 的执行算法不变。不保证未知线上点的命中或收益。

例如 B=3、M=65、N=129、tile=64×64、splits=1、cores=3 时共有 6 个 owner、18 个 Matmul 块：旧流水有 6 次首块立即等待，新流水为 3 次，其余等待都在已有 C 消费之后。这个例子用于解释调度，不是已识别的线上 shape，也不是性能预测。

### 所有权和同步

一个核始终最多一个 Matmul 在途；只在前一次 WaitIterateAll/End 完成后重设 A/B/shape。新提交只写同核的另一个 C 槽，当前 C 的全部 MTE2 读取完成并经过 MTE2_S 围栏后才允许该槽再次被覆盖。两槽奇偶在任务间延续，包括奇数 N 块和不均分片；行 Max 的 output queue 仍等写回后释放，下一任务重新初始化其独立的行 Max。

任务末尾存在下一任务时才预提交，下一任务入口等待并 End。最后任务没有预提交，因此函数返回、全流水屏障和 SyncAll 前全部 Cube 工作已排空。B/M/N 所有权、跨 N 先 Max 再 Sum、补偿求和顺序、分配空间和单 kernel launch 均保持。这里复用已使用的异步 GM 接口，不引入 R18 的 ND LocalTensor 布局。

依据 [CANN 9.0 WaitIterateAll](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0641.html) 与[异步处理说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10015.html)：异步 IterateAll 发出后可做独立工作，使用结果前必须等待。本轮不扩大硬件支持声明；线上 SKU 仍未知，真实重叠和收益由线上确认。

### 验证和提交

默认 `stream_plan.h` 中 `PIPELINE_CHAIN_TASKS=true`；设 false 可恢复 S15 调度。候选评分刻意保持旧模型，因此本轮没有通过重排计划改变路径覆盖。报告单独记录 `pipeline_schedule` 的 core/task 范围，旧报告继续兼容。两项旧实验仍 false。

先在旧实现上加入实际 helper 的调度回归，观察到“每任务重新预热”的预期失败；实现后通过。完整主机 **76/76 通过（78.747 秒）**。Stream 检查 1376 组，交叉四转置、Rows/Partials、单/双槽、旧/新调度，覆盖不均分片、奇偶块数、batch/行跨界、M/N 尾块、负数/零/抵消、空闲核、旧 C 读完前禁止覆盖、配置前 Wait/End 与末尾排空；检查 FP64 golden、输入不变、guard 和 UB 字节数。真实 host 分派在两种 S2 配置下各校验 191 条 JSON，并检查新报告和旧报告兼容。

CPU 替身验证的是调用顺序、地址和数值逻辑，不能证明真实硬件并行、CANN 编译或微秒收益。未做本地 NPU 测试；后续 S16 线上 15/15 Pass，按对话关联 R20，平台源码哈希未核验。已有 R19 时只替换 **stream_plan.h、stream_matmul.asc**，其余九个提交文件保持原版本，不制作提交包。

S16 较 S15 的 8–12 点变化均在 ±1.62% 内，13/14/15 分别 +4.06%/+1.84%/+2.83%；没有出现明显新增响应点。当前证据支持这次提交覆盖路径的正确性，尚不能确认稳定性能收益或实际跨任务重叠覆盖。用户反馈分数微涨，但最优参考也发生变化，详见[迭代记录](ITERATION_LOG.md)；不将参考比值缩小当作代码加速。默认开关保留，两项旧实验继续关闭。


## R21：先折叠列组、延后横向 Max（2026-10-03）

S16 没有显示跨 owner 流水的明显耗时收益。本轮冻结 R20 的分派、SDK tiling、任务数、C 槽、归约 owner 与补偿 Sum，针对 C 进入 UB 之后重复进行的横向归约。比较过改变 Matmul 内块、重新启用连续 Iterate、重排现有 Max 三条路线；当前选第三条，使新增证据集中在 Vector 消费成本，避免再次混合会话、布局和精度修复。

### 数据流与代价

对有限 FP32 C 值，行最大值可以按列号模 64 分组：`max_j C[r,j] = max_l(max_q C[r,64q+l])`，其中 `0 <= l < 64`，只包含有效列。Max 没有加法舍入，完整 K 的 Matmul 和最终补偿 Sum 顺序保持不变。

- `FoldMaxColumns` 在 `kernel.asc` 中实现：将 UB 中每行的第 64/128/192 列组逐元素 Max 到前 64 个位置，每个 repeat 处理一行。一个 256 列块从四次 WholeReduceMax 加四次行 Max，变成三次逐元素 Max、一次 WholeReduceMax 和一次行 Max。
- GM 且 `N > 256` 时，每 32 行任务保留 `32×64` 个逐列最大值。每个 DMA 块先折叠，再更新这些位置；全部 N 块完成后才做一次 WholeReduceMax，直接得到完整行最大值。N=1024 时每个行任务的横向归约从 16 次降到 1 次，N=8192 时从 128 次降到 1 次。这些是调用次数，不是预计加速倍数。
- GM 且 `N <= 256` 只有一个 DMA 块，采用块内折叠，不分配额外累积缓冲；N<=64 时没有列组折叠。Stream/Pipeline 采用块内折叠，每个 C 块的每 32 行只做一次横向归约，随后仍沿用原跨块行 Max。其缓冲大小和 C 槽复用条件保持。

宽 N 的 GM 基础归约 UB 从 47,392 增到 55,456 字节，净增 8,064 字节；Partials 另加 512 字节，最大 55,968，仍在原有 64 KiB 预留内。Matmul 的 UB 预算不减少，Stream UB 不变。`GmReductionUbBase` 统一主机资源判断和 runner 报告；在原合法 plan/caps 下，更新后的检查不会改变候选准入或 Rows/Partials 选择。

代价是更多逐元素 Max 和 UB 读写；收益来自减少横向归约、紧凑行结果合并及相关 Vector 屏障。没有减少 Matmul 次数、C 的 GM 读写字节或全局屏障，所以若这些成本占主导，收益可能很小，也可能退化。原启发式 score 保持冻结，它不能预测新 Max 的微秒耗时。

### 尾部、别名与同步

UB 输入行距固定 256 个 float。折叠的 dst/src0 完全重叠，src1 为当前行内互不重叠的后续列组；不同 repeat 对应不同行。宽 GM 累积缓冲行距为 64，源行距为 256，二者不同分配。所有偏移 32B 对齐，最大 repeat 数为 32。规则依据 [CANN 9 地址重叠约束](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0004.html)；横向归约参数单位见 [WholeReduceMax](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0079.html)。未据此扩大硬件支持声明，线上型号仍未知。

最后不足 64 列的组仅更新有效 mask。例如宽度 65 只将第 64 列并入第 0 个位置，其余前 64 列保留；N=257 的最后一个 DMA 块只更新累积缓冲的第 0 个位置，其他位置保留前面块的最大值。未写的 UB/GM padding 不参与归约。每次依赖的 Vector 操作之间保留 PIPE_V，Stream 的 MTE2→Scalar 围栏和 R20 跨任务提交/Wait/End 次序保留。

### 验证与提交

回归先在旧实现上捕获 GM 和 Stream 重复横向归约；新实现检查每个真实 helper 的归约次数和数值结果。新增最后有效列取最大值的用例覆盖 63/64/65、127/128/129、191/192/193、255/256/257、319/320/321、511/512/513、8191/8192 列；包含 Rows/Partials、M 尾部、全负输入、输入与 guard 保护、补偿抵消及多核唯一写入。原有 1376 组 Stream 检查覆盖四转置、单双槽、分片和新旧跨任务流水，继续检查未完成的 Cube 输出不可读取及 C 复用围栏。CPU 替身同时拒绝部分别名与有依赖的跨 repeat 重叠，不模拟设备时序或真实编译。

runner 为 GM/Stream/Pipeline 记录 `max_schedule={"version":1,"mode":"deferred"|"folded"}`；Small/Iterate 不记录。旧报告缺少此字段时按旧 UB 合约读取，不能把旧报告解释为新算法。字段类型、取值、机制与 shape/family 的对应关系均严格校验。

已有 R20 时同步替换 **kernel.asc、stream_matmul.asc、reduction_plan.h、joint_plan.h、matmul_plan.h**；最后一个文件仅更新资源注释，其余四个包含实际改动。无需新建提交文件或打包。`PIPELINE_CHAIN_TASKS=true`，`S2_UPSTREAM_CONTROL=false`、`ITERATE_MATMUL_MAX=false`。`LOCAL_CANN_BUILD=NOT_RUN`、`LOCAL_NPU_TEST=NOT_RUN`、`ONLINE_EVALUATION=NOT_RUN`；下一次 S17 与 S16 逐点对照，不预设任何点的实际 family 或 N。

最终全套 **77/77 主机检查通过（80.555 秒）**；独立代码审查未发现 Critical/Important 问题，提出的跨 repeat 别名检查缺口已补上。实现提交 `c1d93f1`，尚无线上性能结果。
