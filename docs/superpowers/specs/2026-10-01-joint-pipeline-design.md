# R14：联合执行规划与 Cube/Vector 双槽流水

用户已要求继续深度迭代。本轮落实 ARCHITECTURE_AUDIT 的两个优先级 1 问题。

## 交付与不变量

- Auto 保留 Small 准入；其余输入按同一个结构成本模型联合比较 GM、同步 Stream、异步 Pipeline 的 tile、N 分片与任务负载。
- Pipeline 用两个独占 GM C 槽和一个 UB 输入队列，显式重叠 Cube(t+1) 与 Vector Max(t)。每次只有一个 Matmul 调用在途。
- 全 K 完成后做 Max；N 分片先合并逐行 Max 再求 Sum。保留 R12 补偿 Sum、Small 算法、单 launch 和 ABI。
- 本地强制 gm/stream 保留 R12/R13 对照；新增 pipeline。Auto+固定 tile 仍按旧 GM sweep 约定。元数据记录真实选择、缓冲和 SDK 内层 tile，旧报告可读。
- 只做主机检查；CANN 编译、设备同步语义、正确性与收益等待线上，不增加本地 NPU 前置条件。

## 联合规划

按现有规则生成最多 49 组外层 tile，每组评估 GM、Stream、Pipeline。N 分片候选含 1、2 的幂、完整 N 网格及填满核数附近值；即使行任务超过核数也搜索。每族每 geometry 保留最低分计划，最多 147 个。分数以 Cube 的 16x16x16 工作单元、归一化输入字节、Matmul 调用、Vector Max、部分 Max 合并、屏障和最终 Sum 的结构工作量相加；不是测量的微秒或周期。

同步路径累加两阶段；异步路径按每任务预热/排空和稳定阶段 max(Cube, Vector) 估计，单块 shard 不自动选择 Pipeline。成本模型未校准，排序只是可解释的第一版启发式。SDK 最多尝试 4 个不同 geometry，最后尝试 32x64 回退；拒绝前不分配、不 launch。实际执行必须使用被接受的完整计划，不能选择后再次改变任务分工。

## 双槽协议

预热提交槽 0，WaitIterateAll、End；提交下一块到另一槽后消费当前槽，消费以 MTE2_S fence 完成；等待下一块，重复。设置 shape/A/B 前上一调用必须 Wait/End。每槽重复使用前其全部 DMA 已读完；所有在途调用在任务边界和全核屏障前排空。C 槽按 core、buffer 独占。UB 不翻倍，保留 64 KiB 归约预算和独立 Matmul 预算。

接口依据：[CANN 9.0 异步 Matmul](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10015.html)、[WaitIterateAll](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0641.html)。使用 IterateAll<false>(gm, 0, false, true)。不声称 A 跨调用常驻 L1，也不声称消除 C 的 GM 流量。

## 验证

真实主机 planner 验证候选覆盖、精确 grid-stride 负载、双槽布局、溢出、有界 SDK 回退。提取真实设备 helper 到 CPU 边界检查器，异步模拟只在 Wait 后产出 C，禁止读取未完成结果、覆盖未消费槽或在途改 shape；检查确有消费发生在下一块 pending 时。覆盖四布局、尾块、任务多轮、不同分片、强抵消、Max-before-Sum 反例。保持主机单 launch/失败语义、报告验证和历史报告兼容。
