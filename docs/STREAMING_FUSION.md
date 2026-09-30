# R13：按行块逐块融合 Matmul → Max

R13 的通用默认路径不再保存完整 similarity：每个活跃核组拥有一个 C 临时槽，一个任务拥有 `(batch, M 行块, N 分片)`，遍历分片内所有 N 块。每块沿完整 K 做 Matmul，立即在 Vector 上更新该任务的行最大值，再复用 C 槽。最终 Sum 沿用 R12 的补偿树，Small 的算法和自动门槛不变。

```text
各核组：完整 K Matmul → 每核 C 槽 → 更新行 Max → 下一个 N 块
                                   ↓
                          写出分片的行最大值
                                   ↓
         S=1：全核同步 ─────────→ R12 Sum
         S>1：全核同步 → 跨分片逐行 Max → 全核同步 → R12 Sum
```

这是同步 Matmul 加有界 GM 中转。仍有 O(BMN) 个 C 元素的逻辑写入和读取，尚未实现异步流水、L1 输入复用或 C 直达 UB。可能的收益来自更小的工作集、较短的生产消费间隔，以及 S=1 时少一次全核阶段屏障；逐块 Vector 消费也可能增加 Cube 等待。没有新线上结果前不宣称提速。

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

`LOCAL_CANN_BUILD=NOT_RUN`、`LOCAL_NPU_TEST=NOT_RUN`、`ONLINE_EVALUATION=NOT_RUN`。按既有决定，不安排额外本地 NPU 验证，下一次以 S8 为对照收集全 15 点。

若线上已有 R12，替换 **kernel.asc、small_plan.h**，新增 **stream_plan.h、stream_matmul.asc**。同时保留同目录已有的 **matmul_plan.h、small_vector.h**，共六个源码文件。其余平台原始文件不变，不需要提交包。

官方依据与后续异步方向见[已批准设计](superpowers/specs/2026-09-30-next-wave-optimization-design.md)。
