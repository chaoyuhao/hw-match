# R23：独立物理规划与多层 Cube 流水

状态：实现完成，待线上 CANN 编译、正确性和性能验证。S18/R22 的 15/15 Pass 已归档，不能替代本版本验证。本轮按用户要求直接重写性能主干，无本地 NPU 测试。

## 与 R22 的架构差别

R22 在旧候选排序、SDK tiling 和 GM/Stream 选择结束之后才决定是否替换 Cube 后端，要求 M/N/K 16 对齐；单份 L1/L0/C 缓冲在每层立即等待。R23 在 Small 后直接用 `MakePipelinePlan` 生成资源可行方案，成功时不调用 SDK tiler，也不注册 Matmul 服务端。只在新路径未准入或显式选择旧控制时进入原流程。

| 层次 | R23 的所有权与重叠 |
| --- | --- |
| 分块 | BM=16/32/64/128，BN=32/64/128 的规则候选；BK<=128，按 L1/L0 容量裁剪。BM/BN 编译期专门化 |
| A/L1 | 完整 A 行面板容纳得下时跨 N 驻留；换 batch/行块之前等 MTE1 读完。否则 A 的两个 K 槽随 B 流水 |
| B/L1 | 两个 K 面板槽，预取前两个，LoadData 完成后把槽归还 MTE2 |
| L0A/B | 两套，Mmad 消费后才允许 MTE1 覆写；下一 K 的装载可与当前 Mmad 重叠 |
| L0C | 两个 FP32 tile，Fixpipe 读完才归还 Cube；不再在每次 Store 后立即等待 |
| C/GM | 两个完整物理 tile；先 Compute 再等 FREE，之后 Fixpipe→READY；消费者全部搬完才返 FREE |
| Max/UB | 两个 32×256 FP32 条带缓冲，先预取下一条带再归约当前条带；owner 内持续更新行最大值 |
| Sum | 复用已验证的补偿记录/最终合并；N 分片先合并 Max，再生成 Sum 记录 |

N 分片只用于补充核并行度，实际核数来自接口。资源模型同时约束实际查询容量与 A2 上限，整份双缓冲计入预算；C 槽按 BM×BN 分配，包含尾行。候选分数比较填充工作量、最重核输入和 C 流量、A 驻留与 owner/panel 开销，用 `max` 表达引擎重叠；属于未校准的启发式，不是实测周期预测。它不读取隐藏测试编号、文件或输入数值。

## 尾块与数值

GM leading dimension 使用真实 M/N/K。ND→NZ 的 D 方向能补零到 16 元素，但 N 方向没有相同保证；任一物理尾面板先通过 `InitConstValue` 清零整个槽，MTE2 barrier 后拷入有效元素。L1 NZ 的 stride 使用补齐尺寸。经典 LoadData 生成 zZ/nZ，Mmad 使用完整 BM/BN 和当前补齐 K，保持 FP32 累加。BM 至少 16，避免 `m=1` 自动进入要求不同布局的 GEMV。

Fixpipe 始终向有界 GM 槽写完整 BM×BN FP32 tile；Vector 只拷有效行列，并以负有限极值屏蔽列尾，零填充不会成为负数行的 Max。中间 C 不转回 FP16/BF16，不启用 ReLU。最终求和继续原补偿算法；分块变化会改变归约分组，容差必须通过线上确认。

支持两 dtype、四转置及 M/N/K∈[1,8192]；比赛 K 为 8 倍数，CPU 额外覆盖非比赛 K。支持平台由 `QueryDirectCaps` 判断为 ASCEND910B，设备体针对 A2 架构 220；不假定子型号和 24 核。

## 同步契约

每层两个槽独立分配事件：MTE2→MTE1 为 L1 READY，MTE1→MTE2 为 L1 FREE；MTE1→M 和 M→MTE1 管理 L0A/B；M→FIX 和 FIX→M 管理 L0C。首次 FREE 预置；每次消费后成对归还；结尾等待所有 FREE 再 ReleaseEventID。使用 AllocEventID，避开 TPipe 自身预占事件。resident A 换 owner 的短 Fence 使用未分配事件号。

跨核 mode 2 READY=0/1、FREE=2/3，沿一个核的全部 owner 连续编号。AIC 排空最后 FREE；AIV Max 写出后参加 `SyncAll<true>`，N 分片需要额外合并屏障。只有一次 MIX(1,1) launch，不在失败后重发 kernel。

## 接入、回退和诊断

Small 原先的准入和算法不变。新主干默认适用于 Auto、无显式 tile、无 S2/Iterate 实验、非 R15 控制；显式旧 GM/Stream/Pipeline/tile 仍是原对照路径。`PIPELINE_CORE_ENABLED=false` 恢复 R22；`DIRECT_CUBE_ENABLED=false` 同时关闭 R22/R23 的基础 Cube 路径。新主干未通过资源准入时回退，但运行时错误不会再 launch 第二种算法。

本地 JSON 用 `cube_engine.version=2, mode=pipeline_cube, sdk_inner_tile_role=not_used`、Stream planner 5 和 `physical-pipeline-v1` 评分标记。`inner_tile` 在此记录实际物理 BM/BN/BK；没有 SDK 内块。所有片上大小包含两槽，UB 不再扣除 Matmul 服务端预算。严格验证器仍兼容历史报告。在线不新增日志、计时或读回。

已有 R22 时仅复制 `kernel.asc` 并新增 `pipeline_plan.h`、`pipeline_cube.asc` 到同目录；旧依赖保留。无需修改 main/judge/CMake，也无需准备提交包。

## 验证范围

- 真实物理规划器：规则生成的 576 组维度/核数组合，容量限制、尾部、驻留和拒绝分支。
- 真实 Cube helper：两 dtype × 四转置 × 六组几何（48 组），覆盖 K=1/8/24/137/520、M/N 尾部、A 跨 N 驻留、跨 batch/owner 切换、非驻留面板、完整 FP32 输出与边界哨兵。
- 事件指令按四个独立引擎 FIFO 调度，检查 READY/FREE 不覆写与全部排空；数值模型自身仍为同步 CPU 运算，非 NPU 时序模拟。
- 真实 C/V 循环与双 UB 队列：72 组分片/核数/几何/归约组合，线程交接检查最后条带读完才覆写、部分环及最终补偿合并。
- 真实 host/runner：强制 SDK 不可用时仍选中新主干并恰好一次 launch；四布局两 dtype 和 Rows/Partials，错误不重发，报告资源字段严格校验。

独立审查未发现 Critical/Important 项。验证数字与最终命令结果记录在迭代日志。CPU 检查不能证明 CANN 编译、真实缓存/事件时序、硬件 FP32 累加容差或性能。首轮应先看 15/15 正确性，再与 S18 自身耗时对照。未实现两 AIV 分行、片上 C/V 直传、跨多个 C 的 K-panel 复用；这轮不能称为完整复刻 TileLang。

## 固定来源

- [TileLang 调研与固定提交](TILELANG_REVIEW.md)：分层缓冲和 owner 生命周期依据。
- [CANN v9 ND2NZ 文档](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/docs/api/context/%E9%9A%8F%E8%B7%AF%E8%BD%AC%E6%8D%A2ND2NZ%E6%90%AC%E8%BF%90.md)：ND 真实维度与 NZ 物理 stride 分开。
- [v9 A2 数据搬运实现](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_c220/kernel_operator_data_copy_impl.h)。
- [v9 参数结构](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/include/basic_api/kernel_struct_mm.h)、[接口声明](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/include/basic_api/kernel_operator_mm_intf.h)、[A2 MM 实现](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/dav_c220/kernel_operator_mm_impl.h)。
- [v9 Fill/InitConstValue 约束](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/docs/api/context/Fill.md)：A1/B1 清零，32 B block，两 dtype 支持。
- [v9 TPipe](https://gitcode.com/cann/asc-devkit/blob/v9.0.0/impl/basic_api/kernel_tpipe_impl.h)：事件池分配及释放；原生混合启动依据继续见 [R22](DIRECT_CUBE.md)。
