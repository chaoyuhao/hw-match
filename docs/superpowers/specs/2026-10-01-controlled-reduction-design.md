# R16：冻结 R15 上游计划的受控归约扩围

用户已批准：只做一次覆盖对照，固定算法族/tile/N 分片，仅扩大 Partials；继续单 kernel、不运行本地 NPU、不按官方点号分派，手动复制源码。基线为 R15 `5e27c5a` / S11，当前 main `95d3aad`。

保留 R15 Small 准入和全部联合成本/排序/SDK 接受流程。**SDK 已接受一个完整 R15 计划后**，仅对默认 Auto family、Auto tile、Auto sum 进行覆盖策略。原 Partials 不变。Rows 只有同时满足以下条件才转 Partials：至少两个 M 段；`16*segments < M`（64B 补偿记录的 float 数严格小于有效行 Max 数，不能用行 padding 制造压缩）；累计归约 UB <=64KiB 且加 SDK Matmul UB 预算不超过硬件 UB。GM/S>1 段长32，S1 段长 tileM。没有新增 batch 阈值或未经验证的收益门槛。

仅重建 reduction 与依赖它的 scratch 偏移/大小/UB。family、tile、任务数、核数、split、buffer 数、C 槽、SDK inner tiling 与接受尝试顺序都不变。设备 helper、算术、屏障和 launch 不变。额外成本可能导致退化：本轮测适用边界，不承诺新增覆盖更快。

在 joint_plan.h 增加纯主机 `ExpandPartialReduction(p,h,selected)`。PrepareExecution 在 SDK 接受后调用，RunKernel 新增默认 true 的主机 bool 开关，不改变公开 run_kernel ABI。现有 forced family/tile/sum 控制不扩围。local runner 增加 `CANN_SUM_MODE=r15`：映射原 Auto sum 并关闭扩围，Small 仍可正常选中；其余 auto/rows/partials 语义兼容，auto 默认启用 R16。

报告保留 schema3 / joint planner3 / joint-work-v2（原模型未改），增加适用 Auto 控制的 `reduction_expansion`：version1、enabled、applied、baseline_mode、baseline_score。selection.score 为最终实际计划的结构成本，baseline_score 为冻结的 R15 选择成本；强制扩围后前者可更高。未适用控制不输出该对象，历史报告不要求它；r15 请求必须被报告身份确认。无线上 I/O 或环境读取。

检查：规则网格验证全体上游字段不变；多次 SDK 拒绝/回退仍选择相同计划再扩围；极短段、16行段无压缩、尾段、S>1、预算边界、原 Partials 保留；真实 RunKernel 默认启用、r15 关闭、一次 launch、正确 scratch/metadata。已有72项主机回归并入新检查；记录 CPU 覆盖变化，模型首选候选不是实际 SDK/线上命中统计。新结果记 S12，对照 S11。
