# R22：直接 Cube 执行层

依据：`docs/TILELANG_REVIEW.md`；用户已授权继续实现，维持仅线上验证性能的流程。

现有 Stream/GM 每个 C tile 重新配置 Matmul 客户端并等待服务端。本轮保留外层 tile、任务分配、FP32 C、Max、补偿 Sum，用独立 AIC 循环直接执行 ND→NZ、LoadData、Mmad、Fixpipe。AIV 消费不再提交 Matmul 请求。

适用条件由真实元数据决定：平台报告 ASCEND910B，M/N/K 均为 16 倍数，L1/L0A/L0B/L0C 查询容量足够。资源不足、不支持的架构、非对齐输入继续现有实现；Small 和实验 Iterate 不改变。首版不改变候选排名，不假定线上核心数。

A 面板能与 B 的 K 面板共同放入 L1 时，跨当前 owner 的 N 块驻留；否则分 K 加载。BK 不超过 128，并按独立 L0A/B 容量裁剪。L0C 容纳一个完整输出 tile；必须完成全部 K 才能发布 C。四种 TA/TB 使用经典 LoadData2DParams，C 保持 FP32。初版局部搬运使用明确事件依赖，先实现 C/V 并行，后续再增加局部双缓冲。

Stream 使用原有 1/2 个 GM 槽：每槽独立 READY/FREE，FIX 发 READY，AIV 全部 MTE2 读取完成后发 FREE；首次写不用等待 FREE，最后排空已用槽。GM 路径仍生成完整矩阵，每个 AIC 完成全部写后通知对应 AIV，再执行原全局归约。一个 MIX(1,1) launch；无在线计时、读回或调参。

CPU 测试验证资源边界、四布局物理搬运、完整 K 累加、FP32 输出、owner/槽代次和 host 一次 launch。CPU 不证明 CANN 编译、真实异步时序或性能；本机不运行 NPU。首个线上成绩前，版本状态必须为待验证。

回退开关为编译期 `DIRECT_CUBE_ENABLED`，线上仅复制原有提交文件和新增 `direct_plan.h`、`direct_cube.asc`。文档记录实际覆盖条件、未实现层次和回退方式。
