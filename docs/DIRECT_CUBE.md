# R22：独立 Cube 生产与 Vector 归约

版本状态：实现待线上验证，最新已知成绩仍为 R21 / S17。没有运行本地 CANN 编译或 NPU 测试。用户要求直接通过线上迭代观察收益，本轮不增加本地 NPU 前置流程。

## 改动解决什么

以前 AIV 为每个输出块调用 SetOrgShape / SetSingleShape / SetTensor / IterateAll，再等待 Matmul 服务端。R20 的两槽流水和 R21 的 Max 重排仍保留这些会话。R22 在符合资源条件的输入上，让 AIC 自己遍历任务，用基础 Cube API 生产 C；AIV 自己遍历相同 owner，等待 C 就绪后继续现有 Max 和 Sum。

数据流是 `GM A/B → L1 → L0A/B → FP32 Mmad → L0C → FP32 GM → UB Max → 补偿 Sum`。第一版保留当前外层 tile、N 分片、核数选择、Rows/Partials、GM 或 Stream 家族；不重新调启发式权重。SDK tiler 仍参与原方案准备，但 direct 执行不使用它的内部 baseM/baseN/baseK，也不注册 Matmul server。

潜在收益有三项：减少逐块高层会话；让 AIC 生产与 AIV 消费独立推进；A 面板在容量允许时跨连续 N 块驻留。第三项减少程序要求的 A 装载次数，不等于同倍数降低 HBM 流量或总耗时，缓存和 B/C 成本仍在。短任务可能从会话减少受益，长任务可能从输入复用与 C/V 并行受益；尚无本轮线上数据证明。

## 准入与资源

`direct_plan.h` 用实际 ProblemDesc、已选 tile 和平台查询结果选择，条件如下：

- 平台报告 `SocVersion::ASCEND910B`，设备实现针对编译架构 `__CCE_AICORE__ == 220`；不硬编码子型号或 24 核。
- FP16/BF16、M/N/K 均为 16 的倍数。外层最后一块允许比 tile 小，只要仍 16 对齐。K=8 的倍数但非 16 对齐，以及其他尾部，继续原路径。
- 查询 L1/L0A/L0B/L0C，另以 A2 上限约束。令 `BM=min(M,tileM)`、`BN=min(N,tileN)`，单份 FP32 C 为 `4*BM*BN`，不能超过 128 KiB。超限时回退，首版不拆分已选外层 C tile。
- BK 从不超过 128 的合法值向下裁剪，L0A/B 分别需要 `2*BM*BK`、`2*BN*BK`，各不超过查询容量及 64 KiB；A/B 的最小 L1 面板也必须容纳。
- 完整 A 所需 `2*BM*K` 加 B 面板能放入 L1 时启用 A 驻留，否则按 K 面板搬入 A。AIC 缓存当前 batch/行块，切换 owner 时重新加载。

Small 先于这些条件返回，行为保持。实验 Iterate 仍默认关闭，若显式启用且 SDK 接受则不叠加 direct。`DIRECT_CUBE_ENABLED=false` 可恢复原 Cube 执行层；不改变 Max/Sum，也不在运行失败后偷偷重发第二个 kernel。

## 存储与同步

四种转置使用经典 `LoadData2DParams`，C220 不使用 V2。GM 的真实 leading dimension 与 L1 的物理布局分别计算；A 驻留时 L1 的 K 跨度为完整 K，尾 K 搬入 L0 的宽度仍是当前片宽。每个输出块第一次 Mmad 初始化 C，后续 K 片累加，全部完成后才 Fixpipe。FP32 ND 输出的 `srcStride=有效M`、`dstStride=目标行跨度`，不转为 FP16，也不启用 ReLU。

局部每个层级有一份缓冲，显式设置 MTE2→MTE1、MTE1→M/MTE2、M→MTE1/FIX、FIX→M 的依赖。事件通过 `TPipe::FetchEventID` 获取，避免与 TPipe 预占的 M_MTE1 0/1/2 冲突。TPipe 自身执行 InitSocState。

Stream 保留 1 或 2 个 GM C 槽，编号贯穿一个核的全部任务。AIC 首次使用槽无需等 FREE；后续等待该槽上次的 FREE，写完用 FIX 发 READY。AIV 等 READY，运行原 `ConsumeStreamTile`，在其最后一次 MTE2 读取 fence 后发 FREE。AIC 返回前消耗所有已使用槽最后的 FREE，包含只使用了部分环的情况。READY 使用 0/1，FREE 使用 2/3，避开 SDK 全局同步编号。消费超过 32 行时，释放必须覆盖全部条带。

GM 路径继续写完整 C；每个 AIC 全部写完通知对应 AIV，所有 AIV 再进入 `SyncAll<true>()`，随后执行原全矩阵 Max 和 Sum。该屏障只要求全部已启动 AIV 参加，AIC 不进入。整个调用依然只有一次 MIX(1,1) launch。

原生 `.asc` 混合核的直接 CrossCoreSetFlag 用法与官方基础 API 融合样例一致；官方 msobjdump 文档的融合产物显示 `RUNTIME_IMPLICIT_INFO: FFTS_ADDR`、`CROSS_CORE_SYNC: USE_SYNC`，支持由融合编译/运行时传递隐式同步参数的用法。不把 TileLang 经其他编译入口传递的 `fftsAddr` 参数机械复制到 `<<<>>>` 启动。本机没有 CANN 编译器，尚未检查本版本实际产物的元数据或指令，实际 launch 初始化仍由线上确认。

## 这版尚未实现的层次

L1 的 B 双缓冲、L0A/B 双缓冲、双 L0C 的 Cube/FIX 重叠、两 AIV 分担 M 行、分组通知、资源不适用时的内部 C 再分块、K-panel 跨多个未完成 C 的复用，仍是后续机会。首版局部依赖偏保守；去掉 SDK 会话也可能失去其部分内部流水，长 K 可能退化。不能把本轮称为完整复刻 TileLang 调度。

## 检查和报告

`tests/test_direct_cube.py` 执行真实资源规划和真实 Cube helper，由 CPU 替身按照独立的 ND/NZ/zZ/nZ 布局解释搬运，检查四转置、half/BF16、K 分片、驻留切换、batch/行尾和 FP32 pitched 输出。另一组双线程运行真实 C/V owner 循环，检查单双槽、跨任务、分片、环排空、多条带读取以及原 Rows/Partials 归约。它们不是异步硬件模拟器。

Host 测试执行真实 RunKernel，验证支持/不支持与非对齐回退、开关、四转置两 dtype 的一次 launch、原 scratch 分配和错误不重发。runner 新增 `cube_engine`：direct 时原 `inner_tile` 明确标为 `reference_only`，记录实际 BK、驻留和各层字节数；Matmul API 路径标为 `execution`。报告验证器兼容没有该字段的旧报告，并拒绝矛盾的资源记录。原几何评分继续只解释上游选择，不作为 direct 的耗时预测。

**最终 CPU 回归：80/80 通过，111.440 秒。**

`LOCAL_CANN_BUILD=NOT_RUN`、`LOCAL_NPU_TEST=NOT_RUN`、`ONLINE_EVALUATION=NOT_RUN`。CPU 通过不能证明编译支持、真实硬件 flag/缓存时序、数值累加的线上容差或性能。下一步仍由用户复制代码提交，记录 15 点自身用时与正确率。

## 原始依据

- [TileLang 调研和固定版本](TILELANG_REVIEW.md)：设计来源，未引入 TileLang/CATLASS 依赖。
- [官方 native .asc 基础 Cube/Vector 融合样例](https://gitcode.com/cann/cann-samples/blob/master/Samples/0_Introduction/01_simd_cpp_api/03_fusion_operation/matmul_leakyrelu_basic_api/matmul_leakyrelu_basic_api.asc)：直接混合核、基础 API、跨核通知。
- [官方 msobjdump 示例](https://gitcode.com/cann/asc-tools/blob/master/examples/04_msobjdump/README.md)：融合编译产物的隐式 FFTS 参数和硬件同步元数据。
- [SDK 9 C220 LoadData 实现](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/impl/basic_api/dav_c220/kernel_operator_mm_impl.h)：经典 2D 与 V2 支持边界。
- [SDK 9 TPipe](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/impl/basic_api/kernel_tpipe_impl.h)：预占事件、片上分配、初始化与析构。
- [SDK 9 C220 同步](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/impl/basic_api/dav_c220/kernel_operator_sync_impl.h)：mode 2、设备等待、AIV-only 全局屏障。
- [Fixpipe 参数](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/include/basic_api/kernel_struct_fixpipe.h)、[LoadData/Mmad 参数](https://raw.gitcode.com/cann/asc-devkit/raw/9.0.0/include/basic_api/kernel_struct_mm.h)。
- [CATLASS 布局](https://gitee.com/ascend/catlass/blob/master/include/catlass/layout/matrix.hpp)、[L0A](https://gitee.com/ascend/catlass/blob/master/include/catlass/gemm/tile/copy_l1_to_l0a.hpp)、[L0B](https://gitee.com/ascend/catlass/blob/master/include/catlass/gemm/tile/copy_l1_to_l0b.hpp)：核对四转置映射，未复制依赖库。
