# 本地 BatchMatmulMaxSum baseline

当前源码为 R15 补偿部分和候选，72 项主机检查通过，未在本机做 CANN/NPU 测试，等待 S11 线上验证。最新已确认的反馈仍是 R14/S10 的 15/15 Pass，不能沿用到 R15；保留 R12/S8、R13/S9 与 R14/S10 的收益/退化对照，平台源码哈希未核验。最新机制见 [流式融合与异步流水](STREAMING_FUSION.md)，完整历史见 [迭代记录](ITERATION_LOG.md)。按开发决策，下面 NPU 命令仅作为可选工具，不是当前提交前置步骤。

这是用于本地正确性调试的候选实现，最终以 **CANN 9.0.0 线上平台**评测为准。旧版本 `377f283685ff77c425690e41981e723450398c80` 已在用户的 910B2C / CANN 9.1.0 上通过 smoke 19/19 和 full 55/55；对应 `kernel.asc` SHA256 为 `06cff43ba438d4ecb4003444c459d9712c4777a1f2cc3c1ced3cebaf3c573c1e`。

线上空白模板返回 `Profiling rule violated: each iteration must launch exactly 1 kernel. Expected 75 launches, got 0.`，说明每次迭代必须恰好启动一次；75 是平台累计预期次数，不是在一次调用中启动 75 次。旧版有两次启动，不能满足这项限制。当前版本合为一个 MIX kernel，并在 `2e079f8` 清理调试代码。用户反馈清理后通过线上全部 15 项，见 [线上耗时记录](ONLINE_BASELINE.md)。55 项本地记录仍属于旧版本，不能沿用为新版回归结果。

## 线上修改范围

用户确认：只能修改原有 `kernel.asc`，或新增 `.asc` / `.h` 文件；原有其他文件不能修改。当前候选共九个源码文件：`kernel.asc`、`matmul_plan.h`、`small_plan.h`、`small_vector.h`、`stream_plan.h`、`stream_matmul.asc`、`joint_plan.h`、`reduction_plan.h`、`partial_sum.asc`；复制到线上同目录即可，不需要提交包。保留线上原有 `main.asc`、`CMakeLists.txt`、`run.sh` 和 Python 脚本。本仓库的 `local/`、`scripts/run_local.sh` 等仅供本地调试；已有本地环境适配也不复制到线上。

原模板还明确要求：`kernel.asc` 被外部直接 include，不添加 `main()`、`#pragma once` 或 include guard，不重复定义已有的 TensorInfo/TensorGroupInfo。注释里的 `__cube__` 是示例，没有写明禁止 MIX；workspace、host 检查及调试 API 的许可不能从这段注释推断。

### 提交阶段的拒绝反馈

用户只替换 `0624c72` 的 `kernel.asc` 后，收到“提交代码中存在不合规内容，请检查并删除后提交。如需Debug请在本地进行。”，没有具体函数或行号。因此只能提出待验证的假设：原提交文件中的显式打印、退出或诊断数据回读被检查到。尚无证据确认具体命中项。

`2e079f8` 清理时删除了 `kernel.asc` 中的 `fprintf`、`abort`、诊断 `aclrtMemcpy` / `aclrtMemset` 及相关头文件。所有打印、诊断回读和文件输出均在 `local/runner.asc` 中。内部执行函数返回已完成计算的设备 scratch 所有权：正式入口直接释放；本地 runner 可在释放前回读。提交文件不包含宏隐藏的调试分支，也不依赖本地 runner。当时的设备计算和同步代码与 `0624c72` 逐字节一致；清理后用户反馈提交成功，具体触发拒绝的内容仍未确认。后续算法改动应独立回归，不能沿用提交清理时的结果。

ACL、参数或 tiling 错误仍会抛出异常，不被吞掉。若 stream 持续同步失败，scratch 不提前释放，由设备 context 清理；本地 runner 在输入/输出缓冲析构前检查 stream，必要时终止测试进程。已用 mock ACL 的 C++14 检查验证返回值移动、正常释放、异常恢复及同步持续失败四种所有权路径；这不验证 NPU 或平台规则。

## 运行

在已加载 CANN 环境的仓库目录执行：

```bash
git pull --ff-only
bash scripts/run_local.sh
```

默认运行 19 个 smoke case，每个执行两次。脚本会尝试加载 `ASCEND_HOME_PATH` 下或上一级的 `set_env.sh`，然后使用 `local/CMakeLists.txt` 独立构建，不覆盖原工程的 `build/`。如果没有加载过环境，先 source 本机实际安装的脚本，例如：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
```

更多选项：

```bash
# 55 个扩展 case：长 K、大 M/N、更多 batch 和跨多核任务循环
bash scripts/run_local.sh --suite full

# 仅跑一个 case；设备号仍为 ACL 逻辑编号
bash scripts/run_local.sh --suite full --case long_k_bf16_11 --device 0 --repeat 3

# 无 CANN 的机器可只生成数据，状态为 GENERATED，不会报告 NPU PASS
bash scripts/run_local.sh --generate-only --suite full

# 仅比较最终输出，省去中间矩阵回读与比对
bash scripts/run_local.sh --no-dump-similarity
```

脚本打印 `build-baseline/run-时间-随机后缀/run.log` 路径。配置、编译及逐 case 结果都在该日志；详细结果位于 `cases/report.json`，每个 case 目录有 `runtime.log`、输入、golden、设备输出与 `result.json`。所有生成物被 Git 忽略。

失败时把 `run.log` 和对应 case 的 `runtime.log` 贴回来。CMake 配置限时 120 秒，编译 600 秒，每个 case 进程默认 120 秒；任何失败最终返回非零状态。失败用例保留，后续其他 case 仍会运行。

若已有成功构建，可以直接使用 Python 入口，避免重复编译；输出目录必须是新目录，以防旧输出掩盖失败：

```bash
python3 scripts/local_baseline.py \
  --binary /实际路径/build/baseline_runner \
  --output-dir /tmp/cann-baseline-retest \
  --suite full --case tail_fp16_11
```

## R12 GM 对照路径

以下四步描述强制 GM 路径；当前 Auto 未命中 Small 时走 [R13 流式路径](STREAMING_FUSION.md)，不保存完整 similarity。

`kernel.asc` 保留原始 `run_kernel` 签名，由原 `main.asc` 和本地 runner 共用。实际计算都在 NPU 执行，host 只校验元数据、计算 tiling、分配空间和调度：

1. 一个 `__mix__(1,1)` kernel 内，由 AIV 调用 Matmul 高阶 API，通过通信框架在 AIC 执行矩阵乘。FP16/BF16 输入、FP32 输出；每个任务负责一个 batch 内的输出块，外层分块由规则生成并按实际元数据选择，详见 [规划接口](MATMUL_TILING.md)，沿完整 K 计算。四种转置存储布局直接通过地址偏移和 API 的 transpose 参数解释。
2. 每个 AIV 等待自己负责的 Matmul 完成后，全体 AIV 执行 `SyncAll<true>()`，使归约能够读取其他核产生的中间结果。启动块数不超过入口传入的可用 Cube 核数，且设置 `__schedmode__(1)`，满足同步的调度要求。所有 AIV 都参加同步，包括不负责最终输出的核。
3. 同一次启动内执行 Vector 归约：每个任务处理 32 行、每次读最多 256 列；WholeReduceMax 的 repeat 同时处理多行，再用向量 Max 合并列段。所有行最大值写入独占且对齐的 GM 区域后，再次进行 AIV 核间同步。
4. 最终输出 owner 每次读取最多 1024 个行最大值，使用 R12 向量补偿树缩减，再在 Scalar 上补偿合并。没有浮点原子加，也没有第二次 kernel 启动。

系统 workspace 用 `__kfc_workspace__` 参数传递；移除旧版的纯 Cube 编译宏和 `SetSysWorkspace`。Matmul 的 UB 预算设为 128 KiB，为归约缓冲和通信留出空间。Matmul 与归约共用 TPipe，局部同步事件通过 `FetchEventID` 获取。

中间矩阵行跨度为 `round_up(N,16)`，空间为 `4*B*M*round_up(N,16)` 字节；同一 allocation 尾部增加 `4*B*round_up(M,32)` 字节行最大值，另有 Matmul 系统 workspace。本地诊断继续读取 allocation 开头的中间矩阵，并在 host 保留 FP64 golden。M/N/K 支持范围为 `[1,8192]`；实际可运行规模还受内存限制。README 示例中的 K=2 也可作为诊断输入，生成的常规用例遵守 K 为 8 的倍数。

输出按每 8 个 batch 一组划分任务，避免不同核同时写入同一个 32 字节区域；最后不足 32 字节使用 `DataCopyPad` 精确写回。每次调用分配临时 GM，stream 同步后释放。中间矩阵不预填 NaN，所有有效元素由本次计算写入，padding 不参与归约；本地 runner 仍在每次执行前把最终输出填为 NaN。当前候选仍包含分配、同步及全量中间矩阵读写的开销。

## 验证范围

- golden 使用量化后的实际输入值进行 FP64 Matmul，再做 Max/Sum，最终转为 FP32。
- 数据工具仅依赖 NumPy。BF16 文件按 FP32 → BF16 最近偶数舍入编码，无需在数据生成机器安装 `ml_dtypes`；远端现有包仍可用于交叉核验。
- 检查输出字节数严格等于 `4*B`，拒绝 NaN/Inf，所有元素满足 `abs(actual-golden) <= 1e-4 + 1e-4*abs(golden)`。
- GM 路径可额外比较设备中间矩阵与 FP64 Matmul，以及设备最终输出与从设备中间矩阵计算的归约结果，方便区分 Matmul 与归约问题。
- runner 每次把输出填为 NaN，检查输出前后 guard、输入内容未改变、两次执行输出逐位一致。
- `smoke` 覆盖两 dtype × 四布局、全负值、非对齐 M/N/K、batch、已知值、原模板 shape。`full` 增加 K=8192、M/N=8192、N=1、零输入、抵消和多任务复用；B=257 用例在 24 核机器上覆盖归约输出的完整分组、尾组及同核分组复用。
- 所有报告包含 `ONLINE_EVALUATION=NOT_RUN`。报告里的 wall time 包含进程启动、文件 I/O 和诊断，不能视为 kernel 耗时或官方成绩。

本开发工作区执行：

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/run_local.sh
bash scripts/run_local.sh --generate-only --suite full
```

这些检查验证数据工具和失败处理，不替代真实 CANN 编译、内存检查器或 NPU 测试。后续优化仍需要完整 55 项回归及线上确认，重点检查 MIX 通信 workspace、跨核同步及重复输出一致性。新增本地计时和压力测试入口见 [性能分析](LOCAL_PERFORMANCE.md)，计时不修改 kernel。

## 使用的官方接口

- [CANN 9.0 MIX Matmul 融合实现](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0050.html)
- [SyncAll：同步范围、核数与调度约束](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0204.html)
- [FetchEventID](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0116.html)
- [CANN 9.0 Matmul 实现流程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0038.html)
- [GetTiling：标准 C++ TCubeTiling](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0692.html)
- [SetOrgShape：输入跨度与输出跨度](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0651.html)
- [系统 workspace](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0171.html)
- [PlatformAscendCManager](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1039.html)


## 当前路径与本地对照（可选诊断，不是提交前置条件）

默认先按原规则分派 Small，其余联合选择 GM/Stream/Pipeline、tile、N 分片和 Rows/Partials。Stream/Pipeline/Small 不产生完整 similarity，报告相应标记不可回读。`CANN_EXECUTION_FAMILY=gm|small|stream|pipeline|auto` 仅在本地 runner 使用，线上入口不读取环境变量。强制 small 遇到不支持的尺寸/布局会明确报错；与固定 `CANN_MATMUL_TILE` 冲突也报错。固定 tile + auto 走 GM；tile sweep 自动强制 GM。

`CANN_SUM_MODE=auto|rows|partials` 仅用于本地求和对照。显式 rows/partials 跳过 Small，与强制 small 冲突时报错。sum=auto 时强制 gm/stream 保留旧 Rows 行为；强制 pipeline 比较两种归约。固定 tile + auto family 默认 Rows，但可显式选 partials；tile sweep 继承 sum 设置。报告 schema 3 扩展 `requested_sum`、`reduction`（段长/段数、字节数和 UB），联合模型为 v3 / `joint-work-v2`，历史无新字段的报告继续按旧 Rows 校验。源码快照包含九个提交文件。

`execution_plan.json` schema 3 记录实际算法族、变体、任务/核数、输入身份和资源；GM 继续输出旧 `matmul_plan.json`。small 的 `similarity_available=false` 表示核内直接产出 y，没有中间矩阵回读；最终输出精度、重复一致性、输入不变和 guard 检查仍保留。

`--suite small` 用规则生成各维边界并交叉两 dtype/四布局，包含命中和回退，不代表线上点的真实 shape。CPU 替身只能检查地址和逻辑，不能证明 CANN 指令/流水/舍入行为。本轮按用户要求直接等待线上结果，不安排额外 NPU 采样。
