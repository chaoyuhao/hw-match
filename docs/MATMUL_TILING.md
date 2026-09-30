# 规则生成 Matmul 计划：第一阶段

2026-09-30：替换四种固定候选的选择器。当前候选尚需 CANN 9.1 本地编译、NPU 精度回归及 CANN 9.0 线上评测；主机测试不能证明提速。旧四候选设计与实验记录可在 `a220315` 的此文件中查看。

## 规划与执行边界

`matmul_plan.h` 是不依赖 CANN 的 C++14 规划器，`kernel.asc` 负责调用 SDK tiler 和设备执行。本轮保留完整 similarity、并行行最大值、串行补偿求和及两次 AIV 屏障。正式入口 ABI 不变，每次成功调用恰好启动一个 MIX kernel。

流水为：

```text
ProblemDesc(B/M/N/K/dtype/TA/TB) + HardwareCaps(可用核数/UB)
  → 规则生成并排序外层任务块
  → CANN GetTiling 检查内部配置
  → PreparedMatmul(外层计划 + SDK tiling)
  → 原有单次启动设备路径
```

候选轴包含 16 起的倍增尺寸、输入尺寸向上对齐值、按每 batch 可用核数推导的尺寸。去重后每轴最多 7 个值，交叉组合最多 49 个；容器上限 64。当前外层块每边为 16 的倍数，范围 16..256。256 是当前搜索工程边界，不是硬件上限，也不要求用户手写输入 shape。小维度由真实尾块裁剪。

选择评分考虑任务轮数、代表性块计算量（包含 K）、布局相关的 A/B 访存估计和每次 Matmul 调用的固定成本代理。系数是未校准的启发式，不是延迟预测，也没有保证不退化。FP16/BF16 占用相同输入字节数，dtype 影响 SDK 类型合法性，不人为添加两套性能权重。正式调用不做设备试跑或按测试点编号选路。

自动模式最多尝试排序前 4 个候选，再尝试 `32x64` 参照；SDK 全部拒绝则明确失败。本地固定 `MxN` 请求只尝试该配置，不静默换成别的块。`GetTiling` 接受只代表规划通过，仍需验证设备运行。

UB 从平台接口查询；Matmul 预算为 `min(128 KiB, UB - 64 KiB)`。已有显式归约缓冲占 37152 字节，64 KiB 是含余量的保守保留值，并非精确通信占用模型。内部 baseM/baseN/baseK 由 CANN 决定，本地报告实际返回值。后续双缓冲、融合路径须更新资源核算，不能直接沿用这一预算。

## 日常迭代

在 NPU 机器运行：

```bash
git pull --ff-only
source /usr/local/Ascend/cann-9.1.0/set_env.sh
CANN_MATMUL_TILE=auto bash scripts/run_local.sh --suite full &&
CANN_MATMUL_TILE=auto bash scripts/run_local.sh --suite generated --case-count 64 --case-seed 20260930 &&
CANN_MATMUL_TILE=auto bash scripts/run_local.sh --suite reduction
```

`generated` 默认组合 8 类规则与两 dtype × 四布局，覆盖小输入、对齐、尾块、长条、宽矩阵、核数边界、长 K 和 batch 边界。K 为 8 的倍数；每例 `B*M*N*K <= 32*1024*1024`，保守临时内存估计不超过 128 MiB，拒绝采样最多 128 次。数量和 seed 可调整；`--case-cores` 只是生成边界的提示，不改变执行时真实核数。这些限制控制本地回归成本，不宣称还原线上隐藏数据分布。

用例保存生成规则版本、seed、索引、族别和完整输入身份的 hash。报告比较按 B/M/N/K/dtype/TA/TB/pattern/seed 匹配，名称只用于显示和目录。已有 full/reduction/tiling 等固定回归锚点继续保留。

本地可指定任意合法几何，如 `CANN_MATMUL_TILE=80x144`；SDK 不接受则失败。环境变量解析和全部调试输出都在 runner，线上代码不读取环境变量。

本轮线上需要复制 **`kernel.asc` 和新增的 `matmul_plan.h`**，置于同目录；其余原始文件不改。不需要提交包。本地 runner、脚本和文档不复制。

## 可选的动态候选采样

需要解释瓶颈或退化时再使用；不要求每轮先跑全量矩阵：

```bash
bash scripts/run_perf.sh --suite quick --tile-sweep --sweep-candidates 6
# 同样支持规则生成用例，以及可选 --profile timeline
bash scripts/run_perf.sh --suite generated --case-count 16 --tile-sweep --sweep-candidates 4
```

runner 的 `--list-plans CASE_DIR LOGICAL_DEVICE OUTPUT_JSON` 只做元数据规划和 SDK 检查，不启动计算 kernel。它返回按 C++ 评分排序的候选及拒绝状态。Python 取前若干个接受的候选，再补已接受的 `32x64` 和 `auto`；默认每例最多 8 个策略、2 轮，实际数量随输入与 SDK 变化。所有运行串行且轮换顺序，使用相同输入数据。

每轮实际计划必须与发现结果一致，包括问题描述、硬件能力、外层块、任务网格、UB 预算和内部基本块；profile 与普通运行也必须一致。精度、计划或 profile 失败时，该输入不宣布赢家。旧无 schema 的四候选报告仍可读取，但新 sweep 要求支持 schema 2 的 runner。

`plan_id` 标识算法族/规划版本/外层几何，不独自标识一次完整执行。完整关联使用问题、硬件、全部实际计划和源码/二进制 hash；变更配置语义应增加 planner/schema 版本。报告同时保存 `kernel.asc`、`matmul_plan.h` 快照与工具源码 hash。

| 文件 | 内容 |
| --- | --- |
| `report.md` | 每例候选数、观测最快、auto/最快、near_best、错误 |
| `report.json` | 候选发现原文、逐轮精度/实际计划/统计、源码和环境信息 |
| `sweep.csv` | 每个输入与策略一行，两种计时口径及几何特征 |
| `<case>/discovery/` | 无 launch 的规划输入、候选 JSON、SDK 日志 |
| `<case>/round-N/MxN/` | 正确性输出、Host 原始样本和可选 profile |

Host 耗时包含每次规划、分配、同步和释放；设备 Task Duration 来自独立 profiling，含首次调用。两者不能混用或相减。聚合使用轮中位数的中位数；near_best 的 3%/波动范围规则仅用于候选提示，不是统计显著性。几何利用率不是硬件利用率。

无 NPU 可以运行：

```bash
bash scripts/run_perf.sh --suite generated --case-count 64 --tile-sweep --generate-only
```

此模式每例只生成一份输入，标记 `discovery_pending=true`、`expected_runs=null`，不虚构 SDK 接受列表、测量或排名。

## 后续扩展契约

新增融合/流水算法时，先定义计划中的 owner、N 分片、结果缓冲和同步参与者，再加入候选族与资源检查。不要仅增加 tile 名称。第二阶段首先实现逐块 Matmul → 行最大值；第三阶段再开展双缓冲、固定结构求和及小规模路径。详细依赖见 [架构审计](ARCHITECTURE_AUDIT.md)。

接口实现参考用户克隆的官方 learning hub 中 `02.06/add_custom_template.asc` 的 `GetCoreMemSize`，以及 `03.05/matmul_abs.asc` 的 TCubeTiling 基本块访问；以实际安装 SDK 编译结果为准。
