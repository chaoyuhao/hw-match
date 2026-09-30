# 第二轮候选：按 shape 选择 Matmul 分块

参照版本 `72abad1` 已获用户反馈线上 15/15 通过。其本地 `B=1,M=8192,N=65,K=32,FP16,00` 的设备 Task Duration 中位数为 324.048 μs。此文描述的新候选尚未在真实 CANN/NPU 上编译和测量，不能沿用参照版本的通过记录或承诺提速。

## 改动与选择规则

只调整 Matmul 外层任务尺寸，保留完整 similarity、并行行最大值、按原顺序的 FP32 补偿求和、scratch 布局和两次 AIV 屏障。每次调用仍恰好一个 kernel launch，线上只替换 `kernel.asc`。

Host 在 `32×64`、`32×128`、`64×128`、`128×128` 四个候选中依次选择：

1. 任务数为 `B*ceil(M/tileM)*ceil(N/tileN)`；参与核数为任务数与 availableCoreNum 的较小值。
2. 从 `32×64` 开始。只有候选任务数更少，且参与核数与参照相同时，才替换当前选择。任务数相同时保留较小 tile。
3. 仅使用本次输入的实际 shape 和 availableCoreNum，无缓存、计时选路或测试点编号。

在 availableCoreNum=24 时：

| B,M,N | 自动 tile | 参照任务数 → 新任务数 | 参与核数 |
|---|---|---:|---:|
| 1,8192,65 | 128×128 | 512 → 64 | 24 |
| 1,1024,65 | 32×128 | 64 → 32 | 24 |
| 1,33,8192 | 64×128 | 256 → 64 | 24 |
| 2,65,129 | 32×64 | 18 → 18 | 18 |

Host 的 SetShape、kernel 的任务网格、各 tile 的尾块和启动核数使用同一份选择。A/B 保留原物理行跨度，C 保留 round_up(N,16) 跨度，K 不拆分。SetShape 设定的是一次 Matmul 任务的范围，内部 baseM/baseN/baseK 仍由 CANN tiler 决定，不强制等于外层 tile。

这个选择规则是待测的性能假设。保留核数不保证负载均衡和性能不退化；大 tile 会影响内部 tiling、资源和访存复用，K/dtype/转置尚未作为性能分派条件。GetTiling 失败会明确报错，不会将失败候选伪装成回退成功。

## 先验证正确性

```bash
git pull --ff-only
source /usr/local/Ascend/cann-9.1.0/set_env.sh
CANN_MATMUL_TILE=auto bash scripts/run_local.sh --suite full &&
CANN_MATMUL_TILE=auto bash scripts/run_local.sh --suite tiling &&
CANN_MATMUL_TILE=auto bash scripts/run_local.sh --suite reduction
```

原有 full 55 项和 reduction 62 项保留；新增 tiling 34 项覆盖 FP16/BF16、四种布局、63/65/127/129 行边界、127/129/257 列边界、全负值、多 batch 及 K=2048。在 24 核下，新套件会实际选中三种较大 tile，不只是用小 shape 测旧路径。

任何失败先保留 `run.log`、对应 case 的 `runtime.log` 和 `result.json`。CPU 检查验证的是实际整数分块/地址函数、GM 输出覆盖和报告处理，不验证 Cube 数值行为、CANN 编译器或硬件流水线。

## 同一新版二进制可测试不同分块

仅本地 runner 读取环境变量 `CANN_MATMUL_TILE`：

| 值 | 行为 |
|---|---|
| auto（默认） | 正式 run_kernel 默认的 shape 选择逻辑 |
| 32x64 | 固定原来的任务尺寸 |
| 32x128 | 固定较宽任务 |
| 64x128 | 固定中等任务 |
| 128x128 | 固定最大候选 |

固定模式跳过“核数不减少”的筛选，可能降低并行度，用于测量取舍。它与 auto 复用同一个 kernel 和 RunKernel 实现；环境变量解析、日志和 JSON 写入都在本地 runner 中，提交的 kernel 不读取环境变量。`32x64` 保留参照几何布局，但不是旧提交的相同二进制；可与旧报告交叉核对额外参数/调度计算的影响。

先测默认候选的完整压力集：

```bash
CANN_MATMUL_TILE=auto bash scripts/run_perf.sh --suite stress
```

固定策略也应先验证 tiling 套件，例如 `CANN_MATMUL_TILE=64x128 bash scripts/run_local.sh --suite tiling`。为了减少反复编译，可以复用刚生成的 binary 做分块对比。下面第一行替换为该 run 的实际目录；所有策略依次运行，不并行争用 NPU：

```bash
tile_run=/实际路径/build-perf/run-时间戳
tile_compare=$(mktemp -d "$PWD/build-perf/tile-compare-XXXXXX")
for tile in 32x64 32x128 64x128 128x128 auto; do
    CANN_MATMUL_TILE="$tile" python3 scripts/perf_local.py \
      --binary "$tile_run/build/baseline_runner" \
      --output-dir "$tile_compare/$tile" --suite stress \
      --case sweep_m8192_fp16_00 --profile timeline || break
done
```

也可以单独使用 `CANN_MATMUL_TILE=32x64 bash scripts/run_perf.sh --suite stress` 创建新构建，并用现有 `--compare /旧run/cases/report.json` 比较 host median。固定策略的大范围使用前也应通过对应的正确性套件。

每个 case 的 `matmul_plan.json` 记录实际 policy、tile、任务数和参与核数，合并进报告。Markdown 显示 `Tile; tasks/blocks`。Host 与独立 profiling 的 plan 不一致会判为失败。旧 binary 没有该文件时保留未知，不从环境变量推测已应用策略。

回传 `report.json` 和 `report.md` 即可先分析；最终验收和得分仍以线上评测为准。

## 一次构建采集分块矩阵

`--tile-sweep` 复用同一次构建，针对相同输入依次测试四种固定 tile 和当前 `auto` 策略。它收集选路所需的数据，不修改 kernel 或自动生成决策树。

在 NPU 机器运行：

```bash
git pull --ff-only
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash scripts/run_perf.sh --suite stress --tile-sweep
```

默认 48 个输入 × 5 种策略 × 2 轮，共 480 次 runner 运行。每次仍有 2 次正确性检查、3 次 warmup 和 30 次 Host 计时；各策略使用相同 shape、数据类型、布局、输入分布和随机种子。候选串行执行，随用例和轮次旋转顺序，减轻顺序偏差；轮换并不能排除温度、频率或其他任务干扰。不要同时运行其他性能测试。

脚本覆盖继承的 `CANN_MATMUL_TILE`，以每个 runner 写出的 `matmul_plan.json` 验证实际策略。缺少计划、请求与实际策略不符、不同轮次的计划变化、精度或 profiling 失败时，不为该用例宣布优胜者，并以非零退出。报告逐次保存，失败的日志仍保留。每次运行使用新目录，不混入旧样本。

默认比较 **Host 调用耗时**，包含 tiling 和分配/释放，适合先检查覆盖和明显趋势。设备比较需要显式采集 profiling，可以先针对一个代表性输入：

```bash
bash scripts/run_perf.sh --suite stress --tile-sweep \
  --case sweep_m8192_fp16_00 --profile timeline
```

省略 `--case` 会对全部输入采集，耗时更长。`--sweep-rounds 1` 可用于初步排查，重复轮次更多才有跨轮波动信息。设备排名只接受预期的单个 `MatmulMaxSum` / `MIX_AIC` 任务组及正确记录数；若 CANN 导出形式不同会报错，保留原始 CSV 后再适配，不能擅自相加。设备统计来自独立 profiling 运行，含首次调用；不会用 Host 耗时补缺，也不与 Host 耗时相减。

报告位于新建 run 目录的 `cases/` 下：

| 文件 | 内容 |
| --- | --- |
| `report.md` | 每个输入的五策略耗时、观测最快策略、`auto/最快` 比值、接近最快的候选和错误信息 |
| `report.json` | 每一轮的原始统计、精度、实际分块、profiling 汇总、版本信息及聚合结果 |
| `sweep.csv` | 每个输入 × 策略一行，便于画图或归纳规则；同时保留 Host/设备两种口径 |
| `<case>/round-<轮次>/<策略>/` | 输入、输出、原始 Host 样本、runner 日志和可选的 profiling 文件 |

聚合耗时是各轮中位数的中位数，同时保存各轮中位数和 p95。`near_best` 包含比观测最快慢不超过 3%，或跨轮中位数范围与其重叠的候选；这是启发式提示，不是统计置信区间。只差一点的配置不应据此立即形成新的选择分支。固定 `32x64` 是同一新版二进制的固定几何配置，不代表旧提交的完整实现。

CSV 的任务数、轮数、活跃核比例、所有轮次的任务槽占用比例、外层 tile 有效面积比例、FLOPs 和 similarity 大小均由实际 plan 与输入元数据推导。任务槽占用比例为 `tasks/(waves*blocks)`，不是仅最后一轮的占用比例。它们是几何/工作量特征，不是测得的硬件利用率，外层 tile 也不等于 Matmul 内部基本块。

无 NPU 的开发机可先验证输入与实验规模：

```bash
bash scripts/run_perf.sh --suite stress --tile-sweep --generate-only
```

此模式每个输入仅生成一份数据，记录全部计划运行项，状态为 GENERATED，不产生耗时或排名。本地脚本只为后续分析收集证据；没有真实设备结果前，不据此更新 Host 选择器或声称提速。

## 开发机检查记录

- `python3 -m unittest discover -s tests -v`：41 项通过，包含实际 C++ 分块/地址函数的 CPU 检查和 host/profile 分块不一致的失败检查。
- tiling 34 项数据生成成功，全部标记为 GENERATED，报告为 `build-baseline/run-20260929T145556Z-pzA5jA/cases/report.json`。
- 独立代码审查未发现阻塞问题。源码检查确认两个归约 helper 与 `72abad1` 相同，官方入口签名/实现相同，一次 launch、两次屏障，kernel 内无调试 I/O 或环境变量读取。
- 开发机没有 CANN 编译器或 NPU；尚无本候选的设备编译、正确性和性能结果。

接口依据：[CANN 9.1 Matmul shape 概念与设置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/programug/Ascendcopdevg/docs/en/guide/operator_practice/simd_operator_impl/matrix_advanced_api/operator_implementation.md)。接口与新分块的实际兼容性仍需在目标 CANN 9.0 和本地 9.1 编译验证。
