# 本地压力测试与性能分析

各轮线上结果统一见 [迭代记录](ITERATION_LOG.md)，最新 S3 用户反馈为 15/15 通过。当前 [规则规划版](MATMUL_TILING.md) 可用 `CANN_MATMUL_TILE` 在本地固定分块；本地对比仍需使用相同输入与计时口径，不能沿用历史本地性能。线上只复制 kernel 及其依赖的 `.asc` / `.h`，本地工具不提交。

批量比较同一输入的 SDK 接受候选与 `auto`，使用 `bash scripts/run_perf.sh --suite stress --tile-sweep`；自动用例、实验轮次、设备计时和 CSV 字段见 [规划与采样说明](MATMUL_TILING.md)。

## 先运行这些命令

在已有 910B2C / CANN 9.1.0 的机器上：

```bash
git pull --ff-only
source /usr/local/Ascend/cann-9.1.0/set_env.sh

# 无需编译，保存工具路径、版本和 --help
bash scripts/run_perf.sh --inspect-tools

# 8 项布局测试，默认 2 次正确性检查 + 3 次 warmup + 30 次测量
bash scripts/run_perf.sh --suite quick

# 48 项压力测试，分别扫描 B、M、N、K
bash scripts/run_perf.sh --suite stress

# 针对一个大 M 用例另起进程采集时间线和流水线指标
bash scripts/run_perf.sh --suite stress --case sweep_m8192_fp16_00 \
  --profile timeline --metrics PipeUtilization
```

默认 ACL **逻辑设备 0**，保留容器现有的可见设备映射，不把 `npu-smi` 的物理 13 填入 `--device`。每次脚本创建独立 `build-perf/run-时间-随机后缀/` 并重新编译本地 runner。目录下 `run.log` 包含构建和汇总；`cases/report.md` 供阅读，`cases/report.json` 供后续比较。工具探测结果为 `cases/tools.json`，帮助信息也在其中。

没有 CANN 的开发机可以使用 `--list-cases` 或 `--generate-only`；生成数据标记为 GENERATED，不代表 NPU 通过。`--suite regression` 使用原有 55 项正确性用例并计时。所有测量用例都比较 FP64 golden、检查 NaN/Inf、输出 guard、输入未修改、重复输出逐位一致。失败保留日志并返回非零。

## 三种时间的区别

| 数据 | 包含什么 | 用途 |
|---|---|---|
| `host_call` | C++ steady clock 测 `run_kernel` 至 stream 完成；包括 host tiling、workspace 分配/释放、设备计算与同步 | 判断一次调用的整体开销 |
| profiler `task_duration` | 独立 profiling 进程中导出的 `Task Duration(us)`，按 Op Name / Task Type / Device 分组 | 判断设备任务开销，查看 Cube、Vector、搬运占用 |
| `wall_seconds_including_startup_io` | 生成 golden、进程启动、文件读写和所有测试 | 判断测试脚本本身运行了多久 |

Host 计时窗口不包含输出预填、输入/输出拷贝、验证、日志和文件 I/O；每次调用后均在窗口外验证，再进行下一次调用。因此它是固定输入重复调用测试，缓存及两次调用间的设备空闲状态可能与线上不同，并非吞吐量测试。warmup 从 host 统计中排除，p95 使用 nearest-rank 算法。

profiling 与无 profiler 的测量分开执行，不能把二者相减就认定是 host 开销。当前 profiler 运行默认包含 5 次完整调用，**包括首次调用**；与 host 稳态统计不同。MIX 任务可能有不同类型记录，CSV 行数不等同于 kernel launch 次数。本地报告不推算线上分数，也不替代平台的单次 launch 检查。

原始输出保留在 `cases/<case>/profile_run/profiling/PROF_*/`。自动报告只读取 `op_summary*.csv`，不重复累加 `op_statistic` 汇总。缺少导出文件、无效耗时或输出错误均为 FAIL。指标 N/A 保持缺失，不当作 0；报告保留每项有效指标的样本数。若本机版本导出列不同，保留原始文件供适配。

按 [msprof 应用参数要求](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0010.html)，采集时用正确引用全部参数的 shell launcher，避免应用路径含空格时被拆开。launcher 内容保存在 `profile_run/profile_app.sh`，执行时临时复制到无空格的 `/tmp/cann-msprof-*` 路径，结束后自动清理；`command.json` 同时记录原应用参数与 profiler 命令。

## 怎么找到瓶颈

Stress 固定随机种子，分别只改变一个维度；两 dtype 都扫描 B=1..257、M/N/K 到 8192，另有四种布局的相同 shape。每例的 `B*M*N <= 2^20`，先控制内存占用，避免把 OOM 当性能问题。这些用例是通用诊断输入，**不代表已知线上 shape**。转置下的长维压力可在现有 case 表基础上继续扩展。

先看维度与耗时的关系，再对慢用例采集指标：

| 观察 | 待验证的原因 | 下一步证据 |
|---|---|---|
| M 增长时设备耗时明显增加、Cube 利用率低 | 按行归约、标量读回及同步可能占比大 | Vector/MTE/Scalar 指标和 `msprof op` 流水线 |
| N 增长时搬运占比明显增加 | 全量 FP32 相似度矩阵写入 GM 后再次读取 | `Memory`、`MemoryUB`、`L2Cache` 指标 |
| K 增长时 Cube 占比上升 | 计算量增加，或 tile / 数据复用不合适 | `ArithmeticUtilization` 与 `PipeUtilization` |
| host 调用慢而设备任务短 | 分配、tiling、同步或 API 调度开销 | 同一条 timeline 中的 ACL/Runtime API 事件 |
| 大 B 好、小 B 差 | 核间负载分配或归约并行度不足 | 任务分布及流水线，而非只看单一平均利用率 |

以上都是诊断假设。当前源码确实会写完整 FP32 中间矩阵，再按行归约，但占时多少必须由测量确认。报告的 `reference_work` 只是按参照实现计算的 FLOPs / 中间矩阵字节数，不是测得的带宽。

同一个用例分别采其他计数器组（每次独立采集）：

```bash
bash scripts/run_perf.sh --suite stress --case sweep_n8192_fp16_00 \
  --profile timeline --metrics Memory
bash scripts/run_perf.sh --suite stress --case sweep_k8192_fp16_00 \
  --profile timeline --metrics ArithmeticUtilization
```

JSON 记录 git commit/dirty 状态、kernel 和 runner 源码 SHA256、可执行文件 SHA256、CANN 元数据、编译器与 NPU 信息，并保存 kernel 快照。不要在构建/测试期间更改源文件。通过 Python 直接指定已有 binary 时，脚本不能证明 binary 与当前源代码一致；优先用 shell 入口新构建。

比较两次测试：

```bash
bash scripts/run_perf.sh --suite stress \
  --compare /实际路径/旧run/cases/report.json
```

只匹配通过正确性检查且 shape、dtype、布局、pattern、seed 完全相同的用例，计算旧/新 host median 比值。运行前确认两份环境、warmup、次数一致，尽量避免其他任务占用同一卡；比较不会自动消除机器状态或 SDK 差异。

## CANN 的开发工具

- **msprof**：时间线、ACL/Runtime API、设备任务时间、AI Core 指标。本脚本已接入。官方文档：[9.0 命令](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0011.html)、[op_summary 字段](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0067.html)。
- **msprof op**：单算子更细的流水线、带宽、资源冲突分析。本脚本先探测其帮助；确认机器组件齐全后再运行，部分分析依赖仿真器或专用构建。示例与参数见 [官方使用指南](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md) 和 [Ascend CATLASS 性能工具](https://gitee.com/ascend/catlass/blob/master/docs/tools/performance_tools.md)，以已安装版本支持的选项为准。
- **mssanitizer**：越界、未初始化读取、数据竞争等检查。这里只探测安装情况，没有宣称执行内存检查。设备检测需要按对应 SDK 要求插桩编译，普通性能 binary 不能当作已经完成 sanitizer 检查；参阅 [官方使用指南](https://github.com/Ascend/mssanitizer/blob/master/docs/en/user_guide/mssanitizer_user_guide.md)。

已经生成 case 并构建后，可依本机 `msprof op --help` 运行单算子采集，例如：

```bash
msprof op --launch-count=1 --output=/tmp/cann-op-profile \
  /实际run目录/build/baseline_runner /实际run目录/cases/sweep_m8192_fp16_00 0 5 0
```

手动命令会覆盖该 case 的 `y-*.bin`，需要保留先前输出时先复制 case 目录。详细采集和 sanitizer 应独立运行，其插桩耗时不与普通性能结果混为一谈。

本开发机没有 CANN/NPU；Python 测试及 mock profiler 验证的是数据、计时汇总与失败处理。新增 runner 仍须在用户机器上实际编译、执行，才能验证 SDK 兼容性并获得瓶颈证据。
