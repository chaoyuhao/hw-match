# 本地 BatchMatmulMaxSum baseline

这是用于本地正确性调试的候选实现，最终以 **CANN 9.0.0 线上平台**评测为准。用户的 910B2C / CANN 9.1.0 镜像已经通过独立向量加法检查；本 baseline 的 Matmul 编译、运行和精度仍需在该镜像验证。

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

## 实现

`kernel.asc` 保留原始 `run_kernel` 签名，由原 `main.asc` 和本地 runner 共用。实际计算都在 NPU 执行，host 只校验元数据、计算 tiling、分配空间和调度：

1. Cube kernel：Matmul 高阶 API，FP16/BF16 输入，FP32 输出；每个任务负责一个 batch 内最多 `32×64` 的输出块，沿完整 K 计算。四种转置存储布局直接通过地址偏移和 API 的 transpose 参数解释。
2. Vector kernel：每次读取一行最多 1024 列，归约时仅包含实际有效的 N 个元素；按固定行顺序，用补偿求和累加 FP32 行最大值。没有浮点原子加。

中间矩阵行跨度为 `round_up(N,16)`，空间约为 `4*B*M*round_up(N,16)` 字节，另有 Matmul 系统 workspace。本地诊断默认还会在 host 保留中间矩阵和 FP64 golden。M/N/K 支持范围为 `[1,8192]`；实际可运行规模还受内存限制。README 示例中的 K=2 也可作为诊断输入，生成的常规用例遵守 K 为 8 的倍数。

输出按每 8 个 batch 一组划分任务，避免不同核同时写入同一个 32 字节区域；最后不足 32 字节使用 `DataCopyPad` 精确写回。每次调用分配临时 GM，stream 同步后释放。因此当前版本包含分配、初始化、同步及全量中间矩阵读写的开销，**不是性能优化版**。

## 验证范围

- golden 使用量化后的实际输入值进行 FP64 Matmul，再做 Max/Sum，最终转为 FP32。
- 数据工具仅依赖 NumPy。BF16 文件按 FP32 → BF16 最近偶数舍入编码，无需在数据生成机器安装 `ml_dtypes`；远端现有包仍可用于交叉核验。
- 检查输出字节数严格等于 `4*B`，拒绝 NaN/Inf，所有元素满足 `abs(actual-golden) <= 1e-4 + 1e-4*abs(golden)`。
- 默认额外比较设备中间矩阵与 FP64 Matmul，以及设备最终输出与从设备中间矩阵计算的归约结果，方便区分 Matmul 与归约问题。
- runner 每次把输出填为 NaN，检查输出前后 guard、输入内容未改变、两次执行输出逐位一致。
- `smoke` 覆盖两 dtype × 四布局、全负值、非对齐 M/N/K、batch、已知值、原模板 shape。`full` 增加 K=8192、M/N=8192、N=1、零输入、抵消和多任务复用；B=257 用例在 24 核机器上覆盖归约输出的完整分组、尾组及同核分组复用。
- 所有报告包含 `ONLINE_EVALUATION=NOT_RUN`。报告里的 wall time 包含进程启动、文件 I/O 和诊断，不能视为 kernel 耗时或官方成绩。

本开发工作区执行：

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/run_local.sh
bash scripts/run_local.sh --generate-only --suite full
```

这些检查验证数据工具和失败处理，不替代真实 CANN 编译、内存检查器或 NPU 测试。当前尚未确认平台允许多次 kernel launch、host 临时 GM 分配和内部同步，因此不能直接把本版本称为可提交版本。

## 使用的官方接口

- [CANN 9.0 Matmul 实现流程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0038.html)
- [GetTiling：标准 C++ TCubeTiling](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0692.html)
- [SetOrgShape：输入跨度与输出跨度](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0651.html)
- [系统 workspace](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0171.html)
- [PlatformAscendCManager](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_1039.html)
