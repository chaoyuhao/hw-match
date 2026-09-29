# 本地环境检查与线上评测

本地昇腾卡用于开发、调试和性能比较。**比赛正确性、性能和最终得分，以统一线上平台评测为准。** 本地环境检查成功、本地算子精度通过、线上全部 case 通过是三个不同的结论。

## 一条命令检查环境

在 Git 仓库中执行：

```bash
git pull --ff-only
bash scripts/check_env.sh
```

默认执行以下检查：

1. 记录系统信息、Git 提交、未提交修改和 `kernel.asc` 的 SHA256。
2. 找到 CANN 环境脚本时加载；镜像没有脚本时使用已有环境。记录工具路径、toolkit 版本元数据、ASC CMake 文件位置和 NPU 信息。
3. 验证 `bisheng`、CMake、Make 可以执行。检查 Python、NumPy、`ml_dtypes`，不需要 PyTorch。
4. 使用与比赛模板一致的 `.asc`、`find_package(ASC)`、默认 `dav-2201` 目标和链接库，在独立目录编译小算子。
5. 通过 ACL 申请内存、传输数据、启动一次向量加法、同步、回读，并检查 64 个 FP32 结果。

脚本不安装软件、不更改系统配置，也不覆盖比赛工程的 `build/`。所有输出写入新建的 `build-envcheck/check-时间-随机后缀/`，已经被 `.gitignore` 排除。结束时会打印 `report.log` 的完整路径。把该文件贴回即可，配置、构建与运行还分别有独立日志。

每个外部检查有超时：通常 20 秒、SDK 文件搜索 30 秒、CMake 配置 120 秒、编译 180 秒、运行 30 秒。若机器繁忙导致超时，请根据对应日志判断，超时本身不证明 SDK 不兼容。

## 选项

```bash
# 收集环境信息，不编译、不启动 kernel；仍会读取 npu-smi 信息
bash scripts/check_env.sh --inspect-only

# 检查并编译，不启动 kernel
bash scripts/check_env.sh --no-run

# 使用 ACL 的逻辑设备编号
bash scripts/check_env.sh --device 0

# 指定已安装环境的脚本；只影响检查进程
bash scripts/check_env.sh --env-script /实际安装路径/set_env.sh

# 指定本地编译目标或报告位置
bash scripts/check_env.sh --arch dav-2201 --output-dir /tmp/cann-envcheck
```

默认设备号是 **ACL 逻辑编号 0**。用户之前的 `npu-smi` 显示物理 NPU 7，这不意味着程序应该使用 `--device 7`。运行日志会显示 ACL 可见设备数量和选用的逻辑编号。

环境脚本自动查找优先使用 `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` 下以及上一级的 `set_env.sh`。变量未设置时，尝试从 `bisheng` 路径推断安装目录，再考虑常见安装路径。可以先手动 source 正确脚本或使用 `--env-script` 指定。

## 如何读结果

| 状态 | 意义 |
| --- | --- |
| `[PASS]` | 该项检查实际成功；不能外推到其他检查。 |
| `[FAIL]` | 检查失败，例如命令无法运行、配置失败、编译失败、ACL 错误、输出不符。 |
| `[WARN]` | 信息缺失或需注意。例如 CANN 版本未知/不同于 9.0.0、无 npu-smi、缺少 Python 数据依赖。 |
| `[SKIP]` | 没执行；包括用户选择跳过或前置步骤失败。 |

退出码：`0` 表示所请求范围内没有 FAIL（可能仍有 WARN/SKIP）；`1` 表示检查失败；`2` 表示参数错误。`--inspect-only` 返回 0 也不代表能编译或运行。

Python 数据依赖缺失是 WARN，因为独立 C++ 测试不依赖 Python。此时即便小算子跑通，比赛的 `gen_data.py` 仍可能无法运行。可以在另一台机器生成测试数据。

CANN 版本仅从 toolkit 元数据读取。**驱动版本、Clang 版本和编译日期都不能代替 CANN 版本。** 版本未知或不是比赛要求的 9.0.0 时会明确警告；本地 SDK 可运行仍不等于匹配线上 SDK。

若 `configure` 失败，首先看 `configure.log`。找到 `bisheng` 不等于安装了 ASC CMake 包；找不到 ASC 也可能是包搜索路径问题。确认包的真实路径后可以通过 `CMAKE_PREFIX_PATH` 配置搜索路径，再重新运行。不要为了让本地旧 SDK 通过，直接改动官方提交入口和构建约定。

若 `runtime` 失败，看 `runtime.log` 中的 ACL 返回码、设备数量或错误结果。不要根据 npu-smi 的物理编号猜逻辑编号。

## 检查的边界

独立小算子只验证最基本的 ASC 工具链、模板链接依赖、ACL 调用、设备搬运和 Vector kernel 执行。它**没有验证 Cube Matmul、高阶 API 的完整支持、BatchMatmulMaxSum 的正确性、15 个官方 case 或性能分数**。

每份报告固定包含：

```text
ONLINE_EVALUATION=NOT_RUN
```

后续工作采用以下顺序：

1. 本地环境检查成功后开发 baseline。
2. 本地覆盖两种 dtype、四种布局、全负值、尾块和大 K，检查精度与重复执行。
3. 尽早向线上平台提交候选版本，记录提交文件 SHA256、Git 提交、平台返回的精度结果、逐 case 耗时和得分。
4. 用线上结果决定是否保留优化；本地耗时只用于辅助分析，不能换算或预测官方分数。

不要把本地缺失的环境能力推断成平台规则；多个 kernel、workspace、可用库、计时范围、提交文件清单仍需按平台规定确认。

## 本仓库的脚本验证

```bash
bash -n scripts/check_env.sh
python3 -m unittest discover -s tests -v
```

这些自动检查验证诊断脚本对跳过、失败和参数错误的处理。测试用受控程序代替不可用的 CANN 编译器与 NPU，不能作为真实 Ascend C 编译或设备运行通过的证据。真实设备需要执行本文开头的检查命令。
