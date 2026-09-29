# BatchMatmulMaxSum baseline 调研

调研日期：2026-09-29。依据：本仓库 README、全部模板源码、官方 CANN 文档，以及用户提供的远端机器信息。

进度更新（2026-09-29）：两次启动的 `377f283` 已在用户 910B2C / CANN 9.1.0 上通过 55/55 本地用例。线上要求每次迭代恰好启动一个 kernel；只能修改 `kernel.asc` 或新增 `.asc` / `.h`。当前改为单次启动的 MIX 候选版，需重新进行 NPU 和线上验证，见[运行说明](LOCAL_BASELINE.md)。以下保留最初调研记录，其中多次启动方案仅作为本地正确性参照，不适用于已明确的线上规则。

本地环境检查入口为 `bash scripts/check_env.sh`，详见[环境检查说明](ENVIRONMENT_CHECK.md)。自有昇腾卡用于开发调试，比赛正确性、性能和最终得分以统一线上平台评测为准。

## 已确认的要求与环境

计算语义：

```text
S[b,m,n] = sum_k X1_logical[b,m,k] * X2_logical[b,k,n]
R[b,m]   = max_n S[b,m,n]
Y[b]     = sum_m R[b,m]
```

- 两个输入同类型，支持 FP16 / BF16；输出 `[B]`，固定 FP32。
- 输入连续、ND；batch 一一对应，没有广播、mask 或额外归一化。
- 全负行的最大值也必须为负，初值不能是 0。
- 矩阵乘累加和后续归约均保留 FP32；golden 用输入量化后的实际存储值做 FP64 计算，再转 FP32。
- README 要求 15 个测试点全部正确才计分。最大 M/N/K 在描述中提到 8192，但详细的维度范围和总规模限制丢失，不应自行补造。
- 用户最初的 openEuler / 910B2C 镜像已确认为 CANN 8.1.RC1，无法配置 ASC 工程。随后切换 Ubuntu 24.04 / CANN 9.1.0 镜像，独立 `.asc` 向量加法已经完成配置、编译、链接与 NPU 运行，64 个输出全部匹配；ACL 可见逻辑设备 0，报告 24 个 Cube 核。新镜像的 NumPy 与 `ml_dtypes` 均可用。9.1.0 与比赛要求的 9.0.0 仍有版本差异。
- 本开发工作区没有 NPU、CANN、`bisheng` 或 `ccec`，只有 NumPy，缺 `ml_dtypes`。本地 baseline 使用 NumPy 自行编码有限 BF16 数据；此处的单元测试不代表远端 Matmul 已通过。

输入物理布局：

| transposeX1 | transposeX2 | x1 storage shape | x2 storage shape |
| --- | --- | --- | --- |
| false | false | `[B,M,K]` | `[B,K,N]` |
| false | true | `[B,M,K]` | `[B,N,K]` |
| true | false | `[B,K,M]` | `[B,K,N]` |
| true | true | `[B,K,M]` | `[B,N,K]` |

不需要先生成一份完整转置后的输入。利用 Matmul API 的布局解释能力完成逻辑乘法即可。API 的 transpose 参数仍需按物理存储正确配置。

## 模板实际上提供了什么

| 文件 | 当前内容与影响 |
| --- | --- |
| `kernel.asc` | 被 `main.asc` include；入口为 `run_kernel`，现已填入本地 baseline。不能加入另一个 `main`。 |
| `main.asc` | 固定跑 FP16、`B=1,M=1,N=1,K=32`、两个属性 false；输入字节数和 shape 都写死。 |
| `CMakeLists.txt` | `.asc` 直调工程，`find_package(ASC REQUIRED)`；默认架构 `dav-2201`，已链接 `tiling_api`、`platform` 等库。 |
| `scripts/gen_data.py` | 只生成上面一个 case。 |
| `scripts/BatchMatmulMaxSum.py` | FP64 golden；注释称 15 个 case，但实际列表仅有一个小示例，不是完整比赛用例。顶层直接依赖 `ml_dtypes`。 |
| `scripts/verify_result.py` | 输出使用 `np.isclose(..., rtol=1e-4, atol=1e-4)`，当前只有 case 0。 |
| `run.sh` | 原版只在 `${ASCEND_HOME_PATH}/set_env.sh` 查找脚本，用户机器在此失败。本次已调整为检查当前目录和上一级的脚本；没有脚本时检查已有 `bisheng`。 |

仅修改生成数据脚本无法扩充测试：测试 main 的 shape、dtype、属性和内存大小也要一起参数化。`np.isclose` 的误差条件是 `abs(actual-golden) <= atol + rtol*abs(golden)`，并不等于同时检查两个独立不等式；平台最终判定仍以实际规则为准。

模板暗示 `kernel.asc` 是核心提交文件，但仓库没有正式提交说明。文件清单、多个 kernel launch 是否允许、临时 GM 分配是否允许、计时范围，都需要从比赛平台规则确认。

## 必要文档：按这个顺序读

优先选 CANN 9.0 系列和 Atlas A2 支持项。下面的 9.0.X 文档是持续更新的系列文档，实际接口以远端安装的 **9.0.0 头文件与配套样例**为准。部分中文网页抓取只返回导航，因此这里保留已经读取到正文的英文页面。

| 顺序 | 官方文档 | 本题用途 |
| --- | --- | --- |
| 1 | [Matmul 使用说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0614.html) | 理解 `MatmulType`、初始化、输入绑定、执行和 `End`；查 FP16/BF16 输入、FP32 输出组合。 |
| 2 | [TCubeTiling](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0673.html) | 区分原矩阵尺寸、单核尺寸、base tile；核对 L1/L0 容量与对齐要求。首版使用 Host tiling API 生成参数。 |
| 3 | [SetTensorA](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0631.html) | 四种存储布局；MatmulType 的 `ISTRANS` 必须允许需要的 transpose，Host tiling 与 Kernel 配置保持一致。B 的入口可从 Matmul 使用说明进入。 |
| 4 | [IterateAll](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0640.html) | 首版把单核负责的矩阵乘结果写到 GM。 |
| 5 | [SetTail](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0647.html) | 告知 Matmul 尾核实际 M/N/K 大小。 |
| 6 | [DataCopyPad](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0265.html) | Vector 侧搬入不对齐行、搬出不足 32B 的结果；注意 GM 和 UB stride 单位不同。 |
| 7 | [ReduceMax](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_10055.html) | 对二维 tile 的 N 轴做归约，`Pattern::Reduce::AR`；核对临时空间和 UB 内轴 padding。 |
| 8 | [Iterate](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0638.html) / [GetTensorC](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0639.html) | 第二版逐块取得 FP32 Matmul 结果并立即归约；了解迭代顺序和同步要求。 |

以上文档各自的调用样例、类型支持表比泛泛教程更有用。不要直接搬用只支持 950 的 SIMT、Reg 或特殊数据通路示例。

## 推荐路线

### 第一版：两阶段正确性 baseline

前提：平台允许 `run_kernel` 中启动多个自定义 Ascend C kernel，并允许合法申请临时 GM。仓库尚不能确认这两条。

```text
输入 x1、x2
    ↓ Cube kernel：Matmul 高阶 API，FP32 累加/输出
S[B,M,N]：临时 GM
    ↓ Vector kernel：逐行 Max(N)，然后 Sum(M)
y[B]：FP32
```

建议做法：

1. Host 从 metadata 和两个属性恢复逻辑 B/M/N/K；按 FP16/BF16 和四种布局分派。
2. 按 B 与 M 分块分配 Cube 任务；若需要也可沿 N 分块，但首版避免拆 K。跨核拆 K 会引入额外部分和合并。
3. 使用官方 Matmul tiling API，输入 GM/ND，输出 GM/ND/FP32，不启用 bias。正确设置原矩阵跨度、任务地址偏移和尾块尺寸。
4. Cube kernel 把完整点积结果写到 S。第一版每个输出元素只有一个写者。
5. 在同一 stream 随后启动 Vector kernel。最简单的正确性实现是每个 batch 由一个逻辑任务完成归约，按行或按块搬运，不能把整个 S 塞进 UB。
6. 最终和使用固定归约顺序。先不对 y 使用跨核浮点 AtomicAdd，避免引入执行顺序相关的差异。

这条路线的好处是可以分别比较 S 与最终 y，较容易定位布局、Matmul、归约哪一步出错。它是候选正确性 baseline，并不保证达到比赛的性能门槛。

代价：S 的逻辑空间为 `4*B*M*N` 字节，若采用填充布局还会更大。以单个 `8192×8192` FP32 矩阵为例，S 为 256 MiB，写一次再读一次产生约 512 MiB 流量。该例仅说明规模，不表示这是一个已确认合法的评测 shape。

若规模或内存不允许全量 S，可以分波处理 B/M；代价是更多 launch。正式优化时优先切换到下一版。

### 第二版：块内 Matmul + MaxSim，末尾合并部分和

每个任务负责一个 batch 的一段 M 行，遍历 N 块：

```text
row_max = -∞
for each N tile:
    tile = Matmul(X1[M_tile, :], X2[:, N_tile])  # 完整 K 点积
    row_max = max(row_max, max(tile, axis=N))
partial[b, M_tile] = sum(row_max)

second_kernel: y[b] = fixed_order_sum(partial[b, :])
```

这是根据本题数学语义提出的设计，不是已经实现的代码。`max` 可以沿 N 块逐步合并；K 方向的点积必须先累加完整，才能对不同 N 候选取最大值。

利用 `Iterate + GetTensorC` 取得 tile，Vector 侧只维护当前行最大值。显式中间结果从 O(BMN) 降到 O(B×M块数)，另计 Matmul API 所需 workspace。

910B 上 Cube/Vector 协同、系统 workspace、逻辑核编号与同步都要按实际 API 模式处理。API 返回 LocalTensor 不代表底层一定完全不经过 GM，不能据此宣称已经消除全部数据搬运。尾块的有效 M/N 和 Matmul 的迭代顺序也必须跟踪。

### 不优先选择的路线

- 标量循环穷举 B/M/N/K：可以验证极小 case 的地址公式，但对大维度的工作量不合适，不能当作覆盖全部评测的实用 baseline。
- 一开始手写 `LoadData / Mmad / Fixpipe` 和 Cube/Vector 双缓冲：控制力强，但会同时引入格式转换、对齐、同步与数值问题，适合正确版本之后再研究。

## 最容易导致全场零分的问题

- MaxSim 初始值必须是负无穷或第一个有效值。N 尾部 padding 不能以 0 参与最大值。
- K 尾部参与乘加的数据可补 0；这与 N 归约 padding 的语义不同。
- 无效 M 行不参与最后的求和，不能把它们的负无穷加进去。
- x1 batch 偏移固定为 `b*M*K` 个输入元素，x2 为 `b*K*N`，与存储转置无关；batch 内的地址则随布局变化。
- 逻辑索引 `(m,k)`：x1 非转置为 `m*K+k`，转置存储为 `k*M+m`。
- 逻辑索引 `(k,n)`：x2 非转置为 `k*N+n`，转置存储为 `n*K+k`。
- Host tiling、MatmulType 和运行时 transpose 配置必须匹配。只传 true 而模板禁止 transpose 会得到错误结果。
- 使用 FP32 输出 S；不要在 MaxSim 之前把点积结果降回 FP16/BF16。
- `B=1` 时输出仅 4 字节，不可无条件按 32 字节向输出 GM 写回。
- 索引和 workspace 字节数计算使用足够宽的整数；缺失的规模约束尚未补齐。
- 临时 GM 必须在所有使用它的异步任务结束前保持有效。实际分配、同步和释放策略要结合平台计时规则；不能提前 free 或缓存输入相关结果。
- 验证应额外要求输出长度严格等于 B、全部有限；不能只依赖当前脚本的容错行为。

## 本地测试需要补齐的覆盖

下面是我们应自行构造的测试，不是泄露或已知的隐藏用例：

| 维度 | 覆盖 |
| --- | --- |
| dtype/layout | 两种 dtype × 四种布局，每种都测 |
| 已知结果 | README 的基础与全负示例；zero 输入 |
| 最小尺寸 | `M=1`、`N=1`、`M=N=1` |
| 尾块 | M/N 为 3、15、17、31、33 等；K 为 8 的倍数但非 16 对齐，例如 24 |
| batch | B>1，且不同 batch 使用不同分布，检查偏移和配对 |
| 长归约 | K=8192 搭配小 M/N；大 M 配小 N；大 N 配小 M，避免无意义的巨大 FP64 golden |
| 数值 | 全负相似度、正负混合、接近零的结果、抵消较强的点积 |
| 一致性 | 同一输入重复执行，检查输出稳定和有限性 |

先使用模板输出 FP32 的 `rtol=atol=1e-4` 作最低验证基准，并报告最大绝对误差和近零结果；不能因为输入是 half 就自行放宽到 1e-3。

## OpenEuler 机器的启动故障

用户当前报错：

```text
run.sh: line 17: /usr/local/Ascend/ascend-toolkit/latest/set_env.sh: No such file or directory
```

这证明当前脚本路径不存在，并不能证明编译器或 CANN 整体不可用。模板要求先设 `ASCEND_HOME_PATH`，随后又假设环境脚本在它下面。官方常见 toolkit 安装方式使用上一级的 `ascend-toolkit/set_env.sh`，见[官方环境变量配置说明](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/envvar/envref_07_0003.html)。新版本安装目录也可能是 `cann`，需要看本机实际文件。

先收集这些信息，不需要安装软件：

```bash
printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-}"
ls -ld /usr/local/Ascend/ascend-toolkit /usr/local/Ascend/ascend-toolkit/latest
find /usr/local/Ascend -maxdepth 5 \( -name set_env.sh -o -name bisheng -o -name ASCConfig.cmake -o -name version.cfg \) -print 2>/dev/null
command -v bisheng ccec cmake python3
```

后续诊断未找到环境脚本，但 `bisheng` 已经在 PATH 中。本次修改的 `run.sh` 会检查 toolkit 目录和上一级的环境脚本；若均不存在，则检查当前 PATH 中的编译器，继续使用已有环境。若没有编译器则明确报错退出。该修改仅处理环境加载入口，不证明 ASC CMake 包、动态库和 CANN 版本全部匹配。不能通过创建假的 `set_env.sh` 或把 `ASCEND_HOME_PATH` 改成错误层级来掩盖问题。

启动逻辑验证：`bash -n run.sh` 通过；在临时目录分别验证了环境已配置且无脚本、脚本位于 toolkit 目录、脚本位于上一级、缺少编译器四个分支，结果符合预期。仅模拟了环境入口，未运行真实 CANN 构建或 NPU kernel。

找到 `bisheng` 不等于具备 `find_package(ASC)` 使用的完整开发组件。下一项有判别力的检查是在远端对比赛工程执行 CMake 配置：如果报告找不到 ASC，应确认 SDK 是否完整、版本是否匹配、CMake 包搜索路径是否正确。目录搜索未命中也可能受符号链接等影响，不应直接据此认定必须重装。不要为了适配旧 SDK，立即改写比赛的 CMake 和提交 ABI。

编译依赖是 CANN ASC 编译环境、CMake 和 C++ 工具链；模板数据生成只需要 NumPy + `ml_dtypes`，不依赖 PyTorch。若远端 Python 依赖难安装，也可以在其他机器生成 `.bin` 文件，再拷贝到 NPU 机器执行 C++ runner。仍需实际核对 CANN 版本，不能用驱动版本代替。

## 下一步顺序

1. 已通过 CANN 9.1.0 环境检查；在该镜像运行 `bash scripts/run_local.sh` 验证新 baseline。
2. 获取平台提交规则与 README 缺失的规模限制，确认多 kernel、workspace 和计时范围。
3. 根据本地两阶段实现的报告核对 FP32 中间矩阵和最终 y，修正编译或精度问题。
4. 运行 `--suite full`，覆盖 dtype、布局、负值、尾块、大 K、大 batch 和重复执行。
5. 用平台验证全 case 后保存首个通过版本，再研究 tile 融合与性能。
