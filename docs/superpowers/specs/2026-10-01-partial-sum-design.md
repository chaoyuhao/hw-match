# R15：Max 完成处生成补偿部分和

用户已授权继续推进 S10 后的 E7 方向。本轮改变 Max→Sum 的数据与所有权接口，覆盖 GM、同步 Stream、异步 Pipeline；保留 Small、单 launch、完整 K 后 Max、完整 N Max 后 Sum。以 R14 `65f7817` / S10 为直接对照，不增加本地 NPU 测试。

## 数据与计算

每个完成全 N Max 的 owner 对所持 M 段做向量 TwoSum 树，停在 8 条 lane，输出 8 个主值和 8 个补偿值。64B 记录独占对齐，无 producer Scalar 读回；保留补偿而不是先舍入为一个 float。GM 以 32 行为段；Stream S=1 以 tileM 为段；S>1 仍先写各 N 片行 Max，由 32 行跨片合并 owner 在完成 Max 后生成记录。不能先对 N 分片求和。

最终每 8 batch 的输出 owner 读取记录，以两个 block-strided DMA 分离主值/补偿，各最多 1024 floats。使用同一 TwoSum 树、固定顺序 Neumaier 合并完成输出；每条输入记录最多读一次主值和一次补偿，保持输出 32B 组所有权。尾段填零，不能把 Max 的负 sentinel 加入 Sum。

`partial_sum.asc` 放设备 writer/finalizer；`reduction_plan.h` 放 POD 与纯主机几何/存储函数。旧 R12 SumRowMaxima 保留用于 rows 对照。记录数最多 ceil(8192/16)=512，每 batch 最多 4 个最终输入 chunk。带补偿树不能证明任意数据准确，需强抵消/近零/长跨度 FP64 golden 检查。

## 存储与预算

ReductionPlan 记录 mode、segmentRows、segments、bytes、foldUbBytes。Rows 区占 B*align32(M)*4；Partials 区占 B*ceil(M/segmentRows)*64。每个局部 writer 预留 power-of-two capacity>=segmentRows 的 high/low/work，共 14*capacity+64 bytes。最终 reducer 用 8192B high/low 输入、6144B work、32B 输出，共 14368B，与 R12 最终归约的预算相同。

GM 保留完整 similarity，后接所选归约区域。Stream 的 C 槽不变；S=1 直接写最终区域；S>1 在最终区域后保存原有 N 分片 rowMax。全部偏移/大小/UB 在 host 检查。流水的 Wait/End 与 MTE2_S 不改，所有记录写回在消费者前经过现有全核屏障。

## 联合选择与对照

为每个 geometry/family/split 比较 rows 和 partials，计入 producer/merge 的额外局部 Fold 和最终读取/求和工作；每个 geometry/family 仍只保留最低分，候选上限 147、SDK 尝试上限不变。保留 R14 rows 成本项，partials 的固定/向量成本是未校准结构代理，不声称预测时间。自动可选回 rows，避免短 M 或已有足够 batch 并行时白白增加分布式树开销。

内部参数 ReductionPolicy=Auto/Rows/Partials。线上 Auto，默认 Small 先选。显式 rows/partials 不用于 Small；本地强制 Small 与显式 sum policy 冲突要报错。强制 gm/stream + sum auto 保持原 rows 控制；强制 pipeline + sum auto 比较两种。Auto+固定 tile 保持旧 GM 控制，显式 sum policy 可覆盖其归约。CANN_SUM_MODE 仅本地 runner 读取。

执行 metadata 保持 schema 3，新增 requested_sum/reduction；联合 planner v3/cost-model joint-work-v2。无新字段的历史报告按旧 rows 验证；partial 报告必须携带完整几何/字节预算，不能把 records 当作 rowMax。GM similarity 诊断继续可用。

## 验证与边界

真实 host C++ 测存储布局、UB、溢出、自动两种选择、强制 policy 和 Rows 旧评分；真实 device helper CPU 替身测 GM/Stream S1/S>1、双槽、尾部、N Max-before-Sum 反例和 producer 无 Scalar 读取。独立 FP64/long-double golden 覆盖跨段/跨 chunk 大数抵消、近零和随机指数；验证可重复输出、完整记录唯一所有权和总 UB。主机分派仍恰好一次 launch，SDK 拒绝前不分配。

CANN 编译、真实 Cube 精度与硬件同步由下一次线上 S11 验证；不将主机通过写成设备通过。接口依据：[CANN 9.0 DataCopyPad](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0265.html)，GM stride 单位 byte、UB stride 单位 32B，blockCount<=4095。本轮每次最多 128 条 32B 块。
