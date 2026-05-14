# Research Contract

本文档固定白板重启后的研究边界。它只说明研究什么、暂时不研究什么、什么证据才算有效。

## Core Question

在 INR / 任意尺度超分的单图拟合或适配过程中，参数、更新、响应或特征轨迹里是否存在可复现、非平凡、能被图像内容或局部结构解释的规律？

## Hypothesis Ladder

| ID | 假设 | 当前状态 |
|---|---|---|
| H0 | fitting / adaptation dynamics 是否有超过步长、终点漂移、层尺度和优化平滑性的结构 | 白板重启，未建立新证据 |
| H1 | 图像内容或局部几何是否解释一部分 response / update 行为 | 白板重启，作为候选方向 |
| H2 | 尺度条件下的 response / update 是否有稳定结构 | 未启动 |
| H3 | 结构化架构是否让上述关系更清楚 | 暂不启动 |
| H4 | 是否能利用这些规律减少训练或预测更新 | 暂不启动 |

## Active Scope

- 使用 scratch SIREN 和 scratch LIIF 作为基础观测平台。
- 优先做小节点、短预算、强对照实验。
- 新节点必须先说明研究对象、主指标、对照、成功/失败标准和产物位置。

## Non-Goals

- 不把 PCA/PC1 当作机制证据。
- 不把单张图片、单个 seed 或单个可视化当作稳定规律。
- 不以 PSNR 排榜作为主贡献。
- 不默认使用外部 checkpoint、等变路线、更新预测或训练加速路线。
- 不从旧结果中继承正结论。

## Evidence Standard

任何可升级为科研结论的 claim 至少需要：

1. 明确研究对象，例如参数、update、response、feature 或 Jacobian；
2. 说明指标为什么回答主问题；
3. 有正对照和负对照；
4. 排除步长、层尺度、终点漂移、loss schedule、空间位置和内容偶然性等平凡解释；
5. 至少覆盖多 seed 或多图像；
6. 保留 command、seed、配置、输出目录、分析脚本和失败案例。
