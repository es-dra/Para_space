# Research Contract

本文档固定当前项目的研究合同：研究什么、不研究什么、每个阶段能证明什么，以及哪些 claim 在当前证据下禁止。它应保持短而稳定；每日进展不要追加到这里。

## Core Question

在任意尺度 INR 超分中，图像局部几何是否能够预测模型参数空间或响应空间中的一部分适配变化？等变性是否会让这种关系更结构化、更可迁移，或更适合用于减少训练？

## Hypothesis Ladder

| ID | 假设 | 当前状态 |
|---|---|---|
| H1 | 局部几何相似 patch 是否对应 response/update similarity | Stage C pilot 中，当前操作化未过主证据标准 |
| H2 | 尺度条件下的 response/update 是否有结构 | 未启动 |
| H3 | 等变性是否正则化 geometry-response/update correspondence | blocked，等待 H1 操作化通过 |
| H4 | 是否能预测部分更新并减少训练 | blocked，等待 H1-H3 足够证据 |

## Stage Boundaries

| Stage | 当前可靠完成物 | 不允许外推 |
|---|---|---|
| A | 问题定义、假设、非目标、负对照、第一轮文献地图 | 论文级 related work、最终 baseline 表 |
| B | scratch fitting-dynamics 平台、schema/summary/readiness checks、trajectory controls | 局部几何机制、等变性、update prediction、training reduction |
| C | pilot / failure audit；局部线索与明确 failure case | 跨图像稳定规律、可用 predictor、机制结论 |
| D | 未启动 | 任何性能或效率 claim |

## Non-Goals

- 不把 PCA/PC1 作为科学机制证据。
- 不以 PSNR 排榜作为主贡献。
- 不预设等变性一定改善参数/响应规律。
- 不在前置 gate 未通过时讨论训练加速。
- 不用 response-label oracle 结果伪装成可用方法。

## Current Forbidden Claims

- “发现了跨图像稳定的 geometry-response law。”
- “PCA 证明参数空间低维结构。”
- “baby/woman 正例证明 Set5 或自然图普遍成立。”
- “bird/head 只是拟合质量差或个别异常。”
- “受控 periodic positive 证明 geometry 机制。”
- “当前证据支持 LIIF-EQ / 等变性比较、update prediction 或训练加速。”

## Evidence Standard

论文级 claim 至少需要：

1. 明确 response/update object；
2. 多 seed 或多图像支持；
3. 负对照通过；
4. 与 content/coordinate/spatial confound 区分；
5. 报告 failure cases；
6. 对应 raw artifacts、analysis command 和 caveat 可追溯。
