# Stage A Research Questions

Historical detailed source: [../../archive/historical_docs/StageA_research_questions.md](../../archive/historical_docs/StageA_research_questions.md).

## Central Question

在任意尺度 INR 超分中，图像局部几何是否能够预测模型参数空间或响应空间中的一部分适配变化？等变性是否会让这种关系更结构化、更可迁移，或更适合用于减少训练？

## Decomposition

1. Observation layer：参数/响应变化能否被可靠测量？
2. Correspondence layer：局部几何相似性是否对应 response/update similarity？
3. Prediction layer：是否存在一部分适配可以由 geometry 和 scale 推断？

## Hypotheses

- H1：局部几何对应。
- H2：尺度条件结构。
- H3：等变性正则化对应关系。
- H4：部分更新可预测。

## Minimum Claim Standard

一个 claim 至少应满足：

- 多 image 或 seed；
- 负对照；
- 明确 response/update object；
- 公平 baseline；
- failure cases；
- 不被 content/coordinate/spatial confound 简单解释。
