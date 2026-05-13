# Current Context Capsule

本文档是低 token 恢复入口，不替代 [project_index.md](project_index.md)、[claims_ledger.md](claims_ledger.md)、daily memory 或原始结果。它只回答：当前主线是什么、什么已经可靠、什么禁止、下一步允许做什么。

## Current Research Question

主问题仍是：

> arbitrary-scale INR super-resolution 中，单图 fitting/adaptation dynamics 是否存在可测、可解释、非平凡的参数或响应轨迹结构；如果存在，它是否能被图像内容、局部几何、尺度或旋转解释？

当前主线不是训练加速、不是等变性比较、也不是证明自然图局部几何机制。当前主线是决定已有 probing / failure-audit 证据是否足以形成收缩论文出口，或是否转向一个更基础、更可控的 fitting-order / spectral-dynamics 节点。

## Current Evidence State

- Stage A：pilot 级 research contract 完成；不是论文级 related work。
- Stage B：scratch fitting-dynamics 平台可靠；只支持平台可用，不支持机制 claim。
- Stage B-prime：参数轨迹有 non-random directed drift，但 LIIF decoder 结构接近 endpoint/update-set controls 且 layer-scale dominated；不能写成稳定低维参数规律。
- Stage B-response：response trajectory 可重建；SIREN 较 line-like，LIIF 低秩但 encoder/decoder/interaction 不可简单分离；不能重开 Stage C。
- LIIF internal-unit：旧 256-query screen 只作 diagnostic；旧 artifact 有 `gen_feat()` 输入归一化 caveat，数值不能作为 formal evidence。
- Stage C natural-image route：patch matching、response-object repair、LR-cell query-support、C-minimal feature trajectory都未通过 clean gate。
- C-minimal：baby/woman `Delta R2_cv` 为小正，bird/head 为负，四 run median 约 `0.0039`；verdict 为 mixed/inconclusive。

## Trusted Claims

- 可以说：当前项目已经建立了一个可复现的 fitting-dynamics probing / failure-audit 工具链。
- 可以说：现有自然图 geometry-response 操作化未通过主证据标准。
- 可以说：多条更合理的 LIIF object 尝试仍未形成 clean positive gate。
- 可以说：当前证据支持暂停自然图 Stage C 修补路线。

## Blocked Or Retired Claims

- `PCA/PC1`：`diagnostic-only`；不能作为机制证据。
- natural-image patch nearest-neighbor geometry-response：`blocked-for-claim`；不能继续开放式 descriptor/rerank 修补。
- C-minimal LR-cell geometry increment：`diagnostic-only`；不能写成 LIIF feature dynamics 被 geometry 解释。
- LIIF-EQ / equivariance comparison：`blocked`；等待 H1 或新的 controlled mechanism gate。
- update prediction / training acceleration：`blocked`；等待 H1-H3 和 fair adaptation baseline。

## Current Route Decision

连续多个正式 gate 为 mixed/negative：

1. Stage C operationalization failed.
2. Stage B-prime did not pass mechanism gate.
3. Stage B-response did not provide a clean reopen gate.
4. LIIF-unit screen did not fix response object.
5. C-minimal was mixed/inconclusive.

按当前工作流，默认动作是暂停同一路线，不自动设计第三个相邻 descriptor / metric / rerank 实验。

2026-05-10 route decision: return to the mainline by contracting near-term work
to fitting-dynamics probing / failure-audit. A controlled mechanism test is an
admissible mechanism-recovery branch, but it is not a continuation of natural
image Stage C.

2026-05-11 planning update: the most coherent next scientific reframing is to
test whether SIREN/LIIF fitting order has measurable spectral and spatial
organization before reviving any local geometry-response mechanism claim. This
is only a proposed next node, not evidence yet.

## Allowed Next Nodes

### Option A - Probing / Failure-Audit Paper Framing

目标：把近期贡献收缩为方法学和诊断框架。

允许动作：

- 整理 claim-evidence table；
- 写 paper outline；
- 提炼 negative gates 和 failure modes；
- 做 bib / related work verification；
- 明确哪些 artifact 可复现、哪些只是 diagnostic。

成功条件：论文故事不声称机制成立，而是清楚说明 probing framework、negative results 和 caveat。

### Option B - Controlled Mechanism Test

目标：在小 controlled setting 中独立操控 content、coordinate、orientation、anisotropy 和 residual difficulty，再测试固定 feature trajectory object。

允许动作：

- 先写 Node Contract 和 experiment card；
- 先定义 positive/negative controls 和 Metric Validity Gate；
- 不复用自然图 patch nearest-neighbor 作为主证据；
- 若 controlled 也失败，停止 geometry mechanism route。

成功条件：controlled positive 能按预期响应，negative/trivial baseline 不能刷高，且指标解释清楚。

### Option C - Stage B-Spectral Audit

目标：不再问 patch geometry 是否预测 response 方向，而是先测量
SIREN/LIIF 在单图 scratch fitting 中的 residual spectral acquisition order
是否稳定、是否有空间组织、是否超过平凡 controls。

允许动作：

- 先写 Node Contract / Metric Validity Gate；
- 只读使用现有 Stage B snapshots；
- 主指标定义为局部 band acquisition order 的空间组织和 control increment；
- 不把全图 low-frequency ratio、PC1 或单图可视化作为主证据；
- 不把通过此节点自动回流为 Stage C、等变性或训练加速支持。

成功条件：局部 spectral fitting order 对空间 shuffle、频带标签 shuffle、
phase/random target 或简单 content baselines 有稳定增量，并能报告
baby 多 seed 与 bird/butterfly 的正负边界。

## Forbidden Next Actions

- 不继续自然图 Stage C descriptor / rerank / response-object 修补。
- 不因为 baby/woman 小正增量扩展到全部 reduced LIIF runs。
- 不启动 pretrained LIIF / LIIF-EQ。
- 不启动 equivariance comparison。
- 不做 update prediction 或 training acceleration。
- 不把工程测试通过、文档完整或 JSON 产物数量写成科学正进展。
- 不把 Stage B-spectral audit 的设计本身写成结果；只有通过 controls 后才允许升级 claim。

## Canonical Artifacts

- Project index: [project_index.md](project_index.md)
- Research contract: [research_contract.md](research_contract.md)
- Claim ledger: [claims_ledger.md](claims_ledger.md)
- Experiment registry: [evidence/experiment_registry.md](evidence/experiment_registry.md)
- Stage B-prime: [stages/stage_b/b_prime_trajectory_audit.md](stages/stage_b/b_prime_trajectory_audit.md)
- Stage B-response: [stages/stage_b/response_trajectory_audit.md](stages/stage_b/response_trajectory_audit.md)
- LIIF unit screen: [stages/stage_b/liif_internal_unit_screen.md](stages/stage_b/liif_internal_unit_screen.md)
- Stage C status: [stages/stage_c/current_status.md](stages/stage_c/current_status.md)
- C-minimal decision: [decisions/2026-05-10_stage_c_lr_cell_feature_trajectory_result.md](decisions/2026-05-10_stage_c_lr_cell_feature_trajectory_result.md)
- Mainline return decision: [decisions/2026-05-10_return_to_mainline.md](decisions/2026-05-10_return_to_mainline.md)

## Next Gate

下一步不是默认实验，而是路线选择：

1. 选择 Option A：进入 failure-audit / probing paper framing；
2. 选择 Option B：先写 controlled mechanism test 的 Node Contract；
3. 选择 Option C：先写 Stage B-spectral audit 的 Node Contract；
4. 若都不选，暂停科研扩展，只做清理或提交。
