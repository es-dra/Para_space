# Claim-Evidence Ledger

本文档管理当前项目允许写、只能弱写、或禁止写的主张。每个 claim 必须绑定证据、反证、允许表述和下一 gate。

Status definitions:

- `supported`：当前证据足以作为项目事实。
- `supported_for_pilot`：足以支撑下一阶段探索，不足以作为论文强 claim。
- `mixed_or_weak`：有线索但反证/限制强。
- `negative_supported`：当前证据支持一个反结论或停止条件。
- `blocked`：当前前置条件未满足。

| ID | Stage | Claim | Status | Allowed wording | Evidence | Counterevidence / Caveat | Forbidden wording | Next gate |
|---|---|---|---|---|---|---|---|---|
| A1 | A | 研究问题、H1-H4、非目标和负对照足以支撑 pilot | supported_for_pilot | Stage A establishes a testable pilot-level research program. | [stages/stage_a/closeout.md](stages/stage_a/closeout.md) | 文献和 bib 仍非论文级完整 | Stage A proves the hypothesis. | 补论文级 related work / baseline 表 |
| A2 | A | 文献地图覆盖必要方向但未闭合 | supported | The current literature map is sufficient for research planning, not final paper writing. | [stages/stage_a/closeout.md](stages/stage_a/closeout.md), `paper/refs/main.bib` | LIIF-EQ / SE-INR 来源与 checkpoint 仍未核清 | Related work is complete. | bib 核查与直接相关工作表 |
| B1 | B | scratch fitting-dynamics 平台可靠可用 | supported | Stage B validates the trajectory recording, checking, aggregation, and control-analysis platform. | [stages/stage_b/closeout.md](stages/stage_b/closeout.md), [evidence/run_manifest_stage_b.md](evidence/run_manifest_stage_b.md) | 只说明平台，不说明 H1 | Stage B proves geometry-response correspondence. | 无；可封版 |
| B2 | B | PCA/PC1 不能作为机制证据 | negative_supported | PCA is retained only as a diagnostic; LIIF PC1 is largely explained by controls. | [stages/stage_b/closeout.md](stages/stage_b/closeout.md) | SIREN PC1 与 permuted 有差距，但不足以支撑主线 | High PC1 reveals meaningful parameter-space law. | 不再围绕 PCA 建 claim |
| B3 | B | reduced LIIF decoder-side movement motivates response-space probing | supported_for_pilot | Decoder-dominant update statistics motivate Stage C response/feature probes. | [stages/stage_b/closeout.md](stages/stage_b/closeout.md) | 轨迹统计不是局部几何证据 | Decoder update ratio proves local geometry mechanism. | Stage C response/unit gate |
| C1 | C | 部分图像/高内容强度 patch 有局部线索 | mixed_or_weak | Stage C pilot shows image- and content-dependent local signals, strongest in baby and woman. | [stages/stage_c/current_status.md](stages/stage_c/current_status.md) | bird/head failure；Spearman 弱；butterfly 弱；woman 单 seed | Cross-image stable law has been found. | geometry-content dissociation 或 model-internal unit gate |
| C2 | C | 当前 patch matching / raw output trajectory 操作化未通过主证据标准 | negative_supported | The current operationalization is insufficient as a main evidence generator. | [stages/stage_c/current_status.md](stages/stage_c/current_status.md), [decisions/2026-05-08_stage_c_operationalization_failure.md](decisions/2026-05-08_stage_c_operationalization_failure.md) | 不证明 H1 整体错误 | The research direction is disproved. | 重新定义 response/unit 或降级 |
| C3 | C | controlled self-similarity 证明 probe/wiring 可工作 | supported_for_pilot | Controlled self-similarity narrows the failure away from broken response reconstruction. | [stages/stage_c/current_status.md](stages/stage_c/current_status.md), [evidence/analysis_manifest_stage_c.md](evidence/analysis_manifest_stage_c.md) | content-only 接近 geometry；seed123 严格 gate 未过 | Controlled periodic proves geometry mechanism. | geometry-vs-content dissociation |
| X1 | D | 等变性增强 correspondence | blocked | Not currently supported; LIIF-EQ/pretrained path is blocked pending source/fairness checks and H1 gate. | [research_contract.md](research_contract.md) | H1 操作化未过；LIIF-EQ 来源/ckpt 未核清 | Equivariance improves fitting dynamics. | H1 gate + LIIF-EQ provenance/fairness audit |
| X2 | D | update prediction / training reduction | blocked | Not currently supported. | [research_contract.md](research_contract.md) | 没有 predictor、adapter/modulation 或 residual training evidence | Current method reduces adaptation training. | H1-H3 evidence + fair training baseline |
| PAPER-OLD-1 | Paper old draft | Group transformations trace low-dimensional INR parameter orbits | blocked | Historical draft only; not a current claim. | [../paper/README.md](../paper/README.md), [archive/historical_docs/Phase1_design.md](archive/historical_docs/Phase1_design.md) | Current Stage B PCA controls do not support a mechanism claim; Stage C shifted to response probing and remains unresolved | Signal transformations induce low-dimensional orbital manifolds in INR weight space. | Full rewrite after new evidence, if ever |
| PAPER-OLD-2 | Paper old draft | Lie algebra homomorphism / symmetry-error theory is empirically supported | blocked | No current support. | [../paper/README.md](../paper/README.md) | No validated experiments or proofs in the current evidence chain | We prove INR parameter tangents form a Lie algebra representation. | New theory + validated experiments |
| PAPER-OLD-3 | Paper old draft | COIN++ low-dimensionality is explained by parameter orbits | blocked | Historical hypothesis only. | [../paper/README.md](../paper/README.md), [stages/stage_a/closeout.md](stages/stage_a/closeout.md) | Current project has not tested COIN++ modulation space | This explains why COIN++ compresses well. | Separate literature/theory/evidence program |
| PAPER-OLD-4 | Paper old draft | Tangent/JVP initialization reduces adaptation steps by >=30% | blocked | Not currently supported. | [../paper/README.md](../paper/README.md), [research_contract.md](research_contract.md) | No current predictor, adapter, residual-training, or fair speed baseline | Current method reduces inner-loop convergence by >=30%. | H1-H3 evidence + fair adaptation experiment |

## Update Rule

新增或修改 claim 时必须同时更新：

1. `Evidence` artifact；
2. `Counterevidence / Caveat`；
3. `Allowed wording`；
4. `Forbidden wording`；
5. 对应 decision log 或 stage closeout。
