# 2026-05-09 Stage B-response Audit Result

## Question

函数响应空间审计是否提供了比参数空间更干净的机制 gate，并足以重开 Stage C geometry-response 路线？

## Evidence Considered

- [stages/stage_b/response_trajectory_audit.md](../stages/stage_b/response_trajectory_audit.md)
- `results/FittingDynamics_StageB_diagnostics/stage_b_response_audit_2026-05-09.json`
- `experiments/Phase1_FittingDynamics/analyze_stage_b_response_audit.py`
- `tests/test_stage_b_response_audit.py`

## Decision

不重开 Stage C。

函数响应轨迹是比 flattened parameter PCA 更合理的下一观测对象，但首轮审计仍没有给出 clean positive gate。SIREN response 相对 line-like；reduced LIIF response 低秩但不够直，并且 encoder-only、decoder-only 和 interaction residual 都很大，不能简单解释为 decoder-only response mechanism。

## Rejected Alternatives

- 把 SIREN response line-like 结果外推到 LIIF。
- 把 LIIF response low-rank 写成 geometry-response 机制。
- 把 LIIF hybrid decoder-only ratio 接近或超过 1 解释为 decoder 独立主导。
- 直接重开 patch geometry matching。

## Consequences

- Stage C 继续保持 pilot / failure-audit 状态。
- 若继续主线，应先固定更细的 LIIF function-space unit，例如 decoder input、feature-conditioned query response 或 local cell response，而不是回到开放式 patch descriptor 搜索。
- 若不做更细对象定义，项目应收缩为 probing/failure-audit framework。

## Conditions To Reopen

只有新的固定协议能同时满足以下条件时，才应重开 Stage C：

- response/update object 在 function space 中有明确物理或模型内部含义；
- 该 object 超过 endpoint/random/permuted controls；
- encoder/decoder interaction 能被解释或被控制；
- bird/head failure 被正面处理；
- content/coordinate/spatial confounds 被报告。
