# 2026-05-09 Stage B-prime Audit Result

## Question

首轮 Stage B-prime 轨迹本体审计是否足以重开 Stage C geometry-response、等变性或训练加速路线？

## Evidence Considered

- [stages/stage_b/b_prime_trajectory_audit.md](../stages/stage_b/b_prime_trajectory_audit.md)
- `results/FittingDynamics_StageB_diagnostics/stage_b_prime_trajectory_audit_2026-05-09.json`
- `experiments/Phase1_FittingDynamics/analyze_stage_b_trajectory_audit.py`
- `tests/test_stage_b_trajectory_audit.py`

## Decision

不重开 Stage C、等变性或训练加速。

Stage B-prime 显示 reduced LIIF decoder 参数存在非随机 directed drift，但该现象接近 permuted endpoint/update-set control，并且 full-space update energy 高度集中在 decoder 与少数 decoder layers。该结果不足以作为强参数空间机制证据。

## Rejected Alternatives

- 把 LIIF decoder 高 PC1 / 低 endpoint-line residual 写成稳定低维参数规律。
- 用“强于 random walk”作为重开 Stage C 的充分理由。
- 将 decoder-dominant update energy 解释为局部几何机制。
- 直接进入等变性、update prediction 或训练加速。

## Consequences

- Stage B-prime 完成后，当前主线仍不能回到开放式 Stage C descriptor 修补。
- 若继续科研推进，应先重定义更 defensible 的 function/response object，或者把近期贡献降级为 fitting-dynamics probing + failure-audit framework。
- 参数空间 PCA/PC1 继续保留为诊断，不进入主 claim。

## Conditions To Reopen

只有在新的固定协议能证明某个 parameter/function/response object 超过 endpoint drift、layer-scale concentration、content/coordinate confounds，并能解释或正面处理 bird/head failure 时，才应重开 Stage C 或更后续路线。
