# 2026-05-09 LIIF Internal-Unit Screen

## Question

LIIF decoder input、local decoder output 或 sampled local output 是否提供比 raw output response 更干净的 function-space object，从而足以重开 Stage C？

## Evidence Considered

- [stages/stage_b/liif_internal_unit_screen.md](../stages/stage_b/liif_internal_unit_screen.md)
- `results/FittingDynamics_StageB_diagnostics/stage_b_liif_unit_audit_screen_2026-05-09.json`
- `experiments/Phase1_FittingDynamics/analyze_stage_b_liif_unit_audit.py`
- `tests/test_stage_b_liif_unit_audit.py`

## Decision

不重开 Stage C。

256-query diagnostic screen 显示 `decoder_input` 与 `encoder_feature_lr` 的 trajectory diagnostics 几乎一致；进入 decoder/local ensemble 后，`local_decoder_output` 和 `sampled_output` 更弯、更高秩。该结果没有修复 LIIF response-object 问题。

## Rejected Alternatives

- 把 decoder input 当成已经解决的 response object。
- 把 local decoder output 当成 geometry-response clean gate。
- 在 full 1024-query audit 尚未优化实现前继续长时间硬跑。

## Consequences

- 当前 LIIF internal-unit route 只能作为 diagnostic screen。
- 若继续，需要先优化实现，再决定是否 formalize 为 gate。
- 更重要的科研决策是：继续搜索更细 LIIF unit，还是把近期贡献降级为 probing/failure-audit framework。

## Conditions To Reopen

只有在更高效实现下，固定的 LIIF unit 能稳定超过 random/permuted controls，并且能解释或规避 encoder/decoder interaction 与 bird/head failure，才应重开 Stage C。
