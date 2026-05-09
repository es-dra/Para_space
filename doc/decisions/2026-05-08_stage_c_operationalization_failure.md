# 2026-05-08 Stage C Operationalization Failure

## Question

当前 Stage C 操作化是否能作为主证据生成器？

Current operationalization:

```text
local patch appearance / geometry nearest neighbor
-> raw output trajectory response similarity
```

## Evidence Considered

- [stages/stage_c/current_status.md](../stages/stage_c/current_status.md)
- [evidence/analysis_manifest_stage_c.md](../evidence/analysis_manifest_stage_c.md)
- [evidence/analysis_manifest_stage_c.md](../evidence/analysis_manifest_stage_c.md)
- `results/FittingDynamics_StageC_diagnostics/candidate_attribute_gate_2026-05-08.json`
- `results/FittingDynamics_StageC_diagnostics/response_object_audit_2026-05-08.json`
- `results/FittingDynamics_StageC_diagnostics/patch_unit_gate_lr_cell_2026-05-08.json`
- `results/FittingDynamics_StageC_controlled_diagnostics/controlled_self_similarity_fitted_gate_2026-05-08.json`

## Decision

当前 Stage C 操作化未通过主证据标准。项目不应继续开放式 descriptor/rerank 小修补，也不应进入等变性、pretrained LIIF/LIIF-EQ、update prediction 或训练加速叙事。

## Rejected Alternatives

- 用更多自然图或 seed 稀释 bird/head failure。
- 把 feature/Jacobian tiny effect 替代 output trajectory failure。
- 把 top-k oracle 作为可用 predictor。
- 把 controlled periodic positive 写成 geometry mechanism。

## Consequences

- Stage C 当前只能写成 pilot / failure audit / negative diagnostic。
- 若继续挽救 H1，应做 geometry-vs-content / coordinate 解耦 gate，或转向 model-internal unit。
- 若不做更重定义，应考虑把贡献降级为 fitting-dynamics probing platform + failure-audit framework。

## Conditions To Reopen

- 预注册的新 gate 能同时：
  - 修复 bird/head primary output response；
  - 保持 baby/woman guardrails；
  - 击败 content/coordinate/spatial controls；
  - 不使用 oracle/response-label predictor；
  - 跨 seed 或受控数据稳定。
