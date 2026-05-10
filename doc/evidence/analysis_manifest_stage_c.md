# Stage C Analysis Manifest

本文档索引 Stage C pilot / diagnostic / gate 产物。所有条目都是诊断证据，不是论文最终图表。

## Natural-Image Failure Audit

| Artifact | Script | Runs | Primary response | Purpose | Interpretation |
|---|---|---|---|---|---|
| `results/FittingDynamics_StageC_diagnostics/bird_head_patch_failure_diagnostic_2026-05-07.json` | `analyze_stage_c_failure_audit.py` | bird/head seed42 | `trajectory_delta`, feature/Jacobian auxiliary | patch-level failure localization | bird/head geometry top-1 often response-far |
| `results/FittingDynamics_StageC_diagnostics/baby123_woman42_patch_failure_reference_2026-05-07.json` | `analyze_stage_c_failure_audit.py` | baby123/woman42 | same | positive guardrail comparison | positive references have lower top1 response percentile |
| `results/FittingDynamics_StageC_diagnostics/candidate_attribute_gate_2026-05-08.json` | `analyze_stage_c_failure_audit.py` | bird/head + baby123/woman42 | `trajectory_delta` | response-blind candidate gate | failed; context/rerank did not use oracle headroom |
| `results/FittingDynamics_StageC_diagnostics/response_object_audit_2026-05-08.json` | `analyze_stage_c_response_object_audit.py` | bird/head + baby123/woman42 | output response variants | test amplitude/time aggregation explanations | failed to explain bird/head |
| `results/FittingDynamics_StageC_diagnostics/patch_unit_gate_lr_cell_2026-05-08.json` | `analyze_stage_c_patch_unit_gate.py` | bird/head + baby123/woman42 | `trajectory_delta` | LIIF-aware LR-cell unit gate | failed; content/coordinate controls often stronger |
| `results/FittingDynamics_StageB_diagnostics/stage_c_lr_cell_feature_trajectory_c_minimal_2026-05-10.json` | `analyze_stage_c_lr_cell_feature_trajectory.py` | baby123/woman42/bird42/head42 | LR-cell encoder feature trajectory `[T,32]` | test geometry increment over coordinate+content with blocked spatial CV | mixed/inconclusive; baby/woman positive, bird/head negative |

## Oracle / Rerank Diagnostics

| Artifact | Purpose | Caveat |
|---|---|---|
| `topk_oracle_rgb_grad_2026-05-08.json` | quantify response-label oracle headroom inside geometry top-k | oracle is not a usable predictor |
| `topk_oracle_rgb_grad_context_2026-05-08.json` | compare context candidate set oracle | oracle is diagnostic only |
| `topk_rerank_rgb_grad_to_context_2026-05-08.json` | test response-blind context rerank | failed / no robust output gain |

## Controlled Self-Similarity

| Artifact | Purpose | Result |
|---|---|---|
| `Data/ControlledSelfSimilarity/controlled_self_similarity_metadata.json` | controlled input metadata | periodic exact repeat + nonperiodic texture control |
| `results/FittingDynamics_StageC_diagnostics/controlled_self_similarity_synthetic_gate_2026-05-08.json` | no-training synthetic response smoke | passed |
| `results/FittingDynamics_StageC_controlled_diagnostics/controlled_self_similarity_fitted_gate_2026-05-08.json` | fitted reduced LIIF controlled gate over 3 periodic + 3 nonrepeat seeds | failed strict gate due to seed stability + content confound |
| `results/FittingDynamics_StageC_controlled_diagnostics/controlled_self_similarity_fitted_seed42_123_plus_nonrepeat_hr_geometry_2026-05-08.json` | HR geometry source check | did not rescue seed123 strict failure |

## Current Negative Finding

The current Stage C operationalization:

```text
local patch appearance / geometry nearest neighbor
-> raw output trajectory response similarity
```

does not meet the main evidence standard. This is not proof that H1 is false; it is proof that the current measurement/matching route is insufficient.

The 2026-05-10 C-minimal diagnostic moved from HR patch response matching to
LIIF LR-cell encoder feature trajectories. It still did not pass the frozen
reopen gate: strict geometry `Delta R2_cv` was positive for baby/woman but
negative for bird/head, with four-run median near zero.
