# Experiment Registry

本文档是实验与分析索引。它不是结果表；它记录每类实验回答什么问题、使用哪些输入、产物在哪里、是否允许论文引用。

## Registry Policy

- 每个非平凡实验先有 experiment card，再有 raw result，再有 derived analysis。
- `results/` 与 `Data/` 被 `.gitignore` 忽略，因此必须在此登记路径语义。
- 论文可引用产物必须能追到脚本、输入目录、seed、协议、输出 JSON/表格和 caveat。

## Stage B Experiments

| ID | Question | Input | Script | Output | Evidence class | Status |
|---|---|---|---|---|---|---|
| B-RUN-SIREN | SIREN scratch trajectory 是否可记录/检查 | `Data/Set5/HR/{baby,bird,butterfly}.png` | `experiments/Phase1_FittingDynamics/run_siren.py` | `results/FittingDynamics_StageB/SIREN_*` | raw official runs | complete |
| B-RUN-LIIF | reduced LIIF scratch SR x4 trajectory 是否可记录/检查 | `Data/Set5/HR/{baby,bird,butterfly}.png` | `experiments/Phase1_FittingDynamics/run.py` | `results/FittingDynamics_StageB/LIIF_reduced_*` | raw official runs | complete |
| B-CHECK | trajectory schema/shape/summary/readiness 是否通过 | Stage B result dirs | `check_outputs.py` | stdout checks, documented in Stage B closeout | validation | complete |
| B-CTRL | PC1 与 random/permuted controls 比较 | Stage B original 10 runs | `analyze_stage_b_controls.py` | documented tables | derived diagnostic | complete |
| B-PRIME | 参数/响应轨迹是否存在超过 endpoint drift、step norm schedule、优化平滑性和参数化伪影的非平凡结构 | Stage B trajectory files | `analyze_stage_b_trajectory_audit.py` | `results/FittingDynamics_StageB_diagnostics/stage_b_prime_trajectory_audit_2026-05-09.json` | trajectory-object audit / negative gate | complete |
| B-RESPONSE | 输出函数响应轨迹是否比参数轨迹更适合作为下一对象 | Stage B trajectory files | `analyze_stage_b_response_audit.py` | `results/FittingDynamics_StageB_diagnostics/stage_b_response_audit_2026-05-09.json` | function-space audit / mixed gate | complete |
| B-LIIF-UNIT | LIIF decoder input / local decoder output 是否提供更干净的 function-space unit | Stage B LIIF trajectory files | `analyze_stage_b_liif_unit_audit.py` | `results/FittingDynamics_StageB_diagnostics/stage_b_liif_unit_audit_screen_2026-05-09.json` | diagnostic screen, not formal gate | complete |

See [run_manifest_stage_b.md](run_manifest_stage_b.md).

## Stage C Natural-Image Diagnostics

| ID | Question | Input | Script | Output | Evidence class | Status |
|---|---|---|---|---|---|---|
| C-GEOM | geometry-response pilot 是否有局部信号 | Stage B LIIF runs | `analyze_stage_c_geometry_response.py` | stdout/table and follow-up JSONs | pilot diagnostic | complete |
| C-FAIL | bird/head failure 是否可定位 | bird/head + baby/woman guardrails | `analyze_stage_c_failure_audit.py` | `results/FittingDynamics_StageC_diagnostics/*` | failure audit | complete |
| C-RESP | response-object 修补是否解释 failure | bird/head + baby/woman guardrails | `analyze_stage_c_response_object_audit.py` | `response_object_audit_2026-05-08.json` | negative diagnostic | complete |
| C-UNIT | LIIF-aware LR-cell unit 是否修复 failure | bird/head + baby/woman guardrails | `analyze_stage_c_patch_unit_gate.py` | `patch_unit_gate_lr_cell_2026-05-08.json` | negative gate | complete |
| C-MINIMAL | LR-cell encoder feature trajectory complexity 是否能被 local geometry 在 coordinate+content 之外增量解释 | baby123, woman42, bird42, head42 reduced LIIF | `analyze_stage_c_lr_cell_feature_trajectory.py` | `results/FittingDynamics_StageB_diagnostics/stage_c_lr_cell_feature_trajectory_c_minimal_2026-05-10.json` | C-minimal diagnostic gate / mixed | complete |

See [analysis_manifest_stage_c.md](analysis_manifest_stage_c.md).

## Stage C Controlled Diagnostics

| ID | Question | Input | Script | Output | Evidence class | Status |
|---|---|---|---|---|---|---|
| C-CSS-DATA | 受控 periodic/nonperiodic 数据是否可生成 | seed 0, 48x48, tile 12 | `generate_controlled_self_similarity_images.py` | `Data/ControlledSelfSimilarity/` | controlled input | complete |
| C-CSS-SMOKE | probe/wiring 在 synthetic response 已知正例上是否工作 | controlled images | `analyze_stage_c_controlled_self_similarity.py --synthetic_response_smoke` | `controlled_self_similarity_synthetic_gate_2026-05-08.json` | positive sanity check | complete |
| C-CSS-FIT | fitted reduced LIIF 是否检出 exact repeats 且不误报 nonrepeat | controlled runs, seeds 42/123/456 | `run.py`, `analyze_stage_c_controlled_self_similarity.py` | `controlled_self_similarity_fitted_gate_2026-05-08.json` | mixed / content-confounded gate | complete |

## Blocked / Deferred

| ID | Path | Reason |
|---|---|---|
| PRETRAINED-LIIF-EQ | `run_finetune.py`, pretrained wrappers | source/checkpoint/fairness/buffer policy not closed |
| STAGE-D-UPDATE | adapter/modulation/update prediction | H1 operationalization not passed |
| EQUIVARIANCE-COMP | LIIF vs LIIF-EQ matched comparison | blocked by H1 gate and LIIF-EQ provenance |
