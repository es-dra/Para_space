# Phase1 Fitting Dynamics Entrypoints

This directory contains training, validation, and analysis scripts for fitting-dynamics research.

## Official Training Entrypoints

| Script | Role | Current status |
|---|---|---|
| `run.py` | scratch conditional INR / LIIF / LTE fitting | official for reduced LIIF Stage B/C-controlled runs |
| `run_siren.py` | scratch SIREN fitting | official for Stage B SIREN baseline |

Official runs should pass an explicit `--save_dir`. Do not rely on legacy defaults such as `results/FittingDynamics/...` for new formal results.

## Official Validation / Analysis Entrypoints

| Script | Role |
|---|---|
| `check_outputs.py` | validate trajectory schema, shapes, summary and analysis readiness |
| `analyze_stage_b_controls.py` | Stage B trajectory controls |
| `analyze_stage_b_trajectory_audit.py` | Stage B-prime parameter trajectory-object audit |
| `analyze_stage_b_response_audit.py` | Stage B function-response trajectory audit |
| `analyze_stage_b_liif_unit_audit.py` | LIIF internal-unit diagnostic screen |
| `analyze_stage_c_geometry_response.py` | Stage C geometry-response pilot probe |
| `analyze_stage_c_failure_audit.py` | reviewer-style Stage C failure audit |
| `analyze_stage_c_response_object_audit.py` | response-object audit for bird/head failure |
| `analyze_stage_c_patch_unit_gate.py` | LIIF-aware LR-cell patch-unit gate |
| `analyze_stage_c_lr_cell_feature_trajectory.py` | C-minimal LR-cell encoder-feature trajectory diagnostic |
| `analyze_stage_c_controlled_self_similarity.py` | controlled self-similarity sanity gate |
| `generate_controlled_self_similarity_images.py` | generate controlled diagnostic inputs |

## Legacy / Blocked Entrypoints

| Script | Status | Reason |
|---|---|---|
| `aggregate_results.py` | legacy diagnostic | PCA-style summary, not current Stage B main evidence |
| `viz_trajectory.py` | legacy visualization utility | useful for diagnostics, not formal claim evidence by itself |
| `run_finetune.py` | blocked / experimental | pretrained LIIF/LIIF-EQ source, checkpoint and fairness protocol not closed |

## No Temporary Launchers

Do not add one-off launchers, scratch dispatchers, or nohup wrapper scripts. If a repeated formal process is needed, implement a named Python CLI with clear inputs, outputs, and stop conditions.
