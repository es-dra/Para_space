# Phase1 Fitting Dynamics Entrypoints

This directory contains training, validation, and analysis scripts for fitting-dynamics research.

## Official Training Entrypoints

| Script | Role | Current status |
|---|---|---|
| `run.py` | scratch conditional INR / LIIF / LTE fitting | official for reduced LIIF Stage B runs |
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

## Removed / Retired Entrypoints

| Removed scripts | Status | Reason |
|---|---|---|
| `aggregate_results.py`, `viz_trajectory.py` | deleted legacy diagnostics | PCA/visualization summaries are not current evidence |
| `run_finetune.py` | deleted blocked path | pretrained LIIF/LIIF-EQ route is not active evidence |
| `analyze_stage_c_*.py`, `generate_controlled_self_similarity_images.py` | deleted retired diagnostics | natural-image Stage C repair and old controlled self-similarity are stopped; conclusions live in canonical docs |

## No Temporary Launchers

Do not add one-off launchers, scratch dispatchers, or nohup wrapper scripts. If a repeated formal process is needed, implement a named Python CLI with clear inputs, outputs, and stop conditions.
