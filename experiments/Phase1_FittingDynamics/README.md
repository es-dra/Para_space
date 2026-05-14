# Phase1 Fitting Dynamics Entrypoints

This directory contains the retained scratch training and validation entrypoints
for the whiteboard restart.

## Official Training Entrypoints

| Script | Role | Current status |
|---|---|---|
| `run.py` | scratch conditional INR / LIIF / LTE fitting | retained scratch conditional baseline |
| `run_siren.py` | scratch SIREN fitting | retained scratch SIREN baseline |

Formal runs should pass an explicit `--save_dir` and record command, seed,
config, device, and output path.

## Official Validation Entrypoints

| Script | Role |
|---|---|
| `check_outputs.py` | validate trajectory schema, shapes, summary and analysis readiness |

## Removed / Retired Entrypoints

| Removed scripts | Status | Reason |
|---|---|---|
| `aggregate_results.py`, `viz_trajectory.py` | deleted legacy diagnostics | PCA/visualization summaries are not current evidence |
| `run_finetune.py` | deleted blocked path | external-checkpoint route is not active evidence |
| old natural-image repair scripts | deleted retired diagnostics | old repair diagnostics are not active evidence |
| old trajectory/response audit scripts | deleted old audits | old audits are not part of the whiteboard entry |

## No Temporary Launchers

Do not add one-off launchers, scratch dispatchers, or nohup wrapper scripts. If a repeated formal process is needed, implement a named Python CLI with clear inputs, outputs, and stop conditions.
