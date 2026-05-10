# Stage B-response Function-Space Audit

## Status

Complete as a first function-space trajectory audit.

This audit moves one level above flattened parameters. It asks whether saved
Stage-B model snapshots produce output-response trajectories with structure
that could justify later image-content or geometry explanations.

## Protocol

Script:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/Phase1_FittingDynamics/analyze_stage_b_response_audit.py \
  --results_dir results/FittingDynamics_StageB \
  --data_root Data \
  --device cuda \
  --n_controls 32 \
  --seed 0 \
  --output results/FittingDynamics_StageB_diagnostics/stage_b_response_audit_2026-05-09.json \
  --format table
```

Artifact:

- `results/FittingDynamics_StageB_diagnostics/stage_b_response_audit_2026-05-09.json`

Validation:

- final PSNR reconstructed from raw snapshots matches run summaries with near-zero error;
- JSON artifact parses successfully.

Metrics:

- response snapshot PC1;
- endpoint-line residual;
- response update effective rank;
- straightness and orthogonal update energy;
- permuted response-update control;
- norm-matched response random-walk control;
- LIIF encoder-only / decoder-only / interaction final-response decomposition.

## Main Results

Mean over reduced LIIF runs:

| family | response PC1 % | endpoint-line RMS | permuted-line RMS | random-walk line RMS | straightness | orthogonal energy | response update effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| LIIF | 67.72 | 0.271 | 0.362 | 0.190 | 0.420 | 0.413 | 1.98 |

Mean over SIREN runs:

| family | response PC1 % | endpoint-line RMS | permuted-line RMS | random-walk line RMS | straightness | orthogonal energy | response update effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIREN | 86.51 | 0.152 | 0.254 | 0.212 | 0.614 | 0.445 | 2.82 |

LIIF hybrid final-response decomposition, mean over runs:

| term | norm ratio vs full final response delta |
|---|---:|
| decoder-only | 1.06 |
| encoder-only | 1.01 |
| interaction residual | 1.16 |

The encoder-only and decoder-only final responses are both large, and their
interaction residual is also large. This means LIIF response change is not
cleanly separable into an independent encoder part plus independent decoder
part under this simple hybrid intervention.

## Interpretation

Function-space results are mixed and more nuanced than parameter-space results.

- SIREN response trajectories are fairly line-like relative to both permuted
  and random-walk controls.
- LIIF response trajectories are less line-like than SIREN. They are more
  line-like than permuted response-update controls, but less line-like than
  norm-matched response random walks under this metric.
- LIIF response update effective rank is low, but the trajectory is not simply
  a clean endpoint line.
- LIIF final response change is strongly nonlinear in the encoder/decoder
  split: decoder-only, encoder-only and interaction terms are all large.

## Decision

Stage B-response does not provide a clean positive gate for reopening Stage C
geometry matching. It does, however, show that function-space dynamics are a
more meaningful next object than flattened parameter-space PCA.

Allowed wording:

> Function-space response trajectories are measurable and reconstruct correctly;
> SIREN responses are relatively line-like, while reduced LIIF responses show
> low-rank but non-separable encoder/decoder dynamics.

Forbidden wording:

> Response-space audit proves local geometry can explain adaptation.

> LIIF response change is decoder-only.

## Next

If continuing the main research direction, the next object should be a more
defensible LIIF function-space unit, such as decoder input / feature-conditioned
query response, not open-ended patch geometry matching.

If the project prioritizes near-term publishability, the safer contribution is
a probing and failure-audit framework with explicit negative gates.
