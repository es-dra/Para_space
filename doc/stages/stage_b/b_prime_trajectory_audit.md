# Stage B-prime Trajectory Audit

## Status

Complete as a first trajectory-object audit.

This audit does not establish a publishable parameter-space mechanism. It
clarifies that Stage B trajectories contain non-random directed drift, but the
strongest apparent LIIF decoder structure is close to endpoint/update-set
controls and is heavily layer-scale dominated.

## Protocol

Script:

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_b_trajectory_audit.py \
  --results_dir results/FittingDynamics_StageB \
  --n_controls 32 \
  --seed 0 \
  --output results/FittingDynamics_StageB_diagnostics/stage_b_prime_trajectory_audit_2026-05-09.json \
  --format table
```

Artifact:

- `results/FittingDynamics_StageB_diagnostics/stage_b_prime_trajectory_audit_2026-05-09.json`

Inputs:

- all current `results/FittingDynamics_StageB/` trajectories;
- SIREN full raw/aligned spaces;
- LIIF full, encoder, decoder raw and decoder aligned spaces.

Metrics:

- endpoint-line residuals;
- update decomposition parallel/orthogonal to final displacement;
- update spectrum effective rank;
- layer-wise update energy;
- norm-matched random-walk control;
- permuted-update control.

## Main Results

Mean over LIIF runs:

| space | snapshot PC1 % | endpoint-line RMS | permuted-line RMS | random-walk line RMS | straightness | orthogonal energy | update effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 93.70 | 0.142 | 0.123 | 0.318 | 0.750 | 0.307 | 2.89 |
| encoder | 75.79 | 0.360 | 0.261 | 0.447 | 0.513 | 0.676 | 5.14 |
| decoder aligned | 94.08 | 0.132 | 0.119 | 0.305 | 0.774 | 0.290 | 2.71 |

Mean over SIREN runs:

| space | snapshot PC1 % | endpoint-line RMS | permuted-line RMS | random-walk line RMS | straightness | orthogonal energy | update effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| full aligned | 83.66 | 0.246 | 0.262 | 0.306 | 0.520 | 0.660 | 4.93 |

Layer energy:

- LIIF full-space path energy is dominated by the decoder: approximately `0.947-0.971` of full update energy.
- The largest LIIF decoder layer often contributes `0.493-0.775` of decoder path energy.
- SIREN path energy is dominated by `W_3` at roughly `0.467-0.511`.

Raw/aligned comparison:

- SIREN raw and aligned metrics are identical at reported precision.
- LIIF decoder raw and aligned metrics are identical at reported precision.
- Alignment is not the driver of the observed Stage B-prime conclusions.

## Interpretation

The trajectories are not random walks. LIIF full/decoder spaces are much closer
to their endpoint line than norm-matched random walks.

However, the apparent LIIF decoder line-like structure is close to the
permuted-update control. Therefore the audit does not support a claim that the
coarse training order induces a rich parameter-space law. The safer explanation
is shared update direction / endpoint drift / update-set geometry, amplified by
layer-scale concentration in the decoder.

The encoder and SIREN trajectories are substantially less line-like and have
higher orthogonal energy and higher effective rank. This weakens any broad
claim that scratch fitting dynamics generally follows a simple low-dimensional
parameter trajectory.

## Decision

Stage B-prime does not pass a strong mechanism gate.

Allowed wording:

> Stage B-prime shows non-random directed drift in saved fitting trajectories,
> especially in reduced LIIF decoder parameters, but this structure is close to
> endpoint/update-set controls and is not sufficient evidence for a meaningful
> parameter-space law.

Forbidden wording:

> Stage B-prime proves stable low-dimensional INR parameter dynamics.

> Stage B-prime justifies reopening Stage C geometry matching, equivariance, or
> training acceleration.

## Next

Do not expand Stage C based on this audit alone. Two defensible options remain:

1. redefine the object in function/response space with a fixed, justified
   object before any geometry link; or
2. downgrade the near-term contribution to a fitting-dynamics probing and
   failure-audit framework.
