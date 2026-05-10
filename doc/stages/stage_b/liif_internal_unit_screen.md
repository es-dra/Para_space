# Stage B LIIF Internal-Unit Screen

## Status

Complete as a diagnostic screen, not a full formal gate.

This screen tests whether a more LIIF-specific function-space unit is cleaner
than raw output response. It uses a deterministic subset of 256 HR query points
and 16 controls to avoid treating a slow high-dimensional diagnostic as
progress.

## Protocol

Script:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/Phase1_FittingDynamics/analyze_stage_b_liif_unit_audit.py \
  --results_dir results/FittingDynamics_StageB \
  --data_root Data \
  --device cuda \
  --n_controls 16 \
  --seed 0 \
  --max_queries 256 \
  --output results/FittingDynamics_StageB_diagnostics/stage_b_liif_unit_audit_screen_2026-05-09.json \
  --format table
```

Artifact:

- `results/FittingDynamics_StageB_diagnostics/stage_b_liif_unit_audit_screen_2026-05-09.json`

Important caveat: the 2026-05-09 artifact was generated before the script's
explicit LIIF input normalization fix. Its qualitative structure signal can
motivate C-minimal, but its numeric values must not be upgraded to formal
evidence unless the screen is rerun with normalized `gen_feat()` input.

Earlier attempted full-size settings with `--max_queries 1024 --n_controls 32`
were stopped because they ran for several minutes without producing an artifact.
This is treated as an implementation-efficiency issue, not a scientific result.

## Units

- `encoder_feature_lr`: LR encoder feature field after `gen_feat`;
- `decoder_input`: query-conditioned local-ensemble decoder inputs;
- `local_decoder_output`: per-candidate decoder outputs before local-ensemble sum;
- `sampled_output`: final aggregated output over the same sampled query set.

## Main Results

Mean over reduced LIIF runs:

| unit | response PC1 % | endpoint-line RMS | permuted-line RMS | random-walk line RMS | straightness | orthogonal energy | update effective rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| encoder feature LR | 65.15 | 0.344 | 0.386 | 0.213 | 0.470 | 0.504 | 1.99 |
| decoder input | 65.09 | 0.343 | 0.333 | 0.213 | 0.469 | 0.503 | 1.98 |
| local decoder output | 41.62 | 0.607 | 0.523 | 0.370 | 0.280 | 0.770 | 4.06 |
| sampled output | 44.80 | 0.574 | 0.535 | 0.334 | 0.299 | 0.711 | 3.29 |

## Interpretation

`encoder_feature_lr` and `decoder_input` are almost identical under these
trajectory diagnostics. This means the decoder input is mostly carrying encoder
feature dynamics plus fixed coordinate/cell structure; it does not define a
cleaner object by itself.

After the decoder and local ensemble, the trajectory becomes less line-like,
more orthogonal, and higher-rank. This is consistent with the Stage B-response
finding that LIIF output response is not a clean endpoint-line mechanism.

## Decision

Do not reopen Stage C geometry matching from this screen.

Allowed wording:

> LIIF internal-unit screening suggests decoder-input dynamics closely tracks
> encoder-feature dynamics, while decoder outputs and sampled outputs are more
> curved and higher-rank.

Forbidden wording:

> Decoder input fixes the response-object problem.

> Local decoder output provides a clean geometry-response gate.

## Next

If continuing this route, optimize the implementation before running a larger
formal unit audit. Scientifically, the more important issue is now whether the
project should keep searching for a cleaner LIIF unit or downgrade to a
probing/failure-audit contribution.
