# Stage B Closeout

## Status

Stage B is complete as a platform-validation stage. It is not complete as a
scientific analysis of the parameter trajectory object.

## What Stage B Established

- Scratch SIREN and reduced LIIF fitting dynamics can be recorded.
- Result directories pass schema, shape, summary and analysis-readiness checks.
- Aggregation and control analysis can run on official Stage B outputs.
- Reduced LIIF summary records `LIIF_CONFIG_REDUCED`, `sr_scale=4`, and parameter count.
- No pretrained LIIF/LIIF-EQ evidence is used in Stage B.

## Official Matrix

- SIREN scratch：baby seeds 42/123/456，bird seed42，butterfly seed42。
- LIIF reduced scratch SR x4：baby seeds 42/123/456，bird seed42，butterfly seed42。
- Stage C extension but same scratch protocol：head seed42，woman seed42。

Detailed run table: [../../evidence/run_manifest_stage_b.md](../../evidence/run_manifest_stage_b.md).

## Control Summary

LIIF PC1 is close to the permuted-update control, so high PC1 is not a mechanism claim. Observed step cosine is higher than permuted control, so coarse snapshot order has consecutive directionality, but this is a weak diagnostic that can be expected from smooth optimization.

Representative LIIF control numbers:

| run | PC1 % | permuted PC1 % | step cosine | permuted step cosine |
|---|---:|---:|---:|---:|
| baby seed123 | 94.78 | 95.76 | 0.679 | 0.405 |
| baby seed42 | 93.38 | 94.71 | 0.662 | 0.368 |
| baby seed456 | 92.02 | 93.27 | 0.635 | 0.330 |
| bird seed42 | 94.22 | 94.28 | 0.497 | 0.239 |

## What Stage B Did Not Establish

- Non-trivial parameter-space trajectory law beyond endpoint drift, step-norm schedule, smooth optimization, and parameterization artifacts.
- Local geometry-to-response/update correspondence.
- Equivariance mechanism.
- Training acceleration or update prediction.
- PCA/PC1 as scientific mechanism evidence.

## Open Gap

The first Stage B-prime trajectory-object audit is complete:
[b_prime_trajectory_audit.md](b_prime_trajectory_audit.md).

The first Stage B-response function-space audit is also complete:
[response_trajectory_audit.md](response_trajectory_audit.md).

It shows non-random directed drift, especially in reduced LIIF decoder
parameters, but the strongest apparent structure is close to
endpoint/update-set controls and is layer-scale dominated. This does not pass a
strong mechanism gate.

The response audit shows that SIREN output responses are relatively line-like,
while reduced LIIF output responses are low-rank but not cleanly separable into
encoder-only and decoder-only effects.

The audit tested whether the saved snapshot/update sequence has structure
beyond simple baselines such as endpoint-linearity, norm-matched random walk,
permuted updates, and layer-wise scale effects.

Minimum audit items:

- endpoint-linearity and path residuals;
- update decomposition parallel and orthogonal to final displacement;
- update covariance / effective rank;
- layer-wise update energy;
- full / encoder / decoder comparison;
- raw / aligned comparison.

## Canonical Evidence

- [../../evidence/run_manifest_stage_b.md](../../evidence/run_manifest_stage_b.md)
- Historical long log: [../../archive/historical_docs/StageB_observation_log.md](../../archive/historical_docs/StageB_observation_log.md)
- Archived concise stage notes: [../../archive/stage_summaries_2026-05-09/](../../archive/stage_summaries_2026-05-09/)

## Allowed Wording

> Stage B validates a scratch fitting-dynamics platform and supports a Stage B-prime trajectory-object audit.

> Stage B-prime finds non-random directed drift but not a strong parameter-space mechanism.

> Stage B-response measures output-response trajectories, but does not prove a geometry mechanism.

## Forbidden Wording

> Stage B proves meaningful low-dimensional parameter dynamics or geometry-response structure.
