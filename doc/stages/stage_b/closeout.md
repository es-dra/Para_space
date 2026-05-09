# Stage B Closeout

## Status

Stage B is complete as a platform-validation stage.

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

LIIF PC1 is close to the permuted-update control, so high PC1 is not a mechanism claim. Observed step cosine is higher than permuted control, so training order has real consecutive directionality.

Representative LIIF control numbers:

| run | PC1 % | permuted PC1 % | step cosine | permuted step cosine |
|---|---:|---:|---:|---:|
| baby seed123 | 94.78 | 95.76 | 0.679 | 0.405 |
| baby seed42 | 93.38 | 94.71 | 0.662 | 0.368 |
| baby seed456 | 92.02 | 93.27 | 0.635 | 0.330 |
| bird seed42 | 94.22 | 94.28 | 0.497 | 0.239 |

## What Stage B Did Not Establish

- Local geometry-to-response/update correspondence.
- Equivariance mechanism.
- Training acceleration or update prediction.
- PCA/PC1 as scientific mechanism evidence.

## Canonical Evidence

- [../../evidence/run_manifest_stage_b.md](../../evidence/run_manifest_stage_b.md)
- Historical long log: [../../archive/historical_docs/StageB_observation_log.md](../../archive/historical_docs/StageB_observation_log.md)
- Archived concise stage notes: [../../archive/stage_summaries_2026-05-09/](../../archive/stage_summaries_2026-05-09/)

## Allowed Wording

> Stage B validates a scratch fitting-dynamics platform that can support subsequent response/geometry pilot studies.

## Forbidden Wording

> Stage B proves meaningful low-dimensional parameter dynamics or geometry-response structure.
