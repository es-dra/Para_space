# Stage C Current Status

## Status

Stage C is a pilot / failure-audit stage. It has not produced a cross-image law.

## Reliable Positive Signals

- baby multi-seed shows the strongest local signal.
- woman seed42 is a useful second positive signal.
- Signals are more visible in high-gradient / high-variance patch strata.

## Reliable Negative Constraints

- bird is a real failure case under current output trajectory probe.
- head remains failure/unresolved despite high final PSNR.
- butterfly is weak support, not strong evidence.
- Spearman correlations are generally weak.
- feature/Jacobian improvements cannot replace primary output trajectory failure.

## Failed Repair Routes

- Descriptor ablations did not rescue bird/head.
- High-gradient / high-variance stratification did not rescue bird/head.
- Context rerank did not exploit top-k oracle headroom.
- Response-object audit did not explain failure.
- LR-cell query-support unit did not rescue failure and often lost to content/coordinate controls.

## C-minimal LR-cell Feature Trajectory Diagnostic

2026-05-10 C-minimal replaced HR patch nearest-neighbor matching with the LIIF
locality unit that is most defensible under the current model: each LR cell's
encoder feature trajectory `[T, 32]` from `gen_feat()` before feature unfolding.
The script explicitly applies LIIF input normalization before `gen_feat()`,
fixing the main risk in the earlier LIIF internal-unit screen.

Primary target: per-cell `orthogonal_energy_fraction`.

Primary test: blocked spatial CV `Delta R2 = R2(coord + content + geometry) -
R2(coord + content)`.

Result:

| Run | Strict `Delta R2_cv` |
|---|---:|
| `LIIF_reduced_baby_sr4_seed123` | `0.0331` |
| `LIIF_reduced_woman_sr4_seed42` | `0.0379` |
| `LIIF_reduced_bird_sr4_seed42` | `-0.0253` |
| `LIIF_reduced_head_sr4_seed42` | `-0.0393` |

Verdict: `inconclusive`. baby/woman pass the positive guardrail, but bird/head
are strongly negative and the four-run median is only `0.0039`. This does not
justify reopening Stage C or expanding to equivariance/update prediction.

## Controlled Self-Similarity

Controlled probes show that patch extraction, response reconstruction, and statistical wiring are not simply broken. The no-training synthetic response smoke passed, and fitted reduced LIIF detected periodic repeated structure without false-positive nonrepeat controls.

This is still not a geometry mechanism proof: seed stability was imperfect, and content-intensity-only controls came close to geometry in periodic cases.

## Current Diagnosis

The current operationalization:

```text
local patch appearance / geometry nearest neighbor
-> raw output trajectory response similarity
```

is insufficient as a main evidence generator.

The deeper project-level diagnosis is that Stage C was advanced before the
Stage B parameter/response trajectory object was sufficiently audited. Stage C
therefore remains useful as a pilot and failure-audit record, but it should not
drive the next research step.

## Current Stop Conditions

Do not proceed to:

- full Stage C expansion;
- pretrained LIIF/LIIF-EQ;
- equivariance comparison;
- update prediction;
- training acceleration;
- PCA story revival.
- new Stage C descriptor/response/rerank branches before Stage B-prime.

## Next Legitimate Gates

The immediate Stage B-prime trajectory-object audit, Stage B-response audit,
LIIF internal-unit diagnostic screen, and C-minimal LR-cell feature trajectory
diagnostic have completed. None provides a clean positive gate for reopening
Stage C.

Stage C should only be reopened after a more defensible LIIF function-space
unit is fixed despite the Stage B-prime / Stage B-response / LIIF-unit negative
constraints.
If reopened, two paths remain scientifically coherent:

1. geometry-vs-content / coordinate dissociation gate;
2. model-internal response/unit redefinition, e.g. LIIF decoder input, encoder feature trajectory, query-conditioned support.

Pass conditions for any new gate:

- response/update object fixed before running;
- bird/head failure directly addressed;
- baby/woman guardrails retained;
- content/coordinate/spatial controls reported;
- no response-label oracle predictor;
- negative cases reported.

## Detailed Evidence

- Stage C artifact manifest: [../../evidence/analysis_manifest_stage_c.md](../../evidence/analysis_manifest_stage_c.md)
- Decision record: [../../decisions/2026-05-08_stage_c_operationalization_failure.md](../../decisions/2026-05-08_stage_c_operationalization_failure.md)
- C-minimal decision record: [../../decisions/2026-05-10_stage_c_lr_cell_feature_trajectory_result.md](../../decisions/2026-05-10_stage_c_lr_cell_feature_trajectory_result.md)

Old Stage C diagnostic scripts, JSONs, patch crops, and interim archive notes
were deleted during the 2026-05-11 cleanup after their conclusions were
summarized in the files above.
