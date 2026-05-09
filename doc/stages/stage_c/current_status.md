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

## Current Stop Conditions

Do not proceed to:

- full Stage C expansion;
- pretrained LIIF/LIIF-EQ;
- equivariance comparison;
- update prediction;
- training acceleration;
- PCA story revival.

## Next Legitimate Gates

Only two paths remain scientifically coherent:

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
- Archived stage notes: [../../archive/stage_summaries_2026-05-09/](../../archive/stage_summaries_2026-05-09/)
