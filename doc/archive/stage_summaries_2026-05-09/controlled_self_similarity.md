# Controlled Self-Similarity Gate

## Purpose

Determine whether the Stage C probe can detect known repeated structure under controlled conditions.

## Controlled Inputs

- `Data/ControlledSelfSimilarity/HR/css_periodic12.png`
- `Data/ControlledSelfSimilarity/HR/css_nonrepeat12.png`
- metadata: `Data/ControlledSelfSimilarity/controlled_self_similarity_metadata.json`

## Synthetic Smoke

Result: `pass_synthetic_response_smoke`.

Key numbers:

- periodic: dup@5 `1.0000`, known effect `1.0000`, percentile `0.0000`, rho `0.8451`;
- nonrepeat: dup@5 `0.0694`, known effect `-0.0199`, percentile `0.5121`.

Interpretation: patch extraction, known-group labeling, response descriptor and statistical wiring work in a no-training known-positive setting.

## Fitted Reduced LIIF Gate

Result: `fail_known_duplicate_stability_and_content_confound`.

Summary:

- periodic seeds 42/123/456 all show positive known-duplicate signal;
- nonrepeat seeds 42/123/456 do not false-positive;
- seed123 fails strict fitted support due to weak Spearman / high percentile;
- content-intensity-only control closely matches geometry in all periodic seeds.

## Interpretation

Reliable:

> The response probe and full-snapshot reconstruction are not simply broken.

Not reliable:

> Controlled exact repeats prove a geometry mechanism.

## Next Gate

Controlled positive results only justify a stricter geometry-vs-content / coordinate dissociation gate.
