# Stage C Future Gates

This file defines legitimate future Stage C routes. It prevents drifting back into unregistered descriptor search.

## Gate 1: Geometry-vs-Content Dissociation

Question:

> Can geometry explain response similarity beyond content intensity, coordinate, spatial distance, and exact repetition?

Requirements:

- pre-register controlled inputs;
- hold response object fixed;
- include content-only, coordinate-only, geometry-shuffled, and spatial/content matched controls;
- report negative cases;
- no response-label oracle predictor.

Pass condition:

- geometry beats content/coordinate controls across seeds or controlled variants;
- negative controls do not false-positive.

## Gate 2: Model-Internal Unit

Question:

> Is the current failure caused by using the wrong local unit for LIIF?

Candidate units:

- decoder input tuple: local feature, relative coordinate, cell;
- encoder feature trajectory;
- query-conditioned support neighborhood;
- response/Jacobian around model query cells.

Requirements:

- one frozen unit definition;
- no per-image rule changes;
- same bird/head failure gate and baby/woman guardrails;
- controls against content/coordinate confounds.

## Stop Rule

If these gates fail, Stage C should be downgraded to a failure-audit / diagnostic framework rather than expanded into equivariance or training acceleration.
