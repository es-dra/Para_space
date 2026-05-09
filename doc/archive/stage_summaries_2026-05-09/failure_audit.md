# Stage C Failure Audit Summary

## Purpose

Record why current Stage C cannot be promoted to a main mechanism claim.

## Natural-Image Failure Cases

| Image | Status | Key reason |
|---|---|---|
| baby | strongest positive | multi-seed local signal but still not a cross-image law |
| woman | useful positive | single seed, content-stratum dependent |
| butterfly | weak support | insufficient for strong evidence |
| bird | failure | geometry top-1 often response-far; not rescued by descriptors/strata |
| head | critical failure/unresolved | high PSNR but output response still fails |

## Failed Repair Routes

- descriptor ablations did not rescue bird/head.
- high-gradient / high-variance stratification did not rescue bird/head.
- context rerank did not exploit top-k oracle headroom.
- response-object audit did not explain failure.
- LR-cell query-support unit did not rescue failure and often lost to content/coordinate controls.

## Reviewer-Style Conclusion

Current results do not disprove the research direction. They do show that the current Stage C operation is too weak or mismatched to support a paper-level geometry-response mechanism.

## Claim Boundary

Allowed:

> Some local signals exist, but current matching/response protocol fails as a robust mechanism test.

Forbidden:

> The method discovers a stable natural-image geometry-response law.
