# Stage A Closeout

## Status

Stage A is complete for pilot purposes.

It establishes:

- central research question;
- H1-H4 hypothesis ladder;
- non-goals and forbidden claims;
- negative controls;
- first-pass literature map.

It does not establish:

- paper-ready related work;
- final baseline table;
- verified LIIF-EQ / SE-INR provenance;
- any empirical mechanism claim.

## Canonical Inputs

- Current project contract: [../../research_contract.md](../../research_contract.md)
- Claim boundary: [../../claims_ledger.md](../../claims_ledger.md)
- Project index: [../../project_index.md](../../project_index.md)

Historical Stage A drafts were deleted during the 2026-05-11 cleanup after their
current conclusions were consolidated here and in the research contract.

## Reliable Conclusions

1. The original PCA/trajectory-centered story was correctly downgraded.
2. Parameter/response fitting dynamics remains the primary research object; local geometry is an explanatory hypothesis, not the whole project.
3. Generic PCA low-dimensionality is not an adequate formulation of parameter dynamics.
4. Equivariance is a hypothesis about regularization/exposure of structure, not a premise.
5. Training reduction is a long-term application hypothesis and remains blocked.

## Hypothesis Ladder

- H0：参数/响应 fitting dynamics 是否存在超过平凡控制的非平凡结构。
- H1：局部几何相似 patch 是否对应 response/update similarity。
- H2：尺度条件下的 response/update 是否有结构。
- H3：等变性是否正则化 geometry-response/update correspondence。
- H4：是否能预测部分更新并减少训练。

## Literature Map

The first-pass map covers arbitrary-scale SR / continuous image representation, INR / neural fields, equivariance, internal image recurrence, parameter-efficient adaptation, and optimization/weight-space geometry.

This map is sufficient for pilot design only. It must be converted into a verified related-work table before paper writing.

## Remaining Risks

- Literature coverage is not paper-complete.
- Some bib entries and recent arbitrary-scale SR methods need final verification.
- Current pretrained LIIF-EQ / SE-INR source, checkpoint and fairness protocol remain unresolved.

## Allowed Wording

> Stage A defines a testable research program and the negative controls required for pilot experiments.

## Forbidden Wording

> Stage A establishes that local geometry predicts INR adaptation.

> Stage A replaces parameter-dynamics research with only patch-geometry matching.
