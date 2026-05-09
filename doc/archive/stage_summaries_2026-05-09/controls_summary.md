# Stage B Controls Summary

## Controls

Stage B uses lightweight trajectory controls:

- norm-matched random walk;
- iid Gaussian snapshots;
- permuted update trajectory.

## Core Observation

Observed trajectories are stronger than norm-matched random walk, but LIIF observed PC1 is close to permuted-update PC1.

This means LIIF high PC1 is likely dominated by update vector collection, step norm schedule, endpoint direction, or related trajectory geometry; it is not evidence of local geometry correspondence.

## Key Numbers

| group | observed PC1 | permuted-update PC1 mean | observed step cosine | permuted step cosine mean |
|---|---:|---:|---:|---:|
| LIIF baby seed123 | 94.78 | 95.76 | 0.679 | 0.405 |
| LIIF baby seed42 | 93.38 | 94.71 | 0.662 | 0.368 |
| LIIF baby seed456 | 92.02 | 93.27 | 0.635 | 0.330 |
| LIIF bird seed42 | 94.22 | 94.28 | 0.497 | 0.239 |
| LIIF butterfly seed42 | 96.35 | 96.74 | 0.662 | 0.404 |

## Interpretation

Reliable:

- The platform can capture non-random training trajectories.
- Training order has direction continuity stronger than permuted control.
- Stage B provides a reliable basis for Stage C response probing.

Not reliable:

- PCA as main scientific story.
- PC1 as local geometry evidence.
- Any update prediction or training acceleration claim.
