# ARIS Research Alignment

This is the human-readable alignment note before handing the project to ARIS.
It should stay short. It is not a claim ledger.

## Plain-Language Direction

We want to study what actually happens inside INR / arbitrary-scale SR models
while they fit or adapt to an image.

The core question is:

> When SIREN or scratch LIIF fits an image, do its updates, responses, or local
> features show repeatable patterns that are related to the image itself?

Examples of image factors worth testing are smooth regions, edges, textures,
repeated structures, scale, and local frequency. These are only candidate
explanations.

## Research Purpose

The near-term purpose is not to write a paper or prove a grand parameter-space
theory. The near-term purpose is to decide whether there is a small, clean,
repeatable signal worth studying further.

If the first node fails under controls, the project should stop or redefine the
observable instead of trying many descriptors until something looks good.

## Current Assets

- Scratch SIREN fitting.
- Scratch LIIF / conditional INR fitting.
- Trajectory schema and output checks.
- Basic alignment and spectral helpers.
- Local image data under `Data/`.

## What We Are Not Doing Yet

- No external checkpoints.
- No equivariant model comparison.
- No update prediction or training acceleration.
- No paper writing.
- No use of deleted old route results as positive evidence.
- No PCA/PC1 story as a mechanism.

## Literature Boundary

The related literature supports the project as a probing study, but it does not
already establish the desired claim.

### Continuous and Local Implicit SR

LIIF introduced a local implicit image function that maps continuous
coordinates plus nearby 2D deep features to RGB, allowing arbitrary-resolution
image representation and arbitrary-scale SR:
https://arxiv.org/abs/2012.09161

Meta-SR is an earlier arbitrary-scale SR line where the upscale operation adapts
to a continuous scale factor:
https://arxiv.org/abs/1903.00875

LTE shows that local texture and Fourier-domain information matter for
arbitrary-scale implicit SR, and that plain coordinate MLPs struggle with
high-frequency detail:
https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_Local_Texture_Estimator_for_Implicit_Representation_Function_CVPR_2022_paper.pdf

These papers justify looking at local features, coordinates, cell/scale, and
frequency content. They do not prove that training updates have a reusable
structure.

### SIREN and Coordinate-Network Dynamics

SIREN shows that sinusoidal activations are strong implicit representations for
signals and their derivatives:
https://arxiv.org/abs/2006.09661

Training-dynamics work on coordinate MLPs connects spectral bias to how low- and
high-frequency content is learned over optimization:
https://arxiv.org/abs/2301.05816

These papers justify using SIREN as a clean fitting baseline and tracking
learning over time. They do not justify treating smooth trajectories as
nontrivial by default.

### Internal Image Recurrence

Cross-scale internal graph SR and related internal-prior work show that similar
patches can recur within and across image scales:
https://arxiv.org/abs/2006.16673

This supports testing whether local content or repeated structures explain model
behavior. It does not guarantee that parameter updates will align with patch
similarity.

### Weight-Space Caution

Work on weight-space alignment and permutation symmetries shows that raw neural
network parameters can be misleading unless symmetries and alignment issues are
handled:
https://arxiv.org/abs/2209.04836

This supports keeping parameter-space claims conservative and checking response
or feature spaces when raw weights are ambiguous.

## Recommended First ARIS Node

The first ARIS node should not run a large experiment. It should produce a
concrete first-node contract:

- choose one observable: update, response, local feature, or gradient;
- choose one small model path: likely SIREN first, LIIF only if justified;
- define the image factor being tested in plain terms;
- define controls that would kill the route;
- define exact commands, seeds, device, and output paths;
- define what result would make the user approve a second node.

The first executable node should be small enough to finish quickly and be easy
to reject.

## Suggested ARIS Start Request

Ask ARIS:

> Read `CLAUDE.md`, `doc/aris_project_brief.md`,
> `doc/aris_research_alignment.md`, `doc/research_contract.md`, and
> `experiments/Phase1_FittingDynamics/README.md`. Propose the first automated
> research node for Para_space. Do not run experiments yet. Output a short node
> contract with question, observable, model/data scope, controls, commands,
> artifacts, stop conditions, and what the project owner must review.

