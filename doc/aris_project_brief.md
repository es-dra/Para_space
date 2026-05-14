# ARIS Project Brief

This is the clean entry point for Para_space after the whiteboard reset.

## Current Goal

Study whether INR / arbitrary-scale SR fitting or adaptation dynamics contain
reproducible, nontrivial structure in parameter updates, model responses, local
features, or related trajectory objects.

The project is not trying to rescue an old result. It should restart from a
small, testable question and let evidence decide whether a route is worth
expanding.

## Starting Ideas

The project owner is interested in questions such as:

- whether different image types produce different parameter or update patterns;
- whether smooth regions, edges, textures, or repeated structures correspond to
  different fitting behavior;
- whether local image content can explain where updates concentrate or how
  response trajectories evolve;
- whether SIREN and scratch LIIF expose different observable spaces.

These ideas are hypotheses, not inherited conclusions.

## Active Code Assets

Use these as the starting code surface:

- `src/siren.py`
- `src/models/liif.py`
- `src/datasets.py`
- `src/trajectory_schema.py`
- `src/alignment.py`
- `src/spectral.py`
- `experiments/Phase1_FittingDynamics/run_siren.py`
- `experiments/Phase1_FittingDynamics/run.py`
- `experiments/Phase1_FittingDynamics/check_outputs.py`

`Data/` may be used as the local image source. New formal results should use
explicit output directories and recorded commands, seeds, and configs.

## Do Not Inherit

The following are explicit warnings, not active evidence:

- do not use old R0/R1 results, tools, or conclusions;
- do not treat old Stage B-prime or Stage C narratives as a current route;
- do not use a claim ledger, project index, or experiment registry from the old
  evidence chain;
- do not use pretrained checkpoints, LIIF-EQ wrappers, equivariance comparison,
  update prediction, or training acceleration as default routes;
- do not treat PCA/PC1, descriptor reranking, local-patch matching, or update
  concentration as a proven mechanism.

If historical context is needed, read only `doc/history_negative_summary.md`.

## ARIS Node Protocol

Each ARIS node should write or update a short node note before execution:

- question being answered;
- model and data scope;
- main metric and controls;
- success, failure, and stop conditions;
- expected artifacts and commands;
- what the user must review before the next node.

Within a node, ARIS may implement scripts, run short experiments, analyze
results, and update concise memory. It must stop before changing the research
question, expanding to a new model family, changing the main metric after seeing
results, or turning a diagnostic signal into a claim.

