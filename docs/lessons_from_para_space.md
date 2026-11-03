# Controls carried over from the earlier Para_space study

The earlier project in this repository (2025-2026, whiteboard reset in May 2026) looked for
image-explainable structure in the parameter and update trajectories of SIREN and scratch LIIF
fits. It ended without a positive claim. What survives is a list of controls that any weight-space
claim about INRs must pass, and they apply to an LLM reading or editing weights as much as to a
descriptor:

- A trajectory or edit can look structured because of the step size, the learning-rate schedule,
  layer scale or endpoint drift. Compare against matched-step random directions and time-permuted
  updates before calling a pattern meaningful.
- Raw parameters are ambiguous under unit permutations and (for sine layers) sign flips. Align
  first (`wsinr/align.py`), or measure in function or response space.
- Content explanations need null-content controls: the same analysis on a shape-only or shuffled
  input must fail. For LLM reading, this is the "shapes only, numbers removed" prompt.
- One image, one seed or one picture is not a result. Use several images and seeds, and keep
  commands, seeds, configs, outputs and failures.
- Do not promote a diagnostic signal to a claim in the same node that found it.

The old code (torch SIREN, scratch LIIF/LTE, trajectory schema) is in the git history before the
commit that introduced this study, if it is ever needed.
