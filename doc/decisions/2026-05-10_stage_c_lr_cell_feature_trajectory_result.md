# 2026-05-10 Stage C LR-cell Feature Trajectory Result

## Question

Can LIIF per-LR-cell encoder feature trajectory complexity be explained by
local geometry after controlling for coordinate and content features?

This was the C-minimal route: use the model's real locality unit instead of the
old HR patch output-response nearest-neighbor object.

## Frozen Protocol

- Runs: `LIIF_reduced_baby_sr4_seed123`, `LIIF_reduced_woman_sr4_seed42`,
  `LIIF_reduced_bird_sr4_seed42`, `LIIF_reduced_head_sr4_seed42`.
- Trajectory object: `full_snapshots[t] -> set_params -> normalized LR input
  -> gen_feat() -> [T, H_lr, W_lr, C]`.
- Per-cell object: `[T, 32]`.
- Main target: `orthogonal_energy_fraction`.
- Main metric: blocked spatial CV `Delta R2 = R2(coord + content + geometry) -
  R2(coord + content)`.
- Gate: baby/woman `Delta R2_cv >= 0.03`, four-run median `>= 0.02`, and
  bird/head not below `-0.01`.

## Evidence Considered

- Script: `experiments/Phase1_FittingDynamics/analyze_stage_c_lr_cell_feature_trajectory.py`
- Tests: `tests/test_stage_c_lr_cell_feature_trajectory.py`
- Artifact:
  `results/FittingDynamics_StageB_diagnostics/stage_c_lr_cell_feature_trajectory_c_minimal_2026-05-10.json`
- Validation:
  - `pytest -q tests/test_stage_c_lr_cell_feature_trajectory.py`
  - related regression suite after the old LIIF-unit normalization fix:
    `83 passed in 6.24s`

Main target strict `Delta R2_cv`:

| Run | `Delta R2_cv` |
|---|---:|
| `LIIF_reduced_baby_sr4_seed123` | `0.0331` |
| `LIIF_reduced_woman_sr4_seed42` | `0.0379` |
| `LIIF_reduced_bird_sr4_seed42` | `-0.0253` |
| `LIIF_reduced_head_sr4_seed42` | `-0.0393` |

Median strict `Delta R2_cv`: `0.0039`.

## Decision

Do not reopen Stage C.

The result is mixed/inconclusive rather than a clean positive gate: baby and
woman pass the positive guardrail, but bird and head are negative under the same
blocked-CV protocol. This means the C-minimal model-internal unit did not solve
the cross-image failure.

## Rejected Alternatives

- Expand immediately to all seven reduced LIIF runs: rejected because the
  frozen diagnostic go signal did not pass.
- Reinterpret baby/woman as sufficient support: rejected because bird/head are
  explicitly part of the gate and remain negative.
- Move to equivariance, update prediction, or training acceleration: rejected
  because the H1/C-minimal gate is still not passed.

## Consequences

- The old natural-image nearest-neighbor route remains stopped.
- C-minimal can be cited as a more model-aware negative/mixed diagnostic, not as
  a mechanism result.
- The project should either accept a probing/failure-audit framing or design a
  smaller controlled setting before making stronger geometry claims.

## Conditions To Reopen

Reopening requires a new pre-registered reason for why bird/head negative
increments are not decisive, or a controlled setting that directly tests the
same feature-trajectory object without content/coordinate confounding. It must
not be an open-ended descriptor/rerank patch.
