# Stage 04: editable training by bounding the harmonic index (2026-11-03)

**Exploratory stage: no predictions were frozen before these runs.** Its numbers are descriptive and will be
re-run under a frozen protocol (with a second dataset and seeds) before any claim is written.

Question: stage 02 showed that the L1 penalty on hidden incoming weights buys blur programmability by switching
units off. Does a projection that bounds every unit's harmonic index `H_j = sum_i |A2_ij| <= B` (without killing
units) give a better fit-versus-programmability trade-off? Zoos: 2-32-32-1 on MNIST test 0..19, 2,000 Adam steps,
projection after every step (`wsinr/zoo_np.py`, `swing_bound`). Figure: `tradeoff.png` (`scripts/plot_tradeoff.py`).

| training | H (median) | fit PSNR | blur program error / no-edit (function space) | Adam 10 steps (same target) |
| --- | --- | --- | --- | --- |
| swing bound 0.5 | 0.50 | 20.8 dB | 0.00 (exact, like a shallow net) | 6.6 |
| swing bound 1 | 1.00 | 23.1 dB | 0.02 | 0.81 |
| swing bound 2 | 2.00 | 29.3 dB | 0.15 | 0.18 |
| swing bound 4 | 4.00 | 40.2 dB | 0.30 | 0.23 |
| L1 penalty 1e-2 | 1.62 | 27.7 dB | 0.40 | 0.55 |
| L1 penalty 1e-3 | 3.13 | 34.3 dB | 0.62 | 0.16 |
| standard | 14.1 | 61.5 dB | 1.63 (worse than no edit) | 0.09 |
| shallow 2-384-1 (reference) | 0 | 26.9 dB | 0.00 | 0.15 |

Reading. The bound traces a continuous trade-off: as H grows from 0.5 to 14 the fit rises from 21 to 62 dB and
the first-order blur program degrades from exact to harmful, monotonically. At equal H the projection beats the
L1 penalty on both axes (bound 2: 29.3 dB and 0.15 against penalty 1e-2: 27.7 dB and 0.40), confirming that the
penalty's gains came from pruning. A bound of 4 gives a 40 dB representation whose blur is a zero-cost weight
edit with 70% of the error removed, roughly what ten Adam steps achieve on that network. Whether this operating
point is useful depends on the application; for a standalone image INR 40 dB is adequate, and the point of the
curve is that the user chooses H. The rms-attenuation rule stays useless at every bound, as stage 02 predicted
for broadband units.

## Confirmatory run (protocol_v2.json, frozen before running; 2026-11-04)

Bounds 1 / 2 / 4 / 8, MNIST and Fashion-MNIST test 0..19, seeds 0 and 1 (16 zoos, 2-32-32-1, 2,000 steps).
Scoring: `outcomes_v2.json` (`scripts/analyze_editable.py`); raw: `v2/`.

| bound | H | MNIST PSNR (seeds 0 / 1) | MNIST blur ratio | Fashion PSNR | Fashion blur ratio |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 22.1 / 22.8 | 0.025 / 0.035 | 21.2 / 21.8 | 0.042 / 0.032 |
| 2 | 2.0 | 28.8 / 28.8 | 0.150 / 0.161 | 26.4 / 26.8 | 0.202 / 0.176 |
| 4 | 4.0 | 39.7 / 40.5 | 0.280 / 0.252 | 38.3 / 37.2 | 0.483 / 0.523 |
| 8 | 8.0 | 67.8 / 68.7 | 1.09 / 0.77 | 61.3 / 68.0 | 2.95 / 2.30 |

| Prediction | Verdict | Numbers |
| --- | --- | --- |
| E1 monotone in both PSNR and blur ratio within every (dataset, seed) | **Passed** | 4 of 4 series monotone in both. |
| E2 datasets agree within a factor 2 at every bound | **Failed at bound 8** | factors 1.24 / 1.21 / 1.89 / 2.82 for bounds 1 / 2 / 4 / 8. Once units leave the linear regime, how harmful the first-order program becomes depends on the image class (Fashion textures drive it to 2.3-2.9 x no-edit, MNIST strokes to about 1 x). |
| E3 projection beats the L1 penalty at matched H (bound 2 vs penalty 1e-2) on both seeds | **Passed** | 28.8 dB / 0.15 and 28.8 dB / 0.16 against 27.7 dB / 0.40. |
| E4 bound 4 reaches >= 36 dB with ratio <= 0.4 on both datasets | **Failed on Fashion** | MNIST 40.1 dB / 0.27 passes; Fashion 37.8 dB passes the PSNR half but its ratio is 0.50. |
| E5 bound 8 exceeds 48 dB with ratio > 0.6 on both datasets | **Passed** | 68.2 dB / 0.93 and 64.6 dB / 2.62. |

Reading. The trade-off is monotone and seed-stable, and the projection dominates the L1 penalty, so the swing
bound is the right knob. The operating point is dataset-dependent: a bound of 4 removes about 70% of the blur
error on digits but only about 50% on clothing textures, and at a bound of 8 the first-order program hurts on
textures while roughly breaking even on strokes. The write-up states the curve per dataset and does not claim a
universal operating point. Bound 2 (29 dB, 85% of the error removed) is the setting that behaves the same on
both datasets.
