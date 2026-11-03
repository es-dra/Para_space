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
