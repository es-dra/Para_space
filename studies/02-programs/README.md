# Stage 02: weight-space programs by layer and depth (2026-11-03)

Protocol and frozen predictions: `protocol.json`. Prediction scoring: `outcomes.json` (from
`scripts/analyze_programs.py`). Raw per-network results: `results/`. 20 networks per zoo, MNIST test images
0..19, targets derived from each network's own rendering. Two target conventions are reported for blur:
pixel-space (`blur_s`: `gaussian_filter` of the clipped 28x28 rendering) and function-space (`blur_s_fn`: the
continuous function rendered at 8x, blurred, resampled). The function-space target is the right reference
for a weight-space program; the pixel-space one adds about 4e-4 of discretisation error.

## Zoos

| tag | architecture | weights | harmonic index H (median) | fit PSNR (median) |
| --- | --- | --- | --- | --- |
| L2w32 | 2-32-1 | 129 | 0 | 20.2 dB |
| L2w64 | 2-64-1 | 257 | 0 | 23.6 dB |
| L3w32 | 2-32-32-1 | 1,185 | 14.1 | 61.5 dB |
| L4w32 | 2-32-32-32-1 | 2,241 | 8.9 | 74.0 dB |
| L3w32_lam1e-3 | 2-32-32-1, L1 penalty 1e-3 on hidden incoming weights | 1,185 | 3.1 | 34.3 dB |
| L3w32_lam1e-2 | same, penalty 1e-2 | 1,185 | 1.6 | 27.7 dB |

## Prediction ledger

| Prediction | Verdict | Numbers |
| --- | --- | --- |
| Q1 exact blur on shallow nets, MSE < 2e-4 vs pixel target | **Failed as written; the program is exact.** | Pixel-space target: max MSE 9.6e-4 (L2w64), 7.8e-4 (L2w32), sigma 1. Against the function-space target the same program has median MSE 3.9e-10, max 7.3e-10 on L2w64 (20/20). The threshold was tied to a target that carries its own discretisation and clipping error; the pixel-space residual is entirely that error. Recorded as a failed prediction caused by a measurement choice, not as a pass. |
| Q2 deep nets: rms attenuation stays above the readout bound (>= 18/20) with ratio to no-edit in [0.3, 0.9] | **Failed, in the worse direction.** | rms attenuation is above the bound on 20/20 of L3w32 and L4w32, but its median ratio to no-edit is 2.12 and 2.08: it makes the picture worse than leaving it alone. The first-order input-attenuation program is 0.92 (L3) and 0.76 (L4) in pixel space, 1.63 and 0.86 in function space. On standard-trained deep SIRENs neither layer-local scalar program blurs. |
| Q3 per-network failure correlates with the network harmonic index (rho > 0.4) | **Failed at the network level, holds at the unit level.** | Across the 40 deep networks rho = -0.04 (rms) and 0.10 (input attenuation): H is 10-18 rad on every trained network, so there is no range to correlate over. Per unit (640 units per zoo, `unit_spectra_*.json`): H vs the attenuation rule's error rho = 0.85 (L3) and 0.84 (L4); median spectral spread 0.77 / 0.82 (share of a unit's energy outside a 20% band around its mean frequency); even the best per-unit scalar leaves a relative error of 0.36 / 0.30. |
| Q4 readout-only cannot do morphology (dilate and erode bound > 0.5 x no-edit on >= 15/20); blur bound < 0.5 on >= 15/20 | **Half failed, and the erosion half is a metric artefact.** | Dilate: 20/20 above half (bound 0.063 vs no-edit 0.096). Erode: 0/20 (bound 0.002 vs no-edit 0.079), but the eroded target of a thin white stroke is almost black: its variance is 0.004 and a constant image already reaches 0.004, so the readout's 0.0025 is only R^2 = 0.37. On variance-normalised terms the readout ceiling is R^2 = 0.62 for dilate, 0.37 for erode, 0.88 for blur and 0.93 for threshold (no-edit: 0.42, below zero, 0.71, 0.92). Morphology is the operation family the frozen dictionary serves worst in both directions; ratios to no-edit mislead when the target's energy changes. |
| Q5 contrast by the readout affine map, MSE < 1e-6 on every network | **Passed.** | max 9.4e-17 over all four zoos. Published learned editors report about 0.02 on the NFN/NFT CIFAR contrast task (different zoo; qualitative comparison until that zoo is reproduced). |

## What the stage established

1. **Exact programs exist for two of three layers.** Coordinate maps on layer 1 and pointwise value maps on
   the readout are exact at every depth (stage 00 and Q5). On one-sine-layer networks the dictionary is a set
   of plane waves and Gaussian blur is exact too: `A2_i *= exp(-|w_i|^2 sigma^2 / 2)` reproduces the
   continuous blur to 1e-9 MSE, where Adam needs about 50 steps to reach 1e-4.
2. **Depth destroys dictionary-level programmability, and the mechanism is the harmonic regime.** After
   standard training the units of a deep SIREN's last sine layer have arguments swinging 9-18 radians;
   their spectra are broadband, no single attenuation factor approximates their blur, and both the
   weight-only attenuation rule and the first-order input-attenuation rule fail outright. The harmonic index,
   computable from the weights alone, predicts which units fail (rho 0.85).
3. **Readout-only edits have a measurable ceiling, and it is operation-specific.** Refitting only the readout
   (an oracle that sees the target) explains R^2 = 0.88 of the blurred target, 0.93 of the thresholded one,
   but only 0.62 of the dilated and 0.37 of the eroded one on L3w32. Morphology needs hidden-layer change.
4. **Programmability can be bought with capacity, but not cheaply.** Penalising the harmonic index during
   training brings the input-attenuation blur to 0.40 x no-edit (L3w32_lam1e-2) at a cost of 61.5 to 27.7 dB
   of fit. The per-unit spectra show *why* this is a poor trade: the L1 penalty does not keep units linear, it
   switches most of them off (median unit H 0.1, the active 25% still at 2-7 rad), so the network degenerates
   towards a shallow one. A constraint that bounds each unit's swing without killing it is the next thing to
   try before any claim about "editable training".

## Reading

The clean positive results are the exact programs; the clean negative result is that trained deep SIRENs are
not blur-programmable by any per-unit scalar, with a weight-only diagnostic that says so in advance. The
rms-attenuation rule should not be presented as a method: it is the natural first guess and it is wrong on
every trained deep network, which is the point. Next node: (a) a bounded-swing training constraint instead of
L1, with programmability measured against fit on a curve; (b) high-pass and band-pass on shallow networks
(exact by the same argument) and their use as analytic sharpening; (c) morphology as the operation family outside the span of the dictionary (the apparent dilate/erode asymmetry was a target-energy artefact, resolved above).
