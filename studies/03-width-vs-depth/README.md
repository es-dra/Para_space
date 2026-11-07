# Stage 03: the price of dictionary programmability (2026-11-03)

Protocol and frozen predictions: `protocol.json`; scoring: `outcomes.json` (`scripts/analyze_width_depth.py`);
zoo indices and the LSI check: `results/`. 20 networks per zoo (MNIST test 0..19, 2,000 Adam steps at 1e-3),
except the optimisation control (5 networks, 10,000 steps).

## Fit quality against parameter count

| family | architecture | weights | fit PSNR, median [95% CI] | blur program (function-space MSE ratio to no-edit) |
| --- | --- | --- | --- | --- |
| shallow | 2-32-1 | 129 | 20.2 [19.5, 21.3] | exact (3e-10 abs) |
| shallow | 2-64-1 | 257 | 23.6 [23.0, 24.7] | exact |
| shallow | 2-128-1 | 513 | 25.5 [24.4, 26.4] | exact |
| shallow | 2-256-1 | 1,025 | 26.2 [25.7, 28.0] | exact |
| shallow | 2-384-1 | 1,537 | 26.9 [26.0, 28.4] | exact |
| shallow | 2-1024-1 | 4,097 | 27.5 [26.5, 29.0] | exact |
| deep | 2-8-8-1 | 105 | 18.9 [17.6, 22.5] | 1.01 (no effect) |
| deep | 2-16-16-1 | 337 | 29.0 [27.5, 29.9] | 1.70 (worse than no edit) |
| deep | 2-32-32-1 | 1,185 | 61.5 [59.1, 65.9] | 1.63 |
| deep | 2-16-16-16-1 | 609 | 48.1 [44.0, 50.7] | 1.12 |
| deep | 2-32-32-32-1 | 2,241 | 74.0 [62.4, 83.9] | 0.86 |
| control | 2-384-1, 10,000 steps (n = 5) | 1,537 | 31.3 [26.1, 34.3] | exact |

## Prediction ledger

| Prediction | Verdict | Numbers |
| --- | --- | --- |
| W1 deep exceeds shallow by > 25 dB at 1,000-1,500 weights | **Passed** | 61.5 vs 26.2 / 26.9 dB: a 35 dB gap. |
| W2 shallow saturates (< 4 dB per doubling beyond 256; 2-1024-1 < 32 dB) | **Passed** | 0.6 dB per doubling from 256 to 1,024 units; 27.5 dB at 4,097 weights. |
| W3 ten thousand steps improve 2-384-1 by < 1 dB | **Failed** | +4.4 dB (26.9 to 31.3 dB, n = 5, wide interval). Training does keep moving the frequencies; the least-squares ceiling check (`protocol.json`) shows the fitted dictionary is the limit at any given time, but that dictionary still improves slowly. The gap to the deep family stays above 30 dB. |
| W4 2-16-16-1 (337 weights) beats 2-1024-1 (4,097 weights) | **Passed** | 29.0 vs 27.5 dB. |
| W5 programmability unchanged across the sweep | **Passed** | every shallow network keeps the exact blur program below 3e-10 MSE; no deep network gets the input-attenuation program below 0.86 x no-edit. |

## Exact linear shift-invariant filtering (shallow networks)

For a one-sine-layer SIREN every hidden unit is a single plane wave, so any LSI filter with frequency response
G acts exactly as `A2_i *= |G(w_i)|`, `c1_i += arg G(w_i)`, `c2 *= G(0)` (`wsinr.programs.lsi_filter`). Verified
on 20 networks against discrete filters of the 9x-supersampled function (`results/lsi_completeness_L2w64.json`;
the references' own discretisation error is the floor):

| filter | program MSE (median) | no-edit MSE | target power |
| --- | --- | --- | --- |
| Gaussian 1 px | 4e-10 | 9.6e-3 | 5.9e-2 |
| anisotropic Gaussian 2 x 0.5 px | 4e-10 | 1.7e-2 | 5.7e-2 |
| box footprint 3 x 3 px (pixel-footprint mean) | 5e-9 | 7.6e-3 | 6.3e-2 |
| box footprint 3 x 1 px (anisotropic) | 1e-8 | 3.3e-3 | 7.5e-2 |
| motion 3 px | 3e-9 | 3.8e-3 | 7.1e-2 |
| derivative d/dx | 4e-7 | 1.2e-1 | 3.2e-2 |
| Laplacian | 4e-7 | 3.1e-1 | 9.9e-2 |
| unsharp mask (1 px, 1.5) | 9e-10 | 2.2e-2 | 1.7e-1 |
| difference of Gaussians 0.7 / 1.4 px | 5e-10 | 5.9e-2 | 7.0e-3 |

Together with the coordinate maps (layer 1) and affine value maps (readout), the exactly programmable algebra
of a shallow SIREN is: affine coordinate maps, composed with any LSI filter, composed with affine value maps.
The box footprint row is the pixel-integral of arbitrary-scale SR decoders written as a weight edit.

## Reading

Programmability of the dictionary layer is exact and complete for LSI filtering, and it is not free: for a
standalone image INR it costs about 35 dB at matched parameters, because the compression that depth buys comes
from composition (harmonics), which is exactly what breaks per-unit programmability. Width cannot buy it back
(0.6 dB per doubling). The setting where the shallow, programmable dictionary is cheap is the one where capacity
lives elsewhere: conditional INRs whose encoder does the compression and whose decoder is a local Fourier field
with a linear readout. That is the footprint decoder of the arbitrary-scale SR study, which matched LTE within
0.10 dB at symmetric scales; it is also the architecture on which anti-aliased resampling is a weight edit. This
stage does not test that; it is the prediction for the stage after the ICME submission.

## Replication on Fashion-MNIST (added the same day)

Same recipe, Fashion-MNIST test images 0..19 (`results/fashion_replication.json`): 2-384-1 (1,537 weights)
24.4 dB [23.7, 28.7]; 2-16-16-1 (337 weights) 26.8 dB [25.9, 29.4]; 2-32-32-1 (1,185 weights) 66.3 dB
[59.4, 72.4]. The matched-parameter gap is 42 dB and the small deep network again beats the large shallow one.

## Caveat added 2026-11-08: these are interpolation-regime numbers

Every zoo above has at least as many weights as pixels (1,185-4,097 weights for 784 pixels; the shallow family
is the exception only by construction of the readout), and training is full-batch. The deep networks therefore
*interpolate* the image (60-74 dB), which is not the regime INRs are used in (parameters well below pixels, about
30-40 dB on Kodak-scale images). The direction of the gap is expected to persist, its magnitude is not: the
"35-42 dB price" is a property of this regime. Stage 09 re-measures the price, and the harmonic index of trained
deep networks, with parameters below pixels (128 x 128 crops, ratio 0.12-0.27; 256 x 256 on GPU later). Until
those numbers are in, the price should be quoted with the regime attached. The exactness results (LSI programs,
anti-aliasing) do not depend on the regime.
