# Stage 08: prefiltered rendering as a weight edit (2026-11-07)

Protocol: `protocol.json` (frozen before running); outcomes: `outcomes.json`; per-network PSNRs: `results_*.json`.
Rendering an INR on a coarser grid by evaluating it at the new pixel centres aliases; the correct value is the
mean of the function over each output pixel's footprint. On a programmable dictionary the box-footprint program
(`box_transfer`, stage 03) makes point sampling of the *edited* network equal to footprint integration of the
original. The comparison is cost-matched against supersampling. Ground truth: the footprint mean from an 8x
oversampled rendering. 20 networks per zoo, median PSNR against the truth.

| zoo | factor | point (1 spp) | **program (1 spp)** | 2x2 supersampling (4 spp) | 4x4 supersampling (16 spp) |
| --- | --- | --- | --- | --- | --- |
| one sine layer, 2-64-1 | x2 | 26.9 | **77.2** | 40.7 | 53.5 |
| | x4 | 18.2 | **85.0** | 35.5 | 48.7 |
| | x7 | 14.2 | **91.8** | 25.7 | 45.0 |
| depth 3, swing bound 2 (first-order program) | x2 | 25.7 | 31.9 | 40.3 | 53.2 |
| | x4 | 17.7 | 26.0 | 34.1 | 48.1 |
| | x7 | 14.7 | 25.5 | 25.5 | 43.5 |
| depth 3, standard (first-order program) | x2 | 21.7 | 21.3 | 38.3 | 52.4 |
| | x4 | 16.7 | 14.5 | 28.3 | 45.9 |
| | x7 | 13.9 | 13.3 | 24.1 | 37.2 |

| Prediction | Verdict | Numbers |
| --- | --- | --- |
| AA1 one-sine-layer program > 60 dB at every factor; point sampling < 30 dB at x4 | **Passed** | 77-92 dB; 18.2 dB |
| AA2 program (1 spp) beats 4x4 supersampling (16 spp) by >= 10 dB at x4 | **Passed** | 85.0 vs 48.7 dB: a 36 dB margin at one sixteenth of the samples |
| AA3 swing-2 deep: >= 6 dB over point sampling and within 6 dB of 4x4 supersampling at x4 | **Half failed** | +8.3 dB over point sampling, but 22 dB below 4x4 supersampling: the first-order program removes part of the aliasing, not most of it |
| AA4 standard deep: no gain over point sampling | **Passed** | -2.2 dB |

Reading. For the programmable class, anti-aliased rendering at any footprint, isotropic or anisotropic, costs
one weight edit and then one sample per pixel; it is exact where supersampling only converges. For bounded deep
networks the first-order program helps but does not replace supersampling, and for standard deep networks it
does nothing. This is the cleanest statement of what programmability buys and where: a rendering primitive, not
an accuracy number against training.
