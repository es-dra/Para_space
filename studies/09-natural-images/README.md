# Stage 09: natural images (2026-11-07/08)

Protocol: `protocol.json` (frozen; amended on 11-08 with the compression-regime run, predictions C1-C4 frozen
before it started). 64 x 64 results: `results.json`, `outcomes_64.json`. 128 x 128 results: `compression128/`.
Twenty 64 x 64 grey crops of OpenCV sample images; 3,000 Adam steps; median over crops.

## 64 x 64 (weights >= pixels: interpolation regime)

| zoo | weights | fit PSNR | H | blur program / no-edit | box program / no-edit | AA x4: point | program | 2x2 | 4x4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2-64-64-1 | 4,417 | 46.9 | 20.8 | 2.55 | 2.82 | 26.9 | 22.4 | 38.4 | 56.6 |
| 2-64-64-64-1 | 8,577 | 59.9 | 11.7 | 1.42 | 1.43 | 27.0 | 25.5 | 39.3 | 55.3 |
| 2-1024-1 | 4,097 | 26.5 | 0 | 3e-7 | 5e-7 | 33.6 | **95.8** | 47.2 | 59.7 |
| 2-2048-1 | 8,193 | 26.6 | 0 | 3e-7 | 5e-7 | 33.0 | **95.5** | 46.9 | 59.4 |
| 2-64-64-1, swing bound 2 | 4,417 | 28.0 | 2 | 0.30 | 0.29 | 31.5 | 36.8 | 47.4 | 60.2 |
| 2-64-64-1, swing bound 4 | 4,417 | 29.7 | 4 | 0.83 | 0.82 | 30.4 | 30.5 | 46.3 | 59.7 |

| Prediction | Verdict | Numbers |
| --- | --- | --- |
| N1 deep exceeds shallow by >= 4 dB at both sizes | **Passed** | 20.4 and 33.3 dB |
| N2 shallow programs below 1e-4 x no-edit on every crop | **Passed** | max 6e-7 |
| N3 anti-aliasing program beats 4x4 supersampling by >= 15 dB | **Passed** | 95.8 vs 59.7 |
| N4 swing 4 within 5 dB of standard with blur ratio <= 0.4; swing 2 within 10 dB with <= 0.2 | **Failed** | swing 4: 17 dB below, ratio 0.83; swing 2: 19 dB below, ratio 0.30. On natural content the bounded networks lose far more fit than on digits and their programs are less exact; the bound-2 network is still the only deep one whose program helps (0.30) and whose anti-aliasing edit beats point sampling (+5 dB), but it is nowhere near the one-sine-layer class. |
| N5 standard deep not programmable (ratio >= 0.7) | **Passed** | 2.55 |

Reading. On natural content the exact results hold to the floor and the deep-network failures are as sharp as
on digits; the editable-training middle ground is weaker than the MNIST curve suggested. The PSNRs of the deep
networks are interpolation numbers (4,096 pixels, 4,417-8,577 weights) and must not be quoted as representative;
see the amendment and the 128 x 128 run below.

## 128 x 128 (weights below pixels: compression regime; amendment, predictions C1-C4 frozen before running)

Ten 128 x 128 crops (16,384 pixels); fitting in chunks of two crops; `compression128/results.json`,
`outcomes_128.json`, fitted parameters in `compression128/*_params.npz`.

| zoo | weights | weights / pixels | fit PSNR | H | blur program / no-edit | AA x4: point | program | 2x2 | 4x4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2-64-64-1 | 4,417 | 0.27 | 30.5 | 31 | 1.24 | 25.4 | 24.4 | 41.7 | 57.0 |
| 2-1024-1 | 4,097 | 0.25 | 21.0 | 0 | 5e-7 | 43.9 | **105** | 56.4 | 68.6 |
| 2-32-32-32-1 | 2,241 | 0.14 | 28.2 | 11 | 1.12 | 26.6 | 26.2 | 42.2 | 57.6 |
| 2-512-1 | 2,049 | 0.125 | 21.0 | 0 | 5e-7 | 44.8 | **106** | 57.3 | 69.5 |

| Prediction | Verdict | Numbers |
| --- | --- | --- |
| C1 gap persists but shrinks: 4-15 dB at 4k weights, >= 2 dB at 2k | **Passed** | 9.5 dB and 7.2 dB |
| C2 representative absolute numbers: deep 27-38 dB, shallow 19-26 dB | **Passed** | 30.5 and 21.0 dB |
| C3 harmonic regime persists: H >= 6 and blur ratio >= 0.7 for deep nets | **Passed** | H = 31 and 11; ratios 1.24 and 1.12 |
| C4 exactness and anti-aliasing margin unchanged | **Passed** | max program ratio 7e-7; 105 vs 68.6 dB |

Reading. In the regime INRs are used in, the price of a programmable dictionary is about 7-10 dB at matched
parameters (not the 20-42 dB of the interpolation-regime zoos), the deep networks are even further into the
harmonic regime (H rises as the fit gets harder), and everything exact stays exact. The paper quotes the price
with the regime attached and uses these numbers as the headline.
