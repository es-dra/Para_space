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
