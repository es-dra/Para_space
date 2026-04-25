# Research Methodology

> Core research constraints and known pitfalls.
> These are learned from experience — don't rediscover them.

---

## 1. The Self-Reconstruction Trap

**Phenomenon**: Conditional INRs (models with an encoder that sees the target image) achieve trivially high PSNR (>90 dB) because the encoder can simply copy information from the input to the output. This makes fitting dynamics analysis meaningless — the model isn't "learning," it's "copying."

**Detection**: If PSNR exceeds 50 dB within the first 1000 training steps, you're in the trap.

**Mitigations**:
- Use **non-conditional models** (SIREN) for fitting dynamics experiments
- Use **super-resolution mode** (degraded/low-res input → full output) for conditional models
- Use **partial masking** (encoder sees only part of the image)
- Always report whether the encoder sees the ground truth in experiment documentation

---

## 2. Group Structure Predicts Trajectory Shape

The core theoretical prediction of this project: parameter trajectories induced by group transformations inherit the group's topological structure.

| Group Type | Example | Predicted Trajectory | Key Metric |
|------------|---------|---------------------|------------|
| Compact | SO(2) rotation | Closed orbit (ellipse in PCA) | Closure distance C < 0.15 |
| Non-compact | R+ scaling | Open, monotonic along dominant PC | PC1 vs parameter correlation r > 0.85 |

Every experiment should test these specific geometric predictions, not just report "PC1 is high."

---

## 3. Encoder / Decoder Separation

For conditional INRs, always analyze encoder and decoder parameter trajectories **separately**. They operate in different regimes:
- **Encoder**: Learns global image structure; larger parameter changes (Δθ_enc >> Δθ_dec)
- **Decoder**: Learns local interpolation; smaller, finer adjustments

Merging them into a single PCA hides this structure.

---

## 4. Spectral Bias Tracking

Neural networks fit low frequencies first, then high frequencies. In parameter space, this manifests as:
- Early Δθ aligns with directions corresponding to low-frequency fitting
- Late Δθ rotates toward directions corresponding to high-frequency refinement

Track this via:
- Frequency-band error decomposition (FFT-based)
- Cosine similarity between Δθ_t and NTK principal eigenfunctions

---

## 5. Model Comparison Design

When comparing models (e.g., LIIF vs LTE), change exactly **one** architectural component at a time:
- LIIF ↔ LTE: only the coordinate encoding changes (Fourier ↔ DCT)
- All other variables (layer count, hidden dim, training steps, data) stay fixed

This isolates the causal effect of the encoding choice on parameter geometry.
