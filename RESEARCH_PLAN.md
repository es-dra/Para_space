# Research Plan: Parameter Space Geometry of Neural Implicit Representations

## Overview

This project investigates the **parameter space geometry** of Neural Implicit
Representations (INRs) during single-image fitting. Rather than treating
networks as black boxes that map coordinates to signals, we analyze the
trajectory their parameters trace during optimization — a perspective that
reveals the underlying structure of the learning dynamics.

The core thesis: **parameter trajectories induced by different tasks,
architectures, and transformations are not random walks.** They occupy
low-dimensional subspaces, follow characteristic shapes determined by
architectural symmetries, and exhibit systematic differences between
equivariant and non-equivariant models.

---

## Phase 1: Fitting Dynamics Phenomenology

**Goal**: Establish the basic phenomenology of parameter trajectories
during single-image fitting, across architectures and fitting regimes.

### Key Questions

1. **Low-dimensional structure**: How many effective degrees of freedom
   does a parameter trajectory occupy? Is the participation ratio PR
   stable across images?

2. **Encoder/decoder separation**: For conditional INRs, do encoder and
   decoder parameters move in different subspaces? Is one more
   low-dimensional than the other?

3. **Spectral bias in parameter space**: Does the trajectory rotate from
   low-frequency to high-frequency directions — mirroring the known
   spectral bias in function space?

4. **Architecture comparison**: How does equivariance (C4 group)
   change the trajectory geometry? Is the equivariant model more
   constrained (fewer effective DOF)?

5. **Fine-tuning vs scratch**: Does a pre-trained model move in a
   fundamentally different subspace than a randomly initialized one?

### Experiment Matrix

#### Block A: SIREN Scratch Fitting (non-conditional baseline)

| Variable | Values |
|----------|--------|
| Model | SIREN (5-layer, 256 hidden) |
| Image | Set5: baby, bird, butterfly, head, woman |
| Seed | 42, 123, 456 |
| Steps | 5000 |
| Mode | Self-recon |
| Repeats | 5 images × 3 seeds = **15 runs** |

**Purpose**: Establish baseline trajectory phenomenology for non-conditional
INRs. Largest Δθ signal. Tests whether PCA structure (PC1 ratio, PR) is
consistent across images.

#### Block B: LIIF Scratch Fitting (conditional baseline)

| Variable | Values |
|----------|--------|
| Model | LIIF (EDSR encoder + MLP decoder) |
| Image | Set5: baby, bird, butterfly, head, woman |
| Seed | 42 |
| Steps | 5000 |
| Mode | SR x4 (12×12 → 48×48) |
| Repeats | 5 images × 1 seed = **5 runs** (expand seeds if time) |

**Purpose**: First conditional INR analysis. Encoder/decoder separation.
Is decoder trajectory more structured than encoder?

#### Block C: LIIF-EQ Scratch Fitting (equivariant)

| Variable | Values |
|----------|--------|
| Model | LIIF-EQ (C4 equivariant encoder + EQ_MLP) |
| Image | Set5: baby, bird, butterfly, head, woman |
| Seed | 42 |
| Steps | 5000 |
| Mode | SR x4 |
| Repeats | 5 images × 1 seed = **5 runs** |

**Purpose**: Test whether equivariance constraints reduce trajectory
dimensionality. Compare PC1 ratio, PR, and tortuosity vs LIIF.

#### Block D: Pre-trained Fine-tuning

| Variable | Values |
|----------|--------|
| Models | PretrainedLIIF, PretrainedLIIF_EQ |
| Image | Set5: baby, bird, butterfly, head, woman |
| Seed | 42 |
| Steps | 5000 |
| Mode | SR x4 |
| Repeats | 2 models × 5 images = **10 runs** |

**Purpose**: Transfer learning regime. Do fine-tuning trajectories
differ from scratch? Smaller Δθ but potentially more structured.

---

## Phase 2: Local Similarity in Parameter Space

**Goal**: Investigate how parameter changes for one image relate to
those for another. If two images share local structure (e.g., both
have a smooth gradient region), do their parameter updates align?

Planned experiments:
- Fit image A, then fine-tune from θ_A to image B — measure alignment
  between Δθ_A and Δθ_{A→B}
- Fit two crops/patches from the same image — measure cosine similarity
  of their parameter trajectories
- Vary image similarity (structural similarity index) and measure
  corresponding parameter trajectory similarity

---

## Phase 3: Transformation-Controlled Dynamics

**Goal**: Study how continuous transformations (rotation, scaling) of
the input image induce structure in parameter trajectories.

Planned experiments:
- Fit a reference image. Then for each rotation angle θ ∈ [0, 2π),
  fine-tune and record Δθ(θ). Analyze whether {Δθ(θ)} forms a
  closed manifold whose topology matches SO(2).
- Same for scale factors s ∈ [0.5, 2.0]. Expect open, monotonic
  trajectories reflecting the non-compactness of R⁺.

---

## Protocol for Each Experiment

### Recording

Each experiment run saves (via `run.py` or `run_finetune.py`):

```
results/FittingDynamics/{MODEL_TAG}/
  trajectory.npz        # Parameter snapshots
    full_snapshots      # [n_snapshots, n_params] flattened params
    enc_snapshots       # encoder-only params (conditional models)
    dec_snapshots       # decoder-only params (conditional models)
    snapshot_steps      # step numbers for each snapshot
    losses              # per-step loss values
    psnrs               # PSNR at each snapshot
    freq_ratios         # low-freq error ratio at each snapshot
    target_spectrum     # frequency spectrum of target image
  dynamics_summary.json # Full configuration + final metrics
```

### Analysis Pipeline

For each completed run, the pipeline runs automatically:

1. **Permutation alignment**: Decoder MLP layers aligned via Hungarian
   matching + sign flips (`src/alignment.py`)
2. **PCA**: SVD on centered Δθ matrix → PC1 ratio, participation ratio,
   cumulative explained variance
3. **Trajectory geometry**: Tortuosity (arc length / chord length),
   update direction coherence (mean cos similarity between consecutive
   Δθ)
4. **Spectral bias**: Low-freq error ratio over training, correlation
   with PSNR
5. **Per-layer analysis**: Frobenius norm of ΔW for each layer,
   depth trends
6. **Visualization suite**: PCA animation, dashboard, reconstruction
   grid with error maps, spectral bias plot

### Key Metrics

| Metric | What It Measures | Range | Interpretation |
|--------|-----------------|-------|----------------|
| PC1 ratio (%) | Trajectory low-dimensionality | 0–100 | >80% = highly constrained |
| Participation ratio (PR) | Effective DOF | 1–D | ~1 = single direction dominates |
| Tortuosity T | Path curvature | ≥1 | T ≈ 1 = straight line |
| Coherence C | Update direction stability | −1–1 | >0.8 = consistent, stable opt |
| Enc/Dec ratio R | Which module changes more | >0 | R > 2 = encoder dominant |
| Spectral bias ΔLF | Low-freq error drop | 0–100% | Larger = stronger spectral bias |

---

## Pre-Training Readiness Checklist

- [x] Bug 1: PretrainedLIIF_EQ get_params/set_params fixed to use state_dict
- [x] Bug 2: _get_decoder_layer_keys fixed for EQ_MLP decoder (regex matching)
- [x] Bug 3: viz_trajectory.py supports all model types (including pretrained)
- [x] src/spectral.py created (4 shared FFT utilities, no duplicated code)
- [x] run.py imports from src/spectral, no local FFT code
- [x] run_finetune.py imports from src/spectral, no local FFT code
- [x] src/models/__init__.py includes all 4 model types in MODEL_REGISTRY
- [x] Dead scripts deleted (5 files: analyze_dynamics, compare_siren_liif,
      alignment_analysis, alignment_debug, compute_siren_spectral_bias)
- [x] __pycache__ directories cleared
- [x] viz_trajectory.py upgraded: publication-quality plots, error maps,
      spectral bias tracking, proper layer heatmaps, all model types
- [x] All experiment scripts pass import/syntax validation
- [ ] **Background training**: Launch Block A–D experiments in background
      (see below)

### Launching Training

Once all items above are verified, launch training on all blocks:

```bash
# Block A: SIREN scratch
for img in baby bird butterfly head woman; do
  python experiments/Phase1_FittingDynamics/run.py \
    --model siren --image Data/Set5/HR/${img}.png \
    --save_dir results/FittingDynamics/SIREN_${img}_seed42 --seed 42
done

# Block B: LIIF scratch SR x4
for img in baby bird butterfly head woman; do
  python experiments/Phase1_FittingDynamics/run.py \
    --model liif --image Data/Set5/HR/${img}.png --sr 4 \
    --save_dir results/FittingDynamics/LIIF_${img}_seed42 --seed 42
done

# Block C: LIIF-EQ scratch SR x4
for img in baby bird butterfly head woman; do
  python experiments/Phase1_FittingDynamics/run.py \
    --model lte --image Data/Set5/HR/${img}.png --sr 4 \
    --save_dir results/FittingDynamics/LTE_${img}_seed42 --seed 42
done

# Block D: Pre-trained fine-tuning SR x4
for img in baby bird butterfly head woman; do
  for model in pretrained_liif pretrained_liif_eq; do
    python experiments/Phase1_FittingDynamics/run_finetune.py \
      --model_type "${model#pretrained_}" --image Data/Set5/HR/${img}.png \
      --scale 4 --lr_size 48 --steps 5000 \
      --save_dir results/FittingDynamics/${model}_${img}_seed42
  done
done
```

---

## Expected Outputs

After all experiments complete, the results directory should contain:

```
results/FittingDynamics/
  SIREN_{img}_seed{s}/    (Block A, 15 dirs)
  LIIF_{img}_seed{s}/     (Block B, 5 dirs)
  LTE_{img}_seed{s}/      (Block C, 5 dirs)
  PretrainedLIIF_{img}_seed{s}/   (Block D, 5 dirs)
  PretrainedLIIF_EQ_{img}_seed{s}/ (Block D, 5 dirs)
```

Each experiment directory contains:
- `trajectory.npz` — raw trajectory data
- `dynamics_summary.json` — config + metrics
- `viz/` — visualization suite (optional, generated on demand)

### Summary Analysis

After all blocks complete, a cross-experiment summary should:
1. Aggregate PC1 ratio, PR, tortuosity, coherence across all runs
2. Compare distributions (not just point estimates) between architectures
3. Report statistical significance (effect sizes, confidence intervals)
4. Generate publication-ready comparison figures
