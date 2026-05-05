# Research Plan: Parameter Space Geometry of Neural Implicit Representations

## Overview

**Core Question**: What are the underlying patterns in how neural networks' parameters evolve when learning to represent images?

**Research Focus**: The parameter trajectory during image fitting — not as a black box, but as a window into how networks encode image structure.

**Key Intuition**: When an INR learns to represent an image, its parameters don't move randomly. The trajectory should reflect both the network's inductive biases and the intrinsic structure of the image — particularly local self-similarity.

**Two Complementary Perspectives** (both lead to the same underlying principle):
- Local self-similarity in images: similar patches within an image may share parameter representations
- Group transformations: controlled way to study "similar but different" inputs and their parameter responses

**Applications of Understanding**:
1. Explain how equivariant architectures work in parameter space
2. Accelerate INR training using trajectory patterns

**Three Research Phases**:

| Phase | Focus | Questions |
|-------|-------|-----------|
| Phase 1 | Single-image parameter trajectory patterns | How do parameters evolve? What structure exists? |
| Phase 2 | Local self-similarity representation | Does image similarity map to parameter similarity? |
| Phase 3 | Equivariance & applications | How does equivariance reshape trajectories? Can we accelerate training? |

---

## Phase 1: Parameter Trajectory Fundamentals

**Primary Goal**: Characterize the fundamental patterns of parameter evolution during single-image fitting.

**Architectural Focus**: LIIF vs LIIF-EQ (equivariant vs non-equivariant comparison)
- LTE included as independent architecture for pattern verification
- SIREN as non-conditional baseline reference

**Key Observations to Make**:

| What to Observe | What It Reveals |
|----------------|-----------------|
| Reconstruction error maps at key snapshots | Which image regions are learned at which stage |
| Parameter change rate \|Δθ\|/dt | Optimization dynamics (fast → slow convergence?) |
| Encoder vs Decoder parameter changes | Which module dominates at different stages |
| PC1 ratio, tortuosity, effective dimensionality | Trajectory geometry structure |

**Why This Phase First**:
- Establishes the foundational understanding before extending to transformations
- Single-image focus isolates core patterns from transformation effects
- Results guide subsequent Phase 2 and Phase 3 design

**For detailed experiment design, metrics, and visualization plan, see**: `doc/Phase1_design.md`

---

## Phase 2: Local Self-Similarity in Parameter Space

**Goal**: Investigate whether local self-similarity in images has a corresponding representation in parameter space.

**Key Questions**:
- Do similar image patches produce similar parameter updates?
- Can we observe image similarity structure directly in parameter trajectories?
- Do patterns generalize across different images?

**Status**: Planned after Phase 1 completes

---

## Phase 3: Equivariance and Applications

**Goal**: Use Phase 1-2 insights to understand equivariance effects and enable practical applications.

**Directions**:
1. **Equivariance mechanism**: How does C4 equivariance reshape the parameter trajectory compared to non-equivariant LIIF?
2. **Training acceleration**: Can trajectory patterns guide better initialization or learning rate schedules?
3. **Cross-image transfer**: Do trajectory principles transfer across different images?

**Status**: Planned after Phase 2 completes

---

## Pre-Training Readiness Checklist

- [ ] Verify experiment scripts support planned metrics and visualization
- [ ] Confirm LIIF-EQ alignment issue is resolved or documented
- [ ] Review model configurations for over-parameterization concerns
- [ ] Validate snapshot storage includes necessary data for planned visualizations

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

### Visualization Outputs

Detailed in `doc/Phase1_design.md`:
- Reconstruction comparison grid (key snapshots)
- Convergence curves (PSNR vs step)
- Parameter change rate curves
- Per-module parameter evolution
- Error maps at key stages

---

## Key Metrics

| Metric | What It Measures | Relevance |
|--------|-----------------|-----------|
| PC1 ratio | Trajectory low-dimensionality | Is optimization constrained to a subspace? |
| Tortuosity | Path curvature | Straight line or complex path? |
| \|Δθ\|/dt | Parameter change rate | Optimization dynamics pattern |
| Effective dimensionality | Full dimension structure | Beyond single PC1 |
| Encoder/Decoder Δθ ratio | Module contribution | Which module leads at different stages? |
