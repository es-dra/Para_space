# Phase 1 Experiment Design: Parameter Trajectory Fundamentals

## Objective

Establish foundational understanding of parameter evolution patterns during single-image fitting.

**Core Question**: How do network parameters evolve when learning to represent an image, and what do these trajectories reveal about the image's intrinsic structure?

---

## Research Questions

1. **Trajectory Structure**: Is the parameter trajectory during image fitting constrained to a low-dimensional subspace?

2. **Temporal Dynamics**: Does the parameter change rate follow a characteristic pattern (e.g., fast → slow convergence)?

3. **Module Roles**: Do encoder and decoder parameters evolve independently, or is there a clear leader?

4. **Image-Trajectory Mapping**: Can we observe how different image regions (smooth vs textured) correspond to different phases of parameter evolution?

5. **Architecture Comparison**: How do LIIF vs LIIF-EQ trajectories compare? (Secondary focus — trajectory fundamentals first.)

---

## Experiment Configuration

### Models

| Model | Role | Description |
|-------|------|-------------|
| LIIF | Primary | Non-equivariant conditional INR |
| LIIF-EQ | Primary | C4 equivariant conditional INR |
| LTE | Auxiliary | Alternative architecture for pattern verification |
| SIREN | Reference | Non-conditional INR baseline |

### Image Selection

Selected to represent different levels of local self-similarity:

| Image | Self-Similarity | Characteristics |
|-------|-----------------|-----------------|
| baby.png | Low | Smooth, clear edges, minimal texture |
| butterfly.png | Medium | Prominent subject, textured wings |
| baboon.png | High | Highly repetitive fur texture |

**Initial plan**: Run on baby.png first to establish baseline, then expand to butterfly and baboon for pattern verification.

### Training Configuration

```python
DYNAMICS_CONFIG = {
    "total_steps": 50000,
    "snapshot_interval": 500,    # N = 101 snapshots
    "lr": 5e-4,
    "image_size": 48,
    "sr_scale": 4,               # SR x4 mode: 12x12 → 48x48
}
```

**Snapshot Selection for Visualization**: 8 snapshots at geometric intervals:
```
[0, 500, 1000, 2000, 5000, 10000, 25000, 50000]
```

This captures:
- Step 0: Initial state
- Early phase: Rapid learning (steps 500-2000)
- Mid phase: Refinement (steps 5000-10000)
- Late phase: Convergence (steps 25000-50000)

---

## Core Observation Metrics

### 1. Reconstruction Quality Progression

**Metric**: PSNR vs step curve

**What it reveals**: Overall training dynamics and convergence behavior

### 2. Parameter Change Rate

**Metric**: |Δθ_t| = ||θ_t - θ_{t-1}||_2

**What it reveals**:
- Whether optimization follows fast→slow pattern
- Correlation with PSNR improvement rate
- Potential phase transitions in learning

### 3. Module-Level Parameter Evolution

**Metric**: Per-snapshot Δθ norm for:
- Encoder parameters
- Decoder parameters
- Full model parameters

**What it reveals**:
- Which module leads at different training stages
- Whether encoder/decoder have distinct temporal patterns

### 4. Trajectory Geometry

**Metrics**:
- PC1 ratio: How concentrated is trajectory in primary direction?
- Tortuosity: arc_length / chord_length
- Effective dimensionality: count of principal components to explain 95% variance

**What it reveals**: Low-dimensional structure of optimization landscape

### 5. Frequency-Aligned Parameter Changes (Exploratory)

**Metric**: Correlation between parameter update direction and image frequency components

**What it reveals**: Whether parameter evolution mirrors the spectral bias observed in function space

---

## Visualization Plan

### Panel 1: Reconstruction Comparison Grid

**Content**: 8 snapshot reconstructions in a grid

**Layout**:
```
Step 0    | Step 500  | Step 1000 | Step 2000
Step 5000 | Step 10000 | Step 25000| Step 50000
```

Each cell shows:
- Reconstructed image
- Step number
- PSNR value

**Purpose**: Visual progression of reconstruction quality, showing which image regions are learned at which stage.

### Panel 2: Convergence Curves

**Content**:
- PSNR vs step (primary)
- |Δθ|/dt vs step (secondary)

**Annotations**: Vertical lines marking the 8 snapshot positions

**Purpose**: Understand optimization dynamics and identify phase transitions.

### Panel 3: Per-Module Parameter Evolution

**Content**:
- Encoder Δθ norm vs step
- Decoder Δθ norm vs step
- Ratio (Encoder / Decoder)

**Annotations**: 8 snapshot positions marked

**Purpose**: Identify which module dominates at different stages.

### Panel 4: Error Maps at Key Stages

**Content**: Pixel-wise error maps for selected snapshots

**Calculation**: Current reconstruction - Previous snapshot reconstruction (not GT)

**Layout**: Same 8-snapshot grid as Panel 1

**Color scheme**: Red = high change, Blue = low change

**Purpose**: Directly observe which image regions are changing most between snapshots, revealing the "what is being learned" question.

### Panel 5: Trajectory Summary Statistics

**Content**: Single-figure summary with:
- PC1 ratio
- Tortuosity
- Effective dimensionality
- Final PSNR

**Purpose**: Quick comparison across runs.

---

## Data Storage Requirements

### Per-Experiment Output

```
results/FittingDynamics/{MODEL_TAG}/
  trajectory.npz
    full_snapshots      # [N, n_params]
    enc_snapshots       # [N, n_encoder_params]
    dec_snapshots       # [N, n_decoder_params]
    snapshot_steps       # [N]
    losses              # [total_steps]
    psnrs               # [N]
    freq_ratios         # [N]
    target_spectrum
  dynamics_summary.json
  viz/                  # Generated on demand
    reconstruction_grid.png
    convergence_curves.png
    module_evolution.png
    error_maps_grid.png
    trajectory_summary.png
```

### Additional Storage for Visualization

For Panel 4 (error maps), the following approach is used:
- Error maps are computed on-the-fly during visualization
- Reconstruction images at each snapshot are needed
- These are computed from model state + trajectory snapshots

**Note**: Current implementation saves reconstructions at `recon_interval`. Verify this is compatible with 8-snapshot selection or adjust.

---

## Expected Outcomes

### Should Observe

1. **Smooth regions learned first**: Low-frequency content appears in early snapshots
2. **Texture refinement later**: High-frequency details appear in later snapshots
3. **Encoder leads early, decoder refines late**: Hypothesized module dominance pattern
4. **Trajectory is not random**: PC1 ratio significantly above random baseline

### Will Confirm or Refute

- Parameter evolution follows a characteristic temporal pattern
- Image structure (smooth vs textured) maps to trajectory structure
- LIIF vs LIIF-EQ trajectories differ qualitatively (not just quantitatively)

---

## Current Implementation Status

### Script Locations

| Script | Path |
|--------|------|
| LIIF training | `experiments/Phase1_FittingDynamics/run.py` |
| LIIF-EQ training | `experiments/Phase1_FittingDynamics/run_finetune.py` |
| Visualization | `experiments/Phase1_FittingDynamics/viz_trajectory.py` |

### Known Issues

1. **Alignment for EQ_MLP**: Hungarian matching may not apply correctly to equivariant decoder layers (documented in issue #2 of code review). Needs resolution before LIIF-EQ trajectory analysis.

2. **Reconstruction storage interval**: Current `recon_interval=1000` may not align with planned 8 snapshots. Verify and adjust if needed.

3. **Model over-parameterization**: Pre-trained models may be over-parameterized. Consider model simplification in future.

### Required Validation Before Running

- [ ] Verify snapshot storage covers planned 8 timepoints
- [ ] Verify reconstructions are saved at appropriate intervals
- [ ] Check EQ_MLP alignment issue impact on LIIF-EQ analysis
- [ ] Confirm error map computation is feasible with current data

---

## Next Steps

1. Resolve implementation issues (alignment, storage intervals)
2. Run pilot experiment on baby.png with LIIF
3. Validate visualization outputs match design
4. Extend to LIIF-EQ and other images
5. Analyze results and refine Phase 2 design
