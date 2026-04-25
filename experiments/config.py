"""
Phase 1 Shared Configuration

Key insight from N=11 diagnostic:
- Random baseline PC1 = 100/(N-1) = 10%
- Measured PC1 ~ 14% = only 1.4x baseline
- N must be increased to 20+ for statistical significance
"""

import numpy as np

LIIF_CONFIG = {
    "feature_dim": 64,
    "hidden_dim": 128,
    "num_layers": 4,
    "out_dim": 3,
    "coord_enc_freqs": 8,
    "scale_enc_freqs": 8,
    "modulation": "concat",
    "encoder_layers": 3,
}

LTE_CONFIG = {
    "feature_dim": 64,
    "hidden_dim": 128,
    "num_layers": 4,
    "out_dim": 3,
    "dct_frequencies": 16,
    "scale_enc_freqs": 8,
    "modulation": "concat",
    "encoder_layers": 3,
}

SIREN_CONFIG = {
    "input_dim": 2,
    "hidden_dim": 128,
    "num_layers": 4,
    "output_dim": 3,
    "w0": 30.0,
    "w0_initial": 30.0,
    "use_siren_init": True,
}

MODEL_CONFIGS = {
    "liif": LIIF_CONFIG,
    "lte": LTE_CONFIG,
    "siren": SIREN_CONFIG,
}

TRAINING_CONFIG = {
    "independent_steps": 120000,
    "independent_lr": 5e-4,
    "psnr_target": 28.0,
}

MAML_CONFIG = {
    "meta_lr": 1e-3,
    "inner_lr": 5e-4,
    "meta_steps": 3000,
    "inner_steps": 500,
    "fine_tune_steps": 3000,
    "psnr_target": 28.0,
}

DATA_CONFIG = {
    "dataset": "Set5",
    "images": ["baby.png", "bird.png", "butterfly.png", "head.png", "woman.png"],
    "image_size": 48,
    "normalize": "center",
}

SCALE_CONFIG = {
    "transform_type": "scale",
    "param_range": [round(float(x), 2) for x in np.linspace(0.5, 1.5, 21)],
    "num_tasks": 21,
}

ROTATION_CONFIG = {
    "transform_type": "coordinate_rotation",
    "param_range": [int(x) for x in np.linspace(0, 345, 24)],
    "num_tasks": 24,
}

ANALYSIS_CONFIG = {
    "n_runs": 3,
    "random_seeds": [42, 123, 456],
    "pca_components": 10,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95,
    "pc1_threshold": 0.65,
    "pc1_baseline_ratio_threshold": 3.0,
    "alignment_compression_threshold": 2.0,
    "rotation_closure_threshold": 0.15,
    "scale_monotonicity_threshold": 0.85,
}

OUTPUT_CONFIG = {
    "results_dir": "results/Phase1",
    "save_models": False,
    "save_trajectories": True,
    "save_plots": True,
}

DYNAMICS_CONFIG = {
    "total_steps": 50000,
    "snapshot_interval": 500,
    "lr": 5e-4,
    "image_size": 48,
    "save_reconstructions": True,
    "reconstruction_interval": 1000,
}
