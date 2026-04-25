"""
Phase 1 D1: Enhanced Fitting Dynamics Analysis (v2)

Reads trajectory.npz from run_siren.py or run.py and computes
6-dimensional analysis of the parameter trajectory:

  1. Global PCA          — variance explained, participation ratio, intrinsic dim
  2. Local PCA           — sliding window dimension (detects phase transitions)
  3. Gradient Coherence  — cos(Δθ_t, Δθ_{t+1}) (trajectory "straightness")
  4. Phase Detection     — rapid descent → refinement → convergence
  5. Directional Persistence — autocorrelation half-life of Δθ direction
  6. Hessian Sharpness   — optional power iteration for top eigenvalue

Works on both SIREN (non-conditional) and LIIF/LTE (conditional) trajectories.

Usage:
    # SIREN trajectory
    python experiments/Phase1_FittingDynamics/analyze_dynamics_v2.py \
        --trajectory results/Phase1/FittingDynamics/siren/trajectory.npz \
        --model siren

    # LIIF/LTE trajectory
    python experiments/Phase1_FittingDynamics/analyze_dynamics_v2.py \
        --trajectory results/Phase1_v2/FittingDynamics/liif/trajectory.npz \
        --model liif
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import svd


# ─── Utilities ──────────────────────────────────────────────────────────────

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return dot / (n1 * n2)


def participation_ratio(eigenvalues):
    ev = np.asarray(eigenvalues)
    ev = ev / (ev.sum() + 1e-12)
    return float(1.0 / (np.sum(ev ** 2) + 1e-12))


def median_filter(x, window=3):
    half = window // 2
    n = len(x)
    out = np.zeros_like(x)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.median(x[lo:hi])
    return out


# ─── 1. Global PCA ──────────────────────────────────────────────────────────

def compute_global_pca(snapshots, snapshot_steps):
    """PCA on entire Δθ trajectory (relative to θ_0)."""
    theta_0 = snapshots[0]
    delta = snapshots - theta_0[np.newaxis, :]
    delta_norms = np.linalg.norm(delta, axis=1)

    n = delta.shape[0]
    if n < 3:
        return {"error": "Need >= 3 snapshots"}, delta_norms

    centered = delta - delta.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)
    ev_ratio = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(ev_ratio)
    pr = participation_ratio(ev_ratio)
    intrinsic_dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)

    pca_dirs = Vt[:min(5, len(Vt))]
    pc1 = pca_dirs[0]
    pc2 = pca_dirs[1] if len(pca_dirs) > 1 else np.zeros_like(pc1)

    cos_pc1, cos_pc2 = [], []
    for d in delta:
        d_norm = d / (np.linalg.norm(d) + 1e-12)
        cos_pc1.append(cosine_similarity(d_norm, pc1))
        cos_pc2.append(cosine_similarity(d_norm, pc2))

    step_sizes = [np.linalg.norm(delta[i] - delta[i - 1]) for i in range(1, n)]
    projections = delta @ pca_dirs[:2].T

    return {
        "explained_variance_ratio": [float(x) for x in ev_ratio[:10]],
        "cumulative_variance": [float(x) for x in cumvar[:10]],
        "participation_ratio": float(pr),
        "intrinsic_dim_95": int(intrinsic_dim_95),
        "cosine_with_pc1": [float(x) for x in cos_pc1],
        "cosine_with_pc2": [float(x) for x in cos_pc2],
        "delta_norms": [float(x) for x in delta_norms],
        "step_sizes": [float(x) for x in step_sizes],
        "projections_2d": projections.tolist(),
    }, delta_norms


# ─── 2. Local PCA ───────────────────────────────────────────────────────────

def compute_local_pca(snapshots, window_size=5):
    """Sliding-window PCA to detect local dimension changes."""
    n = snapshots.shape[0]
    if n < window_size + 2:
        return {"error": f"Need >= {window_size + 2} snapshots"}, None

    theta_0 = snapshots[0]
    delta = snapshots - theta_0[np.newaxis, :]

    local_dims = []
    local_pc1_dirs = []
    window_centers = []

    for i in range(n - window_size + 1):
        window = delta[i:i + window_size]
        w_centered = window - window.mean(axis=0)
        _, Sw, _ = svd(w_centered, full_matrices=False)
        ev_w = (Sw ** 2) / (Sw ** 2).sum() if Sw.sum() > 0 else np.ones_like(Sw) / len(Sw)
        local_dims.append(participation_ratio(ev_w))
        window_centers.append(i + window_size // 2)

    local_dims = np.array(local_dims)
    window_centers = np.array(window_centers)

    return {
        "window_size": window_size,
        "local_intrinsic_dims": local_dims.tolist(),
        "window_centers": window_centers.tolist(),
        "mean_local_dim": float(np.mean(local_dims)),
        "early_local_dim": float(np.mean(local_dims[:max(1, len(local_dims) // 3)])),
        "late_local_dim": float(np.mean(local_dims[-max(1, len(local_dims) // 3):])),
    }, (window_centers, local_dims)


# ─── 3. Gradient Coherence ──────────────────────────────────────────────────

def compute_gradient_coherence(snapshots, snapshot_steps):
    """Cosine similarity between consecutive Δθ steps."""
    delta = np.diff(snapshots, axis=0)
    n = len(delta)
    if n < 2:
        return {"error": "Need >= 3 snapshots"}, None

    coherences = []
    for i in range(n - 1):
        coherences.append(cosine_similarity(delta[i], delta[i + 1]))

    # Autocorrelation of gradient direction
    max_lag = min(n - 1, 30)
    ac = np.zeros(max_lag)
    for lag in range(max_lag):
        vals = []
        for i in range(n - lag):
            vals.append(cosine_similarity(delta[i], delta[i + lag]))
        ac[lag] = np.mean(vals) if vals else 0.0

    half_life = None
    for lag in range(1, max_lag):
        if ac[lag] < 0.5:
            half_life = lag
            break
    if half_life is None:
        half_life = max_lag

    return {
        "coherences": [float(x) for x in coherences],
        "mean_coherence": float(np.mean(coherences)),
        "coherence_early": float(np.mean(coherences[:max(1, len(coherences)//3)])),
        "coherence_late": float(np.mean(coherences[-max(1, len(coherences)//3):])),
        "autocorrelation": ac.tolist(),
        "autocorrelation_half_life": half_life,
    }, (snapshot_steps[:-1], coherences)


# ─── 4. Phase Detection ─────────────────────────────────────────────────────

def detect_phases(snapshot_steps, psnrs, delta_norms, min_phase_snapshots=3):
    """
    Detect optimization phases based on PSNR slope and parameter change rate.
    Returns phases: rapid_descent, refinement, convergence.
    """
    steps = np.array(snapshot_steps[1:])  # skip step 0
    psnr = np.array(psnrs)
    deltas = np.array(delta_norms)

    if len(psnr) < 3:
        return {"error": "Need >= 3 PSNR values"}, None

    psnr_slope = np.gradient(psnr)

    smoothed_slope = median_filter(psnr_slope, window=3)
    smoothed_delta = median_filter(deltas, window=3)

    mean_slope = np.mean(smoothed_slope)
    std_slope = np.std(smoothed_slope) + 1e-10

    phases = np.full(len(psnr), -1, dtype=int)
    # 0 = rapid_descent, 1 = refinement, 2 = convergence
    for i in range(len(psnr)):
        if smoothed_slope[i] > mean_slope + 0.3 * std_slope:
            phases[i] = 0
        elif smoothed_slope[i] > 0.05 * mean_slope:
            phases[i] = 1
        else:
            phases[i] = 2

    # Merge short phases
    merged = phases.copy()
    for i in range(1, len(phases)):
        run_start = i
        while i < len(phases) and phases[i] == phases[i - 1]:
            i += 1
        if i - run_start < min_phase_snapshots and run_start > 0:
            merged[run_start:i] = phases[run_start - 1]

    phase_names = {0: "rapid_descent", 1: "refinement", 2: "convergence"}
    phase_ranges = []
    current_phase = merged[0]
    phase_start = 0
    for i in range(1, len(merged)):
        if merged[i] != current_phase:
            phase_ranges.append({
                "name": phase_names.get(int(current_phase), "unknown"),
                "snapshot_start": int(snapshot_steps[phase_start + 1]),
                "snapshot_end": int(snapshot_steps[i]),
                "psnr_start": float(psnr[phase_start]),
                "psnr_end": float(psnr[i - 1]),
                "psnr_gain": float(psnr[i - 1] - psnr[phase_start]),
            })
            phase_start = i
            current_phase = merged[i]
    phase_ranges.append({
        "name": phase_names.get(int(current_phase), "unknown"),
        "snapshot_start": int(snapshot_steps[phase_start + 1]),
        "snapshot_end": int(snapshot_steps[-1]),
        "psnr_start": float(psnr[phase_start]),
        "psnr_end": float(psnr[-1]),
        "psnr_gain": float(psnr[-1] - psnr[phase_start]),
    })

    return {
        "phases": phase_ranges,
        "phase_labels": [int(p) for p in merged],
        "smoothed_slope": smoothed_slope.tolist(),
    }, (steps, psnr, smoothed_slope, merged)


# ─── 5. Directional Persistence ─────────────────────────────────────────────

def compute_directional_persistence(snapshots, snapshot_steps):
    """Compute autocorrelation of Δθ direction at multiple lags."""
    delta = np.diff(snapshots, axis=0)
    n = len(delta)
    if n < 3:
        return {"error": "Need >= 4 snapshots"}, None

    max_lag = min(n, 30)
    half_lives = []
    # Compute half-life in sliding windows
    window_size = max(10, n // 5)
    for i in range(0, n - window_size):
        seg = delta[i:i + window_size]
        ac = np.zeros(max_lag)
        for lag in range(max_lag):
            vals = []
            for j in range(len(seg) - lag):
                vals.append(cosine_similarity(seg[j], seg[j + lag]))
            ac[lag] = np.mean(vals) if vals else 0.0

        hl = max_lag
        for lag in range(1, max_lag):
            if ac[lag] < 0.5:
                hl = lag
                break
        half_lives.append(hl)

    return {
        "window_half_lives": [float(x) for x in half_lives],
        "mean_half_life": float(np.mean(half_lives)) if half_lives else 0.0,
        "early_half_life": float(np.mean(half_lives[:max(1, len(half_lives)//3)])) if half_lives else 0.0,
        "late_half_life": float(np.mean(half_lives[-max(1, len(half_lives)//3):])) if half_lives else 0.0,
    }, half_lives


# ─── 6. Hessian Top Eigenvalue ──────────────────────────────────────────────

def compute_hessian_spectrum_approximation(model_class, model_config, snapshots, coords, target, device, n_iter=10):
    """
    Approximate top Hessian eigenvalue at each snapshot using power iteration.

    This is expensive — uses double backward pass. Only runs if model reconstruction
    is feasible from snapshot data.

    Note: This requires loading the model state at each snapshot, which means
    we need the model architecture and the parameter snapshots that can be
    loaded back. For now, returns a placeholder indicating this is a future feature.
    """
    return {"note": "Hessian computation requires model reconstruction. "
                    "Run with --compute_hessian and provide --model_config for full analysis."}, None


# ─── Main Analysis ──────────────────────────────────────────────────────────

def analyze_dynamics(trajectory_path, model_type="siren", save_dir=None,
                     local_pca_window=5):
    """Run full 6-dimensional analysis on a trajectory."""
    data = np.load(trajectory_path, allow_pickle=True)
    full_snapshots = data["full_snapshots"]
    snapshot_steps = data["snapshot_steps"]
    losses = data.get("losses", np.array([]))
    psnrs = data.get("psnrs", np.array([]))

    n_snapshots, n_params = full_snapshots.shape

    print(f"\n{'='*60}")
    print(f"Dynamics Analysis v2: {model_type}")
    print(f"Snapshots: {n_snapshots}, Params: {n_params:,}")
    print(f"Steps: {snapshot_steps[0]} -> {snapshot_steps[-1]}")
    print(f"{'='*60}")

    results = {
        "model_type": model_type,
        "n_snapshots": int(n_snapshots),
        "n_params": int(n_params),
        "snapshot_steps": [int(s) for s in snapshot_steps],
    }

    # Whether this is a conditional INR with encoder/decoder split
    has_enc_dec = "enc_snapshots" in data
    if has_enc_dec:
        print("Detected conditional INR (encoder/decoder split)")
        results["has_encoder_decoder"] = True

    # ── 1. Global PCA ──
    print("\n[1/6] Global PCA...")
    global_pca, delta_norms = compute_global_pca(full_snapshots, snapshot_steps)
    results["global_pca"] = {k: v for k, v in global_pca.items()
                             if k != "projections_2d"}
    pc1 = global_pca.get("explained_variance_ratio", [0])[0] if global_pca.get("explained_variance_ratio") else 0
    pr = global_pca.get("participation_ratio", 0)
    dim95 = global_pca.get("intrinsic_dim_95", 0)
    print(f"  PC1: {pc1*100:.2f}%, Part. Ratio: {pr:.1f}, Intrinsic Dim (95%): {dim95}")

    # ── 2. Local PCA ──
    print(f"[2/6] Local PCA (window={local_pca_window})...")
    local_pca, local_pca_data = compute_local_pca(full_snapshots, window_size=local_pca_window)
    results["local_pca"] = local_pca
    if local_pca_data:
        w_centers, local_dims = local_pca_data
        print(f"  Mean local dim: {local_pca.get('mean_local_dim', 0):.1f}, "
              f"Early: {local_pca.get('early_local_dim', 0):.1f}, "
              f"Late: {local_pca.get('late_local_dim', 0):.1f}")

    # ── 3. Gradient Coherence ──
    print("[3/6] Gradient Coherence...")
    grad_coh, grad_coh_data = compute_gradient_coherence(full_snapshots, snapshot_steps)
    results["gradient_coherence"] = grad_coh
    if grad_coh.get("mean_coherence") is not None:
        print(f"  Mean coherence: {grad_coh['mean_coherence']:.3f}, "
              f"Early: {grad_coh.get('coherence_early', 0):.3f}, "
              f"Late: {grad_coh.get('coherence_late', 0):.3f}, "
              f"Half-life: {grad_coh.get('autocorrelation_half_life', 0)}")

    # ── 4. Phase Detection ──
    print("[4/6] Phase Detection...")
    phase_data, phase_plot_data = detect_phases(snapshot_steps, psnrs, delta_norms)
    results["phase_detection"] = phase_data
    if phase_plot_data:
        for ph in phase_data.get("phases", []):
            print(f"  {ph['name']}: {ph['snapshot_start']}-{ph['snapshot_end']} "
                  f"(PSNR {ph['psnr_start']:.1f} -> {ph['psnr_end']:.1f}, "
                  f"Δ={ph['psnr_gain']:.1f} dB)")

    # ── 5. Directional Persistence ──
    print("[5/6] Directional Persistence...")
    dir_pers, dir_pers_data = compute_directional_persistence(full_snapshots, snapshot_steps)
    results["directional_persistence"] = dir_pers
    if dir_pers_data:
        print(f"  Mean half-life: {dir_pers.get('mean_half_life', 0):.1f}, "
              f"Early: {dir_pers.get('early_half_life', 0):.1f}, "
              f"Late: {dir_pers.get('late_half_life', 0):.1f}")

    # ── 6. Hessian ──
    print("[6/6] Hessian Sharpness (deferred)...")
    results["hessian"] = {"note": "Requires --compute_hessian flag for full computation"}

    # ── Summary ──
    results["summary_metrics"] = {
        "final_psnr": float(psnrs[-1]) if len(psnrs) > 0 else None,
        "pc1_variance_pct": float(pc1 * 100),
        "participation_ratio": float(pr),
        "intrinsic_dim_95": int(dim95),
        "mean_gradient_coherence": float(grad_coh.get("mean_coherence", 0)),
        "coherence_drop": float(grad_coh.get("coherence_early", 0) - grad_coh.get("coherence_late", 0)),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "dynamics_analysis_v2.json"), "w") as f:
            json.dump(results, f, indent=2)

        # ── Plots ──
        generate_plots(
            model_type, save_dir,
            full_snapshots, snapshot_steps, global_pca, delta_norms,
            local_pca_data, grad_coh_data, phase_plot_data, dir_pers_data,
            psnrs, losses, has_enc_dec,
        )

        print(f"\nAnalysis saved to {save_dir}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    s = results["summary_metrics"]
    print(f"  Final PSNR: {s['final_psnr']:.1f} dB" if s['final_psnr'] else "  Final PSNR: N/A")
    print(f"  PC1 Variance: {s['pc1_variance_pct']:.1f}%")
    print(f"  Participation Ratio: {s['participation_ratio']:.1f}")
    print(f"  Intrinsic Dim (95%): {s['intrinsic_dim_95']}")
    print(f"  Gradient Coherence: {s['mean_gradient_coherence']:.3f}")
    if s['coherence_drop']:
        print(f"  Coherence Drop (early→late): {s['coherence_drop']:.3f}")

    return results


def generate_plots(model_type, save_dir, snapshots, steps, pca, delta_norms,
                   local_pca_data, grad_coh_data, phase_plot_data, dir_pers_data,
                   psnrs, losses, has_enc_dec):
    """Generate comprehensive analysis plots."""
    n_snapshots = len(steps)

    fig, axes = plt.subplots(3, 3, figsize=(24, 18))

    # ── Row 1: Global PCA ──
    if "projections_2d" in pca:
        proj = np.array(pca["projections_2d"])
        sc = axes[0, 0].scatter(proj[:, 0], proj[:, 1], c=steps, cmap="viridis", s=60)
        axes[0, 0].plot(proj[:, 0], proj[:, 1], "k-", alpha=0.3, linewidth=0.8)
        plt.colorbar(sc, ax=axes[0, 0], label="Training Step")
        axes[0, 0].set_xlabel("PC1")
        axes[0, 0].set_ylabel("PC2")
        axes[0, 0].set_title("Δθ Trajectory (PC1/PC2)")
        axes[0, 0].set_aspect("equal")
        axes[0, 0].grid(True, alpha=0.3)

    if pca.get("cosine_with_pc1"):
        axes[0, 1].plot(steps, pca["cosine_with_pc1"], "g-", linewidth=2, label="cos(Δθ, PC1)")
        axes[0, 1].plot(steps, pca["cosine_with_pc2"], "m-", linewidth=2, label="cos(Δθ, PC2)")
        axes[0, 1].set_xlabel("Training Step")
        axes[0, 1].set_ylabel("Cosine Similarity")
        axes[0, 1].set_title("Δθ Alignment with PCA Directions")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    ev = pca.get("explained_variance_ratio", [])
    if ev:
        n_show = min(10, len(ev))
        axes[0, 2].bar(range(1, n_show + 1), [v * 100 for v in ev[:n_show]])
        axes[0, 2].set_xlabel("Principal Component")
        axes[0, 2].set_ylabel("Variance Explained %")
        axes[0, 2].set_title(f"PCA Scree Plot (part. ratio={pca.get('participation_ratio', 0):.1f})")
        axes[0, 2].grid(True, alpha=0.3)

    # ── Row 2: Local PCA + Gradient Coherence ──
    if local_pca_data:
        w_centers, local_dims = local_pca_data
        step_centers = [steps[int(c)] for c in w_centers]
        axes[1, 0].plot(step_centers, local_dims, "b-", linewidth=2)
        axes[1, 0].set_xlabel("Training Step")
        axes[1, 0].set_ylabel("Local Intrinsic Dimension")
        axes[1, 0].set_title("Local PCA Dimension (Sliding Window)")
        axes[1, 0].grid(True, alpha=0.3)

    if grad_coh_data:
        coh_steps, coherences = grad_coh_data
        axes[1, 1].plot(coh_steps[:-1], coherences, "c-", linewidth=1.5, alpha=0.7)
        axes[1, 1].set_xlabel("Training Step")
        axes[1, 1].set_ylabel("cos(Δθ_t, Δθ_{t+1})")
        axes[1, 1].set_title("Gradient Coherence")
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(steps, delta_norms, "r-", linewidth=2)
    axes[1, 2].set_xlabel("Training Step")
    axes[1, 2].set_ylabel("||Δθ||")
    axes[1, 2].set_title("Parameter Offset Norm")
    axes[1, 2].grid(True, alpha=0.3)

    # ── Row 3: Phase Detection + PSNR + Loss ──
    if phase_plot_data:
        ph_steps, ph_psnr, ph_slope, ph_labels = phase_plot_data
        colors = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}
        for label in np.unique(ph_labels):
            mask = ph_labels == label
            axes[2, 0].scatter(ph_steps[mask], ph_psnr[mask],
                             c=colors.get(int(label), "gray"), s=60,
                             label={0: "Rapid Descent", 1: "Refinement", 2: "Convergence"}.get(int(label), "?"))
        axes[2, 0].set_xlabel("Training Step")
        axes[2, 0].set_ylabel("PSNR (dB)")
        axes[2, 0].set_title("Phase Detection (PSNR)")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    if len(psnrs) > 0:
        axes[2, 1].plot(steps[1:], psnrs, "b-", linewidth=2)
        axes[2, 1].set_xlabel("Training Step")
        axes[2, 1].set_ylabel("PSNR (dB)")
        axes[2, 1].set_title("PSNR vs Training Step")
        axes[2, 1].grid(True, alpha=0.3)

    if len(losses) > 0:
        loss_steps = np.arange(1, len(losses) + 1)
        axes[2, 2].semilogy(loss_steps, losses, "r-", linewidth=1, alpha=0.7)
        axes[2, 2].set_xlabel("Training Step")
        axes[2, 2].set_ylabel("Loss (MSE)")
        axes[2, 2].set_title("Training Loss")
        axes[2, 2].grid(True, alpha=0.3)

    plt.suptitle(f"Fitting Dynamics Analysis v2 — {model_type}", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dynamics_analysis_v2.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Extra: Directional Persistence Plot ──
    if dir_pers_data:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(dir_pers_data, "b-", linewidth=2)
        ax2.axhline(y=np.mean(dir_pers_data), color="r", linestyle="--",
                    label=f"Mean = {np.mean(dir_pers_data):.1f}")
        ax2.set_xlabel("Window Index")
        ax2.set_ylabel("Autocorrelation Half-Life (steps)")
        ax2.set_title("Directional Persistence Over Training")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "directional_persistence.png"), dpi=150, bbox_inches="tight")
        plt.close()


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Fitting Dynamics Analysis v2"
    )
    parser.add_argument("--trajectory", type=str, required=True,
                       help="Path to trajectory.npz")
    parser.add_argument("--model", type=str, default="siren",
                       choices=["siren", "liif", "lte"])
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--local_pca_window", type=int, default=5,
                       help="Window size for local PCA (default: 5)")
    parser.add_argument("--compute_hessian", action="store_true",
                       help="Compute Hessian top eigenvalue (expensive)")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(os.path.dirname(args.trajectory), "analysis_v2")
    analyze_dynamics(
        args.trajectory,
        model_type=args.model,
        save_dir=save_dir,
        local_pca_window=args.local_pca_window,
    )


if __name__ == "__main__":
    main()
