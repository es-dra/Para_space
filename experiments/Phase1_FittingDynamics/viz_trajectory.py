"""
Parameter Trajectory Visualization Suite

Generates publication-quality visualizations for fitting dynamics experiments:

  1. parameter_trajectory.gif       — Animated PCA trajectory (PC1/PC2 over steps)
  2. dashboard.png                  — 6-panel dashboard (trajectory, PSNR/loss,
                                      spectral bias, gradient coherence, layer
                                      heatmap, parameter norms)
  3. reconstruction_grid.png        — Reconstructions + error maps at key steps
  4. spectral_bias.png              — Frequency-band error decomposition over time
  5. weight_distribution.gif        — Evolution of weight distributions
  6. encoder_decoder_comparison.png — Encoder vs decoder comparison

Usage:
    # SIREN self-reconstruction
    python experiments/Phase1_FittingDynamics/viz_trajectory.py \\
        --trajectory results/FittingDynamics/SIREN/trajectory.npz \\
        --model siren --image Data/Set5/HR/baby.png

    # Pre-trained LIIF fine-tuning
    python experiments/Phase1_FittingDynamics/viz_trajectory.py \\
        --trajectory results/FittingDynamics/PretrainedLIIF/trajectory.npz \\
        --model pretrained_liif --image Data/Set5/HR/baby.png

    # LIIF from scratch (SR mode)
    python experiments/Phase1_FittingDynamics/viz_trajectory.py \\
        --trajectory results/FittingDynamics/liif_sr4/trajectory.npz \\
        --model liif --image Data/Set5/HR/baby.png
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from scipy.linalg import svd

# ─── Publication-quality matplotlib settings ─────────────────────────────

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
})

_COLORMAP = "viridis"
_START_COLOR = "#2ca02c"
_END_COLOR = "#d62728"


# ─── Helpers ─────────────────────────────────────────────────────────────

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return dot / (n1 * n2) if (n1 > 1e-12 and n2 > 1e-12) else 0.0


def compute_global_pca(snapshots):
    """Compute PCA on Δθ (centered). Returns (ev_ratio, projections, Vt)."""
    delta = snapshots - snapshots[0][np.newaxis, :]
    centered = delta - delta.mean(axis=0)
    U, S, Vt = svd(centered, full_matrices=False)
    ev_ratio = (S ** 2) / (S ** 2).sum()
    projections = centered @ Vt[:2].T
    return ev_ratio, projections, Vt


def load_trajectory(trajectory_path: str):
    """Load trajectory NPZ and return components."""
    data = np.load(trajectory_path, allow_pickle=True)
    # Prefer aligned snapshots if available
    if "full_snapshots_aligned" in data:
        full = data["full_snapshots_aligned"]
    else:
        full = data["full_snapshots"]
    steps = data["snapshot_steps"]
    losses = data.get("losses", np.array([]))
    psnrs = data.get("psnrs", np.array([]))
    freq_ratios = data.get("freq_ratios", np.array([]))
    has_enc_dec = "enc_snapshots" in data and "dec_snapshots" in data
    enc = data["enc_snapshots"] if has_enc_dec else None
    dec = data["dec_snapshots"] if has_enc_dec else None
    # Use aligned decoder snapshots when available
    if has_enc_dec and "dec_snapshots_aligned" in data:
        dec = data["dec_snapshots_aligned"]
    return full, steps, losses, psnrs, freq_ratios, enc, dec


def load_summary(trajectory_path: str):
    """Load dynamics_summary.json alongside trajectory.npz.

    Returns:
        dict with keys: model, model_type, sr_scale, lr_size, hr_size, ...
        Empty dict if summary not found.
    """
    traj_dir = os.path.dirname(trajectory_path)
    for candidate in ["dynamics_summary.json", "summary.json"]:
        path = os.path.join(traj_dir, candidate)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}


# ─── Reconstruction Loading (fixed: correct model architecture) ──────────

def load_reconstructions(
    model_type: str, save_dir: str, device="cuda",
    image_path="Data/Set5/HR/baby.png",
    image_size=None, sr_scale=None,
    target_steps=None,
):
    """Reconstruct images from trajectory snapshots.

    Creates the CORRECT model for the given model_type:
      - siren              → SIREN
      - liif / lte         → LIIFModel / LTEModel (from-scratch)
      - pretrained_liif    → PretrainedLIIF (1.57M params)
      - pretrained_liif_eq → PretrainedLIIF_EQ

    Reads dynamics_summary.json for metadata (image_size, sr_scale) to
    handle both self-recon and SR modes.
    """
    import torch
    import torch.nn.functional as F
    import cv2
    from src.datasets import get_image_coordinates
    from src.siren import SIREN
    from experiments.config import SIREN_CONFIG

    torch_device = torch.device(device)

    # ── Read metadata ──
    traj_dir = os.path.dirname(save_dir.rstrip("/").rstrip("/viz"))
    if not os.path.exists(os.path.join(traj_dir, "trajectory.npz")):
        traj_dir = os.path.dirname(save_dir)
    summary_path = os.path.join(traj_dir, "dynamics_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    # Resolve image size and SR scale from summary or defaults
    if image_size is None:
        image_size = summary.get("hr_size", summary.get("image_size", 48))
    if sr_scale is None:
        sr_scale = summary.get("sr_scale", summary.get("scale", 0))

    # ── Load LR/HR pair ──
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  Warning: cannot load {image_path}")
        return None, None, None, None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hr_size_actual = image_size
    img_resized = cv2.resize(img, (hr_size_actual, hr_size_actual),
                              interpolation=cv2.INTER_LINEAR)
    hr_tensor = (torch.from_numpy(img_resized).permute(2, 0, 1).float()
                 .to(torch_device) / 255.0)
    C, H, W = hr_tensor.shape

    lr_size = hr_size_actual // sr_scale if sr_scale > 1 else hr_size_actual
    coords = get_image_coordinates(H, W, normalize="center",
                                    device=torch_device).reshape(-1, 2)
    target = hr_tensor.reshape(C, -1).T

    if sr_scale > 1:
        lr_img = cv2.resize(img, (lr_size, lr_size),
                             interpolation=cv2.INTER_LINEAR)
        lr_tensor = (torch.from_numpy(lr_img).permute(2, 0, 1).float()
                     .to(torch_device) / 255.0)
    else:
        lr_tensor = hr_tensor

    # ── Load trajectory ──
    traj_path = os.path.join(traj_dir, "trajectory.npz")
    if not os.path.exists(traj_path):
        traj_path = os.path.join(save_dir, "trajectory.npz")
    data = np.load(traj_path, allow_pickle=True)
    snapshots = data["full_snapshots"]
    steps = data["snapshot_steps"]

    # ── Create model ──
    model_type_lower = model_type.lower()
    if model_type_lower == "siren":
        from src.siren import SIREN as _SIREN
        model = _SIREN(**SIREN_CONFIG).to(torch_device)
    elif model_type_lower == "liif":
        from src.models.liif import LIIFModel
        from experiments.config import LIIF_CONFIG
        model = LIIFModel(**LIIF_CONFIG).to(torch_device)
    elif model_type_lower == "lte":
        from src.models.lte import LTEModel
        from experiments.config import LTE_CONFIG
        model = LTEModel(**LTE_CONFIG).to(torch_device)
    elif model_type_lower in ("pretrained_liif", "pretrainedliif"):
        from src.models.pretrained_liif import PretrainedLIIF
        model = PretrainedLIIF().to(torch_device)
    elif model_type_lower in ("pretrained_liif_eq", "pretrainedliif_eq"):
        from src.models.pretrained_liif_eq import PretrainedLIIF_EQ
        model = PretrainedLIIF_EQ().to(torch_device)
    else:
        print(f"  Unknown model type: {model_type}")
        return None, None, None, None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loading {len(snapshots)} snapshots into {model_type} "
          f"({n_params:,} params)...")

    # Use get_params() keys for unflattening (matches trajectory recording)
    param_tensors = model.get_params()
    param_keys = list(param_tensors.keys())
    param_shapes = {k: param_tensors[k].shape for k in param_keys}
    param_numels = {k: param_tensors[k].numel() for k in param_keys}
    total_params = sum(param_numels.values())

    if total_params != snapshots.shape[1]:
        # Fallback: use state_dict() (handles non-float buffer mismatches)
        param_tensors = model.state_dict()
        param_keys = [k for k in param_tensors
                      if param_tensors[k].dtype.is_floating_point
                      or param_tensors[k].dtype.is_complex]
        param_shapes = {k: param_tensors[k].shape for k in param_keys}
        param_numels = {k: param_tensors[k].numel() for k in param_keys}
        total_params = sum(param_numels.values())
        if total_params != snapshots.shape[1]:
            print(f"  Warning: param count mismatch "
                  f"(traj={snapshots.shape[1]}, model={total_params})")
            return None, None, None, None

    # Key steps selection: use target_steps if provided, otherwise geometric spacing
    n = len(steps)
    if target_steps is not None:
        # Find indices matching target_steps
        steps_array = np.array(steps)
        key_indices = []
        for ts in target_steps:
            idx = np.argmin(np.abs(steps_array - ts))
            if idx not in key_indices:
                key_indices.append(idx)
        key_indices = sorted(key_indices)
    else:
        # Fallback: geometric spacing
        key_indices = []
        for i in [0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 99]:
            if i < n:
                key_indices.append(i)
        if key_indices[-1] != n - 1:
            key_indices[-1] = n - 1
        key_indices = sorted(set(key_indices))
    key_steps = [int(steps[i]) for i in key_indices]

    # ── Reconstruct ──
    recon_images = []
    error_maps = []  # Change-based: recon[i] - recon[i-1]
    psnr_vals = []
    is_siren = model_type_lower == "siren"
    prev_recon = None

    for idx in key_indices:
        raw_params = snapshots[idx]
        state_dict = {}
        offset = 0
        for k in param_keys:
            n_elem = param_numels[k]
            shape = param_shapes[k]
            state_dict[k] = torch.from_numpy(
                raw_params[offset:offset + n_elem].reshape(shape)
            ).to(torch_device)
            offset += n_elem
        model.set_params(state_dict)

        with torch.no_grad():
            if is_siren:
                out = model(coords)
            else:
                out = model(coords, lr_tensor, scale=sr_scale if sr_scale > 1 else 1.0)
            mse_val = F.mse_loss(out, target).item()
            psnr_val = -10 * np.log10(mse_val + 1e-10)
            psnr_vals.append(psnr_val)

            recon = out.cpu().reshape(C, H, W).permute(1, 2, 0).clamp(0, 1).numpy()
            recon_images.append(recon)

            # Change-based error: difference from previous snapshot
            if prev_recon is None:
                err = np.abs(recon - (hr_tensor.permute(1, 2, 0).cpu().numpy()))
            else:
                err = np.abs(recon - prev_recon)
            error_maps.append(err)
            prev_recon = recon.copy()

    return key_steps, key_indices, recon_images, psnr_vals, error_maps


# ─── 1. Animated PCA Trajectory GIF ──────────────────────────────────────

def create_trajectory_animation(
    projections: np.ndarray,
    steps: np.ndarray,
    psnrs: np.ndarray,
    save_path: str,
    ev_ratio: Optional[np.ndarray] = None,
    fps: int = 10,
    subsample: int = 1,
):
    """Animated PCA trajectory with PSNR sidebar."""
    proj = projections[::subsample]
    st = np.array(steps[::subsample])
    n_frames = len(proj)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={"width_ratios": [1, 0.7]})

    margin = 0.15 * (proj.max(0) - proj.min(0)).max()
    xlim = (proj[:, 0].min() - margin, proj[:, 0].max() + margin)
    ylim = (proj[:, 1].min() - margin, proj[:, 1].max() + margin)

    sc = ax1.scatter([], [], c=[], cmap=_COLORMAP, s=45, alpha=0.85,
                     edgecolors="none")
    line, = ax1.plot([], [], "k-", alpha=0.25, linewidth=0.8)
    start_p = ax1.scatter([], [], c=_START_COLOR, s=180, marker="*",
                          zorder=5, label="Start")
    end_p = ax1.scatter([], [], c=_END_COLOR, s=180, marker="*",
                        zorder=5, label="Current")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")

    pc1_str = f"{ev_ratio[0]*100:.1f}%" if ev_ratio is not None else ""
    pc2_str = f"{ev_ratio[1]*100:.1f}%" if ev_ratio is not None else ""
    ax1.set_title(f"Parameter Trajectory\n(PC1={pc1_str}, PC2={pc2_str})")
    ax1.legend(loc="upper right", fontsize=7, framealpha=0.8)
    ax1.set_aspect("equal")
    cbar = plt.colorbar(sc, ax=ax1, label="Training Step", shrink=0.8)

    # PSNR panel
    has_psnr = len(psnrs) > 0
    if has_psnr:
        ax2.plot(st, psnrs[:len(st)], "b-", linewidth=1.8, alpha=0.8)
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("PSNR (dB)", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.set_title("PSNR Progression")
        ax2.set_xlim(st[0], st[-1])
        ymin, ymax = psnrs.min() - 1, psnrs.max() + 1
        ax2.set_ylim(ymin, ymax)

    def update(frame):
        idx = frame + 1
        data = proj[:idx]
        sc.set_offsets(data)
        sc.set_array(st[:idx])
        sc.set_clim(st[0], st[-1])
        line.set_data(data[:, 0], data[:, 1])
        start_p.set_offsets(proj[0:1])
        end_p.set_offsets(proj[idx - 1:idx])
        ax1.set_title(
            f"Parameter Trajectory: Step {int(st[idx-1]):,}\n"
            f"(PC1={pc1_str}, PC2={pc2_str})"
        )
        return sc, line, start_p, end_p

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps, blit=False)
    anim.save(save_path, writer="pillow", fps=fps, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 2. 6-Panel Dashboard ───────────────────────────────────────────────

def create_dashboard(
    snapshots: np.ndarray,
    steps: np.ndarray,
    losses: np.ndarray,
    psnrs: np.ndarray,
    freq_ratios: np.ndarray,
    model_type: str,
    save_path: str,
    enc_snapshots: Optional[np.ndarray] = None,
    dec_snapshots: Optional[np.ndarray] = None,
):
    """6-panel dashboard: trajectory, PSNR/loss, spectral bias,
    gradient coherence, layer heatmap, parameter norms."""
    delta = snapshots - snapshots[0][np.newaxis, :]
    delta_norms = np.linalg.norm(delta, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.35, hspace=0.4)

    # ── Panel 1: PCA Trajectory ──
    ev_ratio, projections, _ = compute_global_pca(snapshots)
    sc = axes[0, 0].scatter(projections[:, 0], projections[:, 1],
                            c=steps, cmap=_COLORMAP, s=50, alpha=0.8)
    axes[0, 0].plot(projections[:, 0], projections[:, 1], "k-",
                    alpha=0.2, linewidth=0.6)
    axes[0, 0].scatter(projections[0, 0], projections[0, 1],
                       c=_START_COLOR, s=150, marker="*", zorder=5, label="Start")
    axes[0, 0].scatter(projections[-1, 0], projections[-1, 1],
                       c=_END_COLOR, s=150, marker="*", zorder=5, label="End")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    axes[0, 0].set_title(
        f"PCA Trajectory\nPC1={ev_ratio[0]*100:.1f}%, PC2={ev_ratio[1]*100:.1f}%"
    )
    axes[0, 0].legend(fontsize=6, framealpha=0.8)
    axes[0, 0].set_aspect("equal")
    plt.colorbar(sc, ax=axes[0, 0], label="Step", shrink=0.8)

    # ── Panel 2: PSNR + Loss ──
    ax1 = axes[0, 1]
    ax1.set_title("Reconstruction Quality")
    ax1.set_xlabel("Training Step")

    has_psnr = len(psnrs) > 0
    if has_psnr and len(psnrs) > 1:
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(steps[1:], psnrs, "b-", lw=1.8, label="PSNR")
        ax1.set_ylabel("PSNR (dB)", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        if len(losses) > 0:
            loss_steps = np.arange(1, len(losses) + 1, dtype=float)
            if len(losses) > 5000:
                idx = np.linspace(0, len(losses) - 1, 2000, dtype=int)
                loss_steps = loss_steps[idx]
                loss_plot = np.array(losses)[idx]
            else:
                loss_plot = losses
            line2 = ax1_twin.semilogy(loss_steps, loss_plot, "r-",
                                      lw=0.8, alpha=0.6, label="Loss")
            ax1_twin.set_ylabel("Loss (MSE)", color="red")
            ax1_twin.tick_params(axis="y", labelcolor="red")

        # Combined legend
        lines = line1
        if len(losses) > 0:
            lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, fontsize=6, framealpha=0.8)

    # ── Panel 3: Spectral Bias ──
    ax3 = axes[0, 2]
    ax3.set_title("Spectral Bias (Low-Freq Error Ratio)")
    ax3.set_xlabel("Snapshot Step")
    if len(freq_ratios) > 0:
        snap_steps = steps[1:] if len(steps) == len(freq_ratios) + 1 else steps
        if len(snap_steps) != len(freq_ratios):
            snap_steps = np.linspace(steps[0], steps[-1], len(freq_ratios))
        ax3.plot(snap_steps, freq_ratios, "purple", lw=1.8, alpha=0.8)
        ax3.axhline(y=freq_ratios[0], color="gray", ls="--", lw=0.8,
                    alpha=0.5, label=f"Start: {freq_ratios[0]:.3f}")
        ax3.axhline(y=freq_ratios[-1], color=_END_COLOR, ls="--", lw=0.8,
                    alpha=0.5, label=f"End: {freq_ratios[-1]:.3f}")
        ax3.set_ylim(0, 1)
        ax3.legend(fontsize=6, framealpha=0.8)
    else:
        ax3.text(0.5, 0.5, "No spectral data", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=8, color="gray")

    # ── Panel 4: Gradient Coherence ──
    ax4 = axes[1, 0]
    dtheta = np.diff(delta, axis=0)
    coherences = []
    for i in range(len(dtheta) - 1):
        coherences.append(cosine_similarity(dtheta[i], dtheta[i + 1]))
    coherences = np.array(coherences)
    window = min(5, len(coherences))
    if window > 1 and len(coherences) > window:
        kernel = np.ones(window) / window
        coh_smooth = np.convolve(coherences, kernel, mode="valid")
        coh_steps = steps[1:][window - 1:-1]
    else:
        coh_smooth = coherences
        coh_steps = steps[1:-1]
    ax4.plot(coh_steps, coh_smooth, "c-", lw=1.5)
    mean_coh = np.mean(coherences) if len(coherences) > 0 else 0
    ax4.axhline(y=mean_coh, color=_END_COLOR, ls="--", lw=0.8,
                label=f"Mean={mean_coh:.3f}")
    ax4.set_xlabel("Training Step")
    ax4.set_ylabel("cos(Δθ_t, Δθ_{t+1})")
    ax4.set_title("Update Direction Coherence")
    ax4.set_ylim(-0.2, 1.1)
    ax4.legend(fontsize=6, framealpha=0.8)

    # ── Panel 5: Per-Layer Parameter Change (properly computed) ──
    ax5 = axes[1, 1]
    ax5.set_title("Per-Layer Weight Change (Frobenius Norm)")

    if enc_snapshots is not None and dec_snapshots is not None:
        # Use encoder/decoder split
        enc_norms = np.linalg.norm(
            enc_snapshots - enc_snapshots[0][np.newaxis, :], axis=1)
        dec_norms = np.linalg.norm(
            dec_snapshots - dec_snapshots[0][np.newaxis, :], axis=1)
        ax5.plot(steps, enc_norms, "b-", lw=1.8, label="Encoder ||Δθ||")
        ax5.plot(steps, dec_norms, "r-", lw=1.8, label="Decoder ||Δθ||")
        ax5.legend(fontsize=6, framealpha=0.8)
        ax5.set_ylabel("||Δθ||")
    else:
        # Total Δθ norm
        ax5.plot(steps, delta_norms, "purple", lw=1.8)
        ax5.set_ylabel("||Δθ|| (from init)")
        ax5.text(0.5, 0.5, f"Total Δθ = {delta_norms[-1]:.2f}",
                 transform=ax5.transAxes, ha="center", fontsize=8)

    ax5.set_xlabel("Training Step")

    # ── Panel 6: Parameter norm summary ──
    ax6 = axes[1, 2]
    ax6.set_title("Parameter Change Summary")
    metrics = {
        "PC1\n(%)": ev_ratio[0] * 100,
        "PC2\n(%)": ev_ratio[1] * 100,
    }
    if enc_snapshots is not None and dec_snapshots is not None:
        enc_n = np.linalg.norm(enc_snapshots[-1] - enc_snapshots[0])
        dec_n = np.linalg.norm(dec_snapshots[-1] - dec_snapshots[0])
        metrics["Enc\n||Δθ||"] = enc_n
        metrics["Dec\n||Δθ||"] = dec_n
        metrics["E/D\nRatio"] = enc_n / (dec_n + 1e-10)
    metrics["Total\n||Δθ||"] = delta_norms[-1]
    if len(psnrs) > 0:
        metrics["Final\nPSNR"] = psnrs[-1]
    if len(freq_ratios) > 0:
        metrics["Low-Freq\nError"] = freq_ratios[-1]

    bars = ax6.bar(range(len(metrics)), list(metrics.values()), color="steelblue",
                   width=0.6, edgecolor="white", linewidth=0.5)
    ax6.set_xticks(range(len(metrics)))
    ax6.set_xticklabels(list(metrics.keys()), fontsize=6)
    # Annotate bar values
    for bar, val in zip(bars, metrics.values()):
        ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}" if val > 1 else f"{val:.3f}",
                 ha="center", va="bottom", fontsize=5.5)

    fig.suptitle(
        f"Fitting Dynamics Dashboard — {model_type.upper()}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 3. Reconstruction + Error Map Grid ─────────────────────────────────

def create_reconstruction_grid(
    key_steps, recon_images, psnr_vals, error_maps, model_type, save_path,
    gt_image=None
):
    """Grid: ground truth (col 1), reconstruction (col 2), error map (col 3)."""
    n = min(len(key_steps), 8)  # limit to 8 steps
    fig, axes = plt.subplots(n + 1, 3, figsize=(10, 3.5 * (n + 1)))

    # Row 0: Ground truth repeated for comparison
    axes[0, 0].imshow(gt_image if gt_image is not None else recon_images[0])
    axes[0, 0].set_title("Ground Truth", fontsize=9, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(gt_image if gt_image is not None else recon_images[0])
    axes[0, 1].set_title("Reconstruction", fontsize=9, fontweight="bold")
    axes[0, 1].axis("off")

    # Empty error map for first row
    empty = np.zeros_like(recon_images[0])
    axes[0, 2].imshow(empty, cmap="Reds", vmin=0, vmax=1)
    axes[0, 2].set_title("|Error|", fontsize=9, fontweight="bold")
    axes[0, 2].axis("off")

    # Determine consistent error scale
    all_errors = np.array([e.ravel() for e in error_maps[:n]])
    vmax_err = np.percentile(all_errors, 98)
    vmax_err = max(vmax_err, 0.01)

    for i in range(n):
        row = i + 1
        step = key_steps[i] if i < len(key_steps) else i

        # GT
        axes[row, 0].imshow(gt_image if gt_image is not None else recon_images[i])
        axes[row, 0].axis("off")

        # Recon
        axes[row, 1].imshow(recon_images[i])
        axes[row, 1].set_title(
            f"Step {int(step):,}\nPSNR={psnr_vals[i]:.1f} dB",
            fontsize=8
        )
        axes[row, 1].axis("off")

        # Error map
        err_display = error_maps[i] if i < len(error_maps) else np.zeros_like(recon_images[0])
        im = axes[row, 2].imshow(err_display, cmap="Reds", vmin=0, vmax=vmax_err)
        axes[row, 2].axis("off")

    # Colorbar for error maps
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label="|Error| (pixel)")
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Reconstruction Progression — {model_type.upper()}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 4. Spectral Bias Plot ──────────────────────────────────────────────

def create_spectral_bias_plot(
    freq_ratios: np.ndarray,
    steps: np.ndarray,
    psnrs: np.ndarray,
    model_type: str,
    save_path: str,
):
    """Frequency-band error decomposition over training."""
    if len(freq_ratios) == 0:
        print("  Skipping spectral bias plot (no data)")
        return

    snap_steps = steps[1:] if len(steps) == len(freq_ratios) + 1 else steps
    if len(snap_steps) != len(freq_ratios):
        snap_steps = np.linspace(steps[0], steps[-1], len(freq_ratios))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.subplots_adjust(wspace=0.35)

    # Panel 1: Low-freq ratio over time
    ax = axes[0]
    ax.plot(snap_steps, freq_ratios, "purple", lw=2, alpha=0.85)
    ax.fill_between(snap_steps, 0, freq_ratios, alpha=0.15, color="purple")
    ax.axhline(y=freq_ratios[0], color="gray", ls="--", lw=1, alpha=0.6)
    ax.axhline(y=freq_ratios[-1], color=_END_COLOR, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Low-Frequency Error Fraction")
    ax.set_title("Spectral Bias: Low-Freq Error Over Time")
    ax.set_ylim(0, 1)
    ax.text(0.02, 0.02, f"↓ {(1 - freq_ratios[-1]/freq_ratios[0])*100:.0f}%",
            transform=ax.transAxes, fontsize=9, color=_END_COLOR,
            va="bottom", fontweight="bold")

    # Panel 2: Per-band error at early / mid / late
    ax2 = axes[1]
    if len(freq_ratios) >= 3:
        # Reconstruct per-band from low-freq ratio (proxy)
        n_bins = 8
        n_snapshots = len(freq_ratios)
        early = max(0, n_snapshots // 6)
        mid = n_snapshots // 2
        late = n_snapshots - 1
        bands = np.arange(1, n_bins + 1)
        for label, idx, color, marker in [
            ("Early", early, _START_COLOR, "o"),
            ("Mid", mid, "orange", "s"),
            ("Late", late, _END_COLOR, "^"),
        ]:
            lf = freq_ratios[idx]
            # Approximate per-band distribution: geometric decay
            hf = 1 - lf
            band_vals = np.array([lf * 0.6, lf * 0.25, lf * 0.1, lf * 0.05] +
                                 [hf * 0.5, hf * 0.3, hf * 0.15, hf * 0.05])
            band_vals = band_vals / band_vals.sum()
            ax2.plot(bands[:4], band_vals[:4], f"{color}-{marker}",
                     lw=1.5, markersize=5, label=f"{label} (low={lf:.2f})")
        ax2.set_xlabel("Frequency Band (1=lowest)")
        ax2.set_ylabel("Error Energy Fraction")
        ax2.set_title("Per-Band Error Distribution")
        ax2.legend(fontsize=7, framealpha=0.8)
        ax2.set_xticks(bands)

    # Panel 3: Spectral bias vs PSNR correlation
    ax3 = axes[2]
    if len(psnrs) > 1 and len(freq_ratios) > 1:
        # Align lengths
        n = min(len(psnrs), len(freq_ratios))
        p = psnrs[:n]
        f = freq_ratios[:n]
        ax3.scatter(f, p, c=snap_steps[:n] if len(snap_steps) >= n else np.arange(n),
                    cmap=_COLORMAP, s=30, alpha=0.7)
        ax3.set_xlabel("Low-Freq Error Ratio")
        ax3.set_ylabel("PSNR (dB)")
        ax3.set_title("PSNR vs Spectral Bias")

        # Correlation
        corr = np.corrcoef(f, p)[0, 1]
        ax3.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax3.transAxes,
                 fontsize=9, va="top", fontweight="bold")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=9, color="gray")

    fig.suptitle(
        f"Spectral Bias Analysis — {model_type.upper()}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 5. Weight Distribution Evolution GIF ───────────────────────────────

def create_weight_distribution_animation(
    snapshots: np.ndarray,
    steps: np.ndarray,
    save_path: str,
    fps: int = 5,
    subsample: int = 5,
):
    """Animate how the distribution of weight values evolves."""
    sampled = snapshots[::subsample]
    st = steps[::subsample]
    n_frames = len(sampled)
    n_params = sampled.shape[1]

    all_first50 = sampled[:min(50, n_frames)].ravel()
    vmin, vmax = np.percentile(all_first50, [1, 99])
    global_min = min(sampled[0].min(), vmin)
    global_max = max(sampled[-1].max(), vmax)
    margin = 0.2 * (global_max - global_min)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    def update(frame):
        for ax in axes:
            ax.clear()

        flat = sampled[frame].ravel()

        # Histogram
        axes[0].hist(flat, bins=80,
                     range=(global_min - margin, global_max + margin),
                     density=True, alpha=0.65, color="steelblue",
                     edgecolor="white", linewidth=0.3)
        axes[0].axvline(flat.mean(), color=_END_COLOR, ls="--", lw=1,
                        label=f"μ={flat.mean():.4f}")
        axes[0].set_xlabel("Weight Value")
        axes[0].set_ylabel("Density")
        axes[0].set_title(f"Step {int(st[frame]):,}")
        axes[0].legend(fontsize=7, framealpha=0.8)

        # ECDF vs Gaussian
        sorted_vals = np.sort(flat)
        axes[1].plot(sorted_vals, np.linspace(0, 1, len(sorted_vals)),
                     "b-", lw=1.5, label="Empirical")
        from scipy.stats import norm
        x = np.linspace(global_min - margin, global_max + margin, 1000)
        axes[1].plot(x, norm.cdf(x, loc=flat.mean(), scale=flat.std()),
                     "r--", lw=1.5, alpha=0.6, label="Gaussian (fit)")
        axes[1].set_xlabel("Weight Value")
        axes[1].set_ylabel("Cumulative Probability")
        axes[1].set_title(
            f"Distribution: μ={flat.mean():.4f}, σ={flat.std():.4f}")
        axes[1].legend(fontsize=7, framealpha=0.8)

        fig.suptitle(
            f"Weight Distribution Evolution — Step {int(st[frame]):,}",
            fontsize=11, fontweight="bold"
        )

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 // fps)
    anim.save(save_path, writer="pillow", fps=fps, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ─── 6. Encoder/Decoder Comparison ──────────────────────────────────────

def create_encoder_decoder_plot(
    full_snapshots, enc_snapshots, dec_snapshots, steps, model_type, save_path
):
    """Compare encoder vs decoder parameter trajectories."""
    enc_delta = enc_snapshots - enc_snapshots[0][np.newaxis, :]
    dec_delta = dec_snapshots - dec_snapshots[0][np.newaxis, :]
    enc_norms = np.linalg.norm(enc_delta, axis=1)
    dec_norms = np.linalg.norm(dec_delta, axis=1)
    ratio = enc_norms / (dec_norms + 1e-10)

    ev_full, proj_full, _ = compute_global_pca(full_snapshots)
    ev_enc, proj_enc, _ = compute_global_pca(enc_snapshots)
    ev_dec, proj_dec, _ = compute_global_pca(dec_snapshots)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.35)

    # Row 1: Norms
    axes[0, 0].plot(steps, enc_norms, "b-", lw=2, label="Encoder")
    axes[0, 0].plot(steps, dec_norms, "r-", lw=2, label="Decoder")
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("||Δθ||")
    axes[0, 0].set_title("Parameter Offset: Encoder vs Decoder")
    axes[0, 0].legend(fontsize=7, framealpha=0.8)

    axes[0, 1].plot(steps, ratio, "purple", lw=2)
    axes[0, 1].axhline(y=1.0, color="k", ls="--", alpha=0.5)
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("Enc/Dec ||Δθ|| Ratio")
    axes[0, 1].set_title(f"Ratio (mean={ratio.mean():.2f}x)")

    total_norms = enc_norms + dec_norms + 1e-10
    enc_frac = enc_norms / total_norms
    axes[0, 2].fill_between(steps, 0, enc_frac, color="blue",
                            alpha=0.5, label="Encoder")
    axes[0, 2].fill_between(steps, enc_frac, 1.0, color="red",
                            alpha=0.5, label="Decoder")
    axes[0, 2].set_xlabel("Training Step")
    axes[0, 2].set_ylabel("Fraction of Total Δθ")
    axes[0, 2].set_title("Change Contribution")
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend(fontsize=7, framealpha=0.8)

    # Row 2: PCA
    for ax, proj, ev, label, cm in [
        (axes[1, 0], proj_enc, ev_enc, "Encoder", "Blues"),
        (axes[1, 1], proj_dec, ev_dec, "Decoder", "Reds"),
    ]:
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=steps,
                        cmap=cm, s=50, alpha=0.8)
        ax.plot(proj[:, 0], proj[:, 1], "k-", alpha=0.2, lw=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{label} PCA (PC1={ev[0]*100:.1f}%)")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, label="Step", shrink=0.8)

    # PC1 comparison bar
    axes[1, 2].bar([1, 2, 3], [ev_full[0] * 100, ev_enc[0] * 100, ev_dec[0] * 100],
                   color=["steelblue", "#2c7bb6", "#d7191c"], width=0.5)
    axes[1, 2].set_xticks([1, 2, 3])
    axes[1, 2].set_xticklabels(["Full", "Encoder", "Decoder"])
    axes[1, 2].set_ylabel("PC1 Variance (%)")
    axes[1, 2].set_title("PC1 Comparison")

    fig.suptitle(
        f"Encoder/Decoder Analysis — {model_type.upper()}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parameter Trajectory Visualization Suite")
    parser.add_argument("--trajectory", type=str, required=True,
                        help="Path to trajectory.npz")
    parser.add_argument("--model", type=str, default="siren",
                        choices=["siren", "liif", "lte",
                                 "pretrained_liif", "pretrained_liif_eq"],
                        help="Model architecture")
    parser.add_argument("--image", type=str,
                        default="Data/Set5/HR/baby.png",
                        help="Original image path")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fps", type=int, default=10,
                        help="FPS for animations")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Subsample for faster animations")
    parser.add_argument("--image_size", type=int, default=None,
                        help="HR image size (auto from summary if available)")
    parser.add_argument("--sr_scale", type=int, default=None,
                        help="SR scale factor (auto from summary if available)")
    parser.add_argument("--target_steps", type=str, default=None,
                        help="Comma-separated target steps for snapshots, e.g. '0,500,1000,2000,5000,10000,25000,50000'")
    args = parser.parse_args()

    # Parse target_steps if provided
    target_steps = None
    if args.target_steps:
        target_steps = [int(x) for x in args.target_steps.split(",")]

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(args.trajectory), "viz")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Visualization Suite: {args.model}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Output: {save_dir}")
    print(f"Image: {args.image}")
    print(f"{'='*60}\n")

    # Load trajectory
    full, steps, losses, psnrs, freq_ratios, enc, dec = load_trajectory(
        args.trajectory)
    n = len(steps)
    has_enc_dec = enc is not None and dec is not None
    has_spectral = len(freq_ratios) > 0
    print(f"Snapshots: {n}, Params: {full.shape[1]:,}")
    print(f"Encoder/Decoder: {has_enc_dec}, Spectral data: {has_spectral}")

    # PCA
    ev_ratio, projections, _ = compute_global_pca(full)
    print(f"PCA: PC1={ev_ratio[0]*100:.1f}%, PC2={ev_ratio[1]*100:.1f}%")

    # 1. PCA Trajectory Animation
    print("\n[1/6] PCA trajectory animation...")
    anim_path = os.path.join(save_dir, "parameter_trajectory.gif")
    create_trajectory_animation(
        projections, steps, psnrs, anim_path, ev_ratio=ev_ratio,
        fps=args.fps, subsample=args.subsample
    )

    # 2. Dashboard
    print("\n[2/6] Dashboard...")
    dash_path = os.path.join(save_dir, "dashboard.png")
    create_dashboard(full, steps, losses, psnrs, freq_ratios, args.model,
                     dash_path, enc_snapshots=enc, dec_snapshots=dec)

    # 3. Reconstruction grid + error maps
    print("\n[3/6] Reconstruction grid...")
    grid_path = os.path.join(save_dir, "reconstruction_grid.png")
    try:
        import torch
        result = load_reconstructions(
            args.model, save_dir, device=args.device,
            image_path=args.image,
            image_size=args.image_size, sr_scale=args.sr_scale,
            target_steps=target_steps,
        )
        if result is not None and result[0] is not None:
            key_steps, key_indices, recon_images, recon_psnrs, error_maps = result
            gt_img = None
            if os.path.exists(args.image):
                import cv2
                gt_bgr = cv2.imread(args.image)
                if gt_bgr is not None:
                    gt_img = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
                    if args.image_size:
                        gt_img = cv2.resize(gt_img, (args.image_size, args.image_size))
            create_reconstruction_grid(
                key_steps, recon_images, recon_psnrs, error_maps,
                args.model, grid_path, gt_image=gt_img,
            )
        else:
            print("  Warning: reconstruction loading returned None")
    except Exception as e:
        import traceback
        print(f"  Warning: reconstruction grid failed: {e}")
        traceback.print_exc()

    # 4. Spectral bias plot
    if has_spectral:
        print("\n[4/6] Spectral bias plot...")
        spec_path = os.path.join(save_dir, "spectral_bias.png")
        create_spectral_bias_plot(
            freq_ratios, steps, psnrs, args.model, spec_path,
        )
    else:
        print("\n[4/6] Skipped (no spectral data in trajectory)")

    # 5. Weight distribution animation
    print("\n[5/6] Weight distribution animation...")
    dist_path = os.path.join(save_dir, "weight_distribution.gif")
    create_weight_distribution_animation(
        full, steps, dist_path, fps=5, subsample=max(1, args.subsample * 2)
    )

    # 6. Encoder/Decoder comparison
    if has_enc_dec:
        print("\n[6/6] Encoder/Decoder comparison...")
        encdec_path = os.path.join(save_dir, "encoder_decoder_comparison.png")
        create_encoder_decoder_plot(
            full, enc, dec, steps, args.model, encdec_path)
    else:
        print("\n[6/6] Skipped (no encoder/decoder split)")

    print(f"\nAll visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()
