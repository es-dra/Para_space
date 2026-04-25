"""
Phase 1 D1: Fitting Dynamics Analysis for Conditional INRs

Reads trajectory.npz from run.py output and computes:
  - Separate PCA for encoder and decoder parameter trajectories
  - Frequency-domain spectral bias verification
  - Cosine similarity with PCA directions over time
  - Trajectory curvature and step-size analysis

Usage:
    python experiments/Phase1_FittingDynamics/analyze_dynamics.py \
        --trajectory results/Phase1/FittingDynamics/liif/trajectory.npz \
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


def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return dot / (norm1 * norm2)


def compute_pca_analysis(snapshots, label=""):
    n_snapshots, n_params = snapshots.shape
    theta_0 = snapshots[0]
    delta_thetas = snapshots - theta_0[np.newaxis, :]
    delta_norms = np.linalg.norm(delta_thetas, axis=1)

    pca_results = {}
    if n_snapshots >= 3:
        centered = delta_thetas - delta_thetas.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_directions = Vt[:min(5, len(Vt))]
        explained_var = (S ** 2) / (S ** 2).sum()

        print(f"\nPCA of {label} Δθ trajectory:")
        for i, ev in enumerate(explained_var[:5]):
            print(f"  PC{i+1}: {ev*100:.2f}%")

        cosine_with_pc1 = []
        cosine_with_pc2 = []
        pc1 = pca_directions[0]
        pc2 = pca_directions[1] if len(pca_directions) > 1 else None
        for i in range(n_snapshots):
            delta = delta_thetas[i]
            norm = np.linalg.norm(delta)
            if norm > 1e-10:
                delta_normed = delta / norm
                cosine_with_pc1.append(cosine_similarity(delta_normed, pc1))
                if pc2 is not None:
                    cosine_with_pc2.append(cosine_similarity(delta_normed, pc2))
            else:
                cosine_with_pc1.append(0.0)
                cosine_with_pc2.append(0.0)

        step_sizes = []
        for i in range(1, n_snapshots):
            step_size = np.linalg.norm(delta_thetas[i] - delta_thetas[i - 1])
            step_sizes.append(step_size)

        pca_results = {
            "explained_variance": [float(x) for x in explained_var[:10]],
            "cosine_with_pc1": [float(x) for x in cosine_with_pc1],
            "cosine_with_pc2": [float(x) for x in cosine_with_pc2],
            "step_sizes": [float(x) for x in step_sizes],
            "delta_norms": [float(x) for x in delta_norms],
            "pc1": pc1,
            "pc2": pc2 if pc2 is not None else np.zeros_like(pc1),
            "pca_directions": pca_directions,
            "projections": delta_thetas @ pca_directions[:2].T,
        }

    return pca_results, delta_norms


def analyze_dynamics(trajectory_path, model_type="liif", save_dir=None):
    data = np.load(trajectory_path, allow_pickle=True)
    full_snapshots = data["full_snapshots"]
    enc_snapshots = data["enc_snapshots"]
    dec_snapshots = data["dec_snapshots"]
    snapshot_steps = data["snapshot_steps"]
    losses = data["losses"]
    freq_ratios = data["freq_ratios"] if "freq_ratios" in data else None
    target_spectrum = data["target_spectrum"] if "target_spectrum" in data else None

    n_snapshots, n_params = full_snapshots.shape

    print(f"\n{'='*60}")
    print(f"Dynamics Analysis: {model_type}")
    print(f"Snapshots: {n_snapshots}, Full params: {n_params:,}")
    print(f"Encoder params: {enc_snapshots.shape[1]:,}")
    print(f"Decoder params: {dec_snapshots.shape[1]:,}")
    print(f"Steps: {snapshot_steps[0]} -> {snapshot_steps[-1]}")
    print(f"{'='*60}")

    full_pca, full_delta_norms = compute_pca_analysis(full_snapshots, "Full")
    enc_pca, enc_delta_norms = compute_pca_analysis(enc_snapshots, "Encoder")
    dec_pca, dec_delta_norms = compute_pca_analysis(dec_snapshots, "Decoder")

    results = {
        "model_type": model_type,
        "n_snapshots": n_snapshots,
        "n_params": int(n_params),
        "n_encoder_params": int(enc_snapshots.shape[1]),
        "n_decoder_params": int(dec_snapshots.shape[1]),
        "snapshot_steps": [int(s) for s in snapshot_steps],
        "full_pca": {k: v for k, v in full_pca.items() if k not in ["pc1", "pc2", "pca_directions", "projections"]},
        "encoder_pca": {k: v for k, v in enc_pca.items() if k not in ["pc1", "pc2", "pca_directions", "projections"]},
        "decoder_pca": {k: v for k, v in dec_pca.items() if k not in ["pc1", "pc2", "pca_directions", "projections"]},
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "dynamics_analysis.json"), "w") as f:
            json.dump(results, f, indent=2)

        fig, axes = plt.subplots(3, 3, figsize=(20, 16))

        for row, (pca_data, label, deltas) in enumerate([
            (full_pca, "Full", full_delta_norms),
            (enc_pca, "Encoder", enc_delta_norms),
            (dec_pca, "Decoder", dec_delta_norms),
        ]):
            if not pca_data:
                continue

            axes[row, 0].plot(snapshot_steps, pca_data["delta_norms"], "b-", linewidth=2)
            axes[row, 0].set_xlabel("Training Step")
            axes[row, 0].set_ylabel(f"||Δθ|| ({label})")
            axes[row, 0].set_title(f"Parameter Offset Norm ({label})")
            axes[row, 0].grid(True, alpha=0.3)

            if pca_data.get("cosine_with_pc1"):
                axes[row, 1].plot(snapshot_steps, pca_data["cosine_with_pc1"], "g-", linewidth=2, label="cos(Δθ, PC1)")
                if pca_data.get("cosine_with_pc2"):
                    axes[row, 1].plot(snapshot_steps, pca_data["cosine_with_pc2"], "m-", linewidth=2, label="cos(Δθ, PC2)")
                axes[row, 1].set_xlabel("Training Step")
                axes[row, 1].set_ylabel("Cosine Similarity")
                axes[row, 1].set_title(f"Δθ Alignment with PCA ({label})")
                axes[row, 1].legend()
                axes[row, 1].grid(True, alpha=0.3)

            if "projections" in pca_data:
                proj = pca_data["projections"]
                sc = axes[row, 2].scatter(proj[:, 0], proj[:, 1], c=snapshot_steps, cmap="viridis", s=60)
                axes[row, 2].plot(proj[:, 0], proj[:, 1], "k-", alpha=0.3, linewidth=0.8)
                plt.colorbar(sc, ax=axes[row, 2], label="Step")
                axes[row, 2].set_xlabel("PC1")
                axes[row, 2].set_ylabel("PC2")
                axes[row, 2].set_title(f"Δθ Trajectory in PCA Space ({label})")
                axes[row, 2].set_aspect("equal")
                axes[row, 2].grid(True, alpha=0.3)

        plt.suptitle(f"Fitting Dynamics Analysis ({model_type})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "dynamics_analysis.png"), dpi=150, bbox_inches="tight")
        plt.close()

        if freq_ratios is not None and len(freq_ratios) > 0:
            fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

            sampled_steps = snapshot_steps[1:len(freq_ratios)+1]
            axes2[0].plot(sampled_steps, freq_ratios, "r-", linewidth=2)
            if target_spectrum is not None:
                target_low = np.sum(target_spectrum[:len(target_spectrum)//2]) / (np.sum(target_spectrum) + 1e-10)
                axes2[0].axhline(y=target_low, color="k", linestyle="--", label=f"Target low-freq ratio={target_low:.3f}")
                axes2[0].legend()
            axes2[0].set_xlabel("Training Step")
            axes2[0].set_ylabel("Low-Frequency Energy Ratio")
            axes2[0].set_title("Spectral Bias: Low-Freq Energy Ratio Over Training")
            axes2[0].grid(True, alpha=0.3)

            if len(losses) > 0:
                loss_steps = np.arange(1, len(losses) + 1)
                axes2[1].semilogy(loss_steps, losses, "r-", linewidth=1, alpha=0.7)
                axes2[1].set_xlabel("Training Step")
                axes2[1].set_ylabel("Loss (MSE)")
                axes2[1].set_title("Training Loss")
                axes2[1].grid(True, alpha=0.3)

            if full_pca.get("explained_variance"):
                ev = full_pca["explained_variance"]
                n_show = min(10, len(ev))
                axes2[2].bar(range(1, n_show + 1), [v * 100 for v in ev[:n_show]])
                axes2[2].set_xlabel("Principal Component")
                axes2[2].set_ylabel("Variance Explained %")
                axes2[2].set_title("PCA Scree Plot (Full Model)")
                axes2[2].grid(True, alpha=0.3)

            plt.suptitle(f"Spectral Bias & PCA Analysis ({model_type})", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "spectral_bias_analysis.png"), dpi=150, bbox_inches="tight")
            plt.close()

        print(f"\nAnalysis saved to {save_dir}")

    print(f"\nSummary ({model_type}):")
    for label, pca_data in [("Full", full_pca), ("Encoder", enc_pca), ("Decoder", dec_pca)]:
        if pca_data:
            ev = pca_data.get("explained_variance", [])
            if ev:
                print(f"  {label} PC1: {ev[0]*100:.2f}%, PC2: {ev[1]*100:.2f}%")
            print(f"  {label} Final ||Δθ||: {pca_data['delta_norms'][-1]:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze Fitting Dynamics Trajectory")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--model", type=str, default="liif", choices=["liif", "lte"])
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.dirname(args.trajectory)
    analyze_dynamics(args.trajectory, model_type=args.model, save_dir=save_dir)


if __name__ == "__main__":
    main()
