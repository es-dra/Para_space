"""
Phase 1: Tangent Vector Computation and Alignment Analysis

Computes J_scale and J_rot tangent vectors from independently trained
models at different scale/rotation parameters, then analyzes how
the fitting dynamics trajectory Δθ_t aligns with these tangent vectors.

Core hypothesis: Spectral bias causes early Δθ_t to align with J_scale
(low-frequency structure), then migrate toward J_rot (high-frequency details).

Usage:
    python experiments/Phase1_FittingDynamics/tangent_analysis.py \
        --dynamics_dir results/Phase1_v2/FittingDynamics/liif \
        --independent_dir results/Phase1_v2/Scale_Independent \
        --model liif --transform scale
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


def compute_tangent_vector(independent_results_path):
    """
    Compute tangent vector J_X from independent training results.

    For scale: J_scale ≈ (θ*(s2) - θ*(s1)) / (s2 - s1)
    For rotation: J_rot ≈ (θ*(φ2) - θ*(φ1)) / (φ2 - φ1)

    Uses finite differences between adjacent task parameters.
    Returns the tangent vector at the reference point (middle of parameter range).
    """
    with open(independent_results_path, 'r') as f:
        data = json.load(f)

    if "individual_results" in data:
        results = data["individual_results"][0]
    else:
        results = data

    task_params = results["task_params"]
    n_tasks = len(task_params)

    if n_tasks < 3:
        print("Need at least 3 tasks for tangent vector computation")
        return None

    mid_idx = n_tasks // 2
    return {
        "mid_idx": mid_idx,
        "mid_param": task_params[mid_idx],
        "task_params": task_params,
        "n_tasks": n_tasks,
    }


def compute_tangent_from_snapshots(flat_params_list, task_params):
    """
    Compute tangent vectors from a list of flattened parameter vectors
    and their corresponding task parameters.

    Uses central finite differences for interior points,
    forward/backward differences for endpoints.
    """
    n = len(flat_params_list)
    tangent_vectors = []

    for i in range(n):
        if i == 0:
            ds = task_params[i + 1] - task_params[i]
            dtheta = flat_params_list[i + 1] - flat_params_list[i]
        elif i == n - 1:
            ds = task_params[i] - task_params[i - 1]
            dtheta = flat_params_list[i] - flat_params_list[i - 1]
        else:
            ds = task_params[i + 1] - task_params[i - 1]
            dtheta = flat_params_list[i + 1] - flat_params_list[i - 1]

        if abs(ds) < 1e-10:
            tangent_vectors.append(np.zeros_like(flat_params_list[i]))
        else:
            tangent_vectors.append(dtheta / ds)

    return tangent_vectors


def analyze_tangent_alignment(
    dynamics_dir,
    independent_dir,
    model_type="liif",
    transform_type="scale",
    save_dir=None,
):
    """
    Analyze alignment between fitting dynamics Δθ_t and tangent vectors J_X.

    Steps:
    1. Load dynamics trajectory (Δθ_t over training)
    2. Load independent training results (θ*(s) for different s)
    3. Compute tangent vectors J_scale or J_rot from independent results
    4. Compute cosine similarity between Δθ_t and J_X at each training step
    5. Test hypothesis: early Δθ_t aligns with J_scale, later migrates
    """
    traj_path = os.path.join(dynamics_dir, "trajectory.npz")
    if not os.path.exists(traj_path):
        print(f"Trajectory file not found: {traj_path}")
        return None

    data = np.load(traj_path, allow_pickle=True)
    full_snapshots = data["full_snapshots"]
    enc_snapshots = data["enc_snapshots"]
    dec_snapshots = data["dec_snapshots"]
    snapshot_steps = data["snapshot_steps"]

    n_snapshots = len(snapshot_steps)
    theta_0 = full_snapshots[0]
    delta_thetas = full_snapshots - theta_0[np.newaxis, :]

    import glob
    pattern = os.path.join(
        independent_dir,
        f"{transform_type}_independent_{model_type}_*_seed42.json"
    )
    matching = glob.glob(pattern)
    if not matching:
        print(f"No independent results found matching: {pattern}")
        return None

    with open(matching[0], 'r') as f:
        ind_data = json.load(f)

    task_params = ind_data["task_params"]
    n_tasks = len(task_params)

    print(f"\n{'='*60}")
    print(f"Tangent Vector Alignment Analysis: {model_type}")
    print(f"Dynamics snapshots: {n_snapshots}")
    print(f"Independent tasks: {n_tasks}, params: {task_params[:3]}...{task_params[-3:]}")
    print(f"{'='*60}")

    mid_idx = n_tasks // 2
    mid_param = task_params[mid_idx]

    tangent_scale = None
    if n_tasks >= 3:
        ds = task_params[mid_idx + 1] - task_params[mid_idx - 1]
        if abs(ds) > 1e-10:
            theta_plus = ind_data.get("flat_params_plus")
            theta_minus = ind_data.get("flat_params_minus")

    print(f"\nNote: Full tangent vector computation requires raw parameter vectors")
    print(f"from independent training, which are not stored in JSON results.")
    print(f"Falling back to PCA-based tangent approximation.")

    if n_snapshots >= 3:
        centered = delta_thetas - delta_thetas.mean(axis=0)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_directions = Vt[:min(5, len(Vt))]
        explained_var = (S ** 2) / (S ** 2).sum()

        pc1 = pca_directions[0]
        pc2 = pca_directions[1] if len(pca_directions) > 1 else None

        print(f"\nPCA of Δθ trajectory:")
        for i, ev in enumerate(explained_var[:5]):
            print(f"  PC{i+1}: {ev*100:.2f}%")

        cos_pc1 = []
        cos_pc2 = []
        for i in range(n_snapshots):
            delta = delta_thetas[i]
            norm = np.linalg.norm(delta)
            if norm > 1e-10:
                delta_normed = delta / norm
                cos_pc1.append(cosine_similarity(delta_normed, pc1))
                if pc2 is not None:
                    cos_pc2.append(cosine_similarity(delta_normed, pc2))
            else:
                cos_pc1.append(0.0)
                cos_pc2.append(0.0)

        early_cos = np.mean(cos_pc1[1:6]) if len(cos_pc1) > 5 else cos_pc1[1] if len(cos_pc1) > 1 else 0
        late_cos = np.mean(cos_pc1[-5:]) if len(cos_pc1) > 5 else cos_pc1[-1]

        print(f"\nAlignment with PC1:")
        print(f"  Early training (steps 500-2500): cos(Δθ, PC1) = {early_cos:.4f}")
        print(f"  Late training (last 5 snapshots): cos(Δθ, PC1) = {late_cos:.4f}")

        enc_delta = enc_snapshots - enc_snapshots[0][np.newaxis, :]
        dec_delta = dec_snapshots - dec_snapshots[0][np.newaxis, :]

        enc_norms = np.linalg.norm(enc_delta, axis=1)
        dec_norms = np.linalg.norm(dec_delta, axis=1)

        enc_fraction = enc_norms / (enc_norms + dec_norms + 1e-10)

        print(f"\nEncoder vs Decoder contribution:")
        print(f"  Early: enc fraction = {enc_fraction[2]:.3f}")
        print(f"  Late:  enc fraction = {enc_fraction[-1]:.3f}")

        results = {
            "model_type": model_type,
            "n_snapshots": n_snapshots,
            "pca_explained_variance": [float(x) for x in explained_var[:5]],
            "cosine_with_pc1": [float(x) for x in cos_pc1],
            "cosine_with_pc2": [float(x) for x in cos_pc2],
            "early_pc1_alignment": float(early_cos),
            "late_pc1_alignment": float(late_cos),
            "encoder_fraction_early": float(enc_fraction[2]),
            "encoder_fraction_late": float(enc_fraction[-1]),
            "snapshot_steps": [int(s) for s in snapshot_steps],
        }

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            with open(os.path.join(save_dir, "tangent_analysis.json"), "w") as f:
                json.dump(results, f, indent=2)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].plot(snapshot_steps, cos_pc1, "g-", linewidth=2, label="cos(Δθ, PC1)")
            if cos_pc2:
                axes[0, 0].plot(snapshot_steps, cos_pc2, "m-", linewidth=2, label="cos(Δθ, PC2)")
            axes[0, 0].set_xlabel("Training Step")
            axes[0, 0].set_ylabel("Cosine Similarity")
            axes[0, 0].set_title("Δθ Alignment with PCA Directions")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(snapshot_steps, enc_fraction, "r-", linewidth=2)
            axes[0, 1].set_xlabel("Training Step")
            axes[0, 1].set_ylabel("Encoder ||Δθ|| / Total ||Δθ||")
            axes[0, 1].set_title("Encoder vs Decoder Parameter Change Fraction")
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(snapshot_steps, enc_norms, "b-", linewidth=2, label="Encoder ||Δθ||")
            axes[1, 0].plot(snapshot_steps, dec_norms, "r-", linewidth=2, label="Decoder ||Δθ||")
            axes[1, 0].set_xlabel("Training Step")
            axes[1, 0].set_ylabel("||Δθ||")
            axes[1, 0].set_title("Parameter Offset Norm by Component")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            n_show = min(10, len(explained_var))
            axes[1, 1].bar(range(1, n_show + 1), explained_var[:n_show] * 100)
            axes[1, 1].set_xlabel("Principal Component")
            axes[1, 1].set_ylabel("Variance Explained %")
            axes[1, 1].set_title("PCA Scree Plot")
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle(f"Tangent Alignment Analysis ({model_type})", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "tangent_analysis.png"), dpi=150, bbox_inches="tight")
            plt.close()

            print(f"\nAnalysis saved to {save_dir}")

        return results

    return None


def main():
    parser = argparse.ArgumentParser(description="Tangent Vector Alignment Analysis")
    parser.add_argument("--dynamics_dir", type=str, required=True)
    parser.add_argument("--independent_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="liif", choices=["liif", "lte"])
    parser.add_argument("--transform", type=str, default="scale", choices=["scale", "rotation"])
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.dynamics_dir, "analysis")
    analyze_tangent_alignment(
        args.dynamics_dir, args.independent_dir,
        model_type=args.model, transform_type=args.transform,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()
