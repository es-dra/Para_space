"""
Phase 1 Diagnostic Script

Diagnoses three key questions:
1. Is PC1 significantly above random baseline? (N-dependence)
2. Does weight alignment actually work? (compression ratio)
3. Does the orbit have geometric structure? (ellipse / monotonicity)

Usage:
    python experiments/diagnose_phase1.py --image Data/Set5/HR/baby.png --seed 42
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.siren import SIREN
from src.datasets import get_image_coordinates
from src.alignment import align_siren_parameters, flatten_params, git_rebasin_align_siren
from src.utils import set_seed
from experiments.config import SIREN_CONFIG, TRAINING_CONFIG, DATA_CONFIG


def load_image(image_path, image_size=48):
    import cv2
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def get_scaled_data(image_tensor, scale_factor, device):
    import cv2
    C, H, W = image_tensor.shape
    new_h, new_w = int(H * scale_factor), int(W * scale_factor)
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
    coords = get_image_coordinates(new_h, new_w, normalize="center", device=device).reshape(-1, 2)
    target = resized_tensor.reshape(3, -1).T
    return coords, target


def get_rotation_data(image_tensor, angle_deg, device):
    import cv2
    C, H, W = image_tensor.shape
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img_np, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rotated_tensor = torch.from_numpy(rotated).permute(2, 0, 1).float().to(device) / 255.0
    angle_rad = np.radians(angle_deg)
    R = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad), np.cos(angle_rad)]],
                     dtype=torch.float32, device=device)
    base_coords = get_image_coordinates(H, W, normalize="center", device=device).reshape(-1, 2)
    rotated_coords = base_coords @ R.T
    target = rotated_tensor.reshape(3, -1).T
    return rotated_coords, target


def train_to_convergence(coords, target, device, num_steps=120000, lr=5e-4, seed=42):
    set_seed(seed)
    model = SIREN(**SIREN_CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(coords)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        psnr = -10 * np.log10(F.mse_loss(model(coords), target).item())
    return model, psnr


def diagnose_orbit(theta_traj, group_params, group_name, aligned=True, save_dir=None):
    N, D = theta_traj.shape
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {group_name} ({'aligned' if aligned else 'unaligned'})")
    print(f"{'='*60}")
    print(f"  N = {N}, D = {D}")
    print(f"  Random baseline PC1 = {100/(N-1):.1f}%")

    pca = PCA()
    proj = pca.fit_transform(theta_traj)
    evr = pca.explained_variance_ratio_
    print(f"  PC1 = {evr[0]*100:.2f}%, PC2 = {evr[1]*100:.2f}%")
    print(f"  PC1+PC2 = {(evr[0]+evr[1])*100:.2f}%")
    print(f"  Effective dimension (participation ratio) = {1/np.sum(evr**2):.2f}")
    print(f"  PC1 / random baseline = {evr[0] / (1/(N-1)):.2f}x")

    dists = np.linalg.norm(np.diff(theta_traj, axis=0), axis=1)
    print(f"\n  Path statistics:")
    print(f"    Total path length = {dists.sum():.4f}")
    print(f"    End-to-end distance = {np.linalg.norm(theta_traj[-1]-theta_traj[0]):.4f}")

    if "Rotation" in group_name:
        closure = np.linalg.norm(theta_traj[-1] - theta_traj[0]) / dists.sum()
        print(f"    Closure C = {closure:.4f} (expect < 0.15 for SO(2))")

    pc1_vs_param_corr = np.corrcoef(group_params, proj[:, 0])[0, 1]
    print(f"    PC1 vs parameter Pearson r = {pc1_vs_param_corr:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sc = axes[0, 0].scatter(proj[:, 0], proj[:, 1], c=group_params, cmap='viridis', s=60)
    axes[0, 0].plot(proj[:, 0], proj[:, 1], 'k-', alpha=0.3, linewidth=0.8)
    plt.colorbar(sc, ax=axes[0, 0], label='Transform parameter')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title(f'{group_name}: PC1 vs PC2 trajectory')
    axes[0, 0].set_aspect('equal')

    axes[0, 1].scatter(group_params, proj[:, 0], c='blue', s=60)
    axes[0, 1].set_xlabel('Transform parameter')
    axes[0, 1].set_ylabel('PC1 projection')
    axes[0, 1].set_title(f'{group_name}: Parameter vs PC1 (r={pc1_vs_param_corr:.3f})')

    n_show = min(10, N - 1)
    axes[1, 0].bar(range(1, n_show + 1), evr[:n_show] * 100)
    axes[1, 0].axhline(y=100 / (N - 1), color='r', linestyle='--',
                       label=f'Random baseline={100/(N-1):.1f}%')
    axes[1, 0].set_xlabel('Principal component')
    axes[1, 0].set_ylabel('Variance explained %')
    axes[1, 0].set_title('Scree plot')
    axes[1, 0].legend()

    axes[1, 1].scatter(group_params, proj[:, 1], c='red', s=60)
    axes[1, 1].set_xlabel('Transform parameter')
    axes[1, 1].set_ylabel('PC2 projection')
    axes[1, 1].set_title(f'{group_name}: Parameter vs PC2')

    plt.tight_layout()
    tag = "aligned" if aligned else "unaligned"
    fname = f'diagnose_{group_name}_{tag}.png'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fpath = os.path.join(save_dir, fname)
    else:
        fpath = fname
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")

    return {
        "N": N, "D": D,
        "random_baseline_pc1": 1.0 / (N - 1),
        "pc1": float(evr[0]),
        "pc2": float(evr[1]),
        "pc1_vs_baseline": float(evr[0] / (1 / (N - 1))),
        "effective_dim": float(1 / np.sum(evr ** 2)),
        "path_length": float(dists.sum()),
        "end_to_end": float(np.linalg.norm(theta_traj[-1] - theta_traj[0])),
        "pc1_param_correlation": float(pc1_vs_param_corr),
    }


def run_diagnostic(image_path, seed=42, device="cuda", save_dir=None):
    set_seed(seed)
    device = torch.device(device)
    image_tensor = load_image(image_path, DATA_CONFIG["image_size"]).to(device)

    scale_params = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    rotation_params = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

    all_results = {}

    for group_name, params, get_data_fn in [
        ("Scale", scale_params, lambda s: get_scaled_data(image_tensor, s, device)),
        ("Rotation", rotation_params, lambda a: get_rotation_data(image_tensor, a, device)),
    ]:
        print(f"\n{'#'*60}")
        print(f"# Training {group_name} tasks (N={len(params)})")
        print(f"{'#'*60}")

        models_params = []
        psnrs = []

        for i, p in enumerate(params):
            coords, target = get_data_fn(p)
            model, psnr = train_to_convergence(
                coords, target, device,
                num_steps=TRAINING_CONFIG["independent_steps"],
                lr=TRAINING_CONFIG["independent_lr"],
                seed=seed
            )
            psnrs.append(psnr)
            models_params.append(model.get_params())
            print(f"  {group_name} param={p}: PSNR={psnr:.1f} dB")

        unaligned_flat = torch.stack([flatten_params(p) for p in models_params]).numpy()

        print(f"\nAligning {len(models_params)} models...")
        aligned_params = [models_params[0]]
        for i in range(1, len(models_params)):
            aligned, cost = align_siren_parameters(models_params[0], models_params[i])
            aligned_params.append(aligned)
        aligned_flat = torch.stack([flatten_params(p) for p in aligned_params]).numpy()

        dists_before = np.linalg.norm(np.diff(unaligned_flat, axis=0), axis=1)
        dists_after = np.linalg.norm(np.diff(aligned_flat, axis=0), axis=1)
        compression_ratio = dists_before.mean() / (dists_after.mean() + 1e-10)

        print(f"\n  Alignment effectiveness:")
        print(f"    Avg adjacent distance (before): {dists_before.mean():.4f}")
        print(f"    Avg adjacent distance (after):  {dists_after.mean():.4f}")
        print(f"    Compression ratio: {compression_ratio:.2f}x")

        res_unaligned = diagnose_orbit(unaligned_flat, params, group_name,
                                       aligned=False, save_dir=save_dir)
        res_aligned = diagnose_orbit(aligned_flat, params, group_name,
                                     aligned=True, save_dir=save_dir)

        all_results[group_name] = {
            "psnrs": psnrs,
            "mean_psnr": float(np.mean(psnrs)),
            "alignment_compression_ratio": float(compression_ratio),
            "dists_before_mean": float(dists_before.mean()),
            "dists_after_mean": float(dists_after.mean()),
            "unaligned": res_unaligned,
            "aligned": res_aligned,
        }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "diagnostic_results.json"), 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for g, r in all_results.items():
        print(f"\n{g}:")
        print(f"  PSNR: {r['mean_psnr']:.1f} dB")
        print(f"  Alignment compression: {r['alignment_compression_ratio']:.2f}x")
        print(f"  PC1 (unaligned): {r['unaligned']['pc1']*100:.2f}%")
        print(f"  PC1 (aligned):   {r['aligned']['pc1']*100:.2f}%")
        print(f"  PC1/baseline (aligned): {r['aligned']['pc1_vs_baseline']:.2f}x")
        print(f"  PC1-param correlation: {r['aligned']['pc1_param_correlation']:.4f}")
        print(f"  Effective dim (aligned): {r['aligned']['effective_dim']:.2f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Diagnostic")
    parser.add_argument("--image", type=str, default="Data/Set5/HR/baby.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/Phase1/Diagnostic")
    args = parser.parse_args()

    run_diagnostic(args.image, seed=args.seed, device=args.device, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
