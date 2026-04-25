"""
Phase 1 R1-v2: Rotation Orbit Validation with Fixed Transform

CRITICAL BUG FIX (vs v1):
  v1 rotated BOTH image and coordinates -> SIREN learned same function for all angles
  v2 rotates ONLY the image -> SIREN learns f_phi(x) = I(R_phi^{-1}x), a DIFFERENT function per phi

Theory prediction for SO(2):
  - Closed orbit: theta*(T_{360deg}[f]) ~ theta*(f)
  - Elliptical PC1-PC2 trajectory
  - PC1/baseline > 3.0x
  - Closure C < 0.15

Two rotation modes tested:
  1. image_rotation: rotate image, keep coords -> natural signal transform
  2. coordinate_rotation: rotate coords, keep image -> first-layer theorem applies
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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.siren import SIREN
from src.datasets import get_image_coordinates
from src.alignment import align_siren_parameters, flatten_params
from src.utils import set_seed
from experiments.config import SIREN_CONFIG, TRAINING_CONFIG, DATA_CONFIG


def load_image(image_path, image_size=48):
    import cv2
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def get_image_rotation_data(image_tensor, angle_deg, device):
    import cv2
    C, H, W = image_tensor.shape
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(img_np, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    rotated_tensor = torch.from_numpy(rotated).permute(2, 0, 1).float().to(device) / 255.0
    coords = get_image_coordinates(H, W, normalize="center", device=device).reshape(-1, 2)
    target = rotated_tensor.reshape(3, -1).T
    return coords, target


def get_coord_rotation_data(image_tensor, angle_deg, device):
    C, H, W = image_tensor.shape
    angle_rad = np.radians(angle_deg)
    R = torch.tensor([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad), np.cos(angle_rad)]],
                     dtype=torch.float32, device=device)
    base_coords = get_image_coordinates(H, W, normalize="center", device=device).reshape(-1, 2)
    rotated_coords = base_coords @ R.T
    target = image_tensor.reshape(3, -1).T
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


def full_diagnosis(theta_traj, group_params, group_name, tag="", save_dir=None):
    N, D = theta_traj.shape
    pca = PCA()
    proj = pca.fit_transform(theta_traj)
    evr = pca.explained_variance_ratio_

    dists = np.linalg.norm(np.diff(theta_traj, axis=0), axis=1)
    pc1_param_corr = np.corrcoef(group_params, proj[:, 0])[0, 1]
    pc2_param_corr = np.corrcoef(group_params, proj[:, 1])[0, 1]

    closure = np.linalg.norm(theta_traj[-1] - theta_traj[0]) / (dists.sum() + 1e-10)

    results = {
        "N": N, "D": D,
        "random_baseline_pc1": 1.0 / (N - 1),
        "pc1": float(evr[0]),
        "pc2": float(evr[1]),
        "pc1_pc2": float(evr[0] + evr[1]),
        "pc1_vs_baseline": float(evr[0] / (1 / (N - 1))),
        "effective_dim": float(1 / np.sum(evr ** 2)),
        "path_length": float(dists.sum()),
        "end_to_end": float(np.linalg.norm(theta_traj[-1] - theta_traj[0])),
        "closure": float(closure),
        "pc1_param_correlation": float(pc1_param_corr),
        "pc2_param_correlation": float(pc2_param_corr),
        "explained_variance": evr.tolist(),
    }

    print(f"\n  [{tag}] N={N}, D={D}")
    print(f"    Random baseline PC1 = {100/(N-1):.1f}%")
    print(f"    PC1 = {evr[0]*100:.2f}%, PC2 = {evr[1]*100:.2f}%")
    print(f"    PC1/baseline = {evr[0]/(1/(N-1)):.2f}x")
    print(f"    Effective dim = {1/np.sum(evr**2):.2f}")
    print(f"    PC1-param r = {pc1_param_corr:.4f}")
    print(f"    Closure C = {closure:.4f}")
    print(f"    Path length = {dists.sum():.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sc = axes[0, 0].scatter(proj[:, 0], proj[:, 1], c=group_params, cmap='hsv', s=40)
    axes[0, 0].plot(proj[:, 0], proj[:, 1], 'k-', alpha=0.3, linewidth=0.8)
    if len(group_params) > 2:
        axes[0, 0].plot([proj[-1, 0], proj[0, 0]], [proj[-1, 1], proj[0, 1]],
                       'r--', alpha=0.5, linewidth=1.5, label='closure')
    plt.colorbar(sc, ax=axes[0, 0], label='Angle (deg)')
    axes[0, 0].set_xlabel('PC1')
    axes[0, 0].set_ylabel('PC2')
    axes[0, 0].set_title(f'{group_name} {tag}: PC1 vs PC2 (C={closure:.3f})')
    axes[0, 0].set_aspect('equal')
    axes[0, 0].legend()

    axes[0, 1].scatter(group_params, proj[:, 0], c='blue', s=40)
    axes[0, 1].set_xlabel('Angle (deg)')
    axes[0, 1].set_ylabel('PC1 projection')
    axes[0, 1].set_title(f'PC1 vs Angle (r={pc1_param_corr:.3f})')

    n_show = min(10, N - 1)
    axes[1, 0].bar(range(1, n_show + 1), evr[:n_show] * 100)
    axes[1, 0].axhline(y=100 / (N - 1), color='r', linestyle='--',
                       label=f'Baseline={100/(N-1):.1f}%')
    axes[1, 0].set_xlabel('PC')
    axes[1, 0].set_ylabel('Variance %')
    axes[1, 0].set_title('Scree plot')
    axes[1, 0].legend()

    axes[1, 1].scatter(group_params, proj[:, 1], c='red', s=40)
    axes[1, 1].set_xlabel('Angle (deg)')
    axes[1, 1].set_ylabel('PC2 projection')
    axes[1, 1].set_title(f'PC2 vs Angle (r={pc2_param_corr:.3f})')

    plt.tight_layout()
    fname = f'rotation_N{N}_{tag}.png'
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()

    return results


def run_experiment(image_path, seed=42, device="cuda", save_dir=None,
                   mode="image_rotation"):
    set_seed(seed)
    device = torch.device(device)
    image_tensor = load_image(image_path, DATA_CONFIG["image_size"]).to(device)

    angles = [int(x) for x in np.linspace(0, 345, 24)]

    if mode == "image_rotation":
        get_data_fn = lambda a: get_image_rotation_data(image_tensor, a, device)
        mode_label = "ImageRot"
    elif mode == "coordinate_rotation":
        get_data_fn = lambda a: get_coord_rotation_data(image_tensor, a, device)
        mode_label = "CoordRot"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\n{'='*60}")
    print(f"Rotation Orbit v2: N={len(angles)}, mode={mode_label}")
    print(f"Image: {Path(image_path).name}, Seed: {seed}")
    print(f"Angles: {angles}")
    print(f"{'='*60}")

    models_params = []
    psnrs = []

    for i, angle in enumerate(angles):
        coords, target = get_data_fn(angle)
        model, psnr = train_to_convergence(
            coords, target, device,
            num_steps=TRAINING_CONFIG["independent_steps"],
            lr=TRAINING_CONFIG["independent_lr"],
            seed=seed
        )
        psnrs.append(psnr)
        models_params.append(model.get_params())
        print(f"  [{i+1}/{len(angles)}] angle={angle}deg: PSNR={psnr:.1f} dB")

    unaligned_flat = torch.stack([flatten_params(p) for p in models_params]).numpy()

    print(f"\nAligning {len(models_params)} models...")
    aligned_params = [models_params[0]]
    for i in range(1, len(models_params)):
        aligned, cost = align_siren_parameters(models_params[0], models_params[i])
        aligned_params.append(aligned)
    aligned_flat = torch.stack([flatten_params(p) for p in aligned_params]).numpy()

    dists_before = np.linalg.norm(np.diff(unaligned_flat, axis=0), axis=1)
    dists_after = np.linalg.norm(np.diff(aligned_flat, axis=0), axis=1)
    compression = dists_before.mean() / (dists_after.mean() + 1e-10)

    print(f"\nAlignment compression: {compression:.2f}x")

    res_unaligned = full_diagnosis(unaligned_flat, angles, f"Rotation_{mode_label}",
                                   tag="unaligned", save_dir=save_dir)
    res_aligned = full_diagnosis(aligned_flat, angles, f"Rotation_{mode_label}",
                                 tag="aligned", save_dir=save_dir)

    combined = {
        "image": Path(image_path).name,
        "seed": seed,
        "mode": mode,
        "N": len(angles),
        "angles": angles,
        "psnrs": psnrs,
        "mean_psnr": float(np.mean(psnrs)),
        "alignment_compression": float(compression),
        "unaligned": res_unaligned,
        "aligned": res_aligned,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"rotation_{mode}_v2_results.json"
        with open(os.path.join(save_dir, fname), 'w') as f:
            json.dump(combined, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Mode: {mode_label}, N = {len(angles)}")
    print(f"  Random baseline = {100/(len(angles)-1):.1f}%")
    print(f"  Unaligned: PC1={res_unaligned['pc1']*100:.2f}%, "
          f"PC1/baseline={res_unaligned['pc1_vs_baseline']:.2f}x, "
          f"r={res_unaligned['pc1_param_correlation']:.4f}, "
          f"C={res_unaligned['closure']:.4f}")
    print(f"  Aligned:   PC1={res_aligned['pc1']*100:.2f}%, "
          f"PC1/baseline={res_aligned['pc1_vs_baseline']:.2f}x, "
          f"r={res_aligned['pc1_param_correlation']:.4f}, "
          f"C={res_aligned['closure']:.4f}")

    closure_ok = res_unaligned['closure'] < 0.15
    baseline_ok = res_unaligned['pc1_vs_baseline'] > 3.0
    print(f"\n  Closure C < 0.15: {'PASS' if closure_ok else 'FAIL'}")
    print(f"  PC1/baseline > 3.0x: {'PASS' if baseline_ok else 'FAIL'}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Phase 1 R1-v2: Rotation Orbit Fixed")
    parser.add_argument("--image", type=str, default="Data/Set5/HR/baby.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="results/Phase1/Rotation_Orbit_v2")
    parser.add_argument("--mode", type=str, default="image_rotation",
                       choices=["image_rotation", "coordinate_rotation"],
                       help="image_rotation: rotate image, keep coords; "
                            "coordinate_rotation: rotate coords, keep image")
    args = parser.parse_args()

    run_experiment(args.image, seed=args.seed, device=args.device,
                   save_dir=args.save_dir, mode=args.mode)


if __name__ == "__main__":
    main()
