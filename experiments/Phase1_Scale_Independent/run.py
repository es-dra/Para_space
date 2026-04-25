"""
Phase 1 S2/R2: Independent Training Baseline

Trains SIREN independently on each task (no MAML shared initialization).
After training, aligns parameters using activation matching and analyzes PCA.

Purpose: Confirm MAML effect exceeds independent training baseline
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.siren import SIREN
from src.datasets import get_image_coordinates
from src.alignment import align_siren_parameters, flatten_params
from src.utils import set_seed
from experiments.config import (
    SIREN_CONFIG, TRAINING_CONFIG, DATA_CONFIG,
    SCALE_CONFIG, ROTATION_CONFIG, ANALYSIS_CONFIG, OUTPUT_CONFIG
)


def load_image(image_path, image_size=48):
    import cv2
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def get_scale_task_data(image_tensor, scale_factor, device):
    import cv2
    C, H, W = image_tensor.shape
    new_h, new_w = int(H * scale_factor), int(W * scale_factor)
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
    coords = get_image_coordinates(new_h, new_w, normalize="center", device=device).reshape(-1, 2)
    target = resized_tensor.reshape(3, -1).T
    return coords, target


def get_rotation_task_data(image_tensor, angle_deg, device):
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


def train_independent(coords, target, device, num_steps=100000, lr=5e-4, seed=42):
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


def run_experiment(image_path, transform_type="scale", seed=42, device="cuda", save_dir=None):
    set_seed(seed)
    device = torch.device(device)
    image_tensor = load_image(image_path, DATA_CONFIG["image_size"]).to(device)

    if transform_type == "scale":
        params_list = SCALE_CONFIG["param_range"]
    else:
        params_list = ROTATION_CONFIG["param_range"]

    print(f"\n{'='*60}")
    print(f"Image: {Path(image_path).name}, Seed: {seed}")
    print(f"Transform: {transform_type} (Independent), {len(params_list)} tasks")
    print(f"Training: {TRAINING_CONFIG['independent_steps']} steps, lr={TRAINING_CONFIG['independent_lr']}")
    print(f"{'='*60}")

    models_and_params = []
    psnrs = []

    for i, param_val in enumerate(params_list):
        print(f"\n  Task {i+1}/{len(params_list)}: {transform_type}={param_val}")
        if transform_type == "scale":
            coords, target = get_scale_task_data(image_tensor, param_val, device)
        else:
            coords, target = get_rotation_task_data(image_tensor, param_val, device)

        model, psnr = train_independent(
            coords, target, device,
            num_steps=TRAINING_CONFIG["independent_steps"],
            lr=TRAINING_CONFIG["independent_lr"],
            seed=seed
        )
        psnrs.append(psnr)
        models_and_params.append(model.get_params())
        print(f"    PSNR={psnr:.1f} dB")

    print(f"\nAligning parameters...")
    aligned_params = [models_and_params[0]]
    for i in range(1, len(models_and_params)):
        aligned, cost = align_siren_parameters(
            models_and_params[0], models_and_params[i]
        )
        aligned_params.append(aligned)

    flat_params = torch.stack([flatten_params(p) for p in aligned_params]).numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=ANALYSIS_CONFIG["pca_components"])
    pca.fit(flat_params)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"\nPCA Results (after alignment):")
    print(f"  PC1: {explained_variance[0]*100:.2f}%")
    print(f"  PC2: {explained_variance[1]*100:.2f}%")
    print(f"  2 PCs: {cumulative_variance[1]*100:.2f}%")
    print(f"  Mean PSNR: {np.mean(psnrs):.1f} dB")

    results = {
        "image": Path(image_path).name,
        "seed": seed,
        "transform_type": f"{transform_type}_independent",
        "n_tasks": len(params_list),
        "task_params": params_list,
        "psnrs": psnrs,
        "mean_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "explained_variance": explained_variance.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "pc1_variance": float(explained_variance[0]),
        "pc2_variance": float(explained_variance[1]),
        "n_params": flat_params.shape[1],
        "training_steps": TRAINING_CONFIG["independent_steps"],
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{transform_type}_independent_{Path(image_path).stem}_seed{seed}.json"
        save_path = Path(save_dir) / prefix
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 S2/R2: Independent Training Baseline")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--transform", type=str, default="scale", choices=["scale", "rotation"])
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subdir = "Scale_Independent" if args.transform == "scale" else "Rotation_Independent"
    results_base = args.results_dir or os.path.join(OUTPUT_CONFIG["results_dir"], subdir)
    os.makedirs(results_base, exist_ok=True)

    if args.image:
        image_paths = [args.image]
    else:
        data_root = Path(__file__).parent.parent.parent / "Data"
        image_paths = [
            data_root / "Set5" / "HR" / name
            for name in DATA_CONFIG["images"]
            if (data_root / "Set5" / "HR" / name).exists()
        ]

    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = []

    for img_path in image_paths:
        for seed in seeds:
            result = run_experiment(
                img_path, transform_type=args.transform,
                seed=seed, device=device, save_dir=results_base
            )
            all_results.append(result)

    pc1_vals = [r["pc1_variance"] for r in all_results]
    psnr_vals = [r["mean_psnr"] for r in all_results]

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"PC1: {np.mean(pc1_vals)*100:.2f}% +/- {np.std(pc1_vals)*100:.2f}%")
    print(f"PSNR: {np.mean(psnr_vals):.1f} dB")

    aggregate = {
        "n_experiments": len(all_results),
        "pc1_variance": {"mean": float(np.mean(pc1_vals)), "std": float(np.std(pc1_vals))},
        "mean_psnr": {"mean": float(np.mean(psnr_vals)), "min": float(np.min(psnr_vals))},
        "individual_results": all_results,
    }
    agg_path = Path(results_base) / "aggregate_results.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")


if __name__ == "__main__":
    main()
