"""
Phase 1 R1: Rotation Parameter Orbit Validation

Two-stage strategy:
  Stage A: Independent training (120K steps) → verify orbit existence at high PSNR
  Stage B: MAML + extended fine-tune → verify MAML alignment effect

Target: PC1 > 65%, PSNR > 28 dB
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
    SIREN_CONFIG, MAML_CONFIG, DATA_CONFIG, ROTATION_CONFIG,
    ANALYSIS_CONFIG, OUTPUT_CONFIG, TRAINING_CONFIG
)


def load_image(image_path, image_size=48):
    import cv2
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


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


def train_to_convergence(coords, target, device, num_steps=120000, lr=5e-4, seed=42,
                         init_state_dict=None):
    set_seed(seed)
    model = SIREN(**SIREN_CONFIG).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)
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


def run_stage_a(image_tensor, task_angles, device, seed=42):
    print("\n" + "="*60)
    print("STAGE A: Independent Training (orbit existence at high PSNR)")
    print("="*60)

    models_params = []
    psnrs = []

    for i, angle in enumerate(task_angles):
        rotated_coords, target = get_rotation_task_data(image_tensor, angle, device)
        model, psnr = train_to_convergence(
            rotated_coords, target, device,
            num_steps=TRAINING_CONFIG["independent_steps"],
            lr=TRAINING_CONFIG["independent_lr"],
            seed=seed
        )
        psnrs.append(psnr)
        models_params.append(model.get_params())
        print(f"  angle={angle}deg: PSNR={psnr:.1f} dB")

    print(f"\nAligning {len(models_params)} models...")
    aligned_params = [models_params[0]]
    for i in range(1, len(models_params)):
        aligned, cost = align_siren_parameters(models_params[0], models_params[i])
        aligned_params.append(aligned)

    flat_params = torch.stack([flatten_params(p) for p in aligned_params]).numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(ANALYSIS_CONFIG["pca_components"], len(task_angles) - 1))
    pca.fit(flat_params)

    results = {
        "stage": "A_independent",
        "psnrs": psnrs,
        "mean_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0,
    }

    print(f"\nStage A Results:")
    print(f"  PC1: {results['pc1_variance']*100:.2f}%")
    print(f"  PC2: {results['pc2_variance']*100:.2f}%")
    print(f"  Mean PSNR: {results['mean_psnr']:.1f} dB")

    return results, models_params


def run_stage_b(image_tensor, task_angles, device, seed=42):
    print("\n" + "="*60)
    print("STAGE B: MAML + Extended Fine-tune (alignment effect)")
    print("="*60)

    set_seed(seed)
    meta_model = SIREN(**SIREN_CONFIG).to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=MAML_CONFIG["meta_lr"])
    meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        meta_optimizer, T_max=MAML_CONFIG["meta_steps"]
    )
    n_tasks = len(task_angles)

    for step in tqdm(range(MAML_CONFIG["meta_steps"]), desc="MAML Meta-Train"):
        task_idx = step % n_tasks
        angle = task_angles[task_idx]
        rotated_coords, target = get_rotation_task_data(image_tensor, angle, device)

        inner_model = SIREN(**SIREN_CONFIG).to(device)
        inner_model.load_state_dict(meta_model.state_dict())
        inner_optimizer = torch.optim.Adam(inner_model.parameters(), lr=MAML_CONFIG["inner_lr"])
        inner_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            inner_optimizer, T_max=MAML_CONFIG["inner_steps"]
        )

        for inner_step in range(MAML_CONFIG["inner_steps"]):
            inner_optimizer.zero_grad()
            out = inner_model(rotated_coords)
            loss = F.mse_loss(out, target)
            loss.backward()
            inner_optimizer.step()
            inner_scheduler.step()

        meta_optimizer.zero_grad()
        out = meta_model(rotated_coords)
        loss = F.mse_loss(out, target)
        loss.backward()
        meta_optimizer.step()
        meta_scheduler.step()

        if step % 500 == 0 and step > 0:
            with torch.no_grad():
                psnr = -10 * np.log10(F.mse_loss(meta_model(rotated_coords), target).item() + 1e-10)
            tqdm.write(f"  Step {step}: PSNR@{angle}deg = {psnr:.1f} dB")

    meta_state = meta_model.state_dict()

    print("\nExtended fine-tuning from MAML init...")
    models_params = []
    psnrs = []

    for angle in task_angles:
        rotated_coords, target = get_rotation_task_data(image_tensor, angle, device)
        model, psnr = train_to_convergence(
            rotated_coords, target, device,
            num_steps=TRAINING_CONFIG["independent_steps"],
            lr=TRAINING_CONFIG["independent_lr"],
            seed=seed,
            init_state_dict=meta_state
        )
        psnrs.append(psnr)
        models_params.append(model.get_params())
        print(f"  angle={angle}deg: PSNR={psnr:.1f} dB")

    print(f"\nAligning {len(models_params)} models...")
    aligned_params = [models_params[0]]
    for i in range(1, len(models_params)):
        aligned, cost = align_siren_parameters(models_params[0], models_params[i])
        aligned_params.append(aligned)

    flat_params = torch.stack([flatten_params(p) for p in aligned_params]).numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(ANALYSIS_CONFIG["pca_components"], len(task_angles) - 1))
    pca.fit(flat_params)

    results = {
        "stage": "B_maml_finetune",
        "psnrs": psnrs,
        "mean_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "pc1_variance": float(pca.explained_variance_ratio_[0]),
        "pc2_variance": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0,
    }

    print(f"\nStage B Results:")
    print(f"  PC1: {results['pc1_variance']*100:.2f}%")
    print(f"  PC2: {results['pc2_variance']*100:.2f}%")
    print(f"  Mean PSNR: {results['mean_psnr']:.1f} dB")

    return results


def run_experiment(image_path, seed=42, device="cuda", save_dir=None):
    set_seed(seed)
    device = torch.device(device)
    image_tensor = load_image(image_path, DATA_CONFIG["image_size"]).to(device)
    task_angles = ROTATION_CONFIG["param_range"]

    print(f"\n{'='*60}")
    print(f"Image: {Path(image_path).name}, Seed: {seed}")
    print(f"Transform: Rotation (SO(2)), {len(task_angles)} tasks")
    print(f"Model: SIREN hidden={SIREN_CONFIG['hidden_dim']}, layers={SIREN_CONFIG['num_layers']}")
    print(f"Independent steps: {TRAINING_CONFIG['independent_steps']}")
    print(f"{'='*60}")

    results_a, _ = run_stage_a(image_tensor, task_angles, device, seed=seed)
    results_b = run_stage_b(image_tensor, task_angles, device, seed=seed)

    combined = {
        "image": Path(image_path).name,
        "seed": seed,
        "transform_type": "rotation",
        "n_tasks": len(task_angles),
        "task_angles": task_angles,
        "stage_a": results_a,
        "stage_b": results_b,
        "siren_config": SIREN_CONFIG,
        "training_config": {k: v for k, v in TRAINING_CONFIG.items()},
        "maml_config": {k: v for k, v in MAML_CONFIG.items()},
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = Path(save_dir) / f"rotation_{Path(image_path).stem}_seed{seed}.json"
        with open(save_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\nSaved to {save_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(description="Phase 1 R1: Rotation Orbit Validation")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=str, default="42")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_base = args.results_dir or os.path.join(OUTPUT_CONFIG["results_dir"], "Rotation_Orbit")
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
            result = run_experiment(img_path, seed=seed, device=device, save_dir=results_base)
            all_results.append(result)

    pc1_a = [r["stage_a"]["pc1_variance"] for r in all_results]
    pc1_b = [r["stage_b"]["pc1_variance"] for r in all_results]
    psnr_a = [r["stage_a"]["mean_psnr"] for r in all_results]
    psnr_b = [r["stage_b"]["mean_psnr"] for r in all_results]

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Stage A (Independent): PC1={np.mean(pc1_a)*100:.2f}%, PSNR={np.mean(psnr_a):.1f} dB")
    print(f"Stage B (MAML+Finetune): PC1={np.mean(pc1_b)*100:.2f}%, PSNR={np.mean(psnr_b):.1f} dB")

    pc1_ok = np.mean(pc1_a) >= ANALYSIS_CONFIG["pc1_threshold"]
    psnr_ok = np.mean(psnr_a) >= TRAINING_CONFIG["psnr_target"]
    print(f"\nPC1 >= {ANALYSIS_CONFIG['pc1_threshold']*100}%: {'PASS' if pc1_ok else 'FAIL'}")
    print(f"PSNR >= {TRAINING_CONFIG['psnr_target']:.0f} dB: {'PASS' if psnr_ok else 'FAIL'}")

    aggregate = {
        "n_experiments": len(all_results),
        "stage_a_pc1": {"mean": float(np.mean(pc1_a)), "std": float(np.std(pc1_a))},
        "stage_b_pc1": {"mean": float(np.mean(pc1_b)), "std": float(np.std(pc1_b))},
        "stage_a_psnr": {"mean": float(np.mean(psnr_a)), "min": float(np.min(psnr_a))},
        "stage_b_psnr": {"mean": float(np.mean(psnr_b)), "min": float(np.min(psnr_b))},
        "individual_results": all_results,
    }
    agg_path = Path(results_base) / "aggregate_results.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")


if __name__ == "__main__":
    main()
