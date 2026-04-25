"""
Phase 1 S2/R2: Independent Training Baseline for Conditional INRs

Supports LIIF and LTE models.
Trains independently on each task (no MAML shared initialization).
After training, applies decoder MLP permutation alignment, then analyzes PCA.

Key fix: Decoder MLP has permutation symmetry just like SIREN.
Skipping alignment (as in previous version) allows permutation noise
to dominate PCA, producing artificially low PC1 values.

Usage:
    python experiments/Phase1_Independent/run.py --model liif --transform scale
    python experiments/Phase1_Independent/run.py --model lte --transform scale
    python experiments/Phase1_Independent/run.py --model liif --transform rotation
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

from src.models.liif import LIIFModel
from src.models.lte import LTEModel
from src.datasets import get_image_coordinates
from src.alignment import (
    flatten_params, align_decoder_parameters, get_decoder_state_dict,
    get_encoder_state_dict, AlignmentMethod,
)
from src.utils import set_seed
from experiments.config import (
    LIIF_CONFIG, LTE_CONFIG, TRAINING_CONFIG, DATA_CONFIG,
    SCALE_CONFIG, ROTATION_CONFIG, ANALYSIS_CONFIG, OUTPUT_CONFIG,
)


def load_image(image_path, image_size=48):
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def create_model(model_type, device):
    if model_type == "liif":
        return LIIFModel(**LIIF_CONFIG).to(device)
    elif model_type == "lte":
        return LTEModel(**LTE_CONFIG).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_scale_task_data(image_tensor, scale_factor, device):
    import cv2
    C, H, W = image_tensor.shape
    new_h, new_w = int(H * scale_factor), int(W * scale_factor)
    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_tensor = torch.from_numpy(resized).permute(2, 0, 1).float().to(device) / 255.0
    coords = get_image_coordinates(new_h, new_w, normalize="center", device=device).reshape(-1, 2)
    target = resized_tensor.reshape(3, -1).T
    return coords, target, resized_tensor


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
    return rotated_coords, target, rotated_tensor


def train_independent(model_type, coords, target, device, image_tensor=None,
                      scale=1.0, num_steps=100000, lr=5e-4, seed=42):
    set_seed(seed)
    model = create_model(model_type, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)

    for step in range(num_steps):
        optimizer.zero_grad()
        out = model(coords, image_tensor, scale=scale)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        out = model(coords, image_tensor, scale=scale)
        psnr = -10 * np.log10(F.mse_loss(out, target).item())
    return model, psnr


def run_experiment(image_path, model_type="liif", transform_type="scale",
                   seed=42, device="cuda", save_dir=None,
                   num_steps=None, num_tasks=None):
    set_seed(seed)
    device = torch.device(device)
    image_tensor = load_image(image_path, DATA_CONFIG["image_size"]).to(device)

    if transform_type == "scale":
        all_params = SCALE_CONFIG["param_range"]
    else:
        all_params = ROTATION_CONFIG["param_range"]

    if num_tasks is not None:
        indices = np.linspace(0, len(all_params) - 1, num_tasks, dtype=int)
        params_list = [all_params[i] for i in indices]
    else:
        params_list = all_params

    steps = num_steps or TRAINING_CONFIG["independent_steps"]
    lr = TRAINING_CONFIG["independent_lr"]

    print(f"\n{'='*60}")
    print(f"Image: {Path(image_path).name}, Seed: {seed}")
    print(f"Model: {model_type}, Transform: {transform_type} (Independent)")
    print(f"Tasks: {len(params_list)}, Steps: {steps}, LR: {lr}")
    print(f"{'='*60}")

    models_list = []
    psnrs = []

    for i, param_val in enumerate(params_list):
        print(f"\n  Task {i+1}/{len(params_list)}: {transform_type}={param_val}")
        if transform_type == "scale":
            coords, target, task_image = get_scale_task_data(image_tensor, param_val, device)
            scale = param_val
        else:
            coords, target, task_image = get_rotation_task_data(image_tensor, param_val, device)
            scale = 1.0

        model, psnr = train_independent(
            model_type, coords, target, device,
            image_tensor=task_image, scale=scale,
            num_steps=steps, lr=lr, seed=seed
        )
        psnrs.append(psnr)
        models_list.append(model)
        print(f"    PSNR={psnr:.1f} dB")

    print(f"\nAligning decoder parameters (Hungarian + sign-flip)...")
    decoder_params = [get_decoder_state_dict(m) for m in models_list]
    encoder_params = [get_encoder_state_dict(m) for m in models_list]

    aligned_decoder_params = [decoder_params[0]]
    for i in range(1, len(decoder_params)):
        aligned, cost = align_decoder_parameters(
            decoder_params[0], decoder_params[i],
            method=AlignmentMethod.WEIGHT_MATCHING,
        )
        aligned_decoder_params.append(aligned)

    full_aligned = []
    for i in range(len(models_list)):
        combined = {}
        for k, v in encoder_params[i].items():
            combined[f"encoder.{k}"] = v
        for k, v in aligned_decoder_params[i].items():
            combined[f"decoder.{k}"] = v
        full_aligned.append(combined)

    flat_params = torch.stack([flatten_params(p) for p in full_aligned]).numpy()

    decoder_only_flat = torch.stack([flatten_params(p) for p in aligned_decoder_params]).numpy()
    encoder_only_flat = torch.stack([flatten_params(p) for p in encoder_params]).numpy()

    from sklearn.decomposition import PCA
    n_pca = min(ANALYSIS_CONFIG["pca_components"], len(params_list) - 1)

    pca_full = PCA(n_components=n_pca)
    pca_full.fit(flat_params)
    ev_full = pca_full.explained_variance_ratio_

    pca_dec = PCA(n_components=n_pca)
    pca_dec.fit(decoder_only_flat)
    ev_dec = pca_dec.explained_variance_ratio_

    pca_enc = PCA(n_components=n_pca)
    pca_enc.fit(encoder_only_flat)
    ev_enc = pca_enc.explained_variance_ratio_

    print(f"\nPCA Results ({model_type}, {transform_type}_independent):")
    print(f"  Full model  - PC1: {ev_full[0]*100:.2f}%, PC2: {ev_full[1]*100:.2f}%")
    print(f"  Decoder only - PC1: {ev_dec[0]*100:.2f}%, PC2: {ev_dec[1]*100:.2f}%")
    print(f"  Encoder only - PC1: {ev_enc[0]*100:.2f}%, PC2: {ev_enc[1]*100:.2f}%")
    print(f"  Mean PSNR: {np.mean(psnrs):.1f} dB")

    results = {
        "model_type": model_type,
        "image": Path(image_path).name,
        "seed": seed,
        "transform_type": f"{transform_type}_independent",
        "n_tasks": len(params_list),
        "task_params": [float(p) for p in params_list],
        "psnrs": [float(p) for p in psnrs],
        "mean_psnr": float(np.mean(psnrs)),
        "min_psnr": float(np.min(psnrs)),
        "full_explained_variance": ev_full.tolist(),
        "decoder_explained_variance": ev_dec.tolist(),
        "encoder_explained_variance": ev_enc.tolist(),
        "full_pc1": float(ev_full[0]),
        "decoder_pc1": float(ev_dec[0]),
        "encoder_pc1": float(ev_enc[0]),
        "n_params": int(flat_params.shape[1]),
        "training_steps": steps,
        "aligned": True,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{transform_type}_independent_{model_type}_{Path(image_path).stem}_seed{seed}"
        save_path = Path(save_dir) / f"{prefix}.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        npz_path = Path(save_dir) / f"{prefix}_params.npz"
        np.savez_compressed(
            npz_path,
            full_aligned=flat_params,
            decoder_aligned=decoder_only_flat,
            encoder_only=encoder_only_flat,
            task_params=np.array(params_list),
        )
        print(f"Saved to {save_path} and {npz_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 1 S2/R2: Independent Training Baseline")
    parser.add_argument("--model", type=str, default="liif", choices=["liif", "lte"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--transform", type=str, default="scale", choices=["scale", "rotation"])
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    subdir = f"{args.transform.capitalize()}_Independent"
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
                img_path, model_type=args.model, transform_type=args.transform,
                seed=seed, device=device, save_dir=results_base,
                num_steps=args.steps, num_tasks=args.num_tasks
            )
            all_results.append(result)

    pc1_full = [r["full_pc1"] for r in all_results]
    pc1_dec = [r["decoder_pc1"] for r in all_results]
    pc1_enc = [r["encoder_pc1"] for r in all_results]
    psnr_vals = [r["mean_psnr"] for r in all_results]

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({args.model}, {args.transform}_independent)")
    print(f"{'='*60}")
    print(f"Full PC1: {np.mean(pc1_full)*100:.2f}% +/- {np.std(pc1_full)*100:.2f}%")
    print(f"Decoder PC1: {np.mean(pc1_dec)*100:.2f}% +/- {np.std(pc1_dec)*100:.2f}%")
    print(f"Encoder PC1: {np.mean(pc1_enc)*100:.2f}% +/- {np.std(pc1_enc)*100:.2f}%")
    print(f"PSNR: {np.mean(psnr_vals):.1f} dB")

    aggregate = {
        "model_type": args.model,
        "transform_type": args.transform,
        "n_experiments": len(all_results),
        "full_pc1": {"mean": float(np.mean(pc1_full)), "std": float(np.std(pc1_full))},
        "decoder_pc1": {"mean": float(np.mean(pc1_dec)), "std": float(np.std(pc1_dec))},
        "encoder_pc1": {"mean": float(np.mean(pc1_enc)), "std": float(np.std(pc1_enc))},
        "mean_psnr": {"mean": float(np.mean(psnr_vals)), "min": float(np.min(psnr_vals))},
        "individual_results": all_results,
    }
    agg_path = Path(results_base) / f"aggregate_{args.model}_{args.transform}.json"
    with open(agg_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate saved to {agg_path}")


if __name__ == "__main__":
    main()
