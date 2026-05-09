"""
Phase 1 D1: SIREN Fitting Dynamics (Non-Conditional INR)

Trains SIREN from scratch on a single image and tracks parameter evolution.
Unlike conditional INRs (LIIF/LTE), SIREN has no encoder shortcut — the
parameter trajectory is genuinely meaningful as the model learns the image
representation from coordinates alone.

Expected behavior:
  - PSNR climbs from ~15 dB to ~35-45 dB over full training
  - Parameter trajectory shows distinct phases (rapid descent → refinement → convergence)
  - Gradient coherence decreases over time (trajectory becomes less directed)
  - This provides the "ground truth" fitting dynamics baseline

Usage:
    python experiments/Phase1_FittingDynamics/run_siren.py --image Data/Set5/HR/baby.png
    python experiments/Phase1_FittingDynamics/run_siren.py --image Data/Set5/HR/baby.png --steps 5000 --snapshot 100 --device cpu
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
from src.alignment import flatten_params, align_siren_parameters
from src.spectral import compute_frequency_spectrum, compute_error_spectrum
from src.utils import set_seed
from experiments.config import SIREN_CONFIG, SIREN_DYNAMICS_CONFIG, DATA_CONFIG


def load_image(image_path, image_size=48):
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def compute_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return np.sqrt(total_norm)


def run_siren_dynamics(
    image_path,
    seed=42,
    device="cuda",
    save_dir=None,
    total_steps=None,
):
    set_seed(seed)
    device = torch.device(device)

    cfg = SIREN_DYNAMICS_CONFIG
    total_steps = total_steps or cfg["total_steps"]
    snapshot_steps_list = cfg["snapshot_steps"]
    image_size = cfg["image_size"]

    image_tensor = load_image(image_path, image_size).to(device)
    C, H, W = image_tensor.shape

    model = SIREN(**SIREN_CONFIG).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"SIREN Fitting Dynamics")
    print(f"Image: {Path(image_path).name}, Size: {image_size}x{image_size}")
    print(f"Total params: {n_params:,}")
    print(f"Total steps: {total_steps}, Snapshots: {len(snapshot_steps_list)} (geometric)")
    print(f"{'='*60}")

    coords = get_image_coordinates(H, W, normalize="center", device=device).reshape(-1, 2)
    target = image_tensor.reshape(C, -1).T

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    theta_0 = model.get_params()
    theta_0_flat = flatten_params(theta_0).cpu().numpy()

    full_snapshots = [theta_0_flat]
    param_dicts = [theta_0]  # full dicts for alignment
    snapshot_steps = [0]
    losses = []
    psnrs = []
    freq_ratios = []
    grad_norms = []

    target_spectrum = compute_frequency_spectrum(image_tensor.cpu().numpy())
    n_freq_bins = len(target_spectrum)

    with torch.no_grad():
        initial_out = model(coords)
        initial_mse = F.mse_loss(initial_out, target).item()
        initial_psnr = -10 * np.log10(initial_mse + 1e-10)
    print(f"Initial PSNR: {initial_psnr:.1f} dB")

    for step in tqdm(range(1, total_steps + 1), desc="Fitting (SIREN)"):
        optimizer.zero_grad()
        output = model(coords)
        loss = F.mse_loss(output, target)
        loss.backward()
        grad_norm = compute_grad_norm(model)
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

        if step in snapshot_steps_list:
            with torch.no_grad():
                out = model(coords)
                mse_val = F.mse_loss(out, target).item()
                psnr_val = -10 * np.log10(mse_val + 1e-10)

            psnrs.append(psnr_val)
            grad_norms.append(grad_norm)

            recon_np = out.cpu().reshape(C, H, W).clamp(0, 1).numpy()
            target_np = image_tensor.cpu().numpy()
            error_spectrum = compute_error_spectrum(target_np, recon_np, n_freq_bins)
            low_freq_ratio = np.sum(error_spectrum[:n_freq_bins // 2]) / (
                np.sum(error_spectrum) + 1e-10
            )
            freq_ratios.append(low_freq_ratio)

            current_flat = flatten_params(model.get_params()).cpu().numpy()
            full_snapshots.append(current_flat)
            param_dicts.append(model.get_params())
            snapshot_steps.append(step)

            delta_norm = np.linalg.norm(current_flat - theta_0_flat)

            tqdm.write(
                f"  Step {step}: loss={float(loss.detach().cpu()):.6f}, PSNR={psnr_val:.1f} dB, "
                f"|Δθ|={delta_norm:.4f}, |∇|={grad_norm:.4f}, "
                f"low-freq={low_freq_ratio:.3f}"
            )

    full_snapshots = np.array(full_snapshots)
    n_snapshots = len(snapshot_steps)

    # ── Permutation alignment ────────────────────────────────────────────
    print("  Aligning parameter trajectory (Hungarian matching + sign flips)...")
    aligned_dicts = [param_dicts[0]]
    for i in range(1, len(param_dicts)):
        a, _ = align_siren_parameters(aligned_dicts[-1], param_dicts[i])
        aligned_dicts.append(a)
    full_snapshots_aligned = np.array(
        [flatten_params(d).cpu().numpy() for d in aligned_dicts]
    )
    # ──────────────────────────────────────────────────────────────────────

    print(f"\nFinal PSNR: {psnrs[-1]:.1f} dB, Loss: {losses[-1]:.6e}")
    print(f"Snapshots: {n_snapshots}, Params: {n_params:,}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        np.savez(
            os.path.join(save_dir, "trajectory.npz"),
            full_snapshots=full_snapshots,
            full_snapshots_aligned=full_snapshots_aligned,
            snapshot_steps=np.array(snapshot_steps, dtype=np.int64),
            losses=np.array(losses, dtype=np.float32),
            psnrs=np.array(psnrs, dtype=np.float32),
            freq_ratios=np.array(freq_ratios, dtype=np.float32),
            target_spectrum=target_spectrum,
            grad_norms=np.array(grad_norms, dtype=np.float32),
        )

        summary = {
            "model_type": "siren",
            "image": Path(image_path).name,
            "seed": seed,
            "n_params": n_params,
            "siren_config": SIREN_CONFIG,
            "image_size": image_size,
            "total_steps": total_steps,
            "planned_snapshot_steps": snapshot_steps_list,
            "n_snapshots": n_snapshots,
            "alignment": "hungarian+signflip",
            "initial_psnr": float(initial_psnr),
            "final_psnr": float(psnrs[-1]) if psnrs else 0.0,
            "final_loss": float(losses[-1]) if losses else 0.0,
            "snapshot_steps": snapshot_steps,
            "psnrs": [float(p) for p in psnrs],
            "freq_ratios": [round(float(f), 4) for f in freq_ratios],
            "grad_norms": [float(g) for g in grad_norms],
        }
        with open(os.path.join(save_dir, "dynamics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Results saved to {save_dir}")

    return {
        "full_snapshots": full_snapshots,
        "snapshot_steps": snapshot_steps,
        "losses": losses,
        "psnrs": psnrs,
        "grad_norms": grad_norms,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 D1: SIREN Fitting Dynamics"
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.image:
        image_path = args.image
    else:
        data_root = Path(__file__).parent.parent.parent / "Data"
        image_path = data_root / "Set5" / "HR" / "baby.png"
        if not image_path.exists():
            print(f"Default image not found: {image_path}")
            print("Please provide --image path")
            return

    save_dir = args.save_dir or os.path.join(
        "results", "FittingDynamics", "SIREN"
    )

    run_siren_dynamics(
        image_path,
        seed=args.seed,
        device=device,
        save_dir=save_dir,
        total_steps=args.steps,
    )


if __name__ == "__main__":
    main()
