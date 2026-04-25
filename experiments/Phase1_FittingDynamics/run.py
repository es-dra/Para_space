"""
Phase 1 D1: Fitting Dynamics for Conditional INRs

Tracks parameter evolution during single-image fitting.
Key design decisions for conditional INRs (LIIF/LTE):
  - Track encoder and decoder parameters SEPARATELY
  - Compute per-component PCA (encoder PCA, decoder PCA)
  - Add frequency-domain analysis to verify spectral bias
  - Self-reconstruction PSNR is expected to be very high for conditional INRs
    (encoder sees target image). The meaningful analysis is on the
    parameter trajectory structure, not the PSNR magnitude.

Usage:
    python experiments/Phase1_FittingDynamics/run.py --model liif
    python experiments/Phase1_FittingDynamics/run.py --model lte
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
    flatten_params, flatten_decoder_params, flatten_encoder_params,
    get_decoder_state_dict, get_encoder_state_dict,
)
from src.utils import set_seed
from experiments.config import LIIF_CONFIG, LTE_CONFIG, DYNAMICS_CONFIG, DATA_CONFIG


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


def compute_frequency_spectrum(image_tensor, n_freq_bins=8):
    C, H, W = image_tensor.shape
    spectrum_energies = []
    for c in range(C):
        channel = image_tensor[c].numpy()
        fft = np.fft.fft2(channel)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted) ** 2

        cy, cx = H // 2, W // 2
        max_radius = min(cy, cx)

        freq_energies = []
        for b in range(n_freq_bins):
            r_inner = b * max_radius // n_freq_bins
            r_outer = (b + 1) * max_radius // n_freq_bins
            y, x = np.ogrid[:H, :W]
            radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = (radius >= r_inner) & (radius < r_outer)
            freq_energies.append(np.sum(magnitude[mask]))

        total = sum(freq_energies) + 1e-10
        spectrum_energies.append([e / total for e in freq_energies])

    return np.mean(spectrum_energies, axis=0)


def compute_error_spectrum(target_tensor, recon_tensor, n_freq_bins=8):
    C, H, W = target_tensor.shape
    spectrum_energies = []
    for c in range(C):
        error = (target_tensor[c] - recon_tensor[c]).numpy()
        fft = np.fft.fft2(error)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted) ** 2

        cy, cx = H // 2, W // 2
        max_radius = min(cy, cx)

        freq_energies = []
        for b in range(n_freq_bins):
            r_inner = b * max_radius // n_freq_bins
            r_outer = (b + 1) * max_radius // n_freq_bins
            y, x = np.ogrid[:H, :W]
            radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            mask = (radius >= r_inner) & (radius < r_outer)
            freq_energies.append(np.sum(magnitude[mask]))

        total = sum(freq_energies) + 1e-10
        spectrum_energies.append([e / total for e in freq_energies])

    return np.mean(spectrum_energies, axis=0)


def compute_reconstruction_and_error_spectrum(model, coords, image_tensor, device, n_freq_bins=8):
    model.eval()
    with torch.no_grad():
        output = model(coords, image_tensor, scale=1.0)
    C, H, W = image_tensor.shape
    recon = output.detach().cpu().T.reshape(C, H, W).clamp(0, 1)
    target_cpu = image_tensor.cpu()
    error_spec = compute_error_spectrum(target_cpu, recon, n_freq_bins)
    return error_spec


def run_dynamics(
    image_path,
    model_type="liif",
    seed=42,
    device="cuda",
    save_dir=None,
    total_steps=None,
    snapshot_interval=None,
):
    set_seed(seed)
    device = torch.device(device)

    cfg = DYNAMICS_CONFIG
    total_steps = total_steps or cfg["total_steps"]
    snapshot_interval = snapshot_interval or cfg["snapshot_interval"]
    image_size = cfg["image_size"]

    image_tensor = load_image(image_path, image_size).to(device)
    C, H, W = image_tensor.shape

    model = create_model(model_type, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_decoder = sum(p.numel() for p in model.decoder.parameters())

    print(f"\n{'='*60}")
    print(f"Fitting Dynamics: {model_type}")
    print(f"Image: {Path(image_path).name}, Size: {image_size}x{image_size}")
    print(f"Total params: {n_params:,} (encoder: {n_encoder:,}, decoder: {n_decoder:,})")
    print(f"Total steps: {total_steps}, Snapshot every: {snapshot_interval}")
    print(f"{'='*60}")

    coords = get_image_coordinates(H, W, normalize="center", device=device).reshape(-1, 2)
    target = image_tensor.reshape(C, -1).T

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    theta_0_full = model.get_params()
    theta_0_flat = flatten_params(theta_0_full)
    theta_0_enc = get_encoder_state_dict(model)
    theta_0_dec = get_decoder_state_dict(model)
    enc_0_flat = flatten_params(theta_0_enc)
    dec_0_flat = flatten_params(theta_0_dec)

    full_snapshots = [theta_0_flat.cpu().numpy()]
    enc_snapshots = [enc_0_flat.cpu().numpy()]
    dec_snapshots = [dec_0_flat.cpu().numpy()]
    snapshot_steps = [0]
    losses = []
    psnrs = []
    freq_ratios = []

    target_spectrum = compute_frequency_spectrum(image_tensor.cpu())
    n_freq_bins = len(target_spectrum)
    low_freq_ratio_target = np.sum(target_spectrum[:n_freq_bins // 2]) / (np.sum(target_spectrum) + 1e-10)

    for step in tqdm(range(1, total_steps + 1), desc=f"Fitting ({model_type})"):
        optimizer.zero_grad()
        output = model(coords, image_tensor, scale=1.0)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_val = float(loss.detach().cpu())
        losses.append(loss_val)

        if step % snapshot_interval == 0 or step == total_steps:
            with torch.no_grad():
                out = model(coords, image_tensor, scale=1.0)
                mse_val = F.mse_loss(out, target).item()
                psnr_val = -10 * np.log10(mse_val + 1e-10)

            error_spectrum = compute_reconstruction_and_error_spectrum(model, coords, image_tensor, device, n_freq_bins)
            low_freq_error_ratio = np.sum(error_spectrum[:n_freq_bins // 2]) / (np.sum(error_spectrum) + 1e-10)
            freq_ratios.append(low_freq_error_ratio)

            psnrs.append(psnr_val)

            current_full = flatten_params(model.get_params()).cpu().numpy()
            current_enc = flatten_params(get_encoder_state_dict(model)).cpu().numpy()
            current_dec = flatten_params(get_decoder_state_dict(model)).cpu().numpy()

            full_snapshots.append(current_full)
            enc_snapshots.append(current_enc)
            dec_snapshots.append(current_dec)
            snapshot_steps.append(step)

            delta_full = np.linalg.norm(current_full - theta_0_flat.cpu().numpy())
            delta_enc = np.linalg.norm(current_enc - enc_0_flat.cpu().numpy())
            delta_dec = np.linalg.norm(current_dec - dec_0_flat.cpu().numpy())

            tqdm.write(
                f"  Step {step}: loss={loss_val:.6f}, PSNR={psnr_val:.1f} dB, "
                f"|Δθ|={delta_full:.4f} (enc={delta_enc:.4f}, dec={delta_dec:.4f}), "
                f"low-freq error ratio={low_freq_error_ratio:.3f}"
            )

    full_snapshots = np.array(full_snapshots)
    enc_snapshots = np.array(enc_snapshots)
    dec_snapshots = np.array(dec_snapshots)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        np.savez(
            os.path.join(save_dir, "trajectory.npz"),
            full_snapshots=full_snapshots,
            enc_snapshots=enc_snapshots,
            dec_snapshots=dec_snapshots,
            snapshot_steps=np.array(snapshot_steps),
            losses=np.array(losses),
            freq_ratios=np.array(freq_ratios),
            target_spectrum=target_spectrum,
        )

        summary = {
            "model_type": model_type,
            "image": Path(image_path).name,
            "seed": seed,
            "n_params": n_params,
            "n_encoder_params": n_encoder,
            "n_decoder_params": n_decoder,
            "total_steps": total_steps,
            "snapshot_interval": snapshot_interval,
            "n_snapshots": len(snapshot_steps),
            "final_psnr": float(psnrs[-1]) if psnrs else 0.0,
            "final_loss": float(losses[-1]) if losses else 0.0,
            "snapshot_steps": snapshot_steps,
            "psnrs": [float(p) for p in psnrs],
            "freq_ratios": [float(f) for f in freq_ratios],
        }
        with open(os.path.join(save_dir, "dynamics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {save_dir}")

    return {
        "full_snapshots": full_snapshots,
        "enc_snapshots": enc_snapshots,
        "dec_snapshots": dec_snapshots,
        "snapshot_steps": snapshot_steps,
        "losses": losses,
        "psnrs": psnrs,
        "freq_ratios": freq_ratios,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 D1: Fitting Dynamics for Conditional INRs"
    )
    parser.add_argument(
        "--model", type=str, default="liif",
        choices=["liif", "lte"],
    )
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--snapshot_interval", type=int, default=None)
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
        "results", "Phase1", "FittingDynamics", args.model
    )

    run_dynamics(
        image_path,
        model_type=args.model,
        seed=args.seed,
        device=device,
        save_dir=save_dir,
        total_steps=args.steps,
        snapshot_interval=args.snapshot_interval,
    )


if __name__ == "__main__":
    main()
