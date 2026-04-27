"""
Pre-trained LIIF / LIIF-EQ Fine-tuning Dynamics.

Loads a pre-trained model and fine-tunes on a single SR image pair.
Tracks parameter trajectory for PCA analysis and saves intermediate
reconstructions for visual inspection.

Usage:
    # LIIF baseline
    python experiments/Phase1_FittingDynamics/run_finetune.py \
        --model_type liif --image path/to/image.png --scale 4 --steps 5000

    # LIIF-EQ
    python experiments/Phase1_FittingDynamics/run_finetune.py \
        --model_type liif_eq --image path/to/image.png --scale 4 --steps 5000
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.pretrained_liif import PretrainedLIIF, make_coord
from src.models.pretrained_liif_eq import PretrainedLIIF_EQ
from src.alignment import (
    flatten_params,
    flatten_decoder_params,
    flatten_encoder_params,
    get_decoder_state_dict,
    get_encoder_state_dict,
    align_decoder_parameters,
)
from src.utils import set_seed
from src.spectral import compute_frequency_spectrum, compute_error_spectrum


MODEL_REGISTRY = {
    "liif": PretrainedLIIF,
    "liif_eq": PretrainedLIIF_EQ,
}


def load_sr_image(image_path, lr_size=48, scale=4, device="cpu"):
    """Load an image and create SR pair with PIL bicubic downsampling.

    Returns:
        lr_tensor: [3, lr_size, lr_size] LR image in [0, 1]
        hr_tensor: [3, hr_size, hr_size] HR image in [0, 1]
        hr_coords: [hr_size*hr_size, 2] HR coordinates in [-1, 1] (grid centers)
        cell: [hr_size*hr_size, 2] cell = (2/hr_size, 2/hr_size)
    """
    img = transforms.ToTensor()(Image.open(image_path).convert("RGB"))
    _, h, w = img.shape
    hr_size = round(lr_size * scale)

    if h >= hr_size and w >= hr_size:
        x0 = (h - hr_size) // 2
        y0 = (w - hr_size) // 2
    else:
        img = transforms.ToTensor()(
            transforms.Resize(hr_size, Image.BICUBIC)(
                transforms.ToPILImage()(img)
            )
        )
        x0, y0 = 0, 0

    crop_hr = img[:, x0: x0 + hr_size, y0: y0 + hr_size]
    crop_lr = transforms.ToTensor()(
        transforms.Resize(lr_size, Image.BICUBIC)(
            transforms.ToPILImage()(crop_hr)
        )
    )

    hr_coords = make_coord(crop_hr.shape[-2:], flatten=True).to(device)
    cell = torch.ones_like(hr_coords)
    cell[:, 0] *= 2 / crop_hr.shape[-2]
    cell[:, 1] *= 2 / crop_hr.shape[-1]

    return crop_lr.to(device), crop_hr.to(device), hr_coords, cell


@torch.no_grad()
def save_reconstruction(model, coords, lr_tensor, scale, save_path):
    """Save SR reconstruction as PNG."""
    model.eval()
    out = model(coords, lr_tensor, scale=scale)
    H_hr = int(round(lr_tensor.shape[-2] * scale))
    W_hr = int(round(lr_tensor.shape[-1] * scale))
    img = out.T.reshape(3, H_hr, W_hr).clamp(0, 1)
    transforms.ToPILImage()(img).save(save_path)
    model.train()


def run_finetune_dynamics(
    image_path,
    model_type="liif",
    seed=42,
    device="cuda",
    save_dir=None,
    total_steps=5000,
    snapshot_interval=100,
    recon_interval=500,
    lr_size=48,
    sr_scale=4,
    learning_rate=1e-5,
):
    """Run fine-tuning dynamics experiment.

    Args:
        image_path: Path to HR image
        model_type: 'liif' or 'liif_eq'
        seed: Random seed
        device: Torch device
        save_dir: Output directory
        total_steps: Number of fine-tuning steps
        snapshot_interval: Parameter snapshot frequency
        recon_interval: Reconstruction image save frequency
        lr_size: Low-resolution image size
        sr_scale: SR scale factor
        learning_rate: Fine-tuning learning rate
    """
    set_seed(seed)
    device = torch.device(device)

    lr_tensor, hr_tensor, hr_coords, cell = load_sr_image(
        image_path, lr_size=lr_size, scale=sr_scale, device=device
    )
    hr_size = round(lr_size * sr_scale)
    C, H_hr, W_hr = hr_tensor.shape
    N_coords = hr_coords.shape[0]

    target_01 = hr_tensor.reshape(C, -1).T
    target_norm = ((hr_tensor - 0.5) / 0.5).reshape(C, -1).T

    # Load model
    ModelClass = MODEL_REGISTRY[model_type]
    model = ModelClass().to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    n_encoder = sum(p.numel() for p in model.encoder.parameters())
    n_decoder = sum(p.numel() for p in model.decoder.parameters())

    mode_str = f"SR x{sr_scale}"
    print(f"\n{'=' * 60}")
    print(f"Model: {model_type}")
    print(f"Image: {Path(image_path).name}")
    print(f"LR: {lr_size}x{lr_size}, HR: {hr_size}x{hr_size}")
    print(f"Total coords: {N_coords:,}")
    print(f"Params: {n_params:,} (enc: {n_encoder:,}, dec: {n_decoder:,})")
    print(f"Steps: {total_steps}, Snapshot: {snapshot_interval}, Recon: {recon_interval}")
    print(f"LR: {learning_rate}")
    print(f"{'=' * 60}\n")

    # Initial PSNR
    model.eval()
    with torch.no_grad():
        out = model(hr_coords, lr_tensor, scale=sr_scale)
        psnr_init = -10 * np.log10(F.mse_loss(out, target_01).item() + 1e-10)
    model.train()
    print(f"Initial PSNR: {psnr_init:.2f} dB")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Parameter snapshots
    get_enc = get_encoder_state_dict
    get_dec = get_decoder_state_dict

    full_snapshots = [flatten_params(model.get_params()).cpu().numpy()]
    enc_snapshots = [flatten_params(get_enc(model)).cpu().numpy()]
    dec_snapshots = [flatten_params(get_dec(model)).cpu().numpy()]
    dec_param_dicts = [get_dec(model)]  # full dicts for alignment
    snapshot_steps = [0]
    losses = []
    psnrs = []
    freq_ratios = []

    target_spectrum = compute_frequency_spectrum(hr_tensor.cpu().numpy())
    n_freq_bins = len(target_spectrum)

    for step in tqdm(range(1, total_steps + 1), desc="Fine-tuning"):
        optimizer.zero_grad()
        output = model(hr_coords, lr_tensor, scale=sr_scale)
        output_norm = (output - 0.5) / 0.5
        loss = F.mse_loss(output_norm, target_norm)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))

        if step % snapshot_interval == 0 or step == total_steps:
            model.eval()
            with torch.no_grad():
                out = model(hr_coords, lr_tensor, scale=sr_scale)
                psnr_val = -10 * np.log10(F.mse_loss(out, target_01).item() + 1e-10)
            psnrs.append(psnr_val)

            recon_np = out.T.reshape(C, H_hr, W_hr).cpu().numpy()
            target_np = hr_tensor.cpu().numpy()
            error_spectrum = compute_error_spectrum(target_np, recon_np, n_freq_bins)
            low_freq_ratio = np.sum(error_spectrum[:n_freq_bins // 2]) / (
                np.sum(error_spectrum) + 1e-10
            )
            freq_ratios.append(low_freq_ratio)

            # Restore train mode before capturing params: Fconv_PCA.eval() registers
            # dynamic filter/bias buffers that inflate state_dict() and break
            # consistency across snapshots.
            model.train()

            full_snapshots.append(flatten_params(model.get_params()).cpu().numpy())
            enc_snapshots.append(
                flatten_params(get_enc(model)).cpu().numpy()
            )
            dec_snapshots.append(
                flatten_params(get_dec(model)).cpu().numpy()
            )
            dec_param_dicts.append(get_dec(model))
            snapshot_steps.append(step)

            delta_full = np.linalg.norm(full_snapshots[-1] - full_snapshots[0])
            delta_enc = np.linalg.norm(enc_snapshots[-1] - enc_snapshots[0])
            delta_dec = np.linalg.norm(dec_snapshots[-1] - dec_snapshots[0])
            tqdm.write(
                f"  Step {step}: PSNR={psnr_val:.1f} dB, "
                f"|Δθ|={delta_full:.4f} (enc={delta_enc:.4f}, dec={delta_dec:.4f})"
            )

        # Save intermediate reconstruction
        if recon_interval > 0 and (step % recon_interval == 0 or step == total_steps):
            recon_dir = os.path.join(save_dir, "reconstructions")
            os.makedirs(recon_dir, exist_ok=True)
            recon_path = os.path.join(recon_dir, f"step_{step:05d}.png")
            with torch.no_grad():
                save_reconstruction(model, hr_coords, lr_tensor, sr_scale, recon_path)

    full_snapshots = np.array(full_snapshots)
    enc_snapshots = np.array(enc_snapshots)
    dec_snapshots = np.array(dec_snapshots)

    # ── Permutation alignment (decoder only) ─────────────────────────────
    print("  Aligning decoder parameter trajectory (Hungarian matching + sign flips)...")
    aligned_dec_dicts = [dec_param_dicts[0]]
    for i in range(1, len(dec_param_dicts)):
        a, _ = align_decoder_parameters(aligned_dec_dicts[-1], dec_param_dicts[i])
        aligned_dec_dicts.append(a)
    dec_snapshots_aligned = np.array(
        [flatten_params(d).cpu().numpy() for d in aligned_dec_dicts]
    )
    # ──────────────────────────────────────────────────────────────────────

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, "trajectory.npz"),
            full_snapshots=full_snapshots,
            enc_snapshots=enc_snapshots,
            dec_snapshots=dec_snapshots,
            dec_snapshots_aligned=dec_snapshots_aligned,
            snapshot_steps=np.array(snapshot_steps),
            losses=np.array(losses),
            psnrs=np.array(psnrs),
            freq_ratios=np.array(freq_ratios),
            target_spectrum=target_spectrum,
            model_type=model_type,
        )

        summary = {
            "model_type": model_type,
            "image": Path(image_path).name,
            "seed": seed,
            "mode": mode_str,
            "sr_scale": sr_scale,
            "lr_size": lr_size,
            "hr_size": hr_size,
            "n_params": n_params,
            "n_encoder_params": n_encoder,
            "n_decoder_params": n_decoder,
            "total_steps": total_steps,
            "snapshot_interval": snapshot_interval,
            "n_snapshots": len(snapshot_steps),
            "learning_rate": learning_rate,
            "initial_psnr": round(psnr_init, 2),
            "final_psnr": round(float(psnrs[-1]), 2) if psnrs else 0.0,
            "final_loss": round(float(losses[-1]), 6) if losses else 0.0,
            "snapshot_steps": snapshot_steps,
            "psnrs": [round(float(p), 2) for p in psnrs],
            "freq_ratios": [round(float(f), 4) for f in freq_ratios],
        }
        with open(os.path.join(save_dir, "dynamics_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {save_dir}")
        print(f"  Initial PSNR: {psnr_init:.2f} dB")
        print(f"  Final PSNR:   {psnrs[-1]:.2f} dB")

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
    parser = argparse.ArgumentParser(description="LIIF / LIIF-EQ Fine-tuning Dynamics")
    parser.add_argument("--model_type", type=str, default="liif",
                        choices=["liif", "liif_eq"],
                        help="Model architecture")
    parser.add_argument("--image", type=str, default=None,
                        help="HR image path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Fine-tuning steps")
    parser.add_argument("--snapshot_interval", type=int, default=100,
                        help="Parameter snapshot interval")
    parser.add_argument("--recon_interval", type=int, default=500,
                        help="Reconstruction save interval (0 to disable)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Fine-tuning learning rate")
    parser.add_argument("--lr_size", type=int, default=48,
                        help="Low-resolution input size")
    parser.add_argument("--scale", type=int, default=4,
                        help="SR scale factor")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Default image: Set14 baboon.png
    if args.image:
        image_path = args.image
    elif args.model_type == "liif_eq":
        image_path = str(
            Path(__file__).parent.parent.parent / "Data" / "Set14" / "HR" / "baboon.png"
        )
    else:
        image_path = str(
            Path(__file__).parent.parent.parent / "Data" / "Set14" / "HR" / "baboon.png"
        )

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return

    model_tag = f"{args.model_type}_sr{args.scale}"
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(
            "results", "FittingDynamics",
            "LIIF_EQ" if args.model_type == "liif_eq" else "LIIF"
        )

    run_finetune_dynamics(
        image_path,
        model_type=args.model_type,
        seed=args.seed,
        device=device,
        save_dir=save_dir,
        total_steps=args.steps,
        snapshot_interval=args.snapshot_interval,
        recon_interval=args.recon_interval,
        lr_size=args.lr_size,
        sr_scale=args.scale,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
