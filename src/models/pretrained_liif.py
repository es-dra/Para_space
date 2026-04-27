"""
Pre-trained LIIF model wrapper.

Wraps the EDSR-baseline encoder + MLP decoder architecture from the SE-INR
project (liif_old), loading pre-trained weights from
pretrained/liif/best_model.pth.

Architecture:
  - Encoder: EDSR-baseline (16 ResBlocks, 64 feats, no upsampling, 1.2M params)
  - Decoder: MLP (580 -> 256 -> 256 -> 256 -> 256 -> 3, 347K params)
    - 580 = 576 (9x64 unfolded features) + 2 (relative coord) + 2 (cell)
  - Local ensemble: 4-neighbor area-weighted interpolation
  - Feature unfold: 3x3 neighborhood -> 576-dim feature

Data format (normalization handled internally):
  - Input image: [0, 1] -> internally normalized to [-1, 1]
  - Coordinates: [-1, 1]
  - Output: [0, 1] (denormalized from [-1, 1])
"""

from typing import Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centers."""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class MeanShift(nn.Conv2d):
    """Mean shift layer (frozen, present only for checkpoint compatibility)."""
    def __init__(self, rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    """Residual block: Conv -> ReLU -> Conv + skip."""
    def __init__(self, n_feats=64, kernel_size=3, res_scale=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.ReLU(True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x).mul(self.res_scale) + x


class EDSREncoder(nn.Module):
    """EDSR-baseline encoder without upsampling tail.

    Matches `make_edsr_baseline(no_upsampling=True)` from SE-INR.
    MeanShift layers are present in state dict but bypassed in forward.
    """
    def __init__(self, n_colors=3, n_feats=64, n_resblocks=16,
                 res_scale=1, kernel_size=3, rgb_range=1):
        super().__init__()
        self.out_dim = n_feats

        # MeanShift layers (in checkpoint, but not used in forward pass)
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.head = nn.Sequential(
            nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size // 2)
        )

        body = [
            ResBlock(n_feats, kernel_size, res_scale)
            for _ in range(n_resblocks)
        ]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        return res + x


class MLPDecoder(nn.Module):
    """MLP decoder matching the SE-INR imnet.

    Layers: in_dim -> 256 -> ReLU -> 256 -> ReLU -> 256 -> ReLU -> 256 -> ReLU -> out_dim.
    """
    def __init__(self, in_dim=580, out_dim=3, hidden_list=None):
        super().__init__()
        if hidden_list is None:
            hidden_list = [256, 256, 256, 256]
        layers = []
        last = in_dim
        for h in hidden_list:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.reshape(*shape, -1)


# Default path relative to project root
_DEFAULT_PRETRAINED = str(
    Path(__file__).parent.parent.parent / "pretrained" / "liif" / "best_model.pth"
)


class PretrainedLIIF(nn.Module):
    """Pre-trained LIIF model with EDSR-baseline encoder and MLP decoder.

    Loads SE-INR pre-trained checkpoint and provides the standard
    get_params()/set_params()/forward() interface for dynamics analysis.

    Usage:
        model = PretrainedLIIF().to(device)
        output = model(coords, lr_image, scale=4)
    """

    def __init__(self):
        super().__init__()
        self.encoder = EDSREncoder()
        self.decoder = MLPDecoder()
        self.local_ensemble = True
        self.feat_unfold = True
        self.cell_decode = True

        if Path(_DEFAULT_PRETRAINED).exists():
            self._load_pretrained(_DEFAULT_PRETRAINED)
        else:
            print(f"Warning: pretrained weights not found at {_DEFAULT_PRETRAINED}")

    def _load_pretrained(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["model"]["sd"]
        # Remap checkpoint keys: imnet.* -> decoder.*
        remapped = {}
        for k, v in sd.items():
            if k.startswith("imnet."):
                remapped["decoder." + k[6:]] = v
            else:
                remapped[k] = v
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing:
            print(f"Missing keys (should be empty): {missing}")
        if unexpected:
            print(f"Unexpected keys (should be empty): {unexpected}")

    def gen_feat(self, inp: torch.Tensor):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord: torch.Tensor, cell: torch.Tensor = None) -> torch.Tensor:
        """Query RGB values via local ensemble + MLP decoder.

        Args:
            coord: [B, N, 2] query coordinates in [-1, 1]
            cell: [B, N, 2] cell size = (2/H_hr, 2/W_hr)

        Returns:
            [B, N, 3] RGB predictions in [-1, 1]
        """
        feat = self.feat

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
            )

        if self.local_ensemble:
            vx_lst, vy_lst = [-1, 1], [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = (
            make_coord(feat.shape[-2:], flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(feat.shape[0], 2, *feat.shape[-2:])
        )

        preds, areas = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_ = coord_.clamp(-1 + 1e-6, 1 - 1e-6)

                q_feat = (
                    F.grid_sample(
                        feat, coord_.flip(-1).unsqueeze(1),
                        mode="nearest", align_corners=False,
                    )[:, :, 0, :]
                    .permute(0, 2, 1)
                )
                q_coord = (
                    F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1),
                        mode="nearest", align_corners=False,
                    )[:, :, 0, :]
                    .permute(0, 2, 1)
                )

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.decoder(inp.reshape(bs * q, -1)).reshape(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            # Swap for correct bilinear weighting order
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        ret = torch.zeros_like(preds[0])
        for pred, area in zip(preds, areas):
            ret += pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, coords: torch.Tensor, model_input: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        """Forward pass with internal normalization.

        Args:
            coords: [N, 2] query coordinates in [-1, 1]
            model_input: [3, H, W] or [1, 3, H, W] image in [0, 1]
            scale: SR scale factor

        Returns:
            [N, 3] RGB predictions in [0, 1]
        """
        # Normalize input [0, 1] -> [-1, 1]
        inp = model_input.unsqueeze(0) if model_input.dim() == 3 else model_input
        inp = (inp - 0.5) / 0.5

        H_lr, W_lr = inp.shape[2], inp.shape[3]
        if scale > 0:
            H_hr = int(round(H_lr * scale))
            W_hr = int(round(W_lr * scale))
        else:
            H_hr, W_hr = H_lr, W_lr

        coord = coords.unsqueeze(0)
        cell = torch.tensor([2 / H_hr, 2 / W_hr], device=coords.device, dtype=torch.float32)
        cell = cell.view(1, 1, 2).expand(1, coords.shape[0], 2)

        self.gen_feat(inp)
        rgb = self.query_rgb(coord, cell)  # [1, N, 3] in [-1, 1]
        return (rgb.squeeze(0) * 0.5 + 0.5).clamp(0, 1)  # [N, 3] in [0, 1]

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        self.load_state_dict(params, strict=True)


__all__ = ["PretrainedLIIF"]
