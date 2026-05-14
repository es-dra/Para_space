"""
LIIF: Local Implicit Image Function (CVPR 2021, Chen et al.)

Full architecture with all core components:
  1. EDSR-baseline encoder (simplified: 8 ResBlocks, 64 channels)
  2. Feature unfolding (3×3 neighborhood)
  3. Local ensemble (4-neighbor area-weighted interpolation)
  4. Cell decoding (pixel size as input)
  5. MLP decoder (580 → 256×3 → 3)

Provides get_params() / set_params() interface for dynamics analysis.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_coord(shape, ranges=None, flatten=True):
    """Make coordinates at grid centres."""
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = ranges[i] if ranges is not None else (-1, 1)
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


# ─── EDSR Encoder ───────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block: Conv → ReLU → Conv + skip."""
    def __init__(self, n_feats=64, kernel_size=3, res_scale=1.0):
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
    """EDSR-baseline encoder (no upsampling tail).

    Architecture:
        head → [ResBlock × n_resblocks] → tail → +global_skip

    Args:
        in_ch: Input channels (3 for RGB)
        n_feats: Feature channels
        n_resblocks: Number of residual blocks
    """
    def __init__(self, in_ch=3, n_feats=64, n_resblocks=8):
        super().__init__()
        self.out_dim = n_feats

        self.head = nn.Conv2d(in_ch, n_feats, 3, padding=1)

        body = [ResBlock(n_feats, res_scale=1.0) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.head(x)
        return self.body(x) + x


# ─── LIIF Decoder (MLP) ─────────────────────────────────────────────────────

class LIIFDecoder(nn.Module):
    """MLP decoder for LIIF.

    Input:  580 = 576 (unfolded 9×64 features)
                  + 2 (relative coordinate)
                  + 2 (relative cell)
    Hidden: 256 → 256 → 256 (ReLU)
    Output: 3 (RGB in [-1, 1])
    """
    def __init__(self, in_dim=580, hidden_dim=256, num_layers=3, out_dim=3):
        super().__init__()
        layers = []
        last = in_dim
        for i in range(num_layers):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU(True))
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))

        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features <= 3:
                    # Output layer
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.reshape(*shape, -1)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        self.load_state_dict(params, strict=True)


# ─── Full LIIF Model ────────────────────────────────────────────────────────

class LIIFModel(nn.Module):
    """Full LIIF model with EDSR encoder, feature unfold, local ensemble,
    cell decode, and MLP decoder.

    Input/Output convention (matching common LIIF implementations):
      - Image input:  [0, 1] → internally normalized to [-1, 1]
      - Coordinates:  [-1, 1]
      - Cell:         (2/hr_h, 2/hr_w) in normalized coords
      - RGB output:   [-1, 1] → denormalised back to [0, 1]
    """
    def __init__(
        self,
        n_feats=64,
        n_resblocks=8,
        decoder_hidden=256,
        decoder_layers=3,
        local_ensemble=True,
        feat_unfold=True,
        cell_decode=True,
    ):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.encoder = EDSREncoder(in_ch=3, n_feats=n_feats, n_resblocks=n_resblocks)
        self.encoder.out_dim = n_feats

        # MLP input dim after unfold + coord + cell
        mlp_in = n_feats * 9 if feat_unfold else n_feats
        mlp_in += 2  # rel_coord
        if cell_decode:
            mlp_in += 2  # rel_cell

        self.decoder = LIIFDecoder(
            in_dim=mlp_in,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
            out_dim=3,
        )

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        """Query RGB values via local ensemble + MLP decoder.

        Args:
            coord: [B, N, 2] query coordinates in [-1, 1]
            cell: [B, N, 2] cell size = (2/Hr_h, 2/Hr_w)

        Returns:
            [B, N, 3] RGB predictions in [-1, 1]
        """
        feat = self.feat
        B, C, H_feat, W_feat = feat.shape

        # Feature unfold: 3×3 neighborhood → 9× channels
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                B, C * 9, H_feat, W_feat
            )

        if self.local_ensemble:
            vx_lst, vy_lst = [-1, 1], [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / H_feat / 2
        ry = 2 / W_feat / 2

        feat_coord = (
            make_coord((H_feat, W_feat), flatten=False)
            .to(feat.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(B, 2, H_feat, W_feat)
        )

        preds, areas = [], []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_ = coord_.clamp(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= H_feat
                rel_coord[:, :, 1] *= W_feat

                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode and cell is not None:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= H_feat
                    rel_cell[:, :, 1] *= W_feat
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.decoder(inp.reshape(bs * q, -1)).reshape(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            areas[0], areas[3] = areas[3], areas[0]
            areas[1], areas[2] = areas[2], areas[1]

        ret = torch.zeros_like(preds[0])
        for pred, area in zip(preds, areas):
            ret += pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, coords, image, scale=1.0):
        """Forward pass with internal normalisation.

        Args:
            coords: [N, 2] query coordinates in [-1, 1]
            image: [3, H, W] or [1, 3, H, W] image in [0, 1]
            scale: SR scale factor

        Returns:
            [N, 3] RGB predictions in [0, 1]
        """
        inp = image.unsqueeze(0) if image.dim() == 3 else image
        inp = (inp - 0.5) / 0.5  # [0, 1] → [-1, 1]

        B, C, H_lr, W_lr = inp.shape
        H_hr = int(round(H_lr * scale))
        W_hr = int(round(W_lr * scale))

        coord_ = coords.unsqueeze(0)
        cell = torch.tensor(
            [2 / H_hr, 2 / W_hr], device=coords.device, dtype=torch.float32
        ).view(1, 1, 2).expand(1, coords.shape[0], 2)

        self.gen_feat(inp)
        rgb = self.query_rgb(coord_, cell)  # [1, N, 3] in [-1, 1]
        return (rgb.squeeze(0) * 0.5 + 0.5).clamp(0, 1)  # [N, 3] in [0, 1]

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        self.load_state_dict(params, strict=True)

    def __repr__(self):
        return (
            f"LIIFModel(encoder={self.encoder.__class__.__name__}, "
            f"local_ensemble={self.local_ensemble}, "
            f"feat_unfold={self.feat_unfold}, "
            f"cell_decode={self.cell_decode})"
        )
