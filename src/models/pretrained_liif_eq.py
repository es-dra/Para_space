"""
Pre-trained LIIF-EQ model wrapper.

Rotation-equivariant version of LIIF (Rot-E ASISR style):
  - Encoder: EDSR-eq-baseline (Fconv_PCA equivariant convolutions, C4 group)
  - Decoder: e_mlp_2 (equivariant MLP with group-sum pooling)

Architecture matches the SE-INR project's liif + edsr-eq-baseline + e_mlp_2.
Provides the same get_params()/set_params()/forward() interface as PretrainedLIIF.

Pretrained path: pretrained/liif_eq/best_model.pth (DIV2K, val PSNR 30.63 dB)
"""

from typing import Dict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.pretrained_liif import make_coord
from src.models.eq import make_edsr_baseline, EQ_MLP


# Default path relative to project root
_DEFAULT_PRETRAINED = str(
    Path(__file__).parent.parent.parent / "pretrained" / "liif_eq" / "best_model.pth"
)


class PretrainedLIIF_EQ(nn.Module):
    """Pre-trained LIIF-EQ model with rotation-equivariant encoder and MLP decoder.

    Loads SE-INR checkpoint and provides the standard
    get_params()/set_params()/forward() interface for dynamics analysis.

    Architecture spec (from config.yaml):
      encoder: edsr-eq-baseline (kernel_size=5, res_scale=0.1, no_upsampling=True)
      imnet: e_mlp_2 (hidden_list=[512, 256, 256, 256], out_dim=3)
      tranNum=4 (C4 rotation group)
      feat_unfold=True, cell_decode=True, local_ensemble=True
    """

    def __init__(self, pretrained_path=None):
        super().__init__()
        self.local_ensemble = True
        self.feat_unfold = True
        self.cell_decode = True
        self.tranNum = 4

        # Build encoder: edsr-eq-baseline
        self.encoder = make_edsr_baseline(
            no_upsampling=True, kernel_size=5, res_scale=0.1,
            tranNum=self.tranNum, n_feats=64, n_resblocks=16,
        )
        self.encoder.out_dim = 64  # n_feats

        # Build decoder: e_mlp_2
        imnet_in_dim = self.encoder.out_dim  # 64
        if self.feat_unfold:
            if self.tranNum > 1:
                kernel_size = 5
                from src.models.eq import B_Conv as fn
                self.adjust = fn.Fconv_PCA(
                    kernel_size, imnet_in_dim // self.tranNum,
                    imnet_in_dim // self.tranNum * 9, self.tranNum,
                    inP=kernel_size, padding=(kernel_size - 1) // 2, ifIni=0
                )
            imnet_in_dim *= 9  # 576
        imnet_in_dim += 2  # attach coord -> 578
        if self.cell_decode:
            imnet_in_dim += 2 * self.tranNum  # +8 -> 586

        self.decoder = EQ_MLP(
            in_dim=imnet_in_dim, out_dim=3,
            hidden_list=[512, 256, 256, 256], tranNum=self.tranNum
        )

        if pretrained_path is None:
            pretrained_path = _DEFAULT_PRETRAINED
        if Path(pretrained_path).exists():
            self._load_pretrained(pretrained_path)
        else:
            print(f"Warning: pretrained weights not found at {pretrained_path}")

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
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.feat_unfold:
            if self.tranNum == 1:
                feat = F.unfold(feat, 3, padding=1).view(
                    feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3]
                )
            else:
                feat = self.adjust(feat)
                feat = F.relu(feat)

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

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode="nearest", align_corners=False,
                )[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    rel_cellx = rel_cell[:, :, 0].unsqueeze(2).repeat([1, 1, self.tranNum])
                    rel_celly = rel_cell[:, :, 1].unsqueeze(2).repeat([1, 1, self.tranNum])
                    q_feat = torch.cat([q_feat, rel_cellx, rel_celly], dim=-1)

                inp = torch.cat([q_feat, rel_coord], dim=-1)

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

    def forward(self, coords, model_input, scale=1.0):
        """Forward pass with internal normalization.

        Args:
            coords: [N, 2] query coordinates in [-1, 1]
            model_input: [3, H, W] or [1, 3, H, W] image in [0, 1]
            scale: SR scale factor (also used to infer HR size)

        Returns:
            [N, 3] RGB predictions in [0, 1]
        """
        inp = model_input.unsqueeze(0) if model_input.dim() == 3 else model_input
        # Normalize [0, 1] -> [-1, 1]
        inp = (inp - 0.5) / 0.5

        H_lr, W_lr = inp.shape[2], inp.shape[3]
        H_hr = int(round(H_lr * scale))
        W_hr = int(round(W_lr * scale))

        coord = coords.unsqueeze(0)
        cell = torch.tensor(
            [2 / H_hr, 2 / W_hr], device=coords.device, dtype=torch.float32
        )
        cell = cell.view(1, 1, 2).expand(1, coords.shape[0], 2)

        self.gen_feat(inp)
        rgb = self.query_rgb(coord, cell)  # [1, N, 3] in [-1, 1]
        return (rgb.squeeze(0) * 0.5 + 0.5).clamp(0, 1)  # [N, 3] in [0, 1]

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return full model state (parameters + buffers).

        Uses state_dict() instead of named_parameters() to capture all
        trainable params AND persistent buffers (Basis matrices, rotation
        matrices cosTheta/sinTheta, etc.). Buffers are critical for
        equivariant layers — without them parameter snapshots are incomplete.
        """
        return {k: v.detach().clone() for k, v in self.state_dict().items()
                if v.dtype.is_floating_point or v.dtype.is_complex}

    def set_params(self, params: Dict[str, torch.Tensor]):
        """Load full model state with strict matching.

        Uses load_state_dict for consistency with PretrainedLIIF and to
        catch silent key mismatches.
        """
        self.load_state_dict(params, strict=False)

    def get_encoder_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.encoder.state_dict().items()
                if v.dtype.is_floating_point or v.dtype.is_complex}

    def get_decoder_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.decoder.state_dict().items()
                if v.dtype.is_floating_point or v.dtype.is_complex}

__all__ = ["PretrainedLIIF_EQ"]
