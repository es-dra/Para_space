"""
Local Texture Estimator (LTE)

Based on the CVPR 2022 paper:
  "Local Texture Estimator for Implicit Representation Function"
  Lee, Jin et al. (https://arxiv.org/abs/2111.08918)

Key differences from LIIF:
  1. Learned Fourier encoding (h_f estimates frequencies, h_a estimates amplitudes)
  2. LR skip connection (bilinear upsampled LR added as residual)
  3. The decoder MLP takes LTE features (not raw coords)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .liif import EDSREncoder


class FrequencyEstimator(nn.Module):
    """h_f: z_j ∈ ℝ^C → F_j ∈ ℝ^(K×2)

    Estimates K frequency vectors from the latent code.
    Each frequency vector is 2D, one component per spatial axis.
    """
    def __init__(self, feature_dim: int = 64, K: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, K * 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [N, C] → F: [N, K, 2]"""
        F = self.net(z)  # [N, K*2]
        # Softplus to keep frequencies positive (interpretable)
        F = F.reshape(-1, self.K, 2)
        return F  # [N, K, 2]


class AmplitudeEstimator(nn.Module):
    """h_a: z_j ∈ ℝ^C → A_j ∈ ℝ^(2K)

    Estimates Fourier coefficients from the latent code.
    Outputs 2K amplitudes (K for cos, K for sin).
    """
    def __init__(self, feature_dim: int = 64, K: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * K),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [N, C] → A: [N, 2K]"""
        return self.net(z)  # [N, 2K]


class LTEDecoder(nn.Module):
    """LTE decoder with learned frequency encoding.

    Per the paper:
      h_ψ(z_j, δ) = A_j ⊙ [cos(π F_j δ); sin(π F_j δ)]

    where A_j = h_a(z_j), F_j = h_f(z_j), δ = x - x_j.
    Then f_θ(h_ψ(z_j, δ)) → RGB prediction.
    """
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 3,
        K: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.K = K

        # Frequency and amplitude estimators
        self.freq_estimator = FrequencyEstimator(feature_dim, K)
        self.amp_estimator = AmplitudeEstimator(feature_dim, K)

        # Decoder MLP: takes 2K-dim LTE encoding → RGB
        mlp_input_dim = 2 * K

        layers = []
        for i in range(num_layers):
            in_dim = mlp_input_dim if i == 0 else hidden_dim
            out_dim_i = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim_i))
            if i < num_layers - 1:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.out_dim:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def lte_encode(self, z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Compute LTE encoding: A_j ⊙ [cos(π F_j δ); sin(π F_j δ)]

        Args:
            z: [N, C] latent codes
            delta: [N, 2] relative coordinates δ = x - x_j (in [-2/feat_size, 2/feat_size])

        Returns:
            [N, 2*K] LTE encoded features
        """
        F = self.freq_estimator(z)      # [N, K, 2]
        A = self.amp_estimator(z)       # [N, 2K]

        # F @ δ: [N, K, 2] @ [N, 2, 1] → [N, K, 1] → [N, K]
        freq = torch.bmm(F, delta.unsqueeze(-1)).squeeze(-1)  # [N, K]

        # [cos(πf); sin(πf)] for each frequency
        cos_out = torch.cos(math.pi * freq)   # [N, K]
        sin_out = torch.sin(math.pi * freq)   # [N, K]
        encoding = torch.cat([cos_out, sin_out], dim=-1)  # [N, 2K]

        return A * encoding  # [N, 2K]

    def forward(
        self,
        x_q: torch.Tensor,
        z: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_q: [N, 2] query coordinates (used for relative offset computation)
            z: [N, C] interpolated latent features at query positions
            delta: [N, 2] relative offsets from nearest grid positions

        Returns:
            [N, 3] RGB predictions (before LR residual addition)
        """
        lte_feat = self.lte_encode(z, delta)
        return self.mlp(lte_feat)  # [N, 3]

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        self.load_state_dict(params, strict=True)

    def __repr__(self):
        return (
            f"LTEDecoder(feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, K={self.K})"
        )


class LTEModel(nn.Module):
    """LTE model with encoder, LTE decoder, and LR skip connection.

    Per the paper, the full prediction is:
      ŝ(x) = f_θ(h_ψ(z_j, x - x_j)) + LR↑(x)

    where LR↑ is bilinear upsampled LR image (skip connection for DC component).
    """
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 3,
        K: int = 16,
    ):
        super().__init__()
        self.encoder = EDSREncoder(in_ch=3, n_feats=feature_dim, n_resblocks=8)
        self.decoder = LTEDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            K=K,
        )

    def compute_delta(self, coords: torch.Tensor, feat_size: int) -> torch.Tensor:
        """Compute relative offsets from nearest feature grid positions.

        Maps coordinates from [-1, 1] to the feature grid and computes
        the fractional offset within each cell.

        Args:
            coords: [N, 2] query coordinates in [-1, 1]
            feat_size: spatial size of feature map (e.g. 12 for 12×12)

        Returns:
            [N, 2] δ = x - x_j where x_j is the nearest grid center
        """
        # Map [-1, 1] → [0, feat_size-1] in pixel coordinates
        pixel_coords = (coords + 1) / 2 * (feat_size - 1)  # [N, 2]

        # Nearest grid position
        grid_centers = pixel_coords.round()  # [N, 2]

        # δ in pixel units, then normalize to [-0.5/feat_size, 0.5/feat_size]
        delta_pixels = pixel_coords - grid_centers  # [N, 2], in [-0.5, 0.5]

        # Convert δ back to normalized coordinates
        delta = delta_pixels / (feat_size - 1) * 2  # in normalized coords

        return delta

    def forward(
        self,
        coords: torch.Tensor,
        image: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            coords: [N, 2] query coordinates in [-1, 1]
            image: [3, H, W] or [1, 3, H, W] image in [0, 1]
            scale: SR scale factor (use 0 or 1 for self-recon mode)

        Returns:
            [N, 3] RGB predictions in [0, 1]
        """
        # Handle batched or unbatched image + normalize [0,1] → [-1,1]
        if image.dim() == 4:
            img_batch = image
            H_lr, W_lr = image.shape[2], image.shape[3]
        else:
            img_batch = image.unsqueeze(0)
            H_lr, W_lr = image.shape[1], image.shape[2]
        img_batch = (img_batch - 0.5) / 0.5

        # Compute HR dimensions
        sr = scale if scale > 0 else 1.0
        H_hr = int(round(H_lr * sr))
        W_hr = int(round(W_lr * sr))

        # 1. Encode LR → feature map
        feature_map = self.encoder(img_batch)  # [1, feat_dim, H_lr, W_lr]
        if feature_map.dim() == 4:
            feature_map = feature_map[0]
        feat_h, feat_w = feature_map.shape[1:]

        # 2. Sample features at query coordinates
        sample_grid = coords.view(1, 1, -1, 2)
        z = F.grid_sample(
            feature_map.unsqueeze(0),
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        z = z.squeeze(0).squeeze(1).permute(1, 0)  # [N, feat_dim]

        # 3. Compute relative offsets δ
        delta = self.compute_delta(coords, feat_w)  # [N, 2]

        # 4. LTE decoder → RGB residuals
        rgb_residual = self.decoder(coords, z, delta)  # [N, 3]

        # 5. LR skip connection: bilinear upsampled LR as DC baseline
        with torch.no_grad():
            lr_up = F.interpolate(
                img_batch, size=(H_hr, W_hr),
                mode="bilinear", align_corners=False,
            )
            B, C_up, H_hr_out, W_hr_out = lr_up.shape
            lr_flat = lr_up.reshape(C_up, -1).T  # [N, 3]

        output = (rgb_residual + lr_flat) * 0.5 + 0.5
        return output.clamp(0, 1)

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        self.load_state_dict(params, strict=True)

    def __repr__(self):
        return f"LTEModel(encoder={self.encoder}, decoder={self.decoder})"
