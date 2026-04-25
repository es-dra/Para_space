import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .liif import ScaleEncoding, FeatureEncoder


class DCTEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 16, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.out_dim = num_frequencies * 2 + (2 if include_input else 0)
        n = torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("dct_indices", n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = []
        if self.include_input:
            encodings.append(x)

        for k in self.dct_indices:
            cos_x = torch.cos(math.pi * (k + 0.5) * (x[..., 0:1]))
            cos_y = torch.cos(math.pi * (k + 0.5) * (x[..., 1:2]))
            encodings.append(cos_x)
            encodings.append(cos_y)

        return torch.cat(encodings, dim=-1)


class LTEDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 3,
        dct_frequencies: int = 16,
        scale_enc_freqs: int = 8,
        modulation: str = "concat",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.modulation = modulation

        self.coord_encoding = DCTEncoding(dct_frequencies, include_input=True)
        self.scale_encoding = ScaleEncoding(scale_enc_freqs, include_input=True)

        coord_enc_dim = self.coord_encoding.out_dim
        scale_enc_dim = self.scale_encoding.out_dim

        if modulation == "concat":
            mlp_input_dim = coord_enc_dim + feature_dim + scale_enc_dim
        elif modulation == "film":
            mlp_input_dim = coord_enc_dim + scale_enc_dim
            self.film_layer = nn.Linear(feature_dim, hidden_dim * 2)
        else:
            raise ValueError(f"Unknown modulation: {modulation}")

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(mlp_input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers = nn.ModuleList(layers)
        self._init_weights()

    def _init_weights(self):
        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)
            else:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x_q: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        coord_enc = self.coord_encoding(x_q)
        scale_enc = self.scale_encoding(s)

        if scale_enc.shape[0] != coord_enc.shape[0]:
            scale_enc = scale_enc.expand(coord_enc.shape[0], -1)

        if self.modulation == "concat":
            h = torch.cat([coord_enc, z, scale_enc], dim=-1)
        elif self.modulation == "film":
            h = torch.cat([coord_enc, scale_enc], dim=-1)
            film_params = self.film_layer(z)
            gamma, beta = film_params.chunk(2, dim=-1)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < self.num_layers - 1:
                if self.modulation == "film" and i == 0:
                    h = h * (1 + gamma) + beta
                h = F.relu(h)

        return h

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        self.load_state_dict(params, strict=True)

    def __repr__(self):
        return (
            f"LTEDecoder(feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, out_dim={self.out_dim}, "
            f"modulation={self.modulation})"
        )


class LTEModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 3,
        dct_frequencies: int = 16,
        scale_enc_freqs: int = 8,
        modulation: str = "concat",
        encoder_layers: int = 3,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            in_channels=3, feature_dim=feature_dim, num_layers=encoder_layers
        )
        self.decoder = LTEDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            dct_frequencies=dct_frequencies,
            scale_enc_freqs=scale_enc_freqs,
            modulation=modulation,
        )

    def forward(
        self,
        coords: torch.Tensor,
        image: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        feature_map = self.encoder(image)

        if feature_map.dim() == 4:
            feature_map = feature_map[0]

        sample_grid_2d = coords.view(1, 1, -1, 2)

        local_features = F.grid_sample(
            feature_map.unsqueeze(0),
            sample_grid_2d,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        local_features = local_features.squeeze(0).squeeze(1).permute(1, 0)

        s_tensor = torch.tensor([scale], device=coords.device, dtype=coords.dtype)
        s_tensor = s_tensor.expand(coords.shape[0], 1)

        output = self.decoder(coords, local_features, s_tensor)
        return output

    def get_params(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_params(self, params: Dict[str, torch.Tensor]):
        self.load_state_dict(params, strict=True)

    def __repr__(self):
        return f"LTEModel(encoder={self.encoder}, decoder={self.decoder})"
