import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncoding(nn.Module):
    def __init__(self, in_dim: int, num_frequencies: int, include_input: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)
        self.out_dim = in_dim * num_frequencies * 2 + (in_dim if include_input else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = []
        if self.include_input:
            encodings.append(x)
        for freq in self.freq_bands:
            encodings.append(torch.sin(x * freq * math.pi))
            encodings.append(torch.cos(x * freq * math.pi))
        return torch.cat(encodings, dim=-1)


class ScaleEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 8, include_input: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input
        freq_bands = 2.0 ** torch.linspace(0, num_frequencies - 1, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)
        self.out_dim = num_frequencies * 2 + (1 if include_input else 0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        if s.dim() == 0:
            s = s.unsqueeze(0)
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        encodings = []
        if self.include_input:
            encodings.append(s)
        for freq in self.freq_bands:
            encodings.append(torch.sin(s * freq * math.pi))
            encodings.append(torch.cos(s * freq * math.pi))
        return torch.cat(encodings, dim=-1)


class LIIFDecoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 3,
        coord_enc_freqs: int = 8,
        scale_enc_freqs: int = 8,
        modulation: str = "concat",
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.modulation = modulation

        self.coord_encoding = FourierEncoding(2, coord_enc_freqs, include_input=True)
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
            f"LIIFDecoder(feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, out_dim={self.out_dim}, "
            f"modulation={self.modulation})"
        )


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim

        layers = []
        channels = [in_channels] + [feature_dim] * num_layers
        for i in range(num_layers):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 3, padding=1))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.encoder(image)

    def __repr__(self):
        return f"FeatureEncoder(in_channels={self.in_channels}, feature_dim={self.feature_dim})"


class LIIFModel(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        out_dim: int = 3,
        coord_enc_freqs: int = 8,
        scale_enc_freqs: int = 8,
        modulation: str = "concat",
        encoder_layers: int = 3,
    ):
        super().__init__()
        self.encoder = FeatureEncoder(
            in_channels=3, feature_dim=feature_dim, num_layers=encoder_layers
        )
        self.decoder = LIIFDecoder(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            coord_enc_freqs=coord_enc_freqs,
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
        return f"LIIFModel(encoder={self.encoder}, decoder={self.decoder})"
