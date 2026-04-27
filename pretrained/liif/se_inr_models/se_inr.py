"""
SE-INR: Scale-Equivariant Implicit Neural Representation

Architecture:
- PolarLogFreqPE: Polar coordinate log-frequency positional encoding
- AdditiveInjection: Additive feature fusion
- SE-Decoder: 1D convolution along frequency axis (GELU, bias=False)
- GaussianScaleReadout: Gaussian soft interpolation scale readout
- ResidualBranch: Scale-specific detail branch with learnable alpha
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


class PolarLogFreqPE(nn.Module):
    """
    Polar coordinate log-frequency positional encoding.

    rho = log(||dx|| + softplus(eps_raw))  [smooth, JIT-compatible]
    theta = atan2(dy, dx)

    Frequency encoding along rho: [sin(w_k * rho), cos(w_k * rho)] for k=0..K-1
    Angular encoding: [sin(m * theta), cos(m * theta)] for m=1..M

    Output: [B, Q, K, D_pe] where D_pe = 2 + 2*M
    """

    def __init__(self, K=24, omega_min=1.0, omega_max=30.0, M=8):
        super().__init__()
        self.K = K
        self.M = M
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.D_pe = 2 + 2 * M

        delta_omega = (math.log(omega_max) - math.log(omega_min)) / (K - 1)
        self.register_buffer(
            'frequencies',
            torch.tensor([omega_min * math.exp(k * delta_omega) for k in range(K)])
        )
        self.delta_omega = delta_omega

        self.eps_raw = nn.Parameter(torch.tensor(-5.0))

    def forward(self, rel_coord):
        """
        rel_coord: [B, Q, 2] relative coordinates in LR pixel units
        Returns:   [B, Q, K, D_pe]
        """
        B, Q = rel_coord.shape[:2]

        eps = F.softplus(self.eps_raw)
        r_sq = (rel_coord * rel_coord).sum(dim=-1, keepdim=True)
        rho = 0.5 * torch.log(r_sq + eps)
        theta = torch.atan2(rel_coord[..., 1:2], rel_coord[..., 0:1] + eps)

        freq = self.frequencies.view(1, 1, self.K)
        wr = (rho * freq).unsqueeze(-1)
        E_rho = torch.cat([torch.sin(wr), torch.cos(wr)], dim=-1)

        m_vals = torch.arange(1, self.M + 1, device=theta.device, dtype=theta.dtype)
        m_theta = theta * m_vals.view(1, 1, self.M)
        E_theta = torch.cat([torch.sin(m_theta), torch.cos(m_theta)], dim=-1)
        E_theta = E_theta.unsqueeze(2).expand(B, Q, self.K, 2 * self.M)

        E = torch.cat([E_rho, E_theta], dim=-1)
        return E


class AdditiveInjection(nn.Module):
    """
    Additive feature injection: h = pe_proj(E) + z_proj(z).unsqueeze(1)
    z_proj is shared across all frequency positions (equivariance-safe).
    """

    def __init__(self, feat_dim, D_pe=18, C_h=128):
        super().__init__()
        self.C_h = C_h
        self.pe_proj = nn.Linear(D_pe, C_h, bias=False)
        self.z_proj = nn.Linear(feat_dim, C_h, bias=True)

    def forward(self, z, pe):
        """
        z:  [B*Q, feat_dim] encoder features
        pe: [B, Q, K, D_pe] positional encoding
        Returns: [B, Q, K, C_h]
        """
        B, Q, K = pe.shape[:3]

        pe_flat = pe.view(B * Q, K, -1)
        h_pe = self.pe_proj(pe_flat)
        z_proj = self.z_proj(z)
        h = h_pe + z_proj.unsqueeze(1)

        return h.view(B, Q, K, self.C_h)


class SEDecoderLayer(nn.Module):
    """
    SE-Decoder single layer: 1D convolution along frequency axis + residual.
    Conv1d preserves translation equivariance. bias=False. GELU activation.
    """

    def __init__(self, C_h, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            C_h, C_h, kernel_size,
            padding=padding, bias=False
        )
        self.act = nn.GELU()

    def forward(self, h):
        return self.act(self.conv(h)) + h


class SEDecoder(nn.Module):
    """
    SE-Decoder: multi-layer 1D convolution along frequency axis.
    3 layers of Conv1d with residual connections.
    """

    def __init__(self, C_h=128, num_layers=3, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList([
            SEDecoderLayer(C_h, kernel_size)
            for _ in range(num_layers)
        ])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class GaussianScaleReadout(nn.Module):
    """
    Gaussian soft interpolation scale readout.

    tau = (K-1) * (log s - log s_min) / (log s_max_ood - log s_min), clamp(0, K-1)
    alpha_k = softmax(-(k - tau)^2 / (2 * sigma^2))
    y_eq = sum_k alpha_k * to_rgb_k(H_k)  (per-position projection)
    """

    def __init__(self, C_h, K, inp_size, s_min=1.0, s_max_ood=30.0, sigma=0.8):
        super().__init__()
        self.C_h = C_h
        self.K = K
        self.sigma = sigma
        self.inp_size = inp_size
        self.s_min = s_min
        self.s_max_ood = s_max_ood
        self.register_buffer('k_idx', torch.arange(K, dtype=torch.float32))

        self.to_rgb_weight = nn.Parameter(torch.empty(K, 3, C_h))
        self.to_rgb_bias = nn.Parameter(torch.zeros(K, 3))
        with torch.no_grad():
            for k in range(K):
                nn.init.kaiming_uniform_(self.to_rgb_weight[k], a=math.sqrt(5))

    def _compute_tau(self, cell):
        cell_geomean = (cell[:, 0] * cell[:, 1]).sqrt().clamp(min=1e-10)
        log_s = math.log(2.0 / self.inp_size) - torch.log(cell_geomean)
        tau = (self.K - 1) * (log_s - math.log(self.s_min)) / \
              (math.log(self.s_max_ood) - math.log(self.s_min))
        tau = tau.clamp(0, self.K - 1)
        return tau

    def forward(self, h, cell):
        """
        h:    [B*Q, C_h, K]
        cell: [B*Q, 2]
        Returns: [B*Q, 3]
        """
        BQ, C_h, K = h.shape

        tau = self._compute_tau(cell)

        k_idx = self.k_idx.view(1, K)
        alpha = -(k_idx - tau.unsqueeze(1)) ** 2 / (2 * self.sigma ** 2)
        alpha = F.softmax(alpha, dim=1)

        h_perm = h.permute(0, 2, 1)
        rgb_k = torch.einsum('bkc,koc->bko', h_perm, self.to_rgb_weight) + self.to_rgb_bias

        y_eq = (alpha.unsqueeze(-1) * rgb_k).sum(dim=1)
        return y_eq


class ResidualBranch(nn.Module):
    """
    Residual branch for scale-specific details.
    MLP: [z, rel_coord, cell] -> 256 -> 256 -> 3
    """

    def __init__(self, feat_dim, hidden_dim=256):
        super().__init__()
        in_dim = feat_dim + 4
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z, rel_coord, cell):
        """
        z:         [B*Q, feat_dim]
        rel_coord: [B*Q, 2]
        cell:      [B*Q, 2]
        Returns:   [B*Q, 3]
        """
        inp = torch.cat([z, rel_coord, cell], dim=-1)
        return self.layers(inp)


@register('se-inr')
class SEINR(nn.Module):
    """
    Scale-Equivariant Implicit Neural Representation.

    y = y_eq + sigmoid(alpha_raw) * y_res, alpha_raw init 0 -> alpha=0.5
    """

    def __init__(self, encoder_spec,
                 K=24, omega_min=1.0, omega_max=30.0, M=8,
                 C_h=128, num_layers=3, kernel_size=3,
                 sigma=0.8, s_min=1.0, s_max_ood=30.0,
                 local_ensemble=True, feat_unfold=False,
                 inp_size=48):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.K = K
        self.C_h = C_h

        self.encoder = models.make(encoder_spec)

        self.polar_pe = PolarLogFreqPE(K=K, omega_min=omega_min, omega_max=omega_max, M=M)
        delta_omega = (math.log(omega_max) - math.log(omega_min)) / (K - 1)
        self.delta_omega = delta_omega
        D_pe = 2 + 2 * M

        feat_dim = self.encoder.out_dim
        if feat_unfold:
            feat_dim *= 9

        self.add_inj = AdditiveInjection(feat_dim, D_pe=D_pe, C_h=C_h)
        self.se_decoder = SEDecoder(C_h=C_h, num_layers=num_layers, kernel_size=kernel_size)
        self.scale_readout = GaussianScaleReadout(C_h, K, inp_size, s_min, s_max_ood, sigma)
        self.res_branch = ResidualBranch(feat_dim, hidden_dim=256)
        self.alpha_raw = nn.Parameter(torch.zeros(1))

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None, return_eq=False, cons_ratio=None):
        feat = self.feat

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).to(feat.device) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        y_eq_preds = [] if return_eq else None

        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                B, Q = coord.shape[:2]
                z = q_feat.reshape(B * Q, -1)

                if cons_ratio is not None:
                    rel_coord_pe = rel_coord * cons_ratio
                    cell_readout = cell / cons_ratio
                else:
                    rel_coord_pe = rel_coord
                    cell_readout = cell

                pe = self.polar_pe(rel_coord_pe)

                h = self.add_inj(z, pe)

                h = h.permute(0, 1, 3, 2)
                h = h.reshape(B * Q, self.C_h, self.K)
                h = self.se_decoder(h)

                cell_b_r = cell_readout[:, 0, :]
                cell_bq_r = cell_b_r.unsqueeze(1).expand(B, Q, 2).reshape(B * Q, 2)

                y_eq = self.scale_readout(h, cell_bq_r)
                y_eq = y_eq.view(B, Q, 3)

                cell_b = cell[:, 0, :]
                cell_bq = cell_b.unsqueeze(1).expand(B, Q, 2).reshape(B * Q, 2)

                rel_coord_bq = rel_coord.reshape(B * Q, 2)
                y_res = self.res_branch(z, rel_coord_bq, cell_bq)
                y_res = y_res.view(B, Q, 3)

                pred = y_eq + torch.sigmoid(self.alpha_raw) * y_res

                preds.append(pred)
                if return_eq:
                    y_eq_preds.append(y_eq)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        if return_eq:
            ret_eq = 0
            for y_eq, area in zip(y_eq_preds, areas):
                ret_eq = ret_eq + y_eq * (area / tot_area).unsqueeze(-1)
            return ret, ret_eq

        return ret

    def forward(self, inp, coord, cell, return_eq=False, cons_ratio=None):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell, return_eq=return_eq, cons_ratio=cons_ratio)
