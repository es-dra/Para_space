"""Weight-space learning on SIREN zoos: loading, exact program augmentation, a two-level permutation-invariant
classifier, and an aligned-MLP baseline. Torch, CPU-friendly."""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
import torch.nn as nn

from . import oracle, siren
from .align import align


# ----------------------------------------------------------------------------- data
def load_zoo(zoo_dir: str, n: int | None = None):
    files = sorted(glob.glob(os.path.join(zoo_dir, "*.npz")))
    if n:
        files = files[:n]
    effs, labels = [], []
    for f in files:
        p, meta = siren.load_npz(f)
        effs.append(siren.to_effective(p))
        labels.append(int(meta["label"]))
    return effs, np.array(labels)


def stack(effs):
    """List of effective dicts -> dict of float32 tensors with a batch dimension."""
    return {k: torch.tensor(np.stack([e[k] for e in effs]), dtype=torch.float32) for k in effs[0]}


# ----------------------------------------------------------------------------- augmentation (exact programs)
def augment_programs(E: dict, side: int = 28, rng: np.random.Generator | None = None) -> dict:
    """Batched exact weight-space programs: translation, flip, rotation, zoom (layer 1) and contrast,
    brightness (readout). Works for any depth. E: dict of tensors (B, ...)."""
    rng = rng or np.random.default_rng()
    B = E["A1"].shape[0]
    L = max(int(k[1:]) for k in E if k.startswith("A"))
    out = {k: v.clone() for k, v in E.items()}
    step = 2.0 / (side - 1)
    # coordinate map p -> p R + t, composed from rotation, zoom and flip
    ang = np.deg2rad(rng.uniform(-15, 15, B))
    zoom = rng.uniform(0.9, 1.1, B)
    flip = rng.random(B) < 0.5
    R = np.zeros((B, 2, 2))
    R[:, 0, 0] = np.cos(ang); R[:, 0, 1] = -np.sin(ang); R[:, 1, 0] = np.sin(ang); R[:, 1, 1] = np.cos(ang)
    R = R / zoom[:, None, None]
    R[flip, :, 1] *= -1                                   # mirror x after rotation/zoom
    t = -rng.uniform(-2, 2, (B, 2)) * step                 # translation by up to 2 px
    Rt = torch.tensor(R, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.float32)
    out["c1"] = E["c1"] + torch.einsum("bi,bij->bj", tt, E["A1"])
    out["A1"] = torch.bmm(Rt, E["A1"])
    # value map v -> a v + b
    a = torch.tensor(rng.uniform(0.8, 1.25, B), dtype=torch.float32)
    b = torch.tensor(rng.uniform(-0.1, 0.1, B), dtype=torch.float32)
    out[f"A{L}"] = E[f"A{L}"] * a[:, None, None]
    out[f"c{L}"] = E[f"c{L}"] * a[:, None] + b[:, None] + 0.5 * (1 - a)[:, None]   # contrast about mid-grey
    return out


def augment_permute(E: dict, rng: np.random.Generator | None = None) -> dict:
    """Function-preserving control: random permutations and sign flips of every hidden layer."""
    rng = rng or np.random.default_rng()
    B = E["A1"].shape[0]
    L = max(int(k[1:]) for k in E if k.startswith("A"))
    out = {k: v.clone() for k, v in E.items()}
    for k in range(1, L):
        n = out[f"A{k}"].shape[2]
        for bi in range(B):
            perm = torch.tensor(rng.permutation(n))
            s = torch.tensor(rng.choice([-1.0, 1.0], n), dtype=torch.float32)
            out[f"A{k}"][bi] = out[f"A{k}"][bi][:, perm] * s[None, :]
            out[f"c{k}"][bi] = out[f"c{k}"][bi][perm] * s
            out[f"A{k + 1}"][bi] = out[f"A{k + 1}"][bi][perm, :] * s[:, None]
    return out


def canonicalize_signs(E: dict) -> dict:
    """Fix the sign symmetry of each hidden unit: make its largest-magnitude incoming weight positive."""
    L = max(int(k[1:]) for k in E if k.startswith("A"))
    out = {k: v.clone() for k, v in E.items()}
    for k in range(1, L):
        A = out[f"A{k}"]                                     # (B, in, units)
        idx = A.abs().argmax(dim=1, keepdim=True)            # (B, 1, units)
        s = torch.sign(torch.gather(A, 1, idx)).squeeze(1)   # (B, units)
        s[s == 0] = 1
        out[f"A{k}"] = A * s[:, None, :]
        out[f"c{k}"] = out[f"c{k}"] * s
        out[f"A{k + 1}"] = out[f"A{k + 1}"] * s[:, :, None]
    return out


# ----------------------------------------------------------------------------- models
class DeepSets2(nn.Module):
    """Two-level permutation-invariant classifier for 2-h-h-1 SIRENs (effective form).

    Level 1: each layer-1 unit i -> phi1([w_i (2), c1_i, |w_i|, phase features]).
    Level 2: each layer-2 unit j -> phi2([c2_j, A3_j, pool_i rho(phi1(i) * a_ij, a_ij)]).
    Output: pool_j, then an MLP. Invariant to permutations of both hidden layers (sign symmetry is
    removed beforehand by canonicalize_signs)."""

    def __init__(self, d: int = 64, n_classes: int = 10):
        super().__init__()
        self.phi1 = nn.Sequential(nn.Linear(6, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(d + 2, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.phi2 = nn.Sequential(nn.Linear(2 * d + 2, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, n_classes))

    def forward(self, E: dict):
        A1, c1, A2, c2, A3 = E["A1"], E["c1"], E["A2"], E["c2"], E["A3"]   # (B,2,n1) (B,n1) (B,n1,n2) (B,n2) (B,n2,1)
        w = A1.transpose(1, 2)                                             # (B, n1, 2)
        f1 = torch.cat([w / 15.0, c1.unsqueeze(-1) / 20.0, w.norm(dim=-1, keepdim=True) / 15.0,
                        torch.sin(c1).unsqueeze(-1), torch.cos(c1).unsqueeze(-1)], dim=-1)
        u1 = self.phi1(f1)                                                 # (B, n1, d)
        # messages from layer-1 units to layer-2 units
        a = A2.unsqueeze(-1)                                               # (B, n1, n2, 1)
        msg = torch.cat([u1.unsqueeze(2) * a, a, a.abs()], dim=-1)         # (B, n1, n2, d+2)
        m = self.rho(msg)
        agg = torch.cat([m.mean(dim=1), m.amax(dim=1)], dim=-1)            # (B, n2, 2d)
        f2 = torch.cat([agg, c2.unsqueeze(-1) / 20.0, A3], dim=-1)
        u2 = self.phi2(f2)                                                 # (B, n2, d)
        pooled = torch.cat([u2.mean(dim=1), u2.amax(dim=1)], dim=-1)
        return self.head(pooled)


class AlignedMLP(nn.Module):
    """Flatten the (reference-aligned) weights and classify with an MLP."""

    def __init__(self, n_params: int, d: int = 512, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_params, d), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d, d), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(d, n_classes))

    def forward(self, E: dict):
        keys = sorted(E, key=lambda k: (int(k[1:]), k[0]))
        x = torch.cat([E[k].flatten(1) for k in keys], dim=1)
        return self.net(x)


def align_batch(effs, ref):
    return [align(siren.from_effective(ref), siren.from_effective(e))[0] for e in effs]


class NeuralGraph(nn.Module):
    """Edge-conditioned message passing over the layered graph of a SIREN (any depth).

    Nodes: 2 input coordinates, every hidden unit, the output. Node input features: bias (hidden/output) and a
    layer-type one-hot. Edge features: the effective weight. K rounds of bidirectional messages
    m = MLP([h_src, h_dst, w, |w|]), mean- and max-aggregated, with residual GRU-free updates. Readout: mean and
    max over hidden nodes plus the output node. Invariant to hidden-unit permutations; the sign symmetry is
    removed by canonicalize_signs before the forward pass."""

    def __init__(self, d: int = 64, rounds: int = 3, n_classes: int = 10, n_layer_types: int = 5):
        super().__init__()
        self.d, self.rounds = d, rounds
        self.embed = nn.Linear(1 + n_layer_types, d)
        self.msg_f = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + 2, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(rounds)])
        self.msg_b = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + 2, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(rounds)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(5 * d, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(rounds)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(rounds)])
        self.head = nn.Sequential(nn.Linear(3 * d, d), nn.ReLU(), nn.Linear(d, n_classes))

    def _nodes(self, E):
        L = max(int(k[1:]) for k in E if k.startswith("A"))
        B = E["A1"].shape[0]
        feats, groups = [], []
        feats.append(torch.zeros(B, E["A1"].shape[1], 1))                       # inputs: no bias
        groups.append(0)
        for k in range(1, L + 1):
            feats.append(E[f"c{k}"].unsqueeze(-1) / (20.0 if k < L else 1.0))   # hidden biases are phases (~20 rad)
            groups.append(min(k, 4))
        return feats, groups, L

    def forward(self, E: dict):
        feats, groups, L = self._nodes(E)
        B = E["A1"].shape[0]
        hs = []
        for f, g in zip(feats, groups):
            onehot = torch.zeros(B, f.shape[1], 5)
            onehot[:, :, g] = 1.0
            hs.append(self.embed(torch.cat([f, onehot], dim=-1)))
        Ws = [E[f"A{k}"] / (15.0 if k == 1 else 1.0) for k in range(1, L + 1)]  # (B, n_{k-1}, n_k)
        for r in range(self.rounds):
            new = []
            for li in range(len(hs)):
                parts = [hs[li]]
                if li < L:   # messages from the next layer (backward along edges li -> li+1)
                    W = Ws[li]                                                   # (B, n_li, n_li+1)
                    src, dst = hs[li + 1].unsqueeze(1).expand(-1, W.shape[1], -1, -1), hs[li].unsqueeze(2).expand(-1, -1, W.shape[2], -1)
                    m = self.msg_b[r](torch.cat([src, dst, W.unsqueeze(-1), W.abs().unsqueeze(-1)], dim=-1))
                    parts += [m.mean(dim=2), m.amax(dim=2)]
                else:
                    parts += [torch.zeros_like(hs[li])] * 2
                if li > 0:   # messages from the previous layer (forward along edges li-1 -> li)
                    W = Ws[li - 1]                                               # (B, n_li-1, n_li)
                    src, dst = hs[li - 1].unsqueeze(2).expand(-1, -1, W.shape[2], -1), hs[li].unsqueeze(1).expand(-1, W.shape[1], -1, -1)
                    m = self.msg_f[r](torch.cat([src, dst, W.unsqueeze(-1), W.abs().unsqueeze(-1)], dim=-1))
                    parts += [m.mean(dim=1), m.amax(dim=1)]
                else:
                    parts += [torch.zeros_like(hs[li])] * 2
                new.append(self.norm[r](hs[li] + self.upd[r](torch.cat(parts, dim=-1))))
            hs = new
        hidden = torch.cat(hs[1:L], dim=1)                                          # all hidden nodes
        pooled = torch.cat([hidden.mean(dim=1), hidden.amax(dim=1), hs[L].mean(dim=1)], dim=-1)
        return self.head(pooled)
