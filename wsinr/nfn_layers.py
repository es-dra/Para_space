"""Permutation-equivariant linear layers on MLP weight spaces (NP and HNP of Zhou et al., 2023), written for
this study and verified numerically (tests/test_nfn_layers.py).

Weight space of an MLP with L layers: W_k of shape (n_k, n_{k-1}) and b_k of shape (n_k,), k = 1..L, each carried
with a channel axis: W_k -> (B, c, n_k, n_{k-1}), b_k -> (B, c, n_k). Hidden dimensions n_1..n_{L-1} are permuted
by the symmetry group; n_0 (inputs) and n_L (outputs) are fixed. An equivariant linear map sends each tensor to a
sum of broadcast pooled statistics of all tensors, each with its own channel mixing. HNP additionally mixes
position-specifically along the fixed input and output dimensions.

Our SIRENs in effective form map to this layout with W_k = A_k^T (units x inputs), b_k = c_k.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NPLinear(nn.Module):
    def __init__(self, dims: list[int], c_in: int, c_out: int, hnp: bool = True):
        """dims = [n_0, n_1, ..., n_L]."""
        super().__init__()
        self.dims, self.L, self.c_in, self.c_out, self.hnp = dims, len(dims) - 1, c_in, c_out, hnp
        n0, nL = dims[0], dims[-1]
        self.lin = nn.ModuleDict()
        # for every output tensor, a channel map for each contributing statistic
        for k in range(1, self.L + 1):
            terms_W = ["self", "row", "col", "all", "bk", "bk_mean", "bprev", "bprev_mean", "Wnext_col", "Wnext_all", "Wprev_row", "Wprev_all", "bias"]
            terms_b = ["self", "mean", "Wk_row", "Wk_all", "Wnext_col", "Wnext_all", "bprev_mean", "bnext_mean", "bias"]
            for t in terms_W:
                self.lin[f"W{k}_{t}"] = nn.Linear(c_in, c_out, bias=False)
            for t in terms_b:
                self.lin[f"b{k}_{t}"] = nn.Linear(c_in, c_out, bias=False)
        # HNP: position-specific mixing over the fixed input dim of W1 and output dim of W_L / b_L
        if hnp:
            self.in_mix = nn.Parameter(torch.randn(c_out, c_in, n0, n0) / (c_in * n0) ** 0.5)
            self.out_mix_W = nn.Parameter(torch.randn(c_out, c_in, nL, nL) / (c_in * nL) ** 0.5)
            self.out_mix_b = nn.Parameter(torch.randn(c_out, c_in, nL, nL) / (c_in * nL) ** 0.5)
        self.const = nn.ParameterDict({f"W{k}": nn.Parameter(torch.zeros(c_out)) for k in range(1, self.L + 1)})
        self.const.update({f"b{k}": nn.Parameter(torch.zeros(c_out)) for k in range(1, self.L + 1)})

    def _m(self, name, x):
        """apply channel map lin[name] to x with channels on axis 1."""
        return torch.einsum("oc,bc...->bo...", self.lin[name].weight, x)

    def forward(self, Ws: list[torch.Tensor], bs: list[torch.Tensor]):
        """Ws[k-1]: (B, c, n_k, n_{k-1}); bs[k-1]: (B, c, n_k). Returns lists of the same shapes with c_out."""
        L = self.L
        outW, outb = [], []
        gmean = [W.mean(dim=(2, 3)) for W in Ws]                     # (B, c)
        bmean = [b.mean(dim=2) for b in bs]
        for k in range(1, L + 1):
            W, b = Ws[k - 1], bs[k - 1]
            Bsz, c, nk, nk1 = W.shape
            o = self._m(f"W{k}_self", W)
            o = o + self._m(f"W{k}_row", W.mean(dim=3, keepdim=True)).expand(-1, -1, -1, nk1)       # pooled over inputs -> per unit
            o = o + self._m(f"W{k}_col", W.mean(dim=2, keepdim=True)).expand(-1, -1, nk, -1)        # pooled over units -> per input
            o = o + self._m(f"W{k}_all", gmean[k - 1])[:, :, None, None]
            o = o + self._m(f"W{k}_bk", b)[:, :, :, None]
            o = o + self._m(f"W{k}_bk_mean", bmean[k - 1])[:, :, None, None]
            if k > 1:
                o = o + self._m(f"W{k}_bprev", bs[k - 2])[:, :, None, :]
                o = o + self._m(f"W{k}_bprev_mean", bmean[k - 2])[:, :, None, None]
                o = o + self._m(f"W{k}_Wprev_row", Ws[k - 2].mean(dim=3))[:, :, None, :]            # per n_{k-1} unit
                o = o + self._m(f"W{k}_Wprev_all", gmean[k - 2])[:, :, None, None]
            if k < L:
                o = o + self._m(f"W{k}_Wnext_col", Ws[k].mean(dim=2))[:, :, :, None]               # per n_k unit
                o = o + self._m(f"W{k}_Wnext_all", gmean[k])[:, :, None, None]
            if self.hnp and k == 1:        # fixed input dimension: position-specific mixing
                o = o + torch.einsum("ocij,bcnj->boni", self.in_mix, W)
            if self.hnp and k == L:        # fixed output dimension
                o = o + torch.einsum("ocij,bcjn->boin", self.out_mix_W, W)
            o = o + self.const[f"W{k}"][None, :, None, None]
            outW.append(o)
            ob = self._m(f"b{k}_self", b)
            ob = ob + self._m(f"b{k}_mean", bmean[k - 1])[:, :, None]
            ob = ob + self._m(f"b{k}_Wk_row", W.mean(dim=3))
            ob = ob + self._m(f"b{k}_Wk_all", gmean[k - 1])[:, :, None]
            if k < L:
                ob = ob + self._m(f"b{k}_Wnext_col", Ws[k].mean(dim=2))
                ob = ob + self._m(f"b{k}_Wnext_all", gmean[k])[:, :, None]
                ob = ob + self._m(f"b{k}_bnext_mean", bmean[k])[:, :, None]
            if k > 1:
                ob = ob + self._m(f"b{k}_bprev_mean", bmean[k - 2])[:, :, None]
            if self.hnp and k == L:
                ob = ob + torch.einsum("ocij,bcj->boi", self.out_mix_b, b)
            ob = ob + self.const[f"b{k}"][None, :, None]
            outb.append(ob)
        return outW, outb


class NPPool(nn.Module):
    """Invariant readout: for every tensor, mean and max over permuted dims; fixed dims kept (HNP)."""

    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims, self.L = dims, len(dims) - 1

    def out_features(self, c):
        L, d = self.L, self.dims
        n = 0
        for k in range(1, L + 1):
            rows = d[0] if k == 1 else 1          # fixed input dim kept for W1
            cols = d[L] if k == L else 1          # fixed output dim kept for W_L
            n += 2 * c * rows * cols              # mean and max
            n += 2 * c * (d[L] if k == L else 1)  # bias
        return n

    def forward(self, Ws, bs):
        L = self.L
        feats = []
        for k in range(1, L + 1):
            W, b = Ws[k - 1], bs[k - 1]
            dims = []
            if k != L:
                dims.append(2)   # units of W_k permuted unless output layer
            if k != 1:
                dims.append(3)   # inputs of W_k permuted unless first layer
            if dims:
                feats += [W.mean(dim=dims).flatten(1), W.amax(dim=dims).flatten(1)]
            else:
                feats += [W.flatten(1), W.flatten(1)]
            if k != L:
                feats += [b.mean(dim=2).flatten(1), b.amax(dim=2).flatten(1)]
            else:
                feats += [b.flatten(1), b.flatten(1)]
        return torch.cat(feats, dim=1)


class NFNClassifier(nn.Module):
    def __init__(self, dims: list[int], c: int = 32, n_layers: int = 3, n_classes: int = 10, hnp: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([NPLinear(dims, 1 if i == 0 else c, c, hnp) for i in range(n_layers)])
        self.pool = NPPool(dims)
        self.head = nn.Sequential(nn.Linear(self.pool.out_features(c), 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, n_classes))

    def forward(self, Ws, bs):
        for i, layer in enumerate(self.layers):
            Ws, bs = layer(Ws, bs)
            if i < len(self.layers) - 1:
                Ws, bs = [torch.relu(W) for W in Ws], [torch.relu(b) for b in bs]
        return self.head(self.pool(Ws, bs))


def effective_to_ws(E: dict, scale_first: float = 15.0, scale_bias: float = 20.0):
    """Effective-form batch dict -> (Ws, bs) lists with a channel axis, lightly normalised."""
    L = max(int(k[1:]) for k in E if k.startswith("A"))
    Ws, bs = [], []
    for k in range(1, L + 1):
        W = E[f"A{k}"].transpose(1, 2)                       # (B, n_k, n_{k-1})
        b = E[f"c{k}"]
        if k == 1:
            W = W / scale_first
        if k < L:
            b = b / scale_bias
        Ws.append(W.unsqueeze(1))
        bs.append(b.unsqueeze(1))
    return Ws, bs
