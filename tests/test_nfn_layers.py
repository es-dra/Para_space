import numpy as np
import torch

from wsinr import siren, wsl
from wsinr.nfn_layers import NFNClassifier, NPLinear, NPPool, effective_to_ws


def _batch(n=4):
    effs = [siren.to_effective(siren.init_params(seed=s)) for s in range(n)]
    return wsl.stack(effs)


def _permute(Ws, bs, perms):
    """Apply hidden permutations perms[k] (k = 1..L-1) to a weight-space batch."""
    Ws = [W.clone() for W in Ws]
    bs = [b.clone() for b in bs]
    L = len(Ws)
    for k in range(1, L):
        p = perms[k - 1]
        Ws[k - 1] = Ws[k - 1][:, :, p, :]
        bs[k - 1] = bs[k - 1][:, :, p]
        Ws[k] = Ws[k][:, :, :, p]
    return Ws, bs


def test_np_layer_equivariant_and_classifier_invariant():
    torch.manual_seed(0)
    E = _batch()
    Ws, bs = effective_to_ws(E)
    dims = [2, 32, 32, 1]
    layer = NPLinear(dims, 1, 8, hnp=True)
    perms = [torch.randperm(32), torch.randperm(32)]
    oW, ob = layer(Ws, bs)
    pW, pb = layer(*_permute(Ws, bs, perms))
    eW, eb = _permute(oW, ob, perms)
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(pW, eW))
    assert all(torch.allclose(x, y, atol=1e-5) for x, y in zip(pb, eb))
    clf = NFNClassifier(dims, c=8).eval()
    out = clf(Ws, bs)
    out_p = clf(*_permute(Ws, bs, perms))
    assert torch.allclose(out, out_p, atol=1e-4)
    # not invariant to a non-symmetry (swapping the two input coordinates) when hnp
    Ws2 = [W.clone() for W in Ws]
    Ws2[0] = Ws2[0][:, :, :, [1, 0]]
    assert not torch.allclose(clf(Ws2, bs), out, atol=1e-3)
