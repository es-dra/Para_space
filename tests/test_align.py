import numpy as np

from wsinr import oracle, render, siren
from wsinr.align import align, distance


def test_align_undoes_permutation_and_sign_flips():
    p = siren.init_params(seed=5)
    eff = siren.to_effective(p)
    rng = np.random.default_rng(0)
    e2 = oracle.permute_hidden(eff, 1, rng.permutation(32))
    e2 = oracle.sign_flip_hidden(e2, 1, [1, 4, 9])
    e2 = oracle.permute_hidden(e2, 2, rng.permutation(32))
    e2 = oracle.sign_flip_hidden(e2, 2, [0, 7])
    q = siren.from_effective(e2)
    assert distance(p, q, aligned=False) > 0.1
    assert distance(p, q, aligned=True) < 1e-5
    aligned, costs = align(p, q)
    assert np.allclose(render.render(aligned), render.render(p))
