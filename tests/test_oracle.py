import numpy as np

from wsinr import oracle, render, siren
from wsinr.edits import apply_edits


def _eff():
    p = siren.init_params(seed=3)
    p["W1"] = (np.random.default_rng(1).normal(size=(2, 32)) * 0.5).astype(np.float32)
    return siren.to_effective(p)


def _img(eff):
    return render.render_effective(eff, 28, clip=False)[..., 0]


def test_geometry_exact():
    eff = _eff()
    base = _img(eff)
    assert np.allclose(_img(oracle.flip_lr(eff)), base[:, ::-1])
    assert np.allclose(_img(oracle.flip_ud(eff)), base[::-1, :])
    assert np.allclose(_img(oracle.transpose(eff)), base.T)
    assert np.allclose(_img(oracle.rot90(eff, 1)), np.rot90(base, 1))
    sh = _img(oracle.translate(eff, 3, -2, 28))
    assert np.allclose(sh[3:, :26], base[:25, 2:])


def test_value_and_symmetry():
    eff = _eff()
    base = _img(eff)
    assert np.allclose(_img(oracle.negative(eff)), 1 - base)
    assert np.allclose(_img(oracle.contrast(eff, 2.0)), 0.5 + 2 * (base - 0.5))
    perm = np.random.default_rng(0).permutation(32)
    assert np.allclose(_img(oracle.permute_hidden(eff, 1, perm)), base)
    assert np.allclose(_img(oracle.sign_flip_hidden(eff, 2, [0, 5])), base)


def test_edits_parser():
    eff = _eff()
    new, rep = apply_edits(eff, "A1[0, :] *= -1\nc3[0] = 0.25\nbad line\n")
    assert rep["applied"] == 2 and len(rep["errors"]) == 1
    assert np.allclose(new["A1"][0], -eff["A1"][0]) and new["c3"][0] == 0.25
