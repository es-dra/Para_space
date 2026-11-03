import numpy as np
from scipy import ndimage

from wsinr import oracle, programs, render, siren


def _shallow():
    p = siren.init_params(hidden=64, hidden_layers=0, seed=1)
    p["W1"] = (np.random.default_rng(2).normal(size=(2, 64)) * 0.3).astype(np.float32)
    return siren.to_effective(p)


def _fine(eff, side=28, up=8, pad_px=8):
    t = np.linspace(-1, 1, side)
    step = t[1] - t[0]
    tf = np.arange(-1 - pad_px * step, 1 + pad_px * step + 1e-9, step / up)
    yy, xx = np.meshgrid(tf, tf, indexing="ij")
    F = siren.forward_effective(eff, np.stack([yy.ravel(), xx.ravel()], 1)).reshape(len(tf), len(tf))
    idx = np.round((t - tf[0]) / (step / up)).astype(int)
    return F, idx, step, up


def test_gaussian_is_exact_on_shallow():
    eff = _shallow()
    F, idx, step, up = _fine(eff)
    ref = ndimage.gaussian_filter(F, up, mode="nearest")[np.ix_(idx, idx)]
    pred = render.render_effective(programs.lsi_filter(eff, programs.gaussian_transfer(1.0)), clip=False)[..., 0]
    assert np.mean((pred - ref) ** 2) < 1e-8
    pred2 = render.render_effective(oracle.lowpass_gauss(eff, 1.0), clip=False)[..., 0]
    assert np.allclose(pred, pred2)


def test_derivative_and_unsharp():
    eff = _shallow()
    F, idx, step, up = _fine(eff)
    ddx = (np.gradient(F, step / up, axis=1) * step)[np.ix_(idx, idx)]
    pred = render.render_effective(programs.lsi_filter(eff, programs.derivative_transfer((0, 1))), clip=False)[..., 0]
    assert np.mean((pred - ddx) ** 2) < 1e-3 * np.mean(ddx ** 2)  # reference is a finite difference
    un = (F + 2 * (F - ndimage.gaussian_filter(F, up, mode="nearest")))[np.ix_(idx, idx)]
    pred = render.render_effective(programs.lsi_filter(eff, programs.unsharp_transfer(1.0, 2.0)), clip=False)[..., 0]
    assert np.mean((pred - un) ** 2) < 1e-8


def test_deep_returns_none_and_input_attenuation_linear_regime():
    deep = siren.to_effective(siren.init_params(seed=0))
    assert programs.lsi_filter(deep, programs.gaussian_transfer(1.0)) is None
    deep["A2"] *= 0.01
    F, idx, step, up = _fine(deep)
    ref = ndimage.gaussian_filter(F, up, mode="nearest")[np.ix_(idx, idx)]
    base = F[np.ix_(idx, idx)]
    pred = render.render_effective(programs.input_attenuation_blur(deep, 1.0), clip=False)[..., 0]
    assert np.mean((pred - ref) ** 2) < 0.05 * np.mean((base - ref) ** 2)
