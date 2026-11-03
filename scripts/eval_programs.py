"""Stage 02: programmability of blur / morphology / contrast by layer and depth, per network.

  python scripts/eval_programs.py --zoo zoo/mnist_s32 --tag L3w32 --n 20 --out results/programs_L3w32.json
Writes per-network rows: no_edit, exact (L=2 only), rms_attenuation, lastlayer bound, finetune k, harmonic index.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import ndimage

from wsinr import oracle, programs, siren
from wsinr.baselines.lastlayer import lastlayer_lstsq
from wsinr.metrics import mse
from wsinr.render import render
from wsinr.tasks import TASKS
from wsinr.zoo_np import fit as np_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--sigmas", nargs="*", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--steps", nargs="*", type=int, default=[10, 50, 200])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.zoo, "*.npz")))[: a.n]
    nets = [siren.load_npz(f)[0] for f in files]
    effs = [siren.to_effective(p) for p in nets]
    side = 28
    bases = np.stack([render(p, side)[..., 0] for p in nets])
    L = siren.num_matrices(nets[0])
    out = {"zoo": a.zoo, "tag": a.tag, "L": L, "n": len(nets), "files": [os.path.basename(f) for f in files],
           "harmonic_index": [programs.network_harmonic_index(e) for e in effs],
           "harmonic_index_units_mean": [float(programs.harmonic_index(e).mean()) for e in effs],
           "tasks": {}}

    def row(name, targets, edited_effs: dict, masks=None):
        rec = {"no_edit": [mse(bases[i], targets[i], None if masks is None else masks[i]) for i in range(len(nets))]}
        for prog, es in edited_effs.items():
            rec[prog] = [mse(render(siren.from_effective(es[i]), side)[..., 0], targets[i], None if masks is None else masks[i])
                         for i in range(len(nets))]
        rec["lastlayer"] = [mse(render(siren.from_effective(lastlayer_lstsq(effs[i], targets[i], None if masks is None else masks[i])), side)[..., 0],
                                targets[i], None if masks is None else masks[i]) for i in range(len(nets))]
        rec["finetune"] = {}
        for st in a.steps:
            _, per = np_fit(targets, steps=st, lr=1e-3, init=nets, masks=masks)
            rec["finetune"][str(st)] = [float(x) for x in per]
        out["tasks"][name] = rec
        med = {k: float(np.median(v)) for k, v in rec.items() if k != "finetune"}
        print(f"{a.tag} {name:12s} " + "  ".join(f"{k} {v:.4f}" for k, v in med.items())
              + "  " + "  ".join(f"ft{k} {np.median(v):.4f}" for k, v in rec["finetune"].items()), flush=True)

    for s in a.sigmas:
        targets = np.stack([ndimage.gaussian_filter(b, s) for b in bases])
        progs = {"rms_attenuation": [programs.rms_attenuation_blur(e, s, side) for e in effs]}
        if L == 2:
            progs["exact"] = [oracle.lowpass_gauss(e, s, side) for e in effs]
        row(f"blur_{s:g}", targets, progs)
    for tname in ("dilate", "erode", "threshold"):
        t = TASKS[tname]
        row(tname, np.stack([t.target(b) for b in bases]), {})
    t = TASKS["contrast_2"]
    row("contrast_2", np.stack([t.target(b) for b in bases]), {"affine": [t.oracle(e) for e in effs]})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
