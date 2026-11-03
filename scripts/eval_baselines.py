"""Reference rows for every edit task on a zoo: no-edit, closed-form oracle, and gradient fine-tuning.

  python scripts/eval_baselines.py --zoo zoo/mnist_s32 --n 50 --out results/baselines.json
"""
import argparse
import glob
import json
import os

import numpy as np

from wsinr import siren
from wsinr.baselines.lastlayer import lastlayer_lstsq
from wsinr.metrics import mse, param_change
from wsinr.render import render
from wsinr.tasks import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--steps", nargs="*", type=int, default=[10, 50, 200, 1000])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--backend", choices=["torch", "np"], default="torch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.zoo, "*.npz")))[: a.n]
    nets = [siren.load_npz(f)[0] for f in files]
    side = 28
    bases = np.stack([render(p, side)[..., 0] for p in nets])
    out = {"zoo": a.zoo, "n": len(nets), "tasks": {}}
    for tname, task in TASKS.items():
        if a.tasks and tname not in a.tasks:
            continue
        targets = np.stack([task.target(b) for b in bases])
        masks = None
        if task.mask is not None:
            masks = np.stack([task.mask(side)] * len(nets))
        row = {"level": task.level}
        row["no_edit"] = float(np.mean([mse(bases[i], targets[i], None if masks is None else masks[i]) for i in range(len(nets))]))
        ll = []
        for i, p in enumerate(nets):
            eff_ll = lastlayer_lstsq(siren.to_effective(p), targets[i], None if masks is None else masks[i])
            ll.append(mse(render(siren.from_effective(eff_ll), side)[..., 0], targets[i], None if masks is None else masks[i]))
        row["lastlayer_lstsq"] = float(np.mean(ll))
        if task.oracle is not None and task.oracle(siren.to_effective(nets[0])) is not None:
            o_mse, o_chg = [], []
            for i, p in enumerate(nets):
                eff2 = task.oracle(siren.to_effective(p))
                p2 = siren.from_effective(eff2)
                img = render(p2, side)[..., 0]
                o_mse.append(mse(img, targets[i], None if masks is None else masks[i]))
                o_chg.append(param_change(p, p2)["frac_changed"])
            row["oracle"] = float(np.mean(o_mse))
            row["oracle_frac_changed"] = float(np.mean(o_chg))
        if a.backend == "torch":
            from wsinr.baselines.finetune import finetune
            ft = finetune(nets, targets, steps_list=tuple(a.steps), lr=a.lr, device=a.device, masks=masks)
            row["finetune"] = {str(k): float(np.mean(v[1])) for k, v in ft.items()}
        else:
            from wsinr.zoo_np import fit as np_fit
            row["finetune"] = {}
            for st in a.steps:
                _, per = np_fit(targets, steps=st, lr=a.lr, init=nets, masks=masks)
                row["finetune"][str(st)] = float(np.mean(per))
        out["tasks"][tname] = row
        print(f"{tname:14s} L{task.level}  no-edit {row['no_edit']:.4f}  oracle {row.get('oracle', float('nan')):.4f}  lastlayer {row['lastlayer_lstsq']:.4f}  "
              + "  ".join(f"ft{k} {v:.4f}" for k, v in row["finetune"].items()), flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
