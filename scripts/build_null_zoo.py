"""Networks with the same shapes as a zoo but freshly initialised weights (no image): the null-content
control for reading tasks. Labels are -1.

  python scripts/build_null_zoo.py --like zoo/mnist_s32 --n 20 --out zoo/null_s32
"""
import argparse
import glob
import json
import os

from wsinr import siren


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref, _ = siren.load_npz(sorted(glob.glob(os.path.join(a.like, "*.npz")))[0])
    L = siren.num_matrices(ref)
    hidden = ref["W1"].shape[1]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for i in range(a.n):
        p = siren.init_params(hidden=hidden, hidden_layers=L - 2, out_dim=ref[f"W{L}"].shape[1], seed=a.seed + i)
        name = f"null_{i:05d}"
        siren.save_npz(os.path.join(a.out, name + ".npz"), p, label=-1, index=i, split="null", fit_mse=float("nan"),
                       steps=0, lr=0.0, seed=a.seed + i)
        rows.append({"name": name, "label": -1})
    json.dump({"config": vars(a), "n_params": siren.count_params(p), "rows": rows}, open(os.path.join(a.out, "index.json"), "w"), indent=1)
    print(f"{a.n} null networks ({siren.count_params(p)} params) in {a.out}")


if __name__ == "__main__":
    main()
