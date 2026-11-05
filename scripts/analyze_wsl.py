"""Score G1-G5 of studies/07-weight-space-augmentation/protocol.json from results/wsl/*.json."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/wsl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    acc = defaultdict(list)
    for f in glob.glob(os.path.join(a.results, "*.json")):
        d = json.load(open(f))
        c = d["config"]
        acc[(c["model"], c["aug"], c["n_train"])].append(d["final_test_acc"])
    table = {f"{m}|{g}|{n}": {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_seeds": len(v)} for (m, g, n), v in sorted(acc.items())}
    for k, v in table.items():
        print(f"{k:28s} {v['mean']*100:5.1f} +- {v['std']*100:4.1f} (n={v['n_seeds']})")
    g = lambda m, au, n: table.get(f"{m}|{au}|{n}", {}).get("mean")
    P = {}
    if g("deepsets2", "programs", 1000) is not None and g("deepsets2", "none", 1000) is not None:
        d = g("deepsets2", "programs", 1000) - g("deepsets2", "none", 1000)
        P["G1_programs_help"] = {"gain_points": 100 * d, "pass": d >= 0.05}
    if g("deepsets2", "programs", 1000) is not None and g("deepsets2", "permute", 1000) is not None:
        d = g("deepsets2", "programs", 1000) - g("deepsets2", "permute", 1000)
        P["G2_better_than_permutation"] = {"gain_points": 100 * d, "pass": d >= 0.03}
    if g("deepsets2", "programs", 1000) is not None and g("deepsets2", "none", 2000) is not None:
        P["G3_data_efficiency"] = {"programs_1000": g("deepsets2", "programs", 1000), "none_2000": g("deepsets2", "none", 2000),
                                   "pass": g("deepsets2", "programs", 1000) >= g("deepsets2", "none", 2000)}
    if g("aligned_mlp", "programs", 1000) is not None and g("aligned_mlp", "none", 1000) is not None:
        d = g("aligned_mlp", "programs", 1000) - g("aligned_mlp", "none", 1000)
        P["G4_also_for_the_naive_model"] = {"gain_points": 100 * d, "pass": d >= 0.03}
    if all(g("deepsets2", au, n) is not None for au in ("programs", "none") for n in (500, 5000)):
        g500 = g("deepsets2", "programs", 500) - g("deepsets2", "none", 500)
        g5k = g("deepsets2", "programs", 5000) - g("deepsets2", "none", 5000)
        P["G5_gap_closes_with_data"] = {"gain_500": 100 * g500, "gain_5000": 100 * g5k, "pass": g5k < g500}
    json.dump({"table": table, "predictions": P}, open(a.out, "w"), indent=1)
    print(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
