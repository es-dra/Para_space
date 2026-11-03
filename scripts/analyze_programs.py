"""Score the frozen predictions of studies/02-programs/protocol.json from results/programs_*.json.

  python scripts/analyze_programs.py --results results/programs_L2w64.json results/programs_L2w32.json \
        results/programs_L3w32.json results/programs_L4w32.json --out studies/02-programs/outcomes.json
"""
import argparse
import json

import numpy as np
from scipy import stats


def boot_ci(x, fn=np.median, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    v = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {json.load(open(f))["tag"]: json.load(open(f)) for f in a.results}
    O = {"per_zoo": {}, "predictions": {}}
    for tag, r in R.items():
        z = {"L": r["L"], "n": r["n"]}
        for task, rec in r["tasks"].items():
            ne = np.array(rec["no_edit"])
            row = {"no_edit_median": float(np.median(ne))}
            for k, v in rec.items():
                if k in ("no_edit", "finetune"):
                    continue
                v = np.array(v)
                row[k] = {"median": float(np.median(v)), "ci": boot_ci(v), "ratio_to_no_edit_median": float(np.median(v / np.maximum(ne, 1e-12))),
                          "n_below_half_no_edit": int(np.sum(v < 0.5 * ne)), "max": float(v.max())}
            row["finetune"] = {k: float(np.median(v)) for k, v in rec["finetune"].items()}
            for prog in ("rms_attenuation", "input_attenuation"):
                if prog in rec:
                    row[f"{prog}_above_lastlayer"] = int(np.sum(np.array(rec[prog]) > np.array(rec["lastlayer"])))
            z[task] = row
        O["per_zoo"][tag] = z

    P = {}
    # Q1: exact blur on shallow zoos
    q1 = {}
    for tag in [t for t in R if R[t]["L"] == 2]:
        q1[tag] = {"max_mse_sigma1": O["per_zoo"][tag]["blur_1"]["exact"]["max"]}
    P["Q1_exactness"] = {"detail": q1, "pass": all(v["max_mse_sigma1"] < 2e-4 for v in q1.values()) if q1 else None}
    # Q2: deep: rms above bound on >= 18/20 and ratio in [0.3, 0.9]
    q2 = {}
    for tag in [t for t in R if R[t]["L"] >= 3]:
        b = O["per_zoo"][tag]["blur_1"]
        q2[tag] = {"rms_above_lastlayer": b["rms_attenuation_above_lastlayer"], "ratio": b["rms_attenuation"]["ratio_to_no_edit_median"],
                   "input_above_lastlayer": b.get("input_attenuation_above_lastlayer"),
                   "input_ratio": b.get("input_attenuation", {}).get("ratio_to_no_edit_median")}
    P["Q2_depth_degrades"] = {"detail": q2, "pass": all(v["rms_above_lastlayer"] >= 18 and 0.3 <= v["ratio"] <= 0.9 for v in q2.values()) if q2 else None}
    # Q3: correlation of per-network ratio with harmonic index across deep nets
    xs, ys = [], []
    for tag in [t for t in R if R[t]["L"] >= 3]:
        rec = R[tag]["tasks"]["blur_1"]
        ys += list(np.array(rec["rms_attenuation"]) / np.maximum(np.array(rec["no_edit"]), 1e-12))
        xs += R[tag]["harmonic_index"]
    if xs:
        rho, pval = stats.spearmanr(xs, ys)
        P["Q3_failure_is_predicted_by_harmonics"] = {"spearman_rho": float(rho), "p": float(pval), "n": len(xs), "pass": bool(rho > 0.4)}
        ys2 = []
        for tag in [t for t in R if R[t]["L"] >= 3]:
            rec = R[tag]["tasks"]["blur_1"]
            if "input_attenuation" in rec:
                ys2 += list(np.array(rec["input_attenuation"]) / np.maximum(np.array(rec["no_edit"]), 1e-12))
        if len(ys2) == len(xs):
            rho2, p2 = stats.spearmanr(xs, ys2)
            P["Q3_failure_is_predicted_by_harmonics"]["input_attenuation"] = {"spearman_rho": float(rho2), "p": float(p2)}
    # Q4: readout-only bound for morphology vs blur on L3w32
    if "L3w32" in R:
        z = O["per_zoo"]["L3w32"]
        n = R["L3w32"]["n"]
        above = {t: n - z[t]["lastlayer"]["n_below_half_no_edit"] for t in ("dilate", "erode")}
        P["Q4_readout_only_cannot_do_morphology"] = {"dilate_erode_bound_above_half": above, "blur_bound_below_half": z["blur_1"]["lastlayer"]["n_below_half_no_edit"],
                                                     "pass": all(v >= 15 for v in above.values()) and z["blur_1"]["lastlayer"]["n_below_half_no_edit"] >= 15}
    # Q5: contrast affine below 1e-6 everywhere
    q5 = {tag: O["per_zoo"][tag]["contrast_2"]["affine"]["max"] for tag in R}
    P["Q5_learned_editors_lose_to_algebra"] = {"max_mse": q5, "pass": all(v < 1e-6 for v in q5.values())}
    O["predictions"] = P
    json.dump(O, open(a.out, "w"), indent=1)
    print(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
