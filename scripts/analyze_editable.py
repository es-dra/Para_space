"""Score E1-E5 of studies/04-editable-training/protocol_v2.json.

  python scripts/analyze_editable.py --out studies/04-editable-training/outcomes_v2.json
"""
import argparse
import json
import os

import numpy as np

BOUNDS = [1, 2, 4, 8]
DATASETS = ["mnist", "fashion"]
SEEDS = [0, 1]


def load(tag):
    f = f"results/programs_{tag}.json"
    if not os.path.exists(f):
        return None
    r = json.load(open(f))
    idx = json.load(open(r["zoo"] + "/index.json"))
    b = r["tasks"]["blur_1_fn"]
    ne = np.maximum(np.array(b["no_edit"]), 1e-12)
    return {"psnr": float(np.median([x["psnr"] for x in idx["rows"][:20]])),
            "ratio": float(np.median(np.array(b["input_attenuation"]) / ne)),
            "ft10_ratio": float(np.median(np.array(b["finetune"]["10"]) / ne)),
            "H": float(np.median(r["harmonic_index"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = {}
    for ds in DATASETS:
        for s in SEEDS:
            for B in BOUNDS:
                tag = f"{ds}_s32_swing{B}_seed{s}"
                r = load(tag)
                if r:
                    T[tag] = r
                    print(f"{tag:28s} H {r['H']:5.2f} PSNR {r['psnr']:5.1f} ratio {r['ratio']:.3f} (Adam10 {r['ft10_ratio']:.3f})")
    P = {}
    mono = []
    for ds in DATASETS:
        for s in SEEDS:
            seq = [T.get(f"{ds}_s32_swing{B}_seed{s}") for B in BOUNDS]
            if all(seq):
                r = [x["ratio"] for x in seq]
                p = [x["psnr"] for x in seq]
                mono.append({"series": f"{ds} seed {s}", "ratio_monotone": all(np.diff(r) > 0), "psnr_monotone": all(np.diff(p) > 0)})
    P["E1_monotone"] = {"series": mono, "pass": bool(mono) and all(m["ratio_monotone"] and m["psnr_monotone"] for m in mono)}
    e2 = {}
    for B in BOUNDS:
        m = [T[f"mnist_s32_swing{B}_seed{s}"]["ratio"] for s in SEEDS if f"mnist_s32_swing{B}_seed{s}" in T]
        f = [T[f"fashion_s32_swing{B}_seed{s}"]["ratio"] for s in SEEDS if f"fashion_s32_swing{B}_seed{s}" in T]
        if m and f:
            e2[str(B)] = {"mnist": float(np.median(m)), "fashion": float(np.median(f)),
                          "factor": float(max(np.median(m), np.median(f)) / max(min(np.median(m), np.median(f)), 1e-9))}
    P["E2_dataset_independent"] = {"per_bound": e2, "pass": bool(e2) and all(v["factor"] < 2 for v in e2.values())}
    lam = {}
    for lt in ("L3w32_lam1e-2",):
        f = f"results/programs_{lt}.json"
        if os.path.exists(f):
            r = json.load(open(f))
            idx = json.load(open(r["zoo"] + "/index.json"))
            b = r["tasks"]["blur_1_fn"]
            lam = {"psnr": float(np.median([x["psnr"] for x in idx["rows"][:20]])),
                   "ratio": float(np.median(np.array(b["input_attenuation"]) / np.maximum(np.array(b["no_edit"]), 1e-12)))}
    e3 = {}
    for s in SEEDS:
        t = T.get(f"mnist_s32_swing2_seed{s}")
        if t and lam:
            e3[f"seed{s}"] = {"swing2": t, "penalty1e-2": lam, "pass": t["psnr"] > lam["psnr"] and t["ratio"] < lam["ratio"]}
    P["E3_projection_dominates_penalty"] = {"detail": e3, "pass": bool(e3) and all(v["pass"] for v in e3.values())}
    e4 = {ds: [T[f"{ds}_s32_swing4_seed{s}"] for s in SEEDS if f"{ds}_s32_swing4_seed{s}" in T] for ds in DATASETS}
    P["E4_operating_point"] = {ds: {"psnr_median": float(np.median([x["psnr"] for x in v])) if v else None,
                                    "ratio_median": float(np.median([x["ratio"] for x in v])) if v else None} for ds, v in e4.items()}
    P["E4_operating_point"]["pass"] = all(v and np.median([x["psnr"] for x in v]) >= 36 and np.median([x["ratio"] for x in v]) <= 0.4 for v in e4.values())
    e5 = {ds: [T[f"{ds}_s32_swing8_seed{s}"] for s in SEEDS if f"{ds}_s32_swing8_seed{s}" in T] for ds in DATASETS}
    P["E5_bound_8"] = {ds: {"psnr_median": float(np.median([x["psnr"] for x in v])) if v else None,
                            "ratio_median": float(np.median([x["ratio"] for x in v])) if v else None} for ds, v in e5.items()}
    P["E5_bound_8"]["pass"] = all(v and np.median([x["ratio"] for x in v]) > 0.6 and np.median([x["psnr"] for x in v]) > 48 for v in e5.values())
    json.dump({"table": T, "predictions": P}, open(a.out, "w"), indent=1)
    print(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
