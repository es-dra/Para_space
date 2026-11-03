"""Weight-side identifiability repair for token-style arbitrary-scale SR decoders (SE-INR checkpoints).

Replaces the two cell-input columns of the layer that reads the cell by their mean:
  LTE : phase.weight  (hidden/2 x 2)
  LIIF: imnet.layers.0.weight, last two input columns (coord is attached before cell)
At c_x = c_y the edited network is numerically identical to the original.

  python assr_symmetrize.py epoch-last.pth edited.pth [--dry-run]
"""
import argparse
import sys

import torch


def symmetrize(sd: dict):
    report = []
    if "phase.weight" in sd:                                   # LTE
        W = sd["phase.weight"]
        assert W.ndim == 2 and W.shape[1] == 2, W.shape
        m = 0.5 * (W[:, 0] + W[:, 1])
        new = torch.stack([m, m], dim=1)
        report.append(("phase.weight", int((new != W).sum()), W.numel()))
        sd["phase.weight"] = new
    elif "imnet.layers.0.weight" in sd:                        # LIIF: cell = last two inputs
        W = sd["imnet.layers.0.weight"]
        m = 0.5 * (W[:, -2] + W[:, -1])
        new = W.clone()
        new[:, -2] = m
        new[:, -1] = m
        report.append(("imnet.layers.0.weight[:, -2:]", int((new != W).sum()), W.numel()))
        sd["imnet.layers.0.weight"] = new
    else:
        raise SystemExit("no cell-reading layer found (expected phase.weight or imnet.layers.0.weight)")
    return sd, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    sd = ck["model"]["sd"] if "model" in ck else ck
    sd, report = symmetrize(sd)
    total = sum(v.numel() for v in sd.values())
    for name, changed, n in report:
        print(f"{name}: {changed} entries changed of {n} in the tensor; {changed / total * 100:.4f}% of all {total} weights")
    if not a.dry_run:
        if "model" in ck:
            ck["model"]["sd"] = sd
        else:
            ck = sd
        torch.save(ck, a.dst)
        print("saved", a.dst)


if __name__ == "__main__":
    main()
