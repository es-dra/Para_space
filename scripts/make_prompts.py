"""Write one prompt file per (network, task) for the LLM harness, plus the answer key (never shown to the model).

  python scripts/make_prompts.py --zoo zoo/mnist_s32 --n 5 --tasks flip_lr dilate --out prompts/pilot
  python scripts/make_prompts.py --zoo zoo/mnist_s32 --n 5 --read-digit --out prompts/pilot_read
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np

from wsinr import siren
from wsinr.prompt import edit_prompt, read_digit_prompt
from wsinr.tasks import TASKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--read-digit", action="store_true")
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.zoo, "*.npz")))[a.offset: a.offset + a.n]
    os.makedirs(a.out, exist_ok=True)
    key = {}
    for f in files:
        p, meta = siren.load_npz(f)
        eff = siren.to_effective(p)
        # anonymous id: the model never sees the file name or label
        nid = hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
        if a.read_digit:
            d = os.path.join(a.out, f"read_{nid}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "prompt.txt"), "w").write(read_digit_prompt(eff, decimals=a.decimals))
            key[f"read_{nid}"] = {"file": os.path.basename(f), "label": int(meta["label"]), "kind": "read_digit"}
        for t in a.tasks:
            d = os.path.join(a.out, f"{t}_{nid}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "prompt.txt"), "w").write(edit_prompt(eff, TASKS[t].description, decimals=a.decimals))
            key[f"{t}_{nid}"] = {"file": os.path.basename(f), "label": int(meta["label"]), "kind": "edit", "task": t}
    json.dump({"zoo": a.zoo, "decimals": a.decimals, "cases": key}, open(os.path.join(a.out, "answer_key.json"), "w"), indent=1)
    print(f"{len(key)} prompts in {a.out}")


if __name__ == "__main__":
    main()
