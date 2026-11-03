"""Print base, target and (if an answer exists) edited renderings of a case side by side, as ASCII.

  python scripts/show_case.py --prompts prompts/pilot --case flip_lr_1a2b3c4d
"""
import argparse
import json
import os

import numpy as np

from wsinr import siren
from wsinr.edits import apply_edits, extract_block
from wsinr.render import ascii_art, render
from wsinr.tasks import TASKS


def side_by_side(*imgs, titles=()):
    arts = [ascii_art(i).splitlines() for i in imgs]
    w = max(len(a[0]) for a in arts)
    head = "   ".join(t.ljust(w) for t in titles)
    rows = [head] if titles else []
    for r in range(len(arts[0])):
        rows.append("   ".join(a[r].ljust(w) for a in arts))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    key = json.load(open(os.path.join(a.prompts, "answer_key.json")))
    info = key["cases"][a.case]
    p, meta = siren.load_npz(os.path.join(key["zoo"], info["file"]))
    base = render(p)[..., 0]
    imgs, titles = [base], ["base"]
    if info["kind"] == "edit":
        task = TASKS[info["task"]]
        imgs.append(task.target(base)); titles.append("target")
        ans = os.path.join(a.prompts, a.case, "answer.md")
        if os.path.exists(ans):
            blk = extract_block(open(ans).read())
            if blk:
                eff2, rep = apply_edits(siren.to_effective(p), blk)
                imgs.append(render(siren.from_effective(eff2))[..., 0]); titles.append("edited")
                print("edits applied:", rep["applied"], "errors:", len(rep["errors"]))
    print(f"label {info['label']}  file {info['file']}")
    print(side_by_side(*imgs, titles=titles))


if __name__ == "__main__":
    main()
