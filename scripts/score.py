"""Score answers: for each case folder with answer.md, apply the edits (or read the digit) and compare with the target.

  python scripts/score.py --prompts prompts/pilot --out prompts/pilot/scores.json
"""
import argparse
import json
import os

import numpy as np

from wsinr import siren
from wsinr.edits import apply_edits, extract_block
from wsinr.metrics import mse, param_change
from wsinr.prompt import parse_digit_answer
from wsinr.render import render
from wsinr.tasks import TASKS


def _edited_from_work(eff, case_dir):
    d = os.path.join(case_dir, "work", "edited")
    if not os.path.isdir(d) or not os.listdir(d):
        return None
    out = {k: np.array(v, dtype=np.float64, copy=True) for k, v in eff.items()}
    for k in eff:
        f = os.path.join(d, k + ".npy")
        if os.path.exists(f):
            out[k] = np.load(f).astype(np.float64)
    return out


def score_edit(p, task, text, side=28, case_dir=None):
    eff = siren.to_effective(p)
    base = render(p, side)[..., 0]
    target = task.target(base)
    mask = task.mask(side) if task.mask else None
    rec = {"no_edit_mse": mse(base, target, mask)}
    eff2, rep = None, {"applied": None, "errors": []}
    if case_dir and os.path.isdir(os.path.join(case_dir, "work")):
        eff2 = _edited_from_work(eff, case_dir)
        if eff2 is None:
            rec.update({"status": "no_edit_saved", "mse": rec["no_edit_mse"]})
            return rec
        st = os.path.join(case_dir, "state.json")
        if os.path.exists(st):
            s = json.load(open(st))
            rec.update({"runs": s.get("runs"), "passes_used": s.get("passes_used"), "end": s.get("end"), "turns": s.get("turns")})
    else:
        block = extract_block(text)
        if block is None:
            rec.update({"status": "no_edits_block", "mse": rec["no_edit_mse"]})
            return rec
        eff2, rep = apply_edits(eff, block)
    p2 = siren.from_effective(eff2)
    img = render(p2, side)[..., 0]
    rec.update({"status": "ok", "mse": mse(img, target, mask), "applied": rep["applied"], "errors": rep["errors"],
                **param_change(p, p2)})
    if task.oracle is not None and task.oracle(eff) is not None:
        po = siren.from_effective(task.oracle(eff))
        rec["oracle_mse"] = mse(render(po, side)[..., 0], target, mask)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    key = json.load(open(os.path.join(a.prompts, "answer_key.json")))
    scores = {}
    for case, info in key["cases"].items():
        cdir = os.path.join(a.prompts, case)
        ans = os.path.join(cdir, "answer.md")
        fin = os.path.join(cdir, "final_answer.md")
        if os.path.exists(fin):
            text, sandbox = open(fin).read(), True
        elif os.path.exists(ans):
            text, sandbox = open(ans).read(), False
        else:
            continue
        p, meta = siren.load_npz(os.path.join(key["zoo"], info["file"]))
        if info["kind"] == "read_digit":
            digit, conf, art = parse_digit_answer(text)
            rec = {"true": info["label"], "guess": digit, "correct": digit == info["label"], "confidence": conf}
            if art is not None and art.shape == (28, 28):
                rec["drawing_mse"] = mse(art, render(p, 28)[..., 0])
                rec["drawing_mse_blank"] = mse(np.zeros((28, 28)), render(p, 28)[..., 0])
        else:
            rec = score_edit(p, TASKS[info["task"]], text, case_dir=cdir if sandbox else None)
            rec["task"] = info["task"]
        scores[case] = rec
        print(case, json.dumps(rec, default=str)[:300])
    if a.out:
        json.dump(scores, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
