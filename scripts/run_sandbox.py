"""Run sandbox-protocol sessions over networks x tasks.

  python scripts/run_sandbox.py --zoo zoo/mnist_s32 --n 3 --tasks dilate blur_1 \
      --protocol sandbox-noforward --backend anthropic --model claude-sonnet-5-5 --effort high \
      --max-runs 30 --out runs/sb0
  python scripts/run_sandbox.py ... --protocol sandbox-forward --passes 10
Reading tasks:  --read-digit  (task: name the digit; final answer must contain 'Digit: <d>')
Existing case folders with a final_answer.md are skipped.
"""
import argparse
import glob
import hashlib
import json
import os

from wsinr import siren
from wsinr.harness.session import make_backend, run_session
from wsinr.tasks import TASKS

READ_TASK = ("The weights below store a handwritten digit (one of 0-9, white strokes on a black background, centred in the "
             "frame). Which digit is it? Work from the weights. End your final answer with the lines 'Digit: <one digit 0-9>' "
             "and 'Confidence: <0-100>%'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=[])
    ap.add_argument("--read-digit", action="store_true")
    ap.add_argument("--protocol", choices=["sandbox-noforward", "sandbox-forward"], default="sandbox-noforward")
    ap.add_argument("--passes", type=int, default=0)
    ap.add_argument("--max-runs", type=int, default=30)
    ap.add_argument("--backend", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=32000)
    ap.add_argument("--no-weights-in-prompt", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.zoo, "*.npz")))[a.offset: a.offset + a.n]
    os.makedirs(a.out, exist_ok=True)
    keyf = os.path.join(a.out, "answer_key.json")
    key = json.load(open(keyf)) if os.path.exists(keyf) else {"zoo": a.zoo, "cases": {}}
    jobs = []
    for f in files:
        nid = hashlib.sha256(open(f, "rb").read()).hexdigest()[:8]
        if a.read_digit:
            jobs.append((f, nid, "read_digit", READ_TASK, "read"))
        for t in a.tasks:
            jobs.append((f, nid, t, TASKS[t].description, "edit"))
    for f, nid, tname, text, kind in jobs:
        case = f"{tname}_{nid}"
        cdir = os.path.join(a.out, case)
        if os.path.exists(os.path.join(cdir, "final_answer.md")):
            continue
        p, meta = siren.load_npz(f)
        key["cases"][case] = {"file": os.path.basename(f), "label": int(meta["label"]), "kind": kind,
                              **({"task": tname} if kind == "edit" else {})}
        json.dump(key, open(keyf, "w"), indent=1)
        print(f"== {case} ({a.protocol}, passes {a.passes}, runs {a.max_runs}, {a.model})", flush=True)
        backend = make_backend(a.backend, a.model, a.effort, a.max_tokens)
        st = run_session(cdir, siren.to_effective(p), text, backend, a.protocol, a.passes, a.max_runs, kind=kind,
                         weights_in_prompt=not a.no_weights_in_prompt, model_name=a.model)
        print(f"   end={st['end']} turns={st['turns']} runs={st['runs']} passes={st['passes_used']} edited={st['edited_tensors']}")


if __name__ == "__main__":
    main()
