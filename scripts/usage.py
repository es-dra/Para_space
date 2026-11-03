"""Sum token usage over sandbox runs and estimate cost.

  python scripts/usage.py --runs runs/sb0 runs/sb1 --prices studies/01-sandbox/prices.json
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--prices", default=None)
    a = ap.parse_args()
    prices = json.load(open(a.prices)) if a.prices else {}
    tot = defaultdict(lambda: defaultdict(float))
    n_cases = defaultdict(int)
    for root in a.runs:
        for st in glob.glob(os.path.join(root, "*", "state.json")):
            model = json.load(open(st)).get("model", "?")
            n_cases[model] += 1
            for rj in glob.glob(os.path.join(os.path.dirname(st), "turns", "t*", "run.json")):
                u = json.load(open(rj)).get("usage") or {}
                tot[model]["input"] += u.get("input_tokens", u.get("prompt_tokens", 0)) or 0
                tot[model]["output"] += u.get("output_tokens", u.get("completion_tokens", 0)) or 0
                tot[model]["cache_read"] += u.get("cache_read_input_tokens", 0) or 0
                tot[model]["cache_write"] += u.get("cache_creation_input_tokens", 0) or 0
                tot[model]["seconds"] += json.load(open(rj)).get("seconds", 0) or 0
    for model, t in tot.items():
        line = f"{model}: {n_cases[model]} cases; tokens in {t['input']:.0f} out {t['output']:.0f} cache_read {t['cache_read']:.0f} cache_write {t['cache_write']:.0f}; {t['seconds'] / 3600:.1f} h"
        p = prices.get(model)
        if p and all(p.get(k) is not None for k in ("input", "output", "cache_read", "cache_write")):
            usd = sum(t[k] * p[k] for k in ("input", "output", "cache_read", "cache_write")) / 1e6
            line += f"; about {usd:.2f} USD"
        print(line)


if __name__ == "__main__":
    main()
