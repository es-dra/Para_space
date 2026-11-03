"""Wrap a prompt for an internal subagent probe (no harness enforcement; for feasibility only).

  python scripts/probe_wrap.py --prompts prompts/pilot --case flip_lr_1a2b3c4d > /tmp/probe.txt
"""
import argparse
import os

WRAP = """You are taking part in a single-reply test. Rules: do not use any tool (no code execution, no file reading,
no calculator, no search). Think as long as you need, then write your complete answer as your final report, in
exactly the format the message below asks for. Your final report must contain the whole answer (including any
code block it asks for) and nothing about tools. The message:

=====
{prompt}
=====
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    prompt = open(os.path.join(a.prompts, a.case, "prompt.txt")).read()
    print(WRAP.format(prompt=prompt))


if __name__ == "__main__":
    main()
