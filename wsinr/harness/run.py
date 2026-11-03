"""Tool-free, one-reply LLM runs over a folder of prompts (after the Sparks harness: one message, one answer,
no tools, an audit record per run).

Backends
  claude-cli : `claude -p --bare --tools "" --model <m> --effort <e>` (needs ANTHROPIC_API_KEY; Claude Code >= 2.1)
  anthropic  : the Anthropic Messages API (pip install anthropic)
  openai     : any OpenAI-compatible Chat Completions endpoint (pip install openai; OPENAI_BASE_URL optional)

Each case folder gets answer.md and run.json (model, settings, token usage, wall time, tool audit).
Existing answers are never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

NEUTRAL_SYSTEM = "Answer the user's request."


def run_claude_cli(prompt: str, model: str, effort: str, max_tokens: int):
    cmd = ["claude", "-p", "--bare", "--model", model, "--effort", effort, "--tools", "", "--output-format", "json",
           "--system-prompt", NEUTRAL_SYSTEM]
    t0 = time.time()
    r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=3600)
    dt = time.time() - t0
    try:
        j = json.loads(r.stdout)
        text = j.get("result", "")
        usage = j.get("usage", {})
        cost = j.get("total_cost_usd")
        tool_free = j.get("num_turns", 1) <= 1
    except json.JSONDecodeError:
        text, usage, cost, tool_free = r.stdout, {}, None, None
    return text, {"usage": usage, "cost_usd": cost, "seconds": dt, "tool_free": tool_free, "stderr": r.stderr[-2000:]}


def run_anthropic(prompt: str, model: str, effort: str, max_tokens: int):
    import anthropic  # type: ignore
    client = anthropic.Anthropic()
    budget = {"low": 2048, "medium": 8192, "high": 32000, "max": 64000}.get(effort, 32000)
    t0 = time.time()
    with client.messages.stream(model=model, max_tokens=max_tokens, system=NEUTRAL_SYSTEM,
                                thinking={"type": "enabled", "budget_tokens": min(budget, max_tokens - 1024)},
                                messages=[{"role": "user", "content": prompt}]) as s:
        msg = s.get_final_message()
    dt = time.time() - t0
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text, {"usage": msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else dict(msg.usage),
                  "seconds": dt, "tool_free": True, "stop_reason": msg.stop_reason}


def run_openai(prompt: str, model: str, effort: str, max_tokens: int):
    from openai import OpenAI  # type: ignore
    client = OpenAI()
    t0 = time.time()
    kw = {}
    if effort:
        kw["reasoning_effort"] = effort
    r = client.chat.completions.create(model=model, max_completion_tokens=max_tokens,
                                       messages=[{"role": "system", "content": NEUTRAL_SYSTEM},
                                                 {"role": "user", "content": prompt}], **kw)
    dt = time.time() - t0
    return r.choices[0].message.content or "", {"usage": r.usage.model_dump() if r.usage else {}, "seconds": dt,
                                                  "tool_free": True, "finish_reason": r.choices[0].finish_reason}


BACKENDS = {"claude-cli": run_claude_cli, "anthropic": run_anthropic, "openai": run_openai}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--backend", choices=BACKENDS, default="anthropic")
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-tokens", type=int, default=64000)
    ap.add_argument("--only", nargs="*", default=None, help="case names to run")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cases = sorted(d for d in os.listdir(a.prompts) if os.path.isdir(os.path.join(a.prompts, d)))
    if a.only:
        cases = [c for c in cases if c in a.only]
    done = 0
    for c in cases:
        d = os.path.join(a.prompts, c)
        if os.path.exists(os.path.join(d, "answer.md")):
            continue
        prompt = open(os.path.join(d, "prompt.txt")).read()
        text, rec = BACKENDS[a.backend](prompt, a.model, a.effort, a.max_tokens)
        open(os.path.join(d, "answer.md"), "w").write(text)
        rec.update({"model": a.model, "effort": a.effort, "backend": a.backend, "system": NEUTRAL_SYSTEM,
                    "prompt_chars": len(prompt), "answer_chars": len(text), "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        json.dump(rec, open(os.path.join(d, "run.json"), "w"), indent=1, default=str)
        print(f"{c}: {rec.get('seconds', 0):.0f}s  {rec.get('usage')}")
        done += 1
        if a.limit and done >= a.limit:
            break


if __name__ == "__main__":
    main()
