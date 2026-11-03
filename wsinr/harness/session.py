"""Multi-turn sandbox sessions: one network, one task, one conversation.

Protocols
  sandbox-noforward : Python kernel over the weights, no forward pass (sin/cos refused), no passes
  sandbox-forward   : the same kernel plus a forward-pass tool with a budget of `passes` requests

Each turn the model replies with exactly one ```python block (run in the kernel), one ```forward
block (JSON request to the forward tool), or a final answer without a code block. The harness
returns the output as the next user message. The session ends at the final answer, when the run
limit is reached (the model is then asked for its final answer), or on a hard turn cap.

Logs per case:  turns/tNNN/{prompt.txt, answer.md, run.json}, tool_calls.jsonl, state.json,
final_answer.md, work/ (the kernel's folder, with work/edited/*.npy as the submitted edit).
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

from .. import siren
from ..prompt import network_description, weights_text
from .forward import run_forward
from .kernel import Kernel

NEUTRAL_SYSTEM = "Answer the user's request."


# ----------------------------------------------------------------------------- prompts
def sandbox_task_prompt(eff: dict, task_text: str, protocol: str, passes: int, max_runs: int,
                        side: int = 28, decimals: int = 4, weights_in_prompt: bool = True, kind: str = "edit",
                        work_dir: str = "/work") -> str:
    head = network_description(eff, side)
    if kind == "edit":
        task = ("YOUR TASK: change the network's weights so that the picture it renders is transformed as follows.\n"
                f"  {task_text}\n"
                "Everything else about the picture should stay the same. When you finish, your edited network is rendered "
                "on the same grid and compared pixel by pixel (mean squared error) with the correctly transformed picture.\n")
    else:
        task = ("YOUR TASK: " + task_text + "\n")
    tools = [
        "WHAT YOU HAVE.",
        f"1. A persistent Python kernel (CPU, no internet; numpy and scipy are available). Variables survive between your code "
        f"blocks. Defined in advance: NAMES, SHAPES, load(name) (float64 array: your current edit if there is one, else the "
        f"original), load_original(name), save_edit(name, array), drop_edit(name), edited(), and WORK (a folder kept between "
        f"runs; write notes or files there). To run code, reply with exactly one block that starts with ```python and nothing "
        f"else after it; if its last line is an expression its value is shown. Output is cut at 10,000 characters; a run may "
        f"take up to 300 seconds. You may run code at most {max_runs} times.",
    ]
    if protocol == "sandbox-forward":
        tools.append(
            f"2. A forward-pass tool with a budget of {passes} requests in total. Reply with exactly one block that starts with "
            "```forward and contains JSON. Either {\"type\": \"render\", \"model\": \"original\" | \"edited\" | \"both\", \"side\": 28} "
            "(renders the network(s) on the grid, saves the array to WORK/fwd_<k>_<model>.npy for your code to read, and shows an "
            "ASCII preview) or {\"type\": \"probe\", \"model\": ..., \"points\": [[y, x], ...], \"activations\": true} (at most 256 "
            "points; returns the output values and saves the hidden activations). A request that fails does not count.")
        rules_fwd = ("(b) evaluating the network yourself: the forward-pass tool is the only way to run it; do not implement its "
                     "computation on any input in the kernel; ")
    else:
        rules_fwd = ("(b) evaluating the network on any input, in the kernel or by hand: calls to sin, cos or sinc are refused; "
                     "work from the weights themselves (their structure, norms, frequencies, geometry); ")
    rules = ("RULES. Not allowed: (a) gradient descent or any other fitting of weights to a loss: no autograd, optimizers, "
             "least squares, regression or curve fitting (such calls are refused); " + rules_fwd +
             "(c) internet access. Allowed: any analysis of the weights" +
             (" and of forward-pass results" if protocol == "sandbox-forward" else "") +
             ", and any direct edit you compute and can justify. Edits that only change a global scale or offset of the output "
             "are not what we are looking for unless the task is itself a pointwise value change.\n")
    how = ("HOW TO WORK. State the hypothesis of each code block in its first comment line. Keep outputs compact (print "
           "summaries, not whole arrays). Save edits with save_edit as you go; the tensors in edited() when you finish are "
           "submitted. When you are done, reply WITHOUT a code block: say what you changed, why, and how confident you are "
           f"(0-100%). The session also ends after {max_runs} code runs.\n")
    body = head + "\n" + task + "\n" + "\n".join(tools) + "\n\n" + rules + "\n" + how
    if weights_in_prompt:
        body += "\n" + weights_text(eff, decimals)
    else:
        body += "\nTHE WEIGHTS are not printed here; read them in the kernel with load(name).\n"
    return body


# ----------------------------------------------------------------------------- backends
class AnthropicBackend:
    """Anthropic Messages API, multi-turn. Adaptive thinking with an explicit effort level (the only
    thinking control on Claude 5.5-generation models), streaming, and automatic prompt caching of the
    conversation prefix (the weights and the task text are re-sent every turn)."""

    def __init__(self, model: str, effort: str = "high", max_tokens: int = 32000):
        import anthropic  # type: ignore
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.messages = []

    def send(self, text: str):
        self.messages.append({"role": "user", "content": text})
        t0 = time.time()
        with self.client.messages.stream(model=self.model, max_tokens=self.max_tokens, system=NEUTRAL_SYSTEM,
                                         thinking={"type": "adaptive", "display": "summarized"},
                                         output_config={"effort": self.effort},
                                         cache_control={"type": "ephemeral"},
                                         messages=self.messages) as s:
            msg = s.get_final_message()
        reply = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        thinking = "\n\n".join(getattr(b, "thinking", "") or "" for b in msg.content if getattr(b, "type", "") == "thinking")
        # keep the assistant turn with its thinking blocks unchanged (preserved thinking)
        self.messages.append({"role": "assistant", "content": [b.model_dump() for b in msg.content]})
        usage = msg.usage.model_dump() if hasattr(msg.usage, "model_dump") else dict(msg.usage)
        rec = {"usage": usage, "seconds": time.time() - t0, "stop_reason": msg.stop_reason,
               "thinking_summary": thinking[:20000]}
        if msg.stop_reason == "refusal" and getattr(msg, "stop_details", None):
            rec["stop_details"] = msg.stop_details.model_dump()
        return reply, rec


class OpenAIBackend:
    def __init__(self, model: str, effort: str = "high", max_tokens: int = 32000):
        from openai import OpenAI  # type: ignore
        self.client = OpenAI()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.messages = [{"role": "system", "content": NEUTRAL_SYSTEM}]

    def send(self, text: str):
        self.messages.append({"role": "user", "content": text})
        t0 = time.time()
        kw = {"reasoning_effort": self.effort} if self.effort else {}
        r = self.client.chat.completions.create(model=self.model, messages=self.messages,
                                                max_completion_tokens=self.max_tokens, **kw)
        reply = r.choices[0].message.content or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply, {"usage": r.usage.model_dump() if r.usage else {}, "seconds": time.time() - t0,
                       "finish_reason": r.choices[0].finish_reason}


class ScriptedBackend:
    """Replays prepared replies (a list of strings); for harness tests only."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.i = 0

    def send(self, text: str):
        r = self.replies[self.i] if self.i < len(self.replies) else "Done. I changed nothing further. Confidence: 10%."
        self.i += 1
        return r, {"usage": {}, "seconds": 0.0}


def make_backend(name: str, model: str, effort: str, max_tokens: int):
    if name == "anthropic":
        return AnthropicBackend(model, effort, max_tokens)
    if name == "openai":
        return OpenAIBackend(model, effort, max_tokens)
    raise ValueError(name)


# ----------------------------------------------------------------------------- the loop
_BLOCK = re.compile(r"```(python|forward)[^\n]*\n(.*?)```", re.S)


def extract_action(reply: str):
    blocks = _BLOCK.findall(reply)
    if not blocks:
        return None, None, 0
    kind, body = blocks[-1]
    return kind, body, len(blocks)


def run_session(case_dir: str, eff: dict, task_text: str, backend, protocol: str, passes: int = 0,
                max_runs: int = 40, max_turns: int = 120, kind: str = "edit", weights_in_prompt: bool = True,
                side: int = 28, decimals: int = 4, model_name: str = "", quiet: bool = False) -> dict:
    os.makedirs(case_dir, exist_ok=True)
    work = os.path.join(case_dir, "work")
    wfile = os.path.join(case_dir, "weights.npz")
    np.savez(wfile, **{k: np.asarray(v, np.float32) for k, v in eff.items()})
    kernel = Kernel(wfile, work, no_forward=(protocol == "sandbox-noforward"))
    prompt = sandbox_task_prompt(eff, task_text, protocol, passes, max_runs, side, decimals, weights_in_prompt, kind,
                                 work_dir=work).replace("/work", work)
    open(os.path.join(case_dir, "message.txt"), "w").write(prompt)
    state = {"protocol": protocol, "model": model_name, "passes_budget": passes, "passes_used": 0, "runs": 0,
             "max_runs": max_runs, "turns": 0, "finished": False, "end": None, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    calls = open(os.path.join(case_dir, "tool_calls.jsonl"), "a")
    message = prompt
    reminded_runs = False
    final = ""
    for t in range(1, max_turns + 1):
        tdir = os.path.join(case_dir, "turns", f"t{t:03d}")
        os.makedirs(tdir, exist_ok=True)
        open(os.path.join(tdir, "prompt.txt"), "w").write(message)
        reply, rec = backend.send(message)
        open(os.path.join(tdir, "answer.md"), "w").write(reply)
        rec.update({"turn": t, "model": model_name, "protocol": protocol, "tool_free_api": True})
        json.dump(rec, open(os.path.join(tdir, "run.json"), "w"), indent=1, default=str)
        state["turns"] = t
        kind_b, body, nblocks = extract_action(reply)
        if not quiet:
            print(f"  t{t:03d} {kind_b or 'final'} runs={state['runs']} passes={state['passes_used']}", flush=True)
        if kind_b is None:
            final = reply
            state["finished"], state["end"] = True, "final_answer"
            break
        if kind_b == "python":
            if state["runs"] >= max_runs:
                message = (f"No code was run: the limit of {max_runs} code runs is reached. Reply without a code block with "
                           "your final answer (the tensors in edited() are submitted as they are).")
                if reminded_runs:
                    final = reply
                    state["finished"], state["end"] = True, "run_limit"
                    break
                reminded_runs = True
                continue
            res = kernel.run(body)
            state["runs"] += 1 if "refused" not in res else 0
            calls.write(json.dumps({"turn": t, "kind": "python", "code": body, "out": res["out"][:6000],
                                    "ok": res["ok"], "seconds": res["seconds"], "refused": res.get("refused")}) + "\n")
            calls.flush()
            left = max_runs - state["runs"]
            message = f"[python run {state['runs']}/{max_runs}; {left} left]\n{res['out']}"
            if nblocks > 1:
                message += "\n(Only the last code block of a reply is run.)"
        else:  # forward
            if protocol != "sandbox-forward":
                message = "There is no forward-pass tool in this protocol. Continue with a python block or your final answer."
                continue
            if state["passes_used"] >= passes:
                message = f"The forward-pass budget of {passes} is used up. Continue with a python block or give your final answer."
                continue
            try:
                req = json.loads(body)
            except json.JSONDecodeError as e:
                message = f"ERROR: the forward block is not valid JSON ({e}); request not counted."
                continue
            k = state["passes_used"] + 1
            out = run_forward(req, kernel, eff, k, side)
            counted = not out.startswith("ERROR")
            if counted:
                state["passes_used"] = k
            calls.write(json.dumps({"turn": t, "kind": "forward", "request": req, "out": out[:6000], "counted": counted}) + "\n")
            calls.flush()
            message = f"[forward pass {state['passes_used']}/{passes}; {passes - state['passes_used']} left]\n{out}"
        json.dump(state, open(os.path.join(case_dir, "state.json"), "w"), indent=1)
    else:
        state["end"] = "turn_cap"
    kernel.close()
    calls.close()
    open(os.path.join(case_dir, "final_answer.md"), "w").write(final)
    edited = kernel.edited_params()
    state["edited_tensors"] = sorted(k for k in eff if edited is not None
                                     and not np.array_equal(np.asarray(edited[k], np.float32), np.asarray(eff[k], np.float32)))
    state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    json.dump(state, open(os.path.join(case_dir, "state.json"), "w"), indent=1)
    return state
