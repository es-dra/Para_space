"""A persistent Python kernel over one network's weights, for the sandbox protocols.

The editor's code runs in a child process (CPU only, `python -I`, resource limits, network
unshared when `unshare` is available) with a prelude that exposes the weights:

    NAMES, SHAPES                 tensor names (effective form: A1, c1, ..., AL, cL) and shapes
    load(name)                    current value: the edit if there is one, else the original
    load_original(name)
    save_edit(name, array)        replace a tensor (shape and finiteness checked)
    drop_edit(name), edited()
    WORK = "/work"                a folder kept between runs (notes, intermediate files)

Rules are enforced before a block runs (AST + text checks, see `check_code`) and recorded
after it runs. Forbidden always: gradient descent or any fitting of weights to a loss
(autograd, optimizers, least squares, regression, curve fitting, scikit-learn fitting), the
`wsinr` package itself, network/subprocess access. Forbidden under `no_forward`: evaluating the
network, approximated by refusing any call to sin/cos/sinc (the prelude needs none of them for
weight analysis).
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time

FORBIDDEN_ATTRS = {
    "backward", "requires_grad_", "lstsq", "pinv", "solve", "polyfit", "curve_fit", "minimize",
    "least_squares", "fit", "partial_fit", "fit_transform", "SGD", "Adam", "AdamW", "LBFGS", "RMSprop",
    "nnls", "lsq_linear", "leastsq", "fmin", "lsmr", "lsqr",
}
FORBIDDEN_MODULES = ("torch.optim", "torch.autograd", "sklearn", "scipy.optimize", "wsinr", "socket",
                     "subprocess", "urllib", "requests", "http", "multiprocessing")
FORWARD_NAMES = {"sin", "cos", "sinc", "tan"}


def check_code(code: str, no_forward: bool) -> list[str]:
    """Return a list of rule violations (empty if the block may run)."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if any(n == m or n.startswith(m + ".") for m in FORBIDDEN_MODULES):
                    problems.append(f"import of {n} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            problems.append(f"'.{node.attr}' is not allowed (no training, fitting or least squares)")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_ATTRS:
            problems.append(f"'{node.id}' is not allowed")
        if no_forward:
            fn = None
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    fn = node.func.attr
                elif isinstance(node.func, ast.Name):
                    fn = node.func.id
            if fn in FORWARD_NAMES:
                problems.append(f"'{fn}(' is not allowed in this protocol (no forward-pass computation)")
    if re.search(r"__import__|importlib|exec\(|eval\(|open\((['\"])/(?!work)", code):
        problems.append("dynamic import / exec / file access outside /work is not allowed")
    return sorted(set(problems))


PRELUDE = r'''
import json, os, sys, numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=160)
WORK = "/work"
_ORIG = dict(np.load(os.environ["WSINR_WEIGHTS"]))
NAMES = list(_ORIG)
SHAPES = {k: tuple(v.shape) for k, v in _ORIG.items()}
_EDIT = WORK + "/edited"
def load_original(name):
    return _ORIG[name].astype(np.float64).copy()
def load(name):
    p = f"{_EDIT}/{name}.npy"
    return np.load(p).astype(np.float64) if os.path.exists(p) else load_original(name)
def save_edit(name, a):
    assert name in SHAPES, f"unknown tensor {name}; NAMES = {NAMES}"
    a = np.asarray(a, np.float64)
    assert a.shape == SHAPES[name], (name, a.shape, SHAPES[name])
    assert np.isfinite(a).all(), "non-finite values"
    assert np.abs(a).max() <= 1e6, "values must stay below 1e6 in absolute value"
    os.makedirs(_EDIT, exist_ok=True)
    np.save(f"{_EDIT}/{name}.npy", a.astype(np.float32))
    return f"saved edit of {name}"
def drop_edit(name):
    p = f"{_EDIT}/{name}.npy"
    if os.path.exists(p): os.remove(p)
def edited():
    return sorted(f[:-4] for f in os.listdir(_EDIT)) if os.path.isdir(_EDIT) else []
'''

CHILD = r'''
import sys, json, io, contextlib, traceback, resource, os
resource.setrlimit(resource.RLIMIT_AS, (int(os.environ.get("WSINR_MEM", 4 << 30)),) * 2)
ns = {}
exec(compile(os.environ["WSINR_PRELUDE"], "<prelude>", "exec"), ns)
for line in sys.stdin:
    req = json.loads(line)
    out = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            tree = compile(req["code"], "<cell>", "exec", flags=__import__("ast").PyCF_ONLY_AST)
            import ast as _ast
            if tree.body and isinstance(tree.body[-1], _ast.Expr):
                last = _ast.Expression(tree.body.pop().value)
                exec(compile(tree, "<cell>", "exec"), ns)
                val = eval(compile(last, "<cell>", "eval"), ns)
                if val is not None:
                    print(repr(val))
            else:
                exec(compile(tree, "<cell>", "exec"), ns)
        except BaseException:
            ok = False
            traceback.print_exc(limit=6)
    sys.stdout.write(json.dumps({"ok": ok, "out": out.getvalue()}) + "\n")
    sys.stdout.flush()
'''


class Kernel:
    def __init__(self, weights_npz: str, work_dir: str, timeout: int = 300, max_out: int = 10_000,
                 no_forward: bool = False, mem_bytes: int = 4 << 30):
        self.weights_npz, self.work_dir = os.path.abspath(weights_npz), os.path.abspath(work_dir)
        self.timeout, self.max_out, self.no_forward = timeout, max_out, no_forward
        os.makedirs(self.work_dir, exist_ok=True)
        self.runs = 0
        self._start(mem_bytes)

    def _start(self, mem_bytes):
        env = {"PATH": "/usr/bin:/bin", "HOME": self.work_dir, "OMP_NUM_THREADS": "4",
               "CUDA_VISIBLE_DEVICES": "", "WSINR_WEIGHTS": self.weights_npz, "WSINR_MEM": str(mem_bytes),
               "WSINR_PRELUDE": PRELUDE.replace('WORK = "/work"', f'WORK = {self.work_dir!r}')
               .replace('_EDIT = WORK + "/edited"', f'_EDIT = {self.work_dir + "/edited"!r}')}
        cmd = [sys.executable, "-I", "-c", CHILD]
        if shutil.which("unshare"):
            try:
                subprocess.run(["unshare", "-n", "true"], check=True, capture_output=True, timeout=5)
                cmd = ["unshare", "-n"] + cmd
            except Exception:  # noqa: BLE001
                pass
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, env=env, cwd=self.work_dir)
        self.restarted = False

    def run(self, code: str) -> dict:
        """Run one code block. Returns {ok, out, seconds, refused?}."""
        problems = check_code(code, self.no_forward)
        if problems:
            return {"ok": False, "refused": problems, "out": "REFUSED, not run:\n- " + "\n- ".join(problems), "seconds": 0.0}
        t0 = time.time()
        try:
            self.proc.stdin.write(json.dumps({"code": code}) + "\n")
            self.proc.stdin.flush()
            # read one line with a timeout via a watchdog thread
            import threading
            box = {}

            def reader():
                box["line"] = self.proc.stdout.readline()

            th = threading.Thread(target=reader, daemon=True)
            th.start()
            th.join(self.timeout)
            if th.is_alive() or not box.get("line"):
                self.proc.kill()
                self._start(4 << 30)
                self.restarted = True
                return {"ok": False, "out": f"[the kernel exceeded {self.timeout} s or died and was restarted; its variables are lost]",
                        "seconds": time.time() - t0}
            res = json.loads(box["line"])
        except (BrokenPipeError, ValueError) as e:
            self.proc.kill()
            self._start(4 << 30)
            self.restarted = True
            return {"ok": False, "out": f"[kernel error: {e}; restarted, variables lost]", "seconds": time.time() - t0}
        out = res["out"]
        if len(out) > self.max_out:
            out = out[: self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        self.runs += 1
        return {"ok": res["ok"], "out": out or "(no output)", "seconds": time.time() - t0}

    def edited_params(self) -> dict | None:
        """The current edited network (original tensors where no edit exists), effective form, or None."""
        import numpy as np
        orig = dict(np.load(self.weights_npz))
        ed = os.path.join(self.work_dir, "edited")
        if not os.path.isdir(ed) or not os.listdir(ed):
            return None
        out = {}
        for k, v in orig.items():
            p = os.path.join(ed, k + ".npy")
            out[k] = np.load(p).astype(np.float64) if os.path.exists(p) else v.astype(np.float64)
        return out

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self.proc.kill()
