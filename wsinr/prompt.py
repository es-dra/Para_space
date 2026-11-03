"""Prompt construction in the style of the Sparks of Weight-Space Programming experiments.

The editor/reader sees: the network formula in effective form, the coordinate convention,
the task, an answer format, and every weight as text. No image is shown unless a task says so.
"""
from __future__ import annotations

import numpy as np

from . import siren


def fmt_matrix(name: str, a: np.ndarray, decimals: int, row_label: str, col_label: str) -> str:
    a = np.asarray(a)
    if a.ndim == 1:
        body = " ".join(f"{v:.{decimals}f}" for v in a)
        return f"{name} (shape {a.shape[0]}): {col_label}\n{body}\n"
    lines = [f"{name} (shape {a.shape[0]} x {a.shape[1]}): one line per row; {row_label}; {col_label}"]
    for row in a:
        lines.append(" ".join(f"{v:.{decimals}f}" for v in row))
    return "\n".join(lines) + "\n"


def network_description(eff: dict, side: int) -> str:
    L = sum(1 for k in eff if k.startswith("A"))
    dims = [eff["A1"].shape[0]] + [eff[f"A{k}"].shape[1] for k in range(1, L + 1)]
    lines = [
        "THE NETWORK",
        "A grey-scale picture is stored as a small neural network (an implicit neural representation, or SIREN):",
        "the network maps the coordinates of a point to the grey value of the picture at that point.",
        f"Coordinates p = (y, x): y runs from -1 at the top row to +1 at the bottom row, x from -1 at the left column to +1 at the right column.",
        f"The picture is rendered on a {side} x {side} grid: pixel (row i, column j) has y = -1 + 2 i / {side - 1}, x = -1 + 2 j / {side - 1}.",
        "Grey values: 0 = black, 1 = white; when rendering, the network's output is clipped to [0, 1].",
        "",
    ]
    h = "p"
    for k in range(1, L):
        lines.append(f"  h{k} = sin({h} @ A{k} + c{k})".ljust(32) + f"A{k}: {dims[k - 1]} x {dims[k]}, c{k}: {dims[k]}")
        h = f"h{k}"
    lines.append(f"  v  = {h} @ A{L} + c{L}".ljust(32) + f"A{L}: {dims[L - 1]} x {dims[L]}, c{L}: {dims[L]}   (v = the grey value)")
    lines.append("Row i of a matrix belongs to input i (or to unit i of the previous layer); column j to unit j of the next layer.")
    lines.append(f"For A1, row 0 multiplies y and row 1 multiplies x. The network has {sum(v.size for v in eff.values())} weights in all.")
    return "\n".join(lines) + "\n"


def weights_text(eff: dict, decimals: int = 4) -> str:
    L = sum(1 for k in eff if k.startswith("A"))
    parts = [f"THE WEIGHTS (values rounded to {decimals} decimals)\n"]
    for k in range(1, L + 1):
        rl = "row i = input i (y, x)" if k == 1 else f"row i = unit i of h{k - 1}"
        cl = "column j = unit j of h%d" % k if k < L else "column j = output j"
        parts.append(fmt_matrix(f"A{k}", eff[f"A{k}"], decimals, rl, cl))
        parts.append(fmt_matrix(f"c{k}", eff[f"c{k}"], decimals, "", "entry j = unit j" if k < L else "entry j = output j"))
    return "\n".join(parts)


EDIT_FORMAT = """EDIT FORMAT (numbers only; nothing is run). Write a code block whose opening line is three backticks followed by the word edits, one statement per line ('#' starts a comment; a long list may continue over the next lines):

```edits
c2[5] = 0.25                      # one entry
c2[0:4] = [0.1, 0.2, 0.3, 0.4]    # explicit values along one range (a:b is half-open; ':' alone = the whole axis)
A2[7, :] = 0                      # fill a block
A2[:, 2] *= 0.5                   # scale a block (also += and -=)
ZERO A2                           # set a whole tensor to 0
```
Statements apply in order to the network shown below; a later statement overwrites an earlier one. Only the LAST edits block of your reply is read. A statement that cannot be read is skipped (the others still apply). Values: plain decimal numbers (absolute value at most 1e6). Tensors not mentioned stay as they are.
"""


def edit_prompt(eff: dict, task_text: str, side: int = 28, decimals: int = 4, one_shot: bool = True) -> str:
    head = network_description(eff, side)
    task = (
        "YOUR TASK: change the network's weights so that the picture it renders is transformed as follows.\n"
        f"  {task_text}\n"
        "Everything else about the picture should stay the same. Your edited network is rendered on the same grid and "
        "compared pixel by pixel (mean squared error) with the correctly transformed picture.\n"
    )
    if one_shot:
        task += ("You have no tools: no code execution, no calculator, nothing to run. You write numbers. "
                 "This is a single response; you will get no feedback.\n")
    task += ("\nAnswer in this order:\n"
             "Reasoning: <how the transformation acts on the network; which weights must change and why>\n"
             "Then the edits block.\n"
             "Then at most about 100 words on what your edits do.\n\n")
    return head + "\n" + task + EDIT_FORMAT + "\n" + weights_text(eff, decimals)


def read_digit_prompt(eff: dict, side: int = 28, decimals: int = 4) -> str:
    head = network_description(eff, side)
    q = ("QUESTION. The weights below store a handwritten digit (one of 0-9, white strokes on a black background, "
         f"centred in the {side} x {side} frame). Which digit is it? Work from the weights. You have no tools: no code "
         "execution, no calculator, nothing to run. Do not do any forward-pass calculation (do not evaluate the network "
         "at any point, in your head or on paper). Read what the weights themselves show. This is a single response.\n\n"
         "Answer in this order:\n"
         "Observations: <what you see in the weights; quote the weights you rely on>\n"
         "Reasoning: <which digits fit, which are ruled out, and why>\n"
         "Digit: <one digit 0-9>\n"
         f"Example: <the picture as {side} lines of {side} characters, '#' for white and '.' for black>\n"
         "Confidence: <0-100>%\n\n")
    return head + "\n" + q + weights_text(eff, decimals)


def parse_digit_answer(text: str):
    import re
    m = re.search(r"^\s*Digit:\s*\**\s*(\d)", text, flags=re.M)
    digit = int(m.group(1)) if m else None
    c = re.search(r"Confidence:\s*\**\s*(\d{1,3})", text)
    conf = int(c.group(1)) if c else None
    # ASCII drawing, if any
    rows = []
    started = False
    for line in text.splitlines():
        s = line.strip().strip("`")
        if len(s) >= 20 and set(s) <= set("#."):
            rows.append(s)
            started = True
        elif started and rows:
            break
    art = None
    if rows:
        w = max(len(r) for r in rows)
        art = np.array([[1.0 if ch == "#" else 0.0 for ch in r.ljust(w, ".")] for r in rows])
    return digit, conf, art
