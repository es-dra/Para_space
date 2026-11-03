"""The ``edits`` mini-language (after the Sparks of Weight-Space Programming prompts).

    A1[0, :] *= -1            # scale a block (also += and -=)
    c3[0] = 0.25              # one entry
    c1[0:4] = [0.1, 0.2, 0.3, 0.4]
    A2[3, :] = 0              # fill
    ZERO A3                   # whole tensor to 0
    # comments start with '#'; a long list may continue over following lines

Statements apply in order; one that cannot be read is skipped and reported.
Only the LAST ```edits block of a reply is read.
"""
from __future__ import annotations

import ast
import re

import numpy as np

_STMT = re.compile(r"^\s*([A-Za-z]\w*)\s*\[(.*?)\]\s*(\+=|-=|\*=|=)\s*(.+?)\s*$", re.S)
_ZERO = re.compile(r"^\s*ZERO\s+([A-Za-z]\w*)\s*$")


def extract_block(text: str, tag: str = "edits") -> str | None:
    blocks = re.findall(r"```%s[^\n]*\n(.*?)```" % re.escape(tag), text, flags=re.S)
    return blocks[-1] if blocks else None


def _parse_index(s: str, ndim: int):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != ndim:
        raise ValueError(f"index has {len(parts)} axes, tensor has {ndim}")
    idx = []
    for p in parts:
        if p == ":":
            idx.append(slice(None))
        elif ":" in p:
            a, b = p.split(":", 1)
            idx.append(slice(int(a) if a.strip() else None, int(b) if b.strip() else None))
        else:
            idx.append(int(p))
    return tuple(idx)


def _join_statements(block: str):
    """Merge continuation lines of a list value (brackets not yet balanced)."""
    out, buf, depth = [], "", 0
    for raw in block.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        buf = (buf + " " + line.strip()) if buf else line.strip()
        depth += line.count("[") - line.count("]")
        if depth <= 0:
            out.append(buf)
            buf, depth = "", 0
    if buf:
        out.append(buf)
    return out


def apply_edits(params: dict, block: str, max_abs: float = 1e6):
    """Apply an edits block to a dict of arrays (modified copies are returned). Returns (new_params, report)."""
    new = {k: np.array(v, dtype=np.float64, copy=True) for k, v in params.items()}
    applied, errors = [], []
    for stmt in _join_statements(block):
        try:
            m = _ZERO.match(stmt)
            if m:
                name = m.group(1)
                if name not in new:
                    raise KeyError(name)
                new[name][...] = 0.0
                applied.append(stmt)
                continue
            m = _STMT.match(stmt)
            if not m:
                raise ValueError("cannot read statement")
            name, idx_s, op, val_s = m.groups()
            if name not in new:
                raise KeyError(f"unknown tensor {name}")
            arr = new[name]
            idx = _parse_index(idx_s, arr.ndim)
            val = np.asarray(ast.literal_eval(val_s), dtype=np.float64)
            if not np.all(np.isfinite(val)) or np.any(np.abs(val) > max_abs):
                raise ValueError("value out of range")
            if op == "=":
                arr[idx] = val
            elif op == "+=":
                arr[idx] += val
            elif op == "-=":
                arr[idx] -= val
            elif op == "*=":
                arr[idx] *= val
            applied.append(stmt)
        except Exception as e:  # noqa: BLE001
            errors.append({"statement": stmt, "error": f"{type(e).__name__}: {e}"})
    return new, {"applied": len(applied), "errors": errors}
