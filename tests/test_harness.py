import glob
import json
import os

import numpy as np

from wsinr import siren
from wsinr.harness.kernel import Kernel, check_code
from wsinr.harness.session import ScriptedBackend, extract_action, run_session
from wsinr.tasks import TASKS


def _eff():
    p = siren.init_params(seed=2)
    return siren.to_effective(p)


def test_check_code_rules():
    assert check_code("x = load('A1'); print(x.sum())", True) == []
    assert any("lstsq" in m for m in check_code("np.linalg.lstsq(a, b)", False))
    assert any("Adam" in m for m in check_code("import torch\ntorch.optim.Adam(p)", False))
    assert any("sin(" in m for m in check_code("np.sin(load('A1'))", True))
    assert check_code("np.sin(load('A1'))", False) == []
    assert any("import of sklearn" in m for m in check_code("from sklearn.linear_model import Ridge", False))


def test_kernel_roundtrip(tmp_path):
    eff = _eff()
    w = tmp_path / "w.npz"
    np.savez(w, **{k: np.asarray(v, np.float32) for k, v in eff.items()})
    k = Kernel(str(w), str(tmp_path / "work"), timeout=30, no_forward=True)
    assert k.run("x = 2\nx + 1")["out"].strip() == "3"
    assert "saved" in k.run("save_edit('c3', load('c3') + 1)")["out"]
    assert "REFUSED" in k.run("np.cos(load('A1'))")["out"]
    ed = k.edited_params()
    assert np.allclose(ed["c3"], np.asarray(eff["c3"], np.float32) + 1) and np.allclose(ed["A1"], np.asarray(eff["A1"], np.float32))
    k.close()


def test_extract_action():
    assert extract_action("no code") == (None, None, 0)
    kind, body, n = extract_action("a\n```python\nprint(1)\n```\nb\n```forward\n{\"type\": \"render\"}\n```")
    assert kind == "forward" and json.loads(body)["type"] == "render" and n == 2


def test_scripted_session(tmp_path):
    eff = _eff()
    replies = [
        "```python\n# look\nA1 = load('A1'); A1[1, :] *= -1; save_edit('A1', A1)\n```",
        "```forward\n{\"type\": \"render\", \"model\": \"both\"}\n```",
        "Mirrored left-right. Confidence: 90%.",
    ]
    st = run_session(str(tmp_path / "case"), eff, TASKS["flip_lr"].description, ScriptedBackend(replies),
                     "sandbox-forward", passes=2, max_runs=5, quiet=True)
    assert st["end"] == "final_answer" and st["runs"] == 1 and st["passes_used"] == 1
    assert st["edited_tensors"] == ["A1"]
    assert os.path.exists(tmp_path / "case" / "work" / "edited" / "A1.npy")
    assert len(glob.glob(str(tmp_path / "case" / "turns" / "t*"))) == 3
