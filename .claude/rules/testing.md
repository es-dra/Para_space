---
paths:
  - "**/*.py"
  - "**/*.pyi"
---

# Testing Policy

> Extends `~/.claude/rules/common/testing.md` and `~/.claude/rules/python/testing.md`.
> Adapted for research code: TDD is mandatory for library code, impractical for experiments.

---

## Framework

**pytest** with `pytest-cov` for coverage. Test directory at project root mirrors library structure:

```
tests/
  test_alignment.py
  test_transforms.py
  test_metrics.py
  test_datasets.py
  ...
```

---

## Layered Coverage Targets

| Layer | Coverage Target | What to Test |
|-------|----------------|--------------|
| Utilities (alignment, metrics, transforms) | **>80%** | Correctness of math, edge cases, invariance properties |
| Models (INR architectures) | Forward pass only | Input→output shapes, device placement, get/set_params round-trip |
| Experiment scripts | Smoke test | `--steps 100 --device cpu` runs without crashing |

---

## Test Categories

```python
@pytest.mark.unit       # Pure function, no I/O, no GPU
@pytest.mark.integration  # Uses model forward pass, may need CPU torch
@pytest.mark.slow        # Requires training loop, GPU recommended
```

---

## What NOT to Test

- Experiment scripts (they are exploratory by nature)
- Visualization functions (test by visual inspection)
- Exact PSNR values (test that they are finite and reasonable, not exact)
- Convergence behavior (stochastic; test that loss decreases, not by how much)
