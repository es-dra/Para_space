# Architecture Invariants

> These structural principles persist regardless of implementation changes.
> When refactoring or adding features, preserve these invariants.

---

## 1. Library / Experiment Separation

The project has exactly two kinds of code, with different quality bars:

```
Library (stable, tested)    ←→    Experiments (exploratory, disposable)
```

- **Library**: Pure functions, typed signatures, no side effects. Changes must not break existing experiment scripts.
- **Experiments**: Self-contained scripts that import the library. One experiment = one script with a clear research question.

**Invariant**: Experiment scripts never import from other experiment scripts. The dependency arrow is strictly one-way: `experiments → library`.

---

## 2. Model Interface Contract

Every INR model in this project must provide three capabilities:

```python
params = model.get_params()          # → Dict[str, Tensor]  — serializable
model.set_params(params)             # ← restore from dict
output = model.forward(coords, ...)  # → Tensor               — evaluate
```

This interface is what enables all parameter-space analysis (PCA, alignment, trajectory tracking). It is non-negotiable.

---

## 3. Configuration Hierarchy

```
Global defaults (config.py)
  → CLI overrides (argparse)
    → Result artifact (JSON summary saved alongside outputs)
```

Every experiment run saves its effective configuration as a JSON summary. This is what makes results auditable and reproducible.

---

## 4. Results as Scientific Artifacts

A result is only valid when reproducible from four ingredients tracked together:

1. **Code version** (git commit hash — init git repository when ready)
2. **Full config** (saved as JSON alongside results)
3. **Input data identifier** (image name, dataset, transform parameters)
4. **Random seed** (`set_seed()` called at experiment start)

Without all four, a result is not a scientific artifact.

---

## 5. Parameter Alignment Requirement

MLP-based INRs have neuron permutation symmetry — the same function can be represented by many weight configurations. Before any PCA or trajectory analysis on MLP parameters, run permutation alignment (Hungarian matching). Without this, PCA sees noise from permutation自由度 rather than genuine geometric structure.

**Invariant**: Every analysis pipeline that computes PCA on MLP parameters must first pass through an alignment step.
