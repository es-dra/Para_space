# Engineering Boundary And Refactor Plan

本文档只记录白板重启后的工程边界。它不是研究结论文档，也不是实验日志。

## Current Role

当前代码库是 scratch fitting-dynamics probing platform。它还不是训练加速方法库，也没有可继承的机制结论。

工程目标：

- 保持活跃入口小而清楚；
- 保留 scratch SIREN / LIIF 训练和输出检查能力；
- 删除已经停止的分析分支，而不是让它们继续作为可运行选项存在；
- 新分析必须先有清楚的问题、输入、输出、对照和停止条件。

## Active Code Surfaces

| Area | Path | Status |
|---|---|---|
| Core library | `src/` | Keep stable; changes require tests |
| Training entrypoints | `experiments/Phase1_FittingDynamics/run.py`, `run_siren.py` | Retained scratch training CLIs |
| Validation | `experiments/Phase1_FittingDynamics/check_outputs.py` | Retained output/schema check |
| Tests | `tests/` | Basic schema, alignment, and output validation |

See [../experiments/Phase1_FittingDynamics/README.md](../experiments/Phase1_FittingDynamics/README.md) for entrypoint classification.

## Non-Negotiable Constraints

1. Do not change existing `trajectory.npz` or `dynamics_summary.json` semantics without a migration note and tests.
2. Do not revive PCA, external checkpoints, equivariant comparisons, update prediction, or training acceleration as active routes unless a new node contract authorizes them.
3. Do not add one-off launchers, scratch dispatchers, or nohup wrappers.
4. Do not delete `Data/` without explicit approval.
5. Keep new analysis as named Python CLIs with clear inputs, outputs, and stop conditions.
6. Keep workflow rules out of project docs unless they are project-specific facts.

## Output Schema Contract

SIREN trajectories should retain:

- `full_snapshots`
- `full_snapshots_aligned`
- `snapshot_steps`
- `losses`
- `psnrs`
- `freq_ratios`
- `target_spectrum`
- `grad_norms`

Conditional LIIF/LTE trajectories should retain:

- `full_snapshots`
- `enc_snapshots`
- `dec_snapshots`
- `dec_snapshots_aligned`
- `snapshot_steps`
- `losses`
- `psnrs`
- `freq_ratios`
- `target_spectrum`
- optional `model_type`

Summaries should retain model/data/seed/config fields needed by `src/trajectory_schema.py` and `experiments/Phase1_FittingDynamics/check_outputs.py`.

## Current Refactor Priorities

High value, low risk:

1. Add or maintain tests for any new analysis metric.
2. Convert repeated result-loading code into small library helpers only after behavior is covered.
3. Add structured experiment cards only for analysis outputs that are intended to become canonical.
4. Keep active docs short; delete non-canonical analysis scripts and generated intermediates after their conclusion is summarized.

Deferred:

1. YAML/config-system migration.
2. Large package split under `src/paramspace`, `src/geometry`, `src/response`, etc.
3. Adapter/modulation implementation.
4. External-checkpoint fine-tuning evidence.

## Validation Commands

For documentation-only changes:

```bash
python - <<'PY'
from pathlib import Path
import re, sys
files=[Path('RESEARCH_PLAN.md')] + sorted(Path('doc').rglob('*.md')) + [
    Path('experiments/Phase1_FittingDynamics/README.md'),
    Path('paper/README.md'),
]
pat=re.compile(r'\[[^\]]+\]\(([^)]+)\)')
broken=[]
for f in files:
    if not f.exists():
        continue
    for m in pat.finditer(f.read_text(encoding='utf-8', errors='replace')):
        target=m.group(1).split('#', 1)[0]
        if not target or '://' in target or target.startswith('mailto:'):
            continue
        p=(f.parent / target).resolve()
        if not p.exists():
            broken.append((str(f), target))
if broken:
    print('BROKEN LINKS')
    for item in broken:
        print(item)
    sys.exit(1)
print('no broken local links')
PY
```

For analysis/script behavior changes:

```bash
pytest -q tests/test_check_outputs.py tests/test_trajectory_schema.py tests/test_alignment.py
```
