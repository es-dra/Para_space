# Engineering Boundary And Refactor Plan

本文档只记录当前代码工程边界。它不是研究结论文档，也不是实验日志。科学结论以 [claims_ledger.md](claims_ledger.md) 和 [stages/](stages) 为准。

## Current Role

当前代码库应被视为 fitting-dynamics probing platform，而不是已经完成的训练加速或等变性方法库。

工程目标：

- 保持 Stage B/C 已有结果可读、可检查、可复现；
- 避免大规模迁移破坏旧 artifact；
- 只在测试保护下抽取公共模块；
- 不为尚未通过 gate 的科学路线提前设计复杂框架。
- 保持活跃代码面小而清楚；已停止的分析分支应删除或从入口文档移除，而不是继续作为可运行选项存在。

## Active Code Surfaces

| Area | Path | Status |
|---|---|---|
| Core library | `src/` | Keep stable; changes require tests |
| Training entrypoints | `experiments/Phase1_FittingDynamics/run.py`, `run_siren.py` | Official for scratch reduced LIIF and SIREN evidence |
| Validation / analysis | `check_outputs.py`, `analyze_stage_b_controls.py`, `analyze_stage_b_trajectory_audit.py`, `analyze_stage_b_response_audit.py`, `analyze_stage_b_liif_unit_audit.py` | Official Stage B validation / audit CLIs |
| Retired analysis | old Stage C natural-image repair scripts, old controlled self-similarity scripts, `aggregate_results.py`, `viz_trajectory.py`, `run_finetune.py` | Deleted from active code surface |
| Tests | `tests/` | Required for schema, controls, and retained Stage B analysis logic |

See [../experiments/Phase1_FittingDynamics/README.md](../experiments/Phase1_FittingDynamics/README.md) for entrypoint classification.

## Non-Negotiable Constraints

1. Do not change existing `trajectory.npz` or `dynamics_summary.json` semantics without a migration note and tests.
2. Do not revive PCA, pretrained LIIF/LIIF-EQ, equivariance comparison, update prediction, or training acceleration as active routes unless [claims_ledger.md](claims_ledger.md) gates are updated.
3. Do not add one-off launchers, scratch dispatchers, or nohup wrappers.
4. Do not move or delete `Data/`, `model/`, or `pretrained/` without an artifact mapping and explicit approval. Generated `results/` subtrees that are smoke/debug, legacy, or retired diagnostics should be deleted unless they are current canonical evidence.
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
4. LIIF-EQ buffer/state policy changes.
5. Pretrained fine-tuning evidence.

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
pytest -q tests/test_stage_b_controls.py tests/test_stage_b_trajectory_audit.py \
  tests/test_stage_b_response_audit.py tests/test_stage_b_liif_unit_audit.py \
  tests/test_check_outputs.py tests/test_trajectory_schema.py
```
