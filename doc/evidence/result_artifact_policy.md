# Result Artifact Policy

本文档说明结果目录如何解释和引用。当前不移动任何结果目录。

## Directory Semantics

| Path | Meaning | Paper use |
|---|---|---|
| `results/FittingDynamics_StageB/` | official Stage B scratch runs | may support platform claim |
| `results/FittingDynamics_StageC_diagnostics/` | derived Stage C natural-image diagnostics | diagnostic / failure audit only |
| `results/FittingDynamics_StageC_controlled/` | controlled-image fitted LIIF runs | diagnostic / sanity gate only |
| `results/FittingDynamics_StageC_controlled_diagnostics/` | controlled gate JSON outputs | diagnostic / sanity gate only |
| `results/FittingDynamics_smoke/` | smoke/debug output | not paper evidence |
| `results/FittingDynamics/` | legacy early outputs | historical only unless revalidated |

## Required Metadata For Future Official Runs

Future official runs should record or be accompanied by:

- script path and command;
- code commit or diff state;
- input data path and preprocessing;
- model config and parameter count;
- seed;
- device/environment assumptions;
- output directory;
- validation command and result;
- analysis command and result artifact;
- claim ledger entry affected.

## Handling Ignored Artifacts

`results/`, `Data/`, and `model/` are ignored by git. Therefore:

- do not rely on git status to know whether artifacts exist;
- maintain this policy and manifest files;
- do not move ignored artifacts without a path mapping and user approval;
- do not delete smoke/legacy outputs unless explicitly approved.
