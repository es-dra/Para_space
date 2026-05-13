# Result Artifact Policy

本文档说明结果目录如何解释和引用。当前策略是只保留 current canonical raw runs 和少量 Stage B 派生审计；retired / smoke / debug / old Stage C diagnostic 结果不留在 active tree。

## Directory Semantics

| Path | Meaning | Paper use |
|---|---|---|
| `results/FittingDynamics_StageB/` | official Stage B scratch runs | may support platform claim |
| `results/FittingDynamics_StageB_diagnostics/` | retained Stage B audit JSONs | may support negative/mixed Stage B audit claims |
| retired Stage C diagnostics and old controlled self-similarity results | removed from active tree after conclusions were summarized in decisions/stage docs | not paper evidence |
| `results/FittingDynamics_smoke/`, `results/FittingDynamics/` | deleted smoke/legacy outputs | not paper evidence |

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
- maintain this policy and manifest files only for canonical artifacts;
- do not move or delete `Data/`, `model/`, `pretrained/`, checkpoints, or current raw official runs without a path mapping and explicit approval;
- delete smoke, debug, old Stage C diagnostic, and legacy outputs once their conclusions are summarized.
