# Stage C Analysis Manifest

本文档保留 Stage C pilot / diagnostic / gate 的结论摘要。2026-05-11
cleanup 后，旧 Stage C 分析脚本、派生 JSON、smoke/debug 输出、patch crop
图和旧 archive 长日志不再保留在 active tree；这些条目用于解释路线为何停止，
不是当前可运行入口或论文最终图表。

## Natural-Image Failure Audit

| Retired diagnostic | Runs | Primary response | Purpose | Interpretation |
|---|---|---|---|---|
| patch failure diagnostics | bird/head seed42 | `trajectory_delta`, feature/Jacobian auxiliary | patch-level failure localization | bird/head geometry top-1 often response-far |
| positive guardrail comparison | baby123/woman42 | same | positive reference comparison | positive references have lower top1 response percentile |
| candidate attribute gate | bird/head + baby123/woman42 | `trajectory_delta` | response-blind candidate gate | failed; context/rerank did not use oracle headroom |
| response-object audit | bird/head + baby123/woman42 | output response variants | test amplitude/time aggregation explanations | failed to explain bird/head |
| LR-cell patch-unit gate | bird/head + baby123/woman42 | `trajectory_delta` | LIIF-aware LR-cell unit gate | failed; content/coordinate controls often stronger |
| C-minimal LR-cell feature diagnostic | baby123/woman42/bird42/head42 | LR-cell encoder feature trajectory `[T,32]` | test geometry increment over coordinate+content with blocked spatial CV | mixed/inconclusive; baby/woman positive, bird/head negative |

## Deleted Oracle / Rerank Diagnostics

Oracle and rerank JSONs were deleted because they were response-label or
response-blind debugging aids, not reusable evidence. The only retained fact is
the conclusion: oracle headroom did not translate into a usable response-blind
rerank, so this branch remains stopped.

## Controlled Self-Similarity

| Retired diagnostic | Purpose | Result |
|---|---|---|
| former controlled input metadata | controlled input metadata | periodic exact repeat + nonperiodic texture control |
| synthetic response smoke | no-training probe/wiring smoke | passed |
| fitted controlled gate | fitted reduced LIIF controlled gate over 3 periodic + 3 nonrepeat seeds | failed strict gate due to seed stability + content confound |
| HR geometry source check | HR geometry source check | did not rescue seed123 strict failure |

## Current Negative Finding

The current Stage C operationalization:

```text
local patch appearance / geometry nearest neighbor
-> raw output trajectory response similarity
```

does not meet the main evidence standard. This is not proof that H1 is false; it is proof that the current measurement/matching route is insufficient.

The 2026-05-10 C-minimal diagnostic moved from HR patch response matching to
LIIF LR-cell encoder feature trajectories. It still did not pass the frozen
reopen gate: strict geometry `Delta R2_cv` was positive for baby/woman but
negative for bird/head, with four-run median near zero.
