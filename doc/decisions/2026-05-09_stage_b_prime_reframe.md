# 2026-05-09 Stage B-prime Reframe

## Question

在 Stage C 继续扩展前，当前项目是否已经充分理解了参数/响应 fitting dynamics 的轨迹对象？

## Evidence Considered

- [research_contract.md](../research_contract.md)
- [stages/stage_b/closeout.md](../stages/stage_b/closeout.md)
- [stages/stage_c/current_status.md](../stages/stage_c/current_status.md)
- [claims_ledger.md](../claims_ledger.md)
- [evidence/run_manifest_stage_b.md](../evidence/run_manifest_stage_b.md)
- 用户对 PCA、permuted-update control、step cosine 和 Stage C 过早推进的质疑。

## Decision

Stage B 只能封版为平台验证，不能封版为参数轨迹科学分析。项目下一步应转入 Stage B-prime：参数/响应轨迹本体审计。

Stage C 保留为 pilot / failure-audit 记录，但在 Stage B-prime 完成前不继续扩展 descriptor、response object、rerank、等变性、pretrained LIIF/LIIF-EQ、update prediction 或训练加速路线。

## Rejected Alternatives

- 把 Stage B 的 PCA/PC1 或 step cosine 继续包装为参数空间规律。
- 把 Stage C 的 baby/woman 正例作为继续扩大自然图实验的理由。
- 继续开放式尝试 geometry descriptor、response descriptor 或 patch matching 修补。
- 在参数/响应轨迹对象未讲清楚前进入等变性或训练加速。

## Consequences

- `research_contract.md` 增加 H0 / Stage B-prime 层级。
- Stage A/B/C 当前文档应明确：项目主线是 fitting dynamics，局部几何是解释层而不是整个项目。
- 下一步分析应优先回答轨迹本体问题，例如 endpoint-linearity、parallel/orthogonal update decomposition、update effective rank、layer-wise update energy、full/encoder/decoder 和 raw/aligned 对照。

## Conditions To Reopen Stage C

Stage C 只有在 Stage B-prime 固定了 defensible response/update object 后才应重开。重开时仍必须满足：

- bird/head failure 被正面处理；
- baby/woman guardrails 保留；
- content/coordinate/spatial controls 报告；
- 不使用 response-label oracle；
- negative cases 继续报告。
