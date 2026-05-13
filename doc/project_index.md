# Project Index

本文档是项目当前的论文级入口。它不替代每日记忆，也不替代原始结果；它只说明到哪里查事实、证据、决策和历史材料。

## 当前状态

- Stage A：完成 pilot 级问题定义、可证伪假设、非目标、负对照和第一轮文献地图；尚不是论文级完整 related work。
- Stage B：完成 scratch fitting-dynamics 平台验证；只支持“平台可靠”，不支持非平凡参数轨迹规律、局部几何机制、等变性或训练加速 claim。
- Stage B-prime：完成首轮参数轨迹本体审计；发现 LIIF decoder 非随机 directed drift，但接近 endpoint/update-set controls 且 layer-scale dominated，未通过强机制 gate。
- Stage B-response：完成首轮函数响应轨迹审计；SIREN response 相对 line-like，LIIF response 低秩但 encoder/decoder/interaction 不可简单分离，未给出重开 Stage C 的 clean positive gate。
- Stage B LIIF-unit：完成 256-query diagnostic screen；decoder input 基本跟随 encoder feature，decoder/local output 更弯更高秩，未修复 LIIF response-object 问题。
- Stage C：处于 pilot / failure-audit 阶段；已有局部线索，但当前 patch matching / raw output trajectory 操作化未通过主证据标准，Stage B-prime 也不足以支持直接重开。2026-05-10 的 C-minimal LR-cell encoder-feature diagnostic 为 mixed/inconclusive：baby/woman 有正增量，bird/head 为负，未通过重开 gate。
- 2026-05-11 reframe：下一步更合理的问题候选是先审计 SIREN/LIIF 的 spectral fitting order 是否稳定、是否有空间组织、是否超过平凡 controls；这只是 proposed node，不是结果。
- Stage D：未启动；等变性、update prediction、training reduction 均为 blocked claim。

## Canonical Sources

| 类型 | 入口 | 用途 |
|---|---|---|
| 当前上下文胶囊 | [current_context.md](current_context.md) | 低 token 恢复入口：当前主线、可信结论、禁止动作和下一 gate |
| 研究合同 | [research_contract.md](research_contract.md) | 固定研究问题、阶段边界、非目标和禁止 claim |
| Claim ledger | [claims_ledger.md](claims_ledger.md) | 每条可写/不可写结论的证据、反证和允许表述 |
| 实验注册 | [evidence/experiment_registry.md](evidence/experiment_registry.md) | 实验/分析卡、run manifest、artifact 索引入口 |
| 决策记录 | [decisions/README.md](decisions/README.md) | 为什么封版、暂停、降级或继续某路线 |
| Stage A closeout | [stages/stage_a/closeout.md](stages/stage_a/closeout.md) | Stage A 当前可靠结论与剩余风险 |
| Stage B closeout | [stages/stage_b/closeout.md](stages/stage_b/closeout.md) | Stage B 平台证据与解释边界 |
| Stage B-prime audit | [stages/stage_b/b_prime_trajectory_audit.md](stages/stage_b/b_prime_trajectory_audit.md) | 参数/响应轨迹本体初审结果 |
| Stage B-response audit | [stages/stage_b/response_trajectory_audit.md](stages/stage_b/response_trajectory_audit.md) | 函数响应轨迹初审结果 |
| LIIF unit screen | [stages/stage_b/liif_internal_unit_screen.md](stages/stage_b/liif_internal_unit_screen.md) | LIIF 内部 function-space unit 筛查 |
| Stage C status | [stages/stage_c/current_status.md](stages/stage_c/current_status.md) | Stage C 当前证据、失败和下一 gate |
| 工程边界 | [Refactor_plan.md](Refactor_plan.md) | 当前代码入口、保守重构边界和测试要求 |
| Reduced archive status | [archive/README.md](archive/README.md) | 已删除历史长文档/旧规则的说明；不是事实入口 |

## 证据分层

- Current raw run artifacts：`results/FittingDynamics_StageB/`。
- Current derived diagnostics：`results/FittingDynamics_StageB_diagnostics/` 中保留的 Stage B audit JSON。
- Retired/deleted artifacts：old Stage C diagnostics、controlled self-similarity diagnostics、smoke/debug outputs、legacy `results/FittingDynamics/`、old paper draft、old archive full-text logs/rules。
- Daily memory：`memory/daily/YYYY-MM-DD.md`，append-only 工作记录，不作为 canonical source。

## 使用原则

1. 写论文或摘要前，先查 [claims_ledger.md](claims_ledger.md)。
2. 引用数值前，先查 [evidence/experiment_registry.md](evidence/experiment_registry.md)；只有 current canonical artifact 仍应存在于 `results/`。
3. 解释路线变化前，先查 [decisions/README.md](decisions/README.md)。
4. 旧长文档和旧规则已被删除或缩减到 [archive/README.md](archive/README.md)；若需要事实恢复，以本索引、ledger、stage docs、decision records 和 daily memory 为准。
5. 任何新实验必须先有 experiment card，再有 result artifact，再更新 claim ledger。
