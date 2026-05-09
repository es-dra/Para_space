# Project Index

本文档是项目当前的论文级入口。它不替代每日记忆，也不替代原始结果；它只说明到哪里查事实、证据、决策和历史材料。

## 当前状态

- Stage A：完成 pilot 级问题定义、可证伪假设、非目标、负对照和第一轮文献地图；尚不是论文级完整 related work。
- Stage B：完成 scratch fitting-dynamics 平台验证；只支持“平台可靠”，不支持局部几何机制、等变性或训练加速 claim。
- Stage C：处于 pilot / failure-audit 阶段；已有局部线索，但当前 patch matching / raw output trajectory 操作化未通过主证据标准。
- Stage D：未启动；等变性、update prediction、training reduction 均为 blocked claim。

## Canonical Sources

| 类型 | 入口 | 用途 |
|---|---|---|
| 研究合同 | [research_contract.md](research_contract.md) | 固定研究问题、阶段边界、非目标和禁止 claim |
| Claim ledger | [claims_ledger.md](claims_ledger.md) | 每条可写/不可写结论的证据、反证和允许表述 |
| 实验注册 | [evidence/experiment_registry.md](evidence/experiment_registry.md) | 实验/分析卡、run manifest、artifact 索引入口 |
| 决策记录 | [decisions/README.md](decisions/README.md) | 为什么封版、暂停、降级或继续某路线 |
| Stage A closeout | [stages/stage_a/closeout.md](stages/stage_a/closeout.md) | Stage A 当前可靠结论与剩余风险 |
| Stage B closeout | [stages/stage_b/closeout.md](stages/stage_b/closeout.md) | Stage B 平台证据与解释边界 |
| Stage C status | [stages/stage_c/current_status.md](stages/stage_c/current_status.md) | Stage C 当前证据、失败和下一 gate |
| 工程边界 | [Refactor_plan.md](Refactor_plan.md) | 当前代码入口、保守重构边界和测试要求 |
| 历史材料 | [archive/README.md](archive/README.md) | 旧计划、旧协议、混合日志的状态说明 |

## 证据分层

- Raw run artifacts：`results/FittingDynamics_StageB/`、`results/FittingDynamics_StageC_controlled/`。
- Derived diagnostics：`results/FittingDynamics_StageC_diagnostics/`、`results/FittingDynamics_StageC_controlled_diagnostics/`。
- Smoke / debug artifacts：`results/FittingDynamics_smoke/`。
- Legacy artifacts：`results/FittingDynamics/`。
- Daily memory：`memory/daily/YYYY-MM-DD.md`，append-only 工作记录，不作为 canonical source。

## 使用原则

1. 写论文或摘要前，先查 [claims_ledger.md](claims_ledger.md)。
2. 引用数值前，先查 [evidence/experiment_registry.md](evidence/experiment_registry.md) 和对应 result artifact。
3. 解释路线变化前，先查 [decisions/README.md](decisions/README.md)。
4. 旧文档已归档到 [archive/](archive/)，仍可查，但若与本索引冲突，以本索引和 ledger 为当前事实源。
5. 任何新实验必须先有 experiment card，再有 result artifact，再更新 claim ledger。
