# 研究计划：面向任意尺度 INR 超分的参数/响应空间理论

## 0. 当前定位与范围

本文档是当前项目的顶层路线图。具体事实源以 [`doc/project_index.md`](doc/project_index.md) 为入口；旧 [`paper/`](paper) 正文和旧 [`doc/archive/`](doc/archive) 长日志已在 2026-05-11 cleanup 中删除或缩减，不视为当前证据源。

2026-05-09 更新：论文级证据入口已建立在 [`doc/project_index.md`](doc/project_index.md)。本文件只保留主路线和背景动机；具体 claim 边界、实验索引、stage closeout 和决策记录以 [`doc/claims_ledger.md`](doc/claims_ledger.md)、[`doc/evidence/experiment_registry.md`](doc/evidence/experiment_registry.md)、[`doc/stages/`](doc/stages) 和 [`doc/decisions/`](doc/decisions) 为准。若本文件与这些索引冲突，以新索引为当前事实源。

当前研究主干为：

> 面向任意尺度超分辨率，研究 INR 模型中由图像内部局部几何相似性、尺度连续性与等变性约束共同诱导的参数空间或响应空间结构，并进一步判断这些结构是否能够解释等变架构的作用，以及是否支持部分适配更新的推演，从而减少全量训练或全量微调。

本计划遵循保守原则：

- 任何结论必须由可复现实验支持；
- 群变换只是构造可控扰动的工具，不是研究本体；
- full flattened parameters 只是分析对象之一，decoder response、Jacobian、gradient、feature、adapter、modulation 可能更接近真实规律；
- 等变性是待检验机制，不是预设答案；
- 后续论文中的 claim 必须配套负对照、统计验证与失败案例。

---

## 1. 核心研究目标

### 1.1 主问题

在任意尺度 INR 超分中，图像局部几何相似性是否对应可复用、低复杂度或可预测的参数/响应变化？等变性是否能够使这种对应更稳定、更低维、更可迁移或更可利用？

进一步分解为：

1. 局部几何相似的图像 patch 是否诱导相似的 decoder response、local Jacobian、gradient contribution、adapter update、modulation shift 或 parameter update？
2. 连续尺度变化引起的 INR 适配方向是否足够结构化，以至于可由已有尺度、局部几何或等变约束进行近似预测？
3. 等变架构是否让参数/响应关系更稳定、更可迁移、更低自由度或更容易部分推演？
4. 若上述关系成立，是否能通过预测一部分更新、只训练残差的方式降低训练成本？

### 1.2 本项目不以什么为核心

本项目不是：

- 泛泛研究优化轨迹；
- 证明参数 PCA 低维；
- 以旋转/缩放群为中心的群论论文；
- 以 PSNR 排榜为主的超分 benchmark；
- 预设等变性一定改善参数空间规律的论文。

这些内容可以作为工具、诊断或对照，但不是核心贡献。

### 1.3 研究对象

| 对象 | 价值 | 风险 |
|---|---|---|
| 全模型参数 | 最直接的参数空间对象 | 高维、对称性、buffer 与对齐问题严重 |
| Encoder 参数 | 条件 INR 中承载图像内容 | 可能反映 feature adaptation 而非几何规律 |
| Decoder 参数 | 直接连接 local feature、coord、cell 与 RGB | 任意尺度 SR 中最关键，但仍有对齐问题 |
| Adapter/Modulation | 参数高效适配对象 | 当前仓库尚未实现 |
| Decoder response | 直接反映局部坐标/尺度行为 | 需要设计 probe |
| Local Jacobian | 测量对 feature/coord/cell 的敏感性 | 计算成本较高 |
| Gradient contribution | 连接局部 loss 与参数更新 | 需要谨慎 attribution |
| Feature response | LIIF 类模型中可能更承载几何 | 可能把研究扩展为参数-特征联合空间 |

项目应保持开放：最强信号可能不在 raw full weights，而在 response、feature 或 modulation space。

---

## 2. 核心假设

这些都是待验证假设，不是既有结论。

### H1：局部几何到响应/更新的对应

局部几何相似的图像区域，相比随机区域，应诱导更相似的 INR 响应或更新方向。

局部几何可以包括：patch 外观、边缘方向、局部频率、纹理周期性、跨尺度 recurrence、structure tensor 等。

响应/更新对象可以包括：decoder output response、decoder Jacobian、decoder gradient contribution、adapter/modulation update、low-rank update coefficient 等。

### H2：尺度条件下的更新结构

任意尺度适配不一定是任意参数变化。对某些模型组件，邻近尺度或相似局部几何下的变化可能集中在可复用子空间，或服从局部可预测规则。

必须用随机轨迹、打乱 scale label、足够 snapshot 数量等负对照排除“优化连续性导致的伪低维”。

### H3：等变性作为参数/响应正则化机制

等变架构可能通过结构化共享局部几何变换规律，使局部几何到响应/更新的对应更稳定、更低维或更可迁移。

可测表现包括：cross-scale update consistency 更高、有效更新维度更低、几何-响应相关更强、预测残差更小、未见尺度/图像迁移更好等。

也必须允许相反结果：等变性可能主要改善 feature 或输出一致性，而不直接改善 full parameter regularity。

### H4：部分更新可推演

若 H1-H3 足够强，则 INR 适配中的一部分更新可能可由局部几何和尺度条件推演，只需训练剩余 residual。

潜在目标包括 decoder bias、low-rank decoder update、adapter、modulation、scale-conditioned residual correction 或短微调初始化方向。

这是长期应用假设，不能提前宣称成立。

---

## 3. 当前工作区基础

当前仓库更适合作为观测平台，而不是已经完成的研究证明。

### 3.1 文档状态

| 文件 | 当前作用 | 后续处理 |
|---|---|---|
| [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) | 主路线图 | 持续维护 |
| [`doc/project_index.md`](doc/project_index.md) | 当前项目入口 | 优先阅读 |
| [`doc/research_contract.md`](doc/research_contract.md) | 研究边界与禁止 claim | 当前事实源 |
| [`doc/claims_ledger.md`](doc/claims_ledger.md) | claim-evidence 状态 | 论文写作前必查 |
| [`doc/evidence/experiment_registry.md`](doc/evidence/experiment_registry.md) | 实验与分析索引 | 证据追溯入口 |
| [`doc/Refactor_plan.md`](doc/Refactor_plan.md) | 工程边界 | 代码维护依据 |
| [`paper/`](paper) | 无活跃 manuscript；只保留旧稿删除说明 | 未来论文需从 ledger 重写 |
| [`doc/archive/`](doc/archive) | 旧计划、旧日志、旧规则已缩减为 README | 不是当前事实入口 |

### 3.2 代码资产

| 模块 | 文件 | 作用 |
|---|---|---|
| 参数对齐 | [`src/alignment.py`](src/alignment.py), [`tests/test_alignment.py`](tests/test_alignment.py) | 参数空间比较基础 |
| 模型 | [`src/siren.py`](src/siren.py), [`src/models/`](src/models) | 当前保留 SIREN、scratch LIIF、LTE；pretrained LIIF/LIIF-EQ wrapper 已从活跃代码面删除 |
| 数据/变换 | [`src/datasets.py`](src/datasets.py), [`src/transforms.py`](src/transforms.py) | 图像、坐标、几何变换工具 |
| 指标/频谱 | [`src/metrics.py`](src/metrics.py), [`src/spectral.py`](src/spectral.py) | 重建质量与频谱诊断 |
| 拟合动态 | [`experiments/Phase1_FittingDynamics/`](experiments/Phase1_FittingDynamics) | 当前 observation baseline |
| Schema 校验 | [`src/trajectory_schema.py`](src/trajectory_schema.py), [`tests/test_trajectory_schema.py`](tests/test_trajectory_schema.py) | 非侵入式输出格式验证 |

### 3.3 当前输出约定

现有 fitting dynamics 脚本保存：

```text
results/FittingDynamics_StageB/{MODEL_TAG}/
  trajectory.npz
    full_snapshots
    full_snapshots_aligned
    enc_snapshots
    dec_snapshots
    dec_snapshots_aligned
    snapshot_steps
    losses
    psnrs
    freq_ratios
    target_spectrum
    grad_norms
  dynamics_summary.json
  reconstructions/
```

后续重构必须优先保持兼容。

---

## 4. 阶段路线

## Stage A：研究问题重建与文献基线

### 目标

建立科学问题、文献地图、可证伪假设、负对照清单与保守工程边界。

### 输入

- 当前代码结构与实验脚本；
- 现有模型与参数对齐工具；
- 旧文档，但只作为历史材料；
- 可核查文献：任意尺度 SR、INR、等变性、内部自相似、参数高效适配、优化/权重空间几何。

### 输出

- 更新后的 [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)；
- [`doc/stages/stage_a/closeout.md`](doc/stages/stage_a/closeout.md)；
- [`doc/Refactor_plan.md`](doc/Refactor_plan.md)；
- 必要负对照与 baseline 清单；
- 哪些脚本冻结为 baseline、哪些后续重构的决策。

### 成功标准

- 问题表述不依赖过时 group-orbit claim；
- 文献中已知/未知/不确定项被明确区分；
- 所有强表述都降级为待验证假设；
- 下一阶段实验有清晰问题、指标、负对照和失败条件；
- 工程重构保留既有行为。

## Stage B：观测基础设施与可靠性

目标是把当前 fitting dynamics 代码变成可复现实验观测平台。此阶段不证明主命题，只验证工具可靠性。

2026-05-09 当前状态：Stage B 已按 scratch reduced LIIF / SIREN 证据链封版为“平台可靠”。正式 Stage B 不使用 pretrained LIIF/LIIF-EQ；PCA/PC1 仅作诊断，不支持局部几何、等变性或训练加速 claim。详见 [`doc/stages/stage_b/closeout.md`](doc/stages/stage_b/closeout.md) 与 [`doc/evidence/run_manifest_stage_b.md`](doc/evidence/run_manifest_stage_b.md)。

### 初始实验历史计划与当前状态

1. SIREN 单图像拟合作为非条件 baseline：已完成 scratch Stage B。
2. LIIF/LTE SR x4 scratch fitting，避免 self-reconstruction shortcut：reduced LIIF 已完成 scratch Stage B；LTE 未作为当前正式证据链。
3. Pretrained LIIF 与 pretrained LIIF-EQ 在相同 LR/HR pair 上微调：历史计划，当前不执行，因来源、checkpoint、公平协议和 buffer/alignment policy 未闭合。
4. 至少一个图像/架构的多 seed 运行：baby 多 seed 已完成。
5. alignment vs no-alignment 对照：PCA diagnostic 中已记录 raw/aligned PC1 差异；不能外推为后续 parameter comparison 可忽略 alignment。

### 指标

PSNR/loss、update norm、encoder/decoder ratio、PC1/random baseline、effective dimension、update coherence、low-frequency error ratio、alignment cost、seed reproducibility。

### 风险

PCA 伪低维、self-reconstruction 高 PSNR、参数对称性污染、LIIF-EQ buffer/状态复杂、可视化过度解读。

## Stage C：局部几何到响应/更新的对应

目标是验证核心科学假设：局部图像几何是否对应 INR 响应/更新相似性。

### 实验问题

1. 相似 patch 是否在 decoder response space 中更接近？
2. 相似 patch 是否诱导更相似的 gradient/update direction？
3. 哪类几何描述符最能预测响应相似性？
4. 关系是否跨 scale、跨 image 保持？
5. LIIF-EQ 是否增强、削弱或转移这种关系？
6. 信号出现在 full parameters、decoder、features、Jacobians 还是 modulation-like objects？

### 指标与负对照

- patch similarity 与 response similarity 相关；
- response nearest neighbor 的 geometry retrieval；
- cluster 内/间方差；
- cross-scale consistency；
- shuffled patch/scale label；
- random patch pairs；
- feature-only vs parameter-only 对照。

## Stage D：等变性解释与部分更新推演

目标是在 Stage C 证据充分时，测试是否能利用对应关系解释等变性并减少训练。

### 实验问题

1. 能否由局部几何和目标 scale 预测部分 update？
2. 哪个组件最可预测：bias、low-rank、adapter、modulation、residual correction？
3. predicted update 是否减少 target PSNR 所需训练步数？
4. 等变性是否提高预测或 residual training 效率？
5. 是否泛化到未见 scale/image？

### Baseline

no adaptation、full fine-tuning、decoder-only fine-tuning、random low-rank update、scale interpolation、MAML/meta-init 或 hypernetwork（仅当实现并验证后）。

---

## 5. 工程重构原则

1. 不在缺少测试时重写现有实验脚本；
2. 保留现有 CLI 参数；
3. 保留 `trajectory.npz` 与 `dynamics_summary.json` 兼容性；
4. 优先新增非侵入工具，而不是修改实验语义；
5. 每次行为变更必须配测试或明确人工验证；
6. 科学变更与工程变更分离。

### 近期安全重构目标

| 优先级 | 目标 | 动机 | 安全做法 |
|---|---|---|---|
| 高 | 中文研究文档 | 协作与方向统一 | 更新 Markdown，不改代码语义 |
| 高 | 输出 schema 工具 | 可复现性 | 新增校验，不接入脚本 |
| 高 | alignment 测试 | 参数比较有效性 | 保留并定期运行 |
| 中 | result I/O 工具 | 减少重复逻辑 | 先新增独立模块，暂不替换脚本 |
| 中 | 几何 descriptor | Stage C 必需 | 先定义与测试，再接入实验 |
| 低 | 大规模目录迁移 | 风险高 | 推迟到测试覆盖充分后 |

---

## 6. 评价哲学

### 弱证据

- PCA 图看起来平滑；
- 单图单 seed 的 PC1 较高；
- 低频先学；
- LIIF 与 LIIF-EQ 参数变化不同但无控制；
- PSNR 提升但无参数/响应解释。

### 较强证据

- 局部几何相似性与 response/update similarity 的统计相关；
- 跨 seed/image/scale 可复现；
- 负对照失败而真实配对成功；
- 等变与非等变对照条件明确；
- predicted update 在 unseen scale/image 上优于公平 baseline。

### 实验报告最低标准

每个实验应记录：代码版本、完整配置、数据/图像标识、scale、seed、checkpoint 来源、trainable parameter policy、alignment policy、指标、置信区间（如可行）、失败案例。

---

## 7. 论文产出路径

旧 [`paper/`](paper) 叙事不应继续主导。旧正文已删除；未来 manuscript 必须依据 [`doc/claims_ledger.md`](doc/claims_ledger.md) 和新的 evidence registry 从头重写。

以下只是候选论文命题，当前尚未成立：

> 图像局部几何在任意尺度 INR 超分中诱导结构化响应与适配模式；等变性可以作为正则化和暴露这种模式的机制，并可能支持更可复用、部分可预测的更新。

可能贡献（必须由证据支持）：

1. 一个连接 local geometry 与 INR response/update similarity 的 probing framework；
2. 任意尺度 INR SR 中局部几何自相似预测响应/更新相似性的实证证据；
3. 等变架构是否以及如何增强这种对应的分析；
4. 若验证成立，一个减少适配成本的 partial update prediction 方法。

---

## 8. 当前优先事项

1. 使用 [`doc/project_index.md`](doc/project_index.md) 作为项目入口，使用 [`doc/claims_ledger.md`](doc/claims_ledger.md) 管理所有论文级表述。
2. 将 Stage A/B 维持为封版状态：Stage A 是 pilot 级研究合同，Stage B 是平台可靠性证明。
3. Stage C 暂停开放式 descriptor/rerank 小修补；若继续，只允许 geometry-vs-content dissociation gate 或 model-internal unit gate。
4. 不启动等变性、pretrained LIIF/LIIF-EQ、update prediction 或训练加速，除非前置 gate 通过。
5. 补充候选文献的论文级核查，并把结果同步到 claim ledger / evidence registry。
