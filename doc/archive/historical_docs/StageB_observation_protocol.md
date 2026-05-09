# Stage B 观测实验协议

## 0. 目的

本文档定义阶段 B 的最小可执行实验协议。阶段 B 的目标不是证明核心科学假设，而是把当前 fitting dynamics 代码验证为可靠、可复现、可检查的观测平台。

阶段 B 应直接服务于主线：任意尺度 INR 超分中的参数/响应空间结构、局部几何相似性和等变性分析。当前阶段只处理“能否可靠记录与检查轨迹”这一前置问题，不做强理论结论。

2026-05-07 更新：本文档是阶段 B 的预实验协议，部分条款已经被正式 scratch reduced LIIF / SIREN 实验覆盖。当前正式状态以 [`doc/StageB_observation_log.md`](StageB_observation_log.md) 和 [`doc/StageAB_review_2026-05-07.md`](StageAB_review_2026-05-07.md) 为准。尤其是：本阶段不再要求 pretrained LIIF/LIIF-EQ smoke run 作为进入 Stage C pilot 的前置条件，因为用户已明确当前阶段不使用 pretrained LIIF/LIIF-EQ，正式实验采用 scratch reduced LIIF。

---

## 1. 阶段 A 状态判断

阶段 A 已基本完成，足以进入阶段 B 的预备与最小观测实验。

已完成内容：

- [`RESEARCH_PLAN.md`](../../../RESEARCH_PLAN.md) 已重写为当前主路线图；
- [`doc/StageA_literature_baseline.md`](StageA_literature_baseline.md) 已给出文献类别、已知起点、研究缺口与不确定项；
- [`doc/StageA_research_questions.md`](StageA_research_questions.md) 已明确中心问题、研究对象、可证伪假设、负对照和最低证据标准；
- [`doc/Refactor_plan.md`](../../Refactor_plan.md) 已明确保守重构边界。

仍需补充但不阻塞阶段 B：

- 候选文献的完整 bibliographic verification；
- LIIF-EQ / SE-INR checkpoint 与模型来源的精确查证；
- 任意尺度 SR 后续方法的完整 baseline map；
- Stage C 的 geometry descriptor 与 response probe 详细定义。

上述内容重要，但不影响阶段 B 先验证当前观测平台是否可靠。

---

## 2. 当前已完成工作与阶段 B 的关系

当前已完成的保守重构、schema 工具、shape validation 与测试补充属于：

> 阶段 B 的实验前基础设施建设与预备工作。

它们尚不构成阶段 B 的实质实验内容，因为还没有运行新的观测实验、没有生成新的 trajectory 输出、也没有进行跨 seed/image/scale 的可靠性分析。

当前工作没有偏离主线，原因是：

- schema 校验直接服务于 trajectory 可复现性；
- alignment 测试直接服务于参数空间比较有效性；
- shape validation 避免后续 PCA/response 分析读取错误数据；
- 文档明确禁止把 PCA/轨迹可视化过度解释为科学结论。

需要避免的偏离：

- 长时间停留在工程工具而不进入观测实验；
- 继续沿旧 group-orbit 论文叙事推进；
- 在没有 local geometry/response evidence 前讨论训练加速；
- 过早大规模迁移实验脚本。

---

## 3. 阶段 B 最小目标

阶段 B 最小目标是回答：

1. 当前脚本能否在最小步数下生成结构完整的 trajectory 输出？
2. 输出是否满足 schema 与 shape consistency？
3. summary 是否包含后续复现所需基本字段？
4. aggregator 是否能扫描结果并给出 PC1/random baseline 指标或明确失败信息？
5. 哪些脚本/模型在当前环境下由于数据、checkpoint 或依赖问题暂时不能运行？

---

## 4. 最小实验阶梯

### B0：无训练 schema 与工具测试

目的：确认 schema 工具和 alignment 工具工作正常。

命令：

```bash
python tests/test_alignment.py
pytest -q tests/test_trajectory_schema.py
```

预期：全部通过。

当前状态：已通过。

### B1：SIREN 最小 smoke run

目的：验证 non-conditional INR fitting 脚本可生成 trajectory 输出。

前提：存在可读取图像，例如 `Data/Set5/HR/baby.png`，或用户提供 `--image`。

建议命令：

```bash
python experiments/Phase1_FittingDynamics/run_siren.py \
  --image Data/Set5/HR/baby.png \
  --steps 10 \
  --device cpu \
  --save_dir results/FittingDynamics_smoke/SIREN_baby_steps10
```

注意：当前 [`run_siren.py`](../../../experiments/Phase1_FittingDynamics/run_siren.py) 使用配置中的固定 snapshot steps。如果 `--steps 10` 小于配置中的大部分 snapshot steps，可能只记录 step 0/5/10 等已有 snapshot。若输出为空或不完整，应记录问题，不立即大改脚本。

输出检查：

- `trajectory.npz` 是否存在；
- `dynamics_summary.json` 是否存在；
- [`src/trajectory_schema.py`](../../../src/trajectory_schema.py) 的 schema 与 shape validation 是否通过。

### B2：Conditional LIIF/LTE 最小 smoke run

目的：验证 conditional INR scratch SR mode 输出结构。

前提：存在图像数据。

建议命令：

```bash
python experiments/Phase1_FittingDynamics/run.py \
  --model liif \
  --image Data/Set5/HR/baby.png \
  --sr 4 \
  --steps 50 \
  --device cpu \
  --save_dir results/FittingDynamics_smoke/LIIF_baby_sr4_steps50
```

输出检查：

- full/enc/dec snapshots 是否存在；
- `dec_snapshots_aligned` 是否 shape 一致；
- summary 是否包含 SR 信息。

### B3：Pretrained LIIF/LIIF-EQ smoke run（历史条款，当前不执行）

目的：验证 pretrained wrapper、checkpoint 和 fine-tuning 输出。

当前状态：该条款属于早期预实验协议，不再作为 Stage B 或 Stage C pilot 的进入标准。当前正式证据链只使用 scratch reduced LIIF / SIREN；pretrained LIIF/LIIF-EQ 的来源、checkpoint、公平对照和 buffer/alignment policy 尚未核清，不能作为当前阶段解释性证据。

前提：

- 图像数据存在；
- pretrained checkpoint 存在；
- CPU/GPU 内存足够。

建议先跑 LIIF，再考虑 LIIF-EQ：

```bash
python experiments/Phase1_FittingDynamics/run_finetune.py \
  --model_type liif \
  --image Data/Set5/HR/baby.png \
  --scale 4 \
  --lr_size 12 \
  --steps 2 \
  --snapshot_interval 1 \
  --recon_interval 0 \
  --device cpu \
  --save_dir results/FittingDynamics_smoke/PretrainedLIIF_baby_steps2
```

LIIF-EQ 暂不作为第一 smoke run，因为其 state_dict/buffer 与等变层依赖更复杂。

---

## 5. 输出检查协议

每个 smoke run 完成后应检查：

1. 输出目录是否存在；
2. `trajectory.npz` 是否存在；
3. `dynamics_summary.json` 是否存在；
4. trajectory schema 是否通过；
5. trajectory shape 是否通过；
6. summary schema 是否通过；
7. 若失败，记录失败原因：数据缺失、checkpoint 缺失、依赖缺失、shape mismatch、脚本逻辑问题或资源不足。

---

## 6. 建议新增的轻量工具

为避免直接修改训练脚本，阶段 B 优先新增只读检查脚本：

```text
experiments/Phase1_FittingDynamics/check_outputs.py
```

用途：

- 读取一个已有结果目录；
- 自动判断模型 family 或由用户指定；
- 校验 `trajectory.npz` keys；
- 校验 trajectory shapes；
- 校验 summary keys；
- 打印清晰报告；
- 不修改任何结果文件。

该脚本是非侵入式的，可作为 smoke run 后的统一检查入口。

---

## 7. 阶段 B 进入 Stage C 前的最低标准

历史预实验协议曾要求：

- B0 通过；
- 至少一个 SIREN smoke run 输出完整并通过检查；
- 至少一个 conditional LIIF 或 pretrained LIIF 输出完整并通过检查；
- 明确 LIIF-EQ 是否能运行，若不能则记录阻塞原因；
- 确认 aggregator 能扫描 smoke result；
- 明确下一步 geometry descriptor 与 response probe 的输入输出对象。

2026-05-08 当前正式状态：上述 smoke 条款已被 `results/FittingDynamics_StageB/` 中正式 scratch reduced LIIF / SIREN 输出和检查替代。当前进入 Stage C pilot 的依据是 [`doc/StageB_observation_log.md`](StageB_observation_log.md) 中记录的平台检查与控制分析，不包含 pretrained LIIF/LIIF-EQ。

---

## 8. 当前下一步

当前最小推进任务：新增只读输出检查脚本 [`experiments/Phase1_FittingDynamics/check_outputs.py`](../../../experiments/Phase1_FittingDynamics/check_outputs.py)，并为其添加轻量测试。该任务直接服务于阶段 B，但不会改变现有训练脚本行为。
