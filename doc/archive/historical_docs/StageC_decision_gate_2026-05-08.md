# Stage C 决策门槛：停止开放式 descriptor 尝试

日期：2026-05-08

本文档用于冻结当前 Stage C pilot 的下一节点判断，避免继续通过零散 descriptor / rerank 尝试掩盖核心问题。它记录的是本项目当前事实和决策门槛，不是通用工作流规则。

---

## 1. 当前阶段定位

Stage A 的中心问题仍然成立：在任意尺度 INR 超分中，局部图像几何是否能够预测模型 response/update 的一部分适配变化。

当前实际进度只处在 H1 的 Stage C pilot / 方法诊断阶段：

- Stage B 已证明 scratch fitting-dynamics 平台可可靠记录、检查和重建轨迹；
- Stage C 有 baby 多 seed 和 woman seed42 的局部正线索；
- bird/head 是实质 failure 或 unresolved；
- 还没有进入 scale structure、等变性比较、更新预测或训练加速阶段。

因此下一步不是继续扩大 Stage C，也不是引入 LIIF-EQ，而是判断当前 matching/probe 是否还有方法学可救性。

---

## 2. 可靠证据台账

### 2.1 平台证据

可靠：

- `results/FittingDynamics_StageB/` 中 scratch reduced LIIF / SIREN 轨迹已通过 schema、shape、summary consistency 和 analysis readiness 检查；
- raw `full_snapshots` 可重建 LIIF 函数响应，final PSNR 与 summary 误差约 `0.00000`；
- Stage B 控制分析显示 PCA 只能作为诊断，不能作为科学 claim。

限制：

- LIIF high PC1 接近 permuted-update control，不能解释为几何或训练顺序的强低维结构；
- Stage B 不支持局部几何对应、等变性机制或训练加速。

### 2.2 Stage C 正向线索

可靠但有限：

- baby 多 seed 和 woman seed42 在部分 response object / 高内容强度 patch 中有正信号；
- content stratification 显示 baby/woman 的主要线索集中在 high-gradient / high-variance patch；
- feature trajectory 对 `rgb_grad_context` 更敏感。

限制：

- Spearman 全局相关偏弱；
- butterfly 只能算弱支持；
- coord-Jacobian 多数是 tiny effect，不能作为主证据；
- feature response 的改善不能外推到 output response。

### 2.3 Stage C 反向约束

可靠：

- bird output trajectory 在 `rgb_grad` 下为 negative：`effect=-0.0341`；
- head output trajectory 在 `rgb_grad` 下为 negative：`effect=-0.0063`；
- high-gradient / high-variance 分层没有救回 bird/head；
- head final PSNR 高但 output response 仍失败，说明 failure 不能只归因于 fitting quality。

patch-level failure 定位：

| run | response | top1 response pct mean | worst response frac |
|---|---|---:|---:|
| bird | output trajectory | 0.527 | 0.281 |
| head | output trajectory | 0.456 | 0.190 |
| baby seed123 | output trajectory | 0.252 | 0.041 |
| woman seed42 | output trajectory | 0.337 | 0.066 |

解释：bird/head 的局部 geometry top-1 经常是 response 上相对远的候选；正例参考中该现象明显弱得多。

---

## 3. 最近工作的真正价值

最近 `rgb_grad_context`、top-k oracle 和 context rerank 的价值不在于产生了一个新 descriptor，而在于暴露了一个关键事实：

> `rgb_grad` top-5 candidate set 中存在 response oracle headroom，但现有 response-blind context rerank 不能利用这个 headroom。

关键数值来自：

- `results/FittingDynamics_StageC_diagnostics/topk_oracle_rgb_grad_2026-05-08.json`
- `results/FittingDynamics_StageC_diagnostics/topk_rerank_rgb_grad_to_context_2026-05-08.json`

| run | top1 pct | oracle pct | oracle gain | context rerank pct | rerank gain |
|---|---:|---:|---:|---:|---:|
| bird output | 0.527 | 0.175 | 0.352 | 0.528 | -0.000 |
| head output | 0.456 | 0.148 | 0.308 | 0.469 | -0.013 |
| baby123 output | 0.252 | 0.076 | 0.176 | 0.294 | -0.042 |
| woman42 output | 0.337 | 0.124 | 0.213 | 0.346 | -0.010 |

候选级解释：

| run | oracle is context-best frac | oracle farther than top1 in context frac | rerank-oracle pct gap |
|---|---:|---:|---:|
| bird output | 0.198 | 0.512 | 0.353 |
| head output | 0.207 | 0.463 | 0.320 |
| baby123 output | 0.264 | 0.438 | 0.218 |
| woman42 output | 0.240 | 0.430 | 0.222 |

解释：oracle 候选多数不是 context descriptor 最近者，且在 bird/head output 中约半数 anchor 的 oracle 候选在 context 空间比原 top1 更远。这说明 failure 不是简单“加上下文再排一次”能解决。

---

## 4. 现在必须停止的分支

停止：

- PCA/PC1 叙事；
- 完整 Stage C 扩样；
- 更多 seed / 更多图像来稀释 failure；
- 等变性比较；
- pretrained LIIF/LIIF-EQ 证据链；
- 训练加速、adapter/modulation、update prediction claim；
- coord-Jacobian 作为主证据；
- top-k oracle 作为方法结果；
- 无依据地“再试一个 descriptor”。

允许的唯一下一类工作：围绕冻结协议做 candidate-level、response-blind 的 stop/go 诊断。

---

## 5. 下一节点的单一核心问题

冻结问题：

> 在 `rgb_grad` top-5 candidate 集合内，是否存在一个 response-blind、预注册的 matching/rerank 规则，能稳定利用 bird/head output trajectory 的 oracle headroom，同时不破坏 baby/woman 正例？

这个节点不是 descriptor 搜索。它必须先解释 oracle candidate 和 geometry top1 candidate 在 response-blind 属性上是否存在系统差异；若没有差异，应停止 rerank 路线。

---

## 6. 冻结协议

数据：

- failure gate：`LIIF_reduced_bird_sr4_seed42`、`LIIF_reduced_head_sr4_seed42`；
- positive guardrail：`LIIF_reduced_baby_sr4_seed123`、`LIIF_reduced_woman_sr4_seed42`。

主 response：

- primary：`trajectory_delta`；
- `feature_trajectory_delta` 和 `coord_jacobian_final_delta` 只作辅助解释，不能替代 primary 判定。

固定设置：

- geometry source：`lr_up`；
- baseline candidate set：`rgb_grad` top-5；
- `patch_size=7`；
- `stride=4`；
- `k=5`；
- `min_spatial_distance=8`；
- `n_shuffles=200`；
- `seed=0`。

禁止：

- 按图像单独换规则；
- 看结果后更换 primary metric；
- 用 response label 或 oracle 信息作为 predictor；
- 只报告 high-gradient / high-variance 子集作为主结论；
- 把 feature/Jacobian 改善替代 output 结论。

---

## 7. 成功、失败和灰区标准

### 成功标准

必须同时满足：

- bird 和 head 的 `trajectory_delta` 均达到 `rerank_response_percentile_gain_mean >= 0.10`；
- bird/head 均至少恢复各自 oracle gain 的约 30%；
- bird `rerank_response_percentile_mean < 0.43`；
- head `rerank_response_percentile_mean < 0.36`；
- baby123 / woman42 不明显变差，即 `rerank_response_percentile_gain_mean >= -0.02`；
- 同一冻结规则适用于四个 run。

### 失败标准

任一成立即失败：

- bird 或 head 任一 output gain `< 0.05`；
- 改善只出现在 feature/Jacobian，output 不成立；
- baby/woman 明显变差；
- 需要 response label、oracle 信息、事后图像选择或 stratum 选择才成立；
- oracle candidate 与 top1 candidate 在预注册 response-blind 属性上不可分。

### 灰区标准

灰区只支持继续诊断，不支持进入等变性或训练加速：

- bird/head output gain 在 `0.05-0.10`；
- 只修复 bird 或只修复 head；
- patch percentile 改善但 shuffled-control effect 仍 tiny/negative；
- 改善伴随 baby/woman 小幅受损。

---

## 8. 如果失败，如何降级

若下一节点失败，Stage C 主线应降级为：

> 当前 scratch reduced LIIF 中，某些图像和高内容强度 patch 存在局部 geometry-response 线索，但当前 patch matching / descriptor 协议不能形成跨图像可靠机制。

后续应转向：

1. 重新定义 response object 或 patch matching 问题；或
2. 将论文贡献降级为 Stage B fitting-dynamics 平台、response 重建协议，以及 Stage C failure audit / negative diagnostic framework。

失败后不应启动等变性比较，因为它会把未解决的 matching 问题转移到架构差异上。

---

## 9. 下一次执行要求

若继续推进，应一次性完成一个完整节点：

1. 实现 candidate-level oracle-vs-top1 属性审计；
2. 只使用预注册属性：edge orientation / structure tensor、一阶或二阶梯度、低频上下文布局、空间区域、geometry/context rank、aliasing 或重复纹理指标；
3. 补测试；
4. 跑四个冻结 run；
5. 与已有 oracle/context rerank 对照；
6. 更新文档和每日记忆。

不应每新增一个小字段就升级结论。

---

## 10. 2026-05-08 执行结果：gate 未通过

执行产物：

- `results/FittingDynamics_StageC_diagnostics/candidate_attribute_gate_2026-05-08.json`

执行命令使用冻结协议：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_failure_audit.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_bird_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_head_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_baby_sr4_seed123 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_woman_sr4_seed42 \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad \
  --patch_rerank_descriptor rgb_grad_context \
  --response_modes trajectory_delta \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0 \
  --patch_failure_diagnostics \
  --patch_failure_top_n 10 \
  --format table \
  --output_json results/FittingDynamics_StageC_diagnostics/candidate_attribute_gate_2026-05-08.json
```

primary gate 结果：

| run | top1 pct | oracle pct | oracle gain | context rerank pct | rerank gain | recovered oracle gain |
|---|---:|---:|---:|---:|---:|---:|
| bird output | 0.527 | 0.175 | 0.352 | 0.528 | -0.000 | -0.001 |
| head output | 0.456 | 0.148 | 0.308 | 0.469 | -0.013 | -0.041 |
| baby123 output | 0.252 | 0.076 | 0.176 | 0.294 | -0.042 | -0.237 |
| woman42 output | 0.337 | 0.124 | 0.213 | 0.346 | -0.010 | -0.045 |

按第 7 节标准，该 gate 未通过：

- bird/head output 的 rerank gain 均 `< 0.05`，直接触发失败标准；
- baby123 guardrail 明显变差，`rerank gain=-0.042 < -0.02`；
- context rerank 没有恢复 oracle gain。

candidate-level 属性审计结果也不支持继续构造简单 response-blind rerank：

| run | attribute | oracle better frac | signed advantage | interpretation |
|---|---|---:|---:|---|
| bird output | edge orientation agreement | 0.408 | -0.0774 | oracle 不优于 top1 |
| head output | edge orientation agreement | 0.409 | -0.1717 | oracle 不优于 top1 |
| bird output | context lowfreq distance | 0.423 | -0.0755 | oracle 更远 |
| head output | context lowfreq distance | 0.424 | -0.1230 | oracle 更远 |
| bird output | context descriptor distance | 0.380 | -0.0746 | oracle 更远 |
| head output | context descriptor distance | 0.394 | -0.0810 | oracle 更远 |
| bird output | patch variance absdiff | 0.606 | 0.0058 | 小幅正向，不足以单独成规则 |
| head output | patch variance absdiff | 0.606 | 0.0027 | 小幅正向，不足以单独成规则 |

审稿式判断：

> `rgb_grad` top-5 中存在 response oracle headroom，但在预注册 response-blind 属性中，oracle candidate 与 top1 candidate 没有形成足够稳定、强、可迁移的差异。当前 simple context / edge / low-frequency / content-intensity 属性不足以支持一个可靠 rerank 规则。

因此 Stage C matching/rerank 路线当前应降级为失败或至多灰区偏失败。后续不应继续开放式 descriptor 搜索；若继续研究，应转向重新定义 response object、patch matching 单元，或把当前成果定位为 fitting-dynamics 平台与 failure audit 框架。

---

## 11. 2026-05-08 执行结果：response-object audit 未解释 failure

candidate-level gate 失败后，本轮按第 8 节的降级路径，执行了一个只读 response-object audit。该 audit 不是新的 descriptor/rerank 搜索，而是检查当前 output response object 是否因为幅值处理、归一化或时间聚合而掩盖了 bird/head 的可预测结构。

执行产物：

- `experiments/Phase1_FittingDynamics/analyze_stage_c_response_object_audit.py`
- `results/FittingDynamics_StageC_diagnostics/response_object_audit_2026-05-08.json`

执行命令：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_response_object_audit.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_bird_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_head_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_baby_sr4_seed123 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_woman_sr4_seed42 \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0 \
  --format table \
  --output_json results/FittingDynamics_StageC_diagnostics/response_object_audit_2026-05-08.json
```

预注册 response-object 变体：

- full trajectory：`trajectory_unit`、`trajectory_raw`、`trajectory_norm`；
- endpoint final delta：`final_unit`、`final_raw`、`final_norm`；
- fixed temporal split：`trajectory_early_unit`、`trajectory_mid_unit`、`trajectory_late_unit`。

说明：现有主分析中的 `trajectory_delta` 已经对 patch response descriptor 做 L2 normalize，因此主失败不是简单 raw amplitude 支配。这里的 norm audit 更准确地检查 response norm 是否导致归一化不稳定、norm-only 是否提供替代解释，以及 failure percentile 是否与 response norm/gap 系统相关。

主结果：

| variant | bird label / effect frac / patch pct | head label / effect frac / patch pct | baby123 label / effect frac / patch pct | woman42 label / effect frac / patch pct |
|---|---|---|---|---|
| trajectory_unit | negative / -0.0319 / 0.5275 | negative / -0.0048 / 0.4562 | support / 0.1070 / 0.2518 | support / 0.0502 / 0.3367 |
| trajectory_raw | negative / -0.0287 / 0.4985 | negative / -0.0030 / 0.4734 | support / 0.1083 / 0.2471 | local_only / 0.0420 / 0.3608 |
| trajectory_norm | tiny / 0.1021 / 0.4531 | negative / -0.0535 / 0.5287 | tiny / 0.1101 / 0.4220 | negative / -0.0481 / 0.5083 |
| trajectory_early_unit | negative / -0.0342 / 0.5274 | negative / -0.0060 / 0.4564 | support / 0.1085 / 0.2525 | support / 0.0492 / 0.3369 |
| trajectory_mid_unit | tiny / 0.0085 / 0.4820 | negative / -0.0124 / 0.4781 | local_only / 0.0315 / 0.3469 | tiny / 0.0085 / 0.4739 |
| trajectory_late_unit | tiny / 0.0042 / 0.4871 | negative / -0.0037 / 0.4935 | negative / -0.0004 / 0.4782 | tiny / 0.0001 / 0.5134 |

variant summary 的保守判断：

| variant | failure effect frac mean | failure patch pct mean | guardrail effect frac mean | guardrail patch pct mean | verdict |
|---|---:|---:|---:|---:|---|
| trajectory_unit | -0.0184 | 0.4918 | 0.0786 | 0.2943 | does_not_resolve_failure |
| trajectory_raw | -0.0159 | 0.4860 | 0.0752 | 0.3040 | does_not_resolve_failure |
| trajectory_norm | 0.0243 | 0.4909 | 0.0310 | 0.4651 | does_not_resolve_failure |
| trajectory_early_unit | -0.0201 | 0.4919 | 0.0788 | 0.2947 | does_not_resolve_failure |
| trajectory_mid_unit | -0.0020 | 0.4800 | 0.0200 | 0.4104 | does_not_resolve_failure |
| trajectory_late_unit | 0.0002 | 0.4903 | -0.0002 | 0.4958 | does_not_resolve_failure |

temporal consistency 也没有把 failure 与 guardrail 分开：

| group | segment pair Spearman mean | directed percentile std mean |
|---|---:|---:|
| bird/head failure | 0.1392 | 0.1987 |
| baby/woman guardrail | 0.1511 | 0.2009 |

审稿式判断：

> 当前预注册的 output response-object 分解没有解释 bird/head failure。raw response、log-norm-only response、endpoint final delta，以及 early/mid/late fixed temporal split 都没有同时恢复 bird/head 且保护 baby/woman。temporal consistency 在 failure 和 guardrail 之间也接近，不能支持“full trajectory 聚合混掉了可预测阶段结构”的解释。

结论边界：

- 可以说：简单 response-object 聚合、幅值/归一化和固定三段时间切分不是当前 bird/head failure 的充分解释。
- 不能说：所有可能的 response object 路线都失败；本轮只覆盖预注册的 output trajectory/final、raw/unit/log-norm、early/mid/late。
- 不能说：局部几何假设整体失败；仍有 baby/woman 的局部正线索。
- 不能用 final/raw/norm-only 或 tiny temporal segment 替代 primary `trajectory_delta` 结论。

更新后的下一步：

> 不应继续修补当前 descriptor/rerank 或简单 output response-object。若继续 Stage C 方法研究，下一节点应重新定义 patch matching 单元，例如跨尺度/语义/重复纹理感知的 patch 单元；否则应将当前贡献降级为 Stage B fitting-dynamics 平台、response 重建协议，以及 Stage C failure audit / negative diagnostic framework。

---

## 12. 2026-05-08 执行结果：LR-cell patch-unit gate 未通过

为避免过早降级，本轮按“重新定义 patch matching 单元”的最小路线执行一个 LIIF-aware gate。该 gate 不是继续换 descriptor，也不是 rerank，而是把 matching 单元从 HR 7x7 外观 patch 改成与 x4 LIIF 查询机制更一致的 LR feature-cell / query-support unit。

执行产物：

- `experiments/Phase1_FittingDynamics/analyze_stage_c_patch_unit_gate.py`
- `tests/test_stage_c_patch_unit_gate.py`
- `results/FittingDynamics_StageC_diagnostics/patch_unit_gate_lr_cell_2026-05-08.json`

冻结 unit：

- baseline：旧 `hr7_rgb_grad_baseline`，即 HR `7x7` / stride 4 / `lr_up + rgb_grad`；
- candidate：`lr_cell_query_support`；
- 每个 LR cell 对应一个 HR `4x4` output query block；
- geometry 用该 LR cell 周围固定 `3x3` LR support 的 mean-centered RGB + grayscale gradient；
- response 固定为 output `trajectory_delta`，并做 L2-normalized patch response descriptor；
- 不扫 support size、不换 response object、不做 rerank、不使用 oracle。

执行命令：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_patch_unit_gate.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_bird_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_head_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_baby_sr4_seed123 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_woman_sr4_seed42 \
  --device cpu \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0 \
  --format table \
  --output_json results/FittingDynamics_StageC_diagnostics/patch_unit_gate_lr_cell_2026-05-08.json
```

主结果：

| run | unit / analysis | label | effect frac | patch pct | worst frac |
|---|---|---|---:|---:|---:|
| bird | HR7 geometry | negative | -0.0319 | 0.5275 | 0.2810 |
| bird | LR-cell geometry | tiny | 0.0257 | 0.4507 | 0.1875 |
| bird | LR-cell coordinate-only | support | 0.1494 | 0.3540 | 0.0833 |
| bird | LR-cell content-only | support | 0.1177 | 0.3791 | 0.1042 |
| head | HR7 geometry | negative | -0.0048 | 0.4562 | 0.1901 |
| head | LR-cell geometry | negative | -0.0057 | 0.4669 | 0.1806 |
| head | LR-cell coordinate-only | tiny | 0.0094 | 0.4729 | 0.1389 |
| head | LR-cell content-only | support | 0.0811 | 0.3909 | 0.1042 |
| baby123 | HR7 geometry | support | 0.1070 | 0.2518 | 0.0413 |
| baby123 | LR-cell geometry | support | 0.1156 | 0.3278 | 0.1389 |
| baby123 | LR-cell content-only | support | 0.1944 | 0.2874 | 0.0972 |
| woman42 | HR7 geometry | support | 0.0502 | 0.3367 | 0.0661 |
| woman42 | LR-cell geometry | local_only | 0.0595 | 0.3697 | 0.0764 |
| woman42 | LR-cell content-only | support | 0.1554 | 0.3210 | 0.0903 |

Gate verdict：`fail`，24 个检查中 15 个失败。

关键失败原因：

- bird 的 LR-cell geometry 只从 negative 变成 tiny，未达到 support；patch pct `0.4507 > 0.40`，worst frac `0.1875 > 0.18`。
- head 的 LR-cell geometry 仍为 negative，patch pct `0.4669 > 0.45`，未通过 primary failure gate。
- LR-cell geometry 没有强于弱 controls：
  - bird 中 coordinate-only 和 content-only 都明显强于 LR-cell geometry；
  - head 中 content-only 明显强于 LR-cell geometry；
  - baby/woman 中 content-only 也强于 LR-cell geometry。
- baby/woman 的正信号没有完全消失，但 LR-cell geometry 不是比旧 HR7 或内容强度控制更清晰的机制证据。

审稿式判断：

> 最小 LIIF-aware LR-cell / query-support unit 没有修复 bird/head output failure，也没有证明 geometry matching 优于 coordinate/content weak controls。它支持“当前 HR7 patch 单元不是唯一问题”这个坏消息：即使换成更贴近 LIIF x4 查询的 LR-cell unit，当前 response-blind geometry matching 仍不足以形成跨图像可靠机制。

结论边界：

- 可以说：旧 HR7 patch 和简单 LR-cell query-support unit 都不能作为 Stage C 主证据生成器。
- 不能说：所有 model-aware patch unit 都失败；本轮只测试固定 `3x3 LR support -> 4x4 HR response block`。
- 不能把 coordinate/content-only controls 的强结果写成 geometry 机制；它们更像说明 response 有空间/内容强度结构，而不是当前 geometry descriptor 捕捉了机制。

更新后的下一步：

> 若继续挽救 Stage C，不能再做 descriptor/rerank/response-object/简单 LR-cell 小修补。唯一更重的方向是显式建模 LIIF decoder input 或 encoder feature trajectory 单元，并建立新的冻结 gate；否则应把 Stage C 降级为 failure audit / negative diagnostic 贡献。

---

## 13. 2026-05-08 执行结果：controlled self-similarity sanity gate

为区分“Stage C probe/response 重建本身坏了”和“自然图 patch matching / response 定义不足”，本轮增加了受控自相似 sanity gate。该 gate 不是自然图主证据，也不是 geometry 机制证明；它只回答一个更窄的问题：

> 如果图像中存在已知的精确重复局部结构，当前 fitting-dynamics response probe 是否能检出这些重复结构在 response trajectory 上更接近？

新增产物：

- `experiments/Phase1_FittingDynamics/generate_controlled_self_similarity_images.py`
- `experiments/Phase1_FittingDynamics/analyze_stage_c_controlled_self_similarity.py`
- `tests/test_controlled_self_similarity.py`
- `Data/ControlledSelfSimilarity/HR/css_periodic12.png`
- `Data/ControlledSelfSimilarity/HR/css_nonrepeat12.png`
- `Data/ControlledSelfSimilarity/controlled_self_similarity_metadata.json`
- `results/FittingDynamics_StageC_diagnostics/controlled_self_similarity_synthetic_gate_2026-05-08.json`
- `results/FittingDynamics_StageC_controlled_diagnostics/controlled_self_similarity_fitted_gate_2026-05-08.json`
- `results/FittingDynamics_StageC_controlled_diagnostics/controlled_self_similarity_fitted_seed42_123_plus_nonrepeat_hr_geometry_2026-05-08.json`

受控图像：

- positive：`css_periodic12.png`，一个 `12x12` RGB tile 精确平铺成 `48x48` 图像；
- negative：`css_nonrepeat12.png`，非周期平滑纹理控制图，没有故意构造 tile-phase duplicates；
- known duplicate group：patch start modulo tile size；
- 固定分析设置：`geometry_source=lr_up`、`geometry_descriptor=rgb_grad`、`response_mode=trajectory_delta`、`patch_size=7`、`stride=4`、`k=5`、`min_spatial_distance=8`、`n_shuffles=500`、`seed=0`。

Tier 0 synthetic response smoke：

| run | role | dup@5 | known effect | p_shuf | percentile | rho | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| synthetic periodic | positive known duplicate | 1.0000 | 1.0000 | 0.0020 | 0.0000 | 0.8451 | pass |
| synthetic nonrepeat | negative nonperiodic texture | 0.0694 | -0.0199 | 1.0000 | 0.5121 | 0.3398 | pass negative |

解释：patch extraction、known-group labeling、response descriptor 和统计 wiring 在无训练的已知正例中能工作。若这里失败，后续自然图解释没有意义；本轮没有失败。

Tier 1 fitted reduced LIIF x4：

| run | role | final PSNR | dup@5 | known effect | p_shuf | percentile | rho | std geom | content-only |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| periodic seed42 | positive | 36.56 | 1.0000 | 0.2181 | 0.0020 | 0.1847 | 0.4026 | 0.2867 | 0.2646 |
| periodic seed123 | positive | 36.86 | 1.0000 | 0.1675 | 0.0020 | 0.3058 | 0.0929 | 0.3688 | 0.3464 |
| periodic seed456 | positive | 35.97 | 1.0000 | 0.2710 | 0.0020 | 0.1525 | 0.4462 | 0.3454 | 0.3333 |
| nonrepeat seed42 | negative | 59.66 | 0.0661 | -0.0271 | 1.0000 | 0.5185 | 0.0109 | 0.0103 | -0.0035 |
| nonrepeat seed123 | negative | 59.18 | 0.0661 | -0.0180 | 0.9920 | 0.4950 | -0.0156 | 0.0132 | -0.0068 |
| nonrepeat seed456 | negative | 58.75 | 0.0661 | -0.0149 | 0.9880 | 0.5017 | 0.0369 | -0.0093 | -0.0035 |

Gate verdict：`fail_known_duplicate_stability_and_content_confound`，18 个检查中 4 个失败：

- periodic seed123 未通过严格 fitted-known-duplicate support：虽然 known effect 为正且 p_shuf 通过，但 Spearman `0.0929` 低于阈值，known-group response percentile `0.3058` 偏高；
- 三个 periodic seed 均未通过 `geometry_beats_content`：content-intensity-only control 几乎追上 geometry；
- 三个 nonrepeat seed 均没有误报，negative control 是健康的；
- HR geometry source 复核没有救回 seed123 的严格失败，说明问题不是简单 `lr_up` 下采样/aliasing。

审稿式判断：

> Controlled gate 支持“probe / full snapshot response reconstruction / patch extraction 不是坏的”：它能在 synthetic response 和 fitted exact-repeat 图像上检出已知重复结构，且 nonrepeat negative control 不误报。但它不支持“局部 geometry 机制已经成立”：exact-repeat 中 content-only control 接近 geometry，且 seed123 的 response percentile / Spearman 不稳定，说明当前 response trajectory 仍受内容统计、绝对位置、优化路径或 LIIF 内部表示影响。

对根因的更新判断：

- Stage B 平台和 raw full snapshot response 重建不是当前主要问题；
- 自然图 bird/head failure 不应再解释为“代码没读对 response”；
- 现有 Stage C 正例更可能是 image-internal repeated content / local appearance signal，而不是已经剥离 content/coordinate confound 的 geometry law；
- 当前根本问题不在研究方向本身被证伪，而在 H1 的操作化不足：`local patch appearance/geometry nearest neighbor -> raw output trajectory similarity` 这个定义没有充分对齐 LIIF response 的决定因素。

下一步边界：

- 不应继续用更多自然图 seed、descriptor、context rerank 或简单 LR-cell unit 稀释 failure；
- 不应启动等变性、pretrained LIIF/LIIF-EQ、update prediction 或训练加速；
- 若继续挽救核心目标，下一节点必须是 geometry-vs-content / coordinate 的解耦 gate，或者重新定义 response object 到更接近 LIIF decoder input / encoder feature trajectory 的 model-internal unit；
- 若不做更重定义，Stage C 应降级为 failure-audit / negative diagnostic 贡献，而不是声称发现跨图像稳定规律。
