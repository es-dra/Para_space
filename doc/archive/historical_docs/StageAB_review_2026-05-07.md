# Stage A/B 重新审视记录

日期：2026-05-07

本文档记录对当前研究工作流、阶段 A 文献与问题定义、阶段 B 结果，以及进入 Stage C 前置条件的重新审视。目标是把可靠结论、坏消息和下一步推进边界说清楚。

---

## 1. 对工作流执行的反思

用户要求是：

- 先读本地证据，不迎合式下结论；
- 阶段 A 要有研究问题、文献缺口、可证伪假设和负对照；
- 阶段 B 只证明平台可靠，不把 PCA 当科学结论；
- 当前正式实验不用 pretrained LIIF/LIIF-EQ，避免过参数化与 checkpoint confound；
- 使用 scratch reduced LIIF；
- 不保留一次性 launcher；
- 分析工具、测试、正式文档可以保留；
- 结论要能帮助建立真实信心，但不能吹。

过去工作的完成点：

- 删除了一次性 Stage B launcher，文档改为可复现命令模板；
- 完成 10 个 scratch run 的输出检查、聚合和控制分析；
- 新增 Stage B 控制分析工具与测试；
- 新增 Stage C pilot geometry-response 分析工具与测试；
- 第一版 descriptor ablation 已完成，并记录了 bird failure。

过去工作的遗漏或需要纠正处：

- 阶段 A 文献核查不够及时。之前把 MetaSR、CiaoSR、SRNO、UltraSR 等列为待查证后，没有马上补一个最小 bibliographic verification 表；
- [`doc/StageB_observation_protocol.md`](StageB_observation_protocol.md) 仍保留了早期 “pretrained LIIF/LIIF-EQ smoke run” 的进入 Stage C 前置表达，与当前 scratch reduced LIIF 约束不完全一致；
- Stage B 结果已经足以说明平台可靠，但此前文档中 Stage C pilot 进展和 Stage B 结论混在同一个文件，容易让读者误以为 Stage C 已经正式展开；
- 还没有单独形成“专家审视”文档，导致好消息、坏消息、下一步门槛分散在日志中；
- Stage A 的 paper/bib 条目中至少 LIIF 作者信息与公开 CVPR 页面不一致，本轮已修正；其余新增 bib 仍需在论文写作前统一核查格式和字段。

本轮已修正：

- 在 [`doc/StageA_literature_baseline.md`](StageA_literature_baseline.md) 增加 2026-05-07 文献核查补记；
- 在 [`doc/StageB_observation_protocol.md`](StageB_observation_protocol.md) 标明该协议是早期预实验协议，正式状态以 observation log 和本文档为准；
- 本文档明确当前阶段判断和下一步推进原则。

---

## 2. Stage A 当前状态

Stage A 的核心研究问题是清楚的：

> 任意尺度 INR 超分中，局部图像几何是否对应模型 response/update 的相似结构；等变性是否让这种结构更稳定、更可迁移，最终是否能减少适配训练。

Stage A 的好结果：

- 研究主线已经从“参数轨迹 PCA 低维”纠偏到“local geometry-to-response/update correspondence”；
- 可证伪假设 H1-H4 已写清楚；
- 非目标和反声明明确，尤其是不把 PCA、spectral bias、等变性或训练加速提前写成 claim；
- 负对照清单基本正确：random patch pair、shuffled labels、random trajectory、alignment/no-alignment、same architecture different seed 等；
- 文献地图已覆盖必要类别：任意尺度 SR、INR/神经场、等变性、图像内部自相似、参数高效适配、优化动力学。

Stage A 的坏消息或未闭合点：

- 文献核查还不是论文级完整 related work；
- 当前仓库的 pretrained LIIF-EQ / SE-INR 来源和 checkpoint 细节仍不清楚，不能作为解释性证据；
- `paper/refs/main.bib` 至少存在需要修正的条目；
- 还没有形成最终 baseline 表：哪些只做背景，哪些作为 Stage C/D 必跑实验；
- “是否已有工作直接研究 patch geometry 到 INR response/update similarity” 仍需要进一步检索。

Stage A 结论：

> Stage A 足以支撑 Stage B 和 Stage C pilot，但还不足以支撑论文写作。下一步不应先写强相关工作，而应边做 Stage C pilot，边补最小文献核查表和正式 bib。

---

## 3. Stage B 当前状态

正式结果目录：

```text
results/FittingDynamics_StageB/
```

正式实验对象：

- SIREN scratch：baby seeds 42/123/456，bird seed42，butterfly seed42；
- LIIF reduced scratch SR x4：baby seeds 42/123/456，bird seed42，butterfly seed42；
- LIIF 为 `LIIF_CONFIG_REDUCED`，`n_params=138531`；
- 未使用 pretrained LIIF/LIIF-EQ。

Stage B 的好结果：

- 10 个正式 scratch run 全部成功；
- 所有输出通过 schema、shape、summary consistency、analysis readiness 检查；
- summary 记录了 reduced LIIF 配置，避免与 full LIIF 或 pretrained LIIF 混淆；
- 聚合器能扫描正式输出并跳过非结果目录；
- SIREN 与 LIIF 的 trajectory recording 都是可读、可查、可复现的；
- baby 多 seed 已有，能初步看 seed stability；
- baby/bird/butterfly 多图像已有，能初步看 image specificity；
- 控制分析显示 observed trajectory 强于 norm-matched random walk；
- observed step cosine 高于 permuted update control，说明真实训练顺序存在连续方向性；
- LIIF encoder/decoder update ratio 约 0.12-0.16，说明 scratch reduced LIIF 的适配变化主要集中在 decoder 侧，这为 Stage C 优先分析 response/decoder object 提供了工程依据。

Stage B 的坏结果或限制：

- PC1 高不是科学结论。LIIF observed PC1 与 permuted-update PC1 很接近，说明高 PC1 很可能由 update vector 集合、step norm schedule 或端点方向主导；
- snapshot 数量少，PCA 本身很容易受采样策略影响；
- SIREN 是 self-reconstruction，PSNR 接近饱和，不能代表 SR response/update 规律；
- LIIF bird 的 final PSNR 明显低于 baby/butterfly，说明该图上的 fitting quality 是一个潜在 confound；
- raw/aligned PC1 差异为 0 只说明当前 PC1 不受 alignment 影响，不说明后续 parameter-space comparison 可忽略 alignment；
- Stage B 没有证明局部几何对应、等变性、scale update 可复用或训练加速。

用直白话讲：

> Stage B 真正证明的是：我们现在有一个能稳定记录、检查、重建和分析 scratch fitting dynamics 的平台。它没有证明 idea 成立，但排除了“工具不可靠导致后续分析没意义”这个大风险。Stage B 也给了一个重要警告：不要把高 PC1 当发现，尤其是 LIIF 的高 PC1 经不起 update-order permutation control 的解释压力。

Stage B 结论：

> Stage B 平台层面完成。科学层面只给出进入 Stage C pilot 的资格，不给论文 claim。

---

## 4. Stage C pilot 对 Stage A/B 的反馈

虽然用户当前要求重点是 Stage A/B，但已有 Stage C pilot 对阶段判断有用。

Stage C pilot 的好结果：

- raw `full_snapshots` 可重建 LIIF 函数响应，final PSNR 与 summary 误差为 `0.00000`；
- 这排除了 response probe 读取错误或 state reconstruction 错误；
- baby 三个 seed 上，geometry-neighbor response distance 均小于 shuffled response-label control；
- descriptor ablation 显示 baby 信号主要由 RGB / gradient / rgb_grad 承载，不是单个混合 descriptor 偶然造成；
- butterfly 有弱正信号；
- response-object 拆分显示 baby 信号在 output trajectory、output final、feature trajectory 和 coordinate-Jacobian final response 中都保留；
- head/woman 补齐后，woman 在 output trajectory、feature trajectory、coord-Jacobian final 三个对象上均为正，提供了 baby 之外的新支持图像。

Stage C pilot 的坏结果：

- bird 是明确 failure case，五种 descriptor 都不能产生正效应；
- Spearman rho 整体低，说明不是强全局单调相关；
- structure tensor 和 local spectrum 当前不强；
- 现有结果更像 image/content-dependent local nearest-neighbor 现象，不像跨图像稳定规律；
- feature response 只把 bird 从负效应推到近零小正效应，仍不显著；
- coordinate-Jacobian final response 在 bird 上出现小正效应，但效应很弱，不能当强规律；
- head 的 output trajectory 不支持正效应，只在 feature trajectory 和 coord-Jacobian final 上有弱正信号；
- 还没有 scale-conditioned response 的证据。

对主 idea 的当前信心判断：

> 方向仍可研究，但当前方法没有稳定捕捉到主旨中假设的统一规律。信心来自 baby 多 seed 和 woman 的正信号，以及 response 重建链路可靠；限制来自 bird/head failure、Spearman 全局相关偏弱、coord-Jacobian 多数只是微小效应。现阶段不能把正负例并存解释成“规律强弱不同”，更合理的判断是：descriptor、response object、patch matching 或 fitting quality 中至少有一环还不充分。

---

## 5. 当前可以确定的结论

可以确定：

- 当前正式实验没有使用 pretrained LIIF/LIIF-EQ；
- scratch reduced LIIF 输出完整，且配置可追踪；
- Stage B 观测平台可用；
- PCA 只能作为诊断，不是主 claim；
- LIIF 的高 PC1 不能解释为几何对应；
- response-space probe 比 raw parameter-space 更适合作为下一阶段主线；
- baby 上存在重复出现的 geometry-neighbor response 正信号；
- Set5 seed42 已覆盖 baby、bird、butterfly、head、woman；woman 是新增支持图像，head/bird 是边界或失败图像；
- feature/Jacobian response object 值得继续作为 Stage C pilot 对象；
- bird 是必须解释的 failure case。

不能确定：

- geometry-response correspondence 是否跨图像稳定；
- 等变性是否增强 correspondence；
- 哪个 response object 最可预测；
- 是否能预测 update 或减少训练；
- 当前 LIIF-EQ wrapper 是否适合做 matched control。

---

## 6. 下一步推进原则

当前不建议：

- 不直接跑完整 Stage C；
- 不直接引入 pretrained LIIF/LIIF-EQ；
- 不直接上等变性比较；
- 不围绕 PCA 继续挖故事；
- 不写训练加速 claim；
- 不新增一次性调度脚本。

当前建议：

1. **先补 Stage A 最小文献/bib 修正**
   - 本轮已修正 LIIF 并补入 LTE、MetaSR、ZSSR 等核心条目的最小 bib；
   - 标记 LIIF-EQ / SE-INR 来源未核清；
   - 不需要一次性写完整 related work。

2. **把 Stage B 封版为平台完成**
   - Stage B 不再继续扩大 PCA 分析；
   - 只保留必要控制工具；
   - 后续引用 Stage B 时只说平台可靠和 PC1 风险。

3. **Stage C 下一步先做 failure audit，而不是继续扩展**
   - response-object 拆分第一版已完成；
   - 严格审稿口径下，多数正结果是 mixed / weak，而非一致支持；
   - coord-Jacobian 的小正效应不能作为强证据；
   - 下一步应先解释 bird/head 为什么失败，再决定是否扩展更多 seed 或新模型。

4. **扩展数据前先定义成功/失败标准**
   - baby 多 seed 正信号保留；
   - bird failure 必须记录和解释；
   - head/woman 已完成小规模扩展；
   - 新图结果分裂：woman 支持，head output 失败但 feature/Jacobian 弱正；
   - 后续若继续出现 head/bird 类 failure，要诚实降低主 claim。

5. **等变性比较必须延后**
   - 等 LIIF-EQ 来源、checkpoint、buffer policy、alignment/function-equivalence 问题处理清楚后再做；
   - matched control 需要参数量、训练协议、data pair 和 response probe 都可比。

---

## 7. 推荐的立即行动

最稳的下一步是：

1. 对本轮新增核心 bib 做一次完整格式核验，并继续查清 LIIF-EQ / SE-INR 来源；
2. 将 Set5 全图 seed42 的 response-object 结果整理成正式 Stage C pilot 表，并明确哪些只是微小效应；
3. 对 bird/head failure 做方法审计：边缘强度、纹理/平滑区域、低频误差、response norm、PSNR、patch descriptor 区分力是否解释失败；
4. 暂缓 baby/woman additional seed，除非 failure audit 能说明当前指标定义合理；
5. 只有 failure audit 通过后，再考虑更多 seed、full trajectory Jacobian 或等变性 matched control。

---

## 9. 2026-05-08 追加：controlled self-similarity gate 后的根因判断

本轮新增受控自相似 sanity gate 后，Stage C 的问题边界更清楚：

- synthetic response smoke 通过，说明 patch extraction、known-group labeling、response descriptor 和统计 wiring 在已知正例中能工作；
- fitted reduced LIIF exact-repeat periodic 图像在 seeds 42/123/456 上均有正向 known-duplicate signal，且 nonrepeat seeds 42/123/456 均不误报；
- 但 fitted gate 总体未通过，verdict 为 `fail_known_duplicate_stability_and_content_confound`；
- 关键失败不是负控误报，而是：
  - periodic seed123 的 Spearman 与 known-group response percentile 未过严格阈值；
  - 三个 periodic seed 中 content-intensity-only control 都接近 geometry。

因此当前更可靠的判断是：

> Stage B 平台与 Stage C response 重建链路不是主要故障点；当前失败来自 Stage C 的操作化定义不足。现有自然图正例更像 image-internal repeated content / local appearance signal，而不是已经排除 content/coordinate confound 的 geometry-response law。

这并不等于研究方向被证伪。更准确地说，当前 H1 的实现版本：

```text
local patch appearance/geometry nearest neighbor -> raw output trajectory similarity
```

没有通过审稿级主证据标准。后续若继续挽救核心目标，应转向 geometry-vs-content / coordinate 解耦 gate，或重新定义 response object 到更接近 LIIF decoder input / encoder feature trajectory 的 model-internal unit；不应继续扩自然图、等变性或训练加速叙事。

这一路线的优点是：它先审查方法是否真的能测到主旨中的机制，避免用更多实验掩盖指标不充分的问题。

---

## 8. 2026-05-07 追加：审稿人式 Stage C failure audit

新增只读审计工具：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_failure_audit.py \
  --results_dir results/FittingDynamics_StageB \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad \
  --response_modes trajectory_delta,feature_trajectory_delta,coord_jacobian_final_delta \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0
```

审计规则故意保守：微小正效应、fractional gain 太小、p 值不过关或 Spearman 近零，都不能当成强支持。

审计结果：

| image | seed | verdict | support modes | negative | tiny | mean effect |
|---|---:|---|---:|---:|---:|---:|
| baby | 123 | consistent_support | 3/3 | 0 | 0 | 0.0846 |
| baby | 42 | mixed_weak_support | 2/3 | 0 | 1 | 0.0574 |
| baby | 456 | mixed_weak_support | 2/3 | 0 | 1 | 0.0495 |
| woman | 42 | mixed_weak_support | 2/3 | 0 | 1 | 0.0453 |
| butterfly | 42 | mixed_weak_support | 2/3 | 0 | 0 | 0.0347 |
| bird | 42 | failure_or_unresolved | 0/3 | 1 | 2 | -0.0056 |
| head | 42 | failure_or_unresolved | 0/3 | 1 | 2 | 0.0069 |

关键判断：

- 只有 `baby seed123` 在三个 response object 上达到一致支持；
- baby 其他 seed、woman、butterfly 都是 mixed/weak，不是强结论；
- bird/head 是明确失败或未解决，不应被弱 feature/Jacobian 正数掩盖；
- bird 的 final PSNR 低，存在 fitting-quality confound；
- head 的 final PSNR 高但 output response 仍失败，说明问题不只是拟合质量，当前 geometry/response 定义本身可能不足；
- head 的 patch gradient/variance 较低，提示平滑或弱结构图像可能让当前 descriptor 缺乏区分力；
- coord-Jacobian 小正效应经常被审计为 tiny，不能自我包装成强证据。

更新后的结论：

> Stage C pilot 目前是“可疑但有线索”，不是“扎实进入下一阶段”。下一步必须先做方法审计和 failure-case 解释；否则继续扩展实验只会扩大不确定性。

---

## 9. 2026-05-07 追加：内容分层 failure audit

新增诊断命令：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_failure_audit.py \
  --results_dir results/FittingDynamics_StageB \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad \
  --response_modes trajectory_delta,feature_trajectory_delta,coord_jacobian_final_delta \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0 \
  --content_stratification \
  --format table
```

分层规则：

- 按 LR-up patch 的平均梯度强度和 patch 方差分别做 low / mid / high tercile；
- 每个 stratum 内重复同一 geometry-neighbor vs shuffled-response control；
- 该分析只用于解释 failure，不用于挑选有利子集写 claim。

关键结果：

| image | seed | high-gradient support | high-gradient mean effect | high-variance support | high-variance mean effect |
|---|---:|---:|---:|---:|---:|
| baby | 42 | 3/3 | 0.0734 | 3/3 | 0.0620 |
| baby | 123 | 3/3 | 0.0913 | 3/3 | 0.0878 |
| baby | 456 | 3/3 | 0.0683 | 3/3 | 0.0577 |
| woman | 42 | 3/3 | 0.0611 | 2/3 | 0.0695 |
| butterfly | 42 | 1/3 | 0.0190 | 2/3 | 0.0319 |
| bird | 42 | 0/3 | -0.0213 | 0/3 | -0.0178 |
| head | 42 | 0/3 | 0.0026 | 0/3 | 0.0056 |

审稿式解释：

- 正例不是全图均匀成立；baby/woman 的主要信号集中在高梯度或高方差 patch；
- bird/head failure 没有被高梯度或高方差分层救回，因此不能说它们只是被平滑区域稀释；
- head final PSNR 高但仍失败，说明当前问题不只是拟合质量；
- bird final PSNR 低仍是 confound，但分层结果显示即使只看高内容强度 patch，也没有恢复支持；
- butterfly 仍是弱支持，不应与 baby/woman 放在同一证据等级。

更新后的阶段判断：

> 当前最扎实的 Stage C 线索是：在 scratch reduced LIIF 中，baby 多 seed 和 woman seed42 的高内容强度 patch 存在局部 geometry-response correspondence。当前最扎实的反证是：bird/head 在同样审计下仍失败，所以“统一规律已经被当前 probe 捕捉到”这个说法不成立。下一步必须先改进 patch geometry/response 定义或定位非唯一匹配问题，再决定是否扩展 Stage C。

---

## 10. 2026-05-07 追加：bird/head patch-level failure 定位

新增同一只读审计入口下的 patch-level 诊断：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_failure_audit.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_bird_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_head_sr4_seed42 \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad \
  --response_modes trajectory_delta,feature_trajectory_delta,coord_jacobian_final_delta \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0 \
  --patch_failure_diagnostics \
  --patch_failure_top_n 10 \
  --format table \
  --output_json results/FittingDynamics_StageC_diagnostics/bird_head_patch_failure_diagnostic_2026-05-07.json \
  --patch_crop_dir results/FittingDynamics_StageC_diagnostics/patch_crops_2026-05-07 \
  --patch_crop_scale 14
```

正例参考保存为：

```text
results/FittingDynamics_StageC_diagnostics/baby123_woman42_patch_failure_reference_2026-05-07.json
results/FittingDynamics_StageC_diagnostics/patch_crops_2026-05-07/
```

核心诊断：

| run | response | top1 response percentile mean | p90 | worst patch frac | strict ties frac | top2 close frac |
|---|---|---:|---:|---:|---:|---:|
| bird | output trajectory | 0.527 | 0.912 | 0.281 | 0.066 | 0.306 |
| bird | feature trajectory | 0.401 | 0.847 | 0.132 | 0.066 | 0.306 |
| bird | coord-Jacobian final | 0.402 | 0.833 | 0.132 | 0.066 | 0.306 |
| head | output trajectory | 0.456 | 0.865 | 0.190 | 0.107 | 0.281 |
| head | feature trajectory | 0.415 | 0.856 | 0.149 | 0.107 | 0.281 |
| head | coord-Jacobian final | 0.444 | 0.807 | 0.107 | 0.107 | 0.281 |
| baby seed123 | output trajectory | 0.252 | 0.613 | 0.041 | 0.083 | 0.314 |
| woman seed42 | output trajectory | 0.337 | 0.728 | 0.066 | 0.058 | 0.248 |

这里 `top1 response percentile` 是每个 anchor patch 的 geometry top-1 近邻在同一 anchor 的 eligible response-distance 分布中的分位；越高表示几何近邻在 response 上越不像近邻。`worst patch frac` 是分位 `>=0.8` 的 patch 比例。`strict ties frac` 是 top-k 内至少两个 geometry candidate 与 top-1 距离在 1% 内的比例；`top2 close frac` 是 top2/top1 geometry distance `<=1.05` 的比例。

审稿式解释：

- bird/head failure 可以定位到 patch 层：geometry top-1 近邻经常对应 response 上相对远的 patch。这个现象在正例参考 baby seed123 / woman seed42 中明显弱得多。
- 对上一版诊断的修正：此前的非唯一近邻定义过严，已拆成 strict ties 与 top2 close。top2 close 在 bird/head 和 baby/woman 中都不低，因此“近邻候选接近”是普遍现象，不能单独解释 failure，也不能完全排除。
- 严重 response 错配与 strict ties 的重叠仍很小，因此最坏错配不主要来自“多个 candidate 几乎完全等距”；但语义层面的多义、aliasing 或重复纹理仍未排除。
- 严重错配 patch 的 response norm rank 接近中位或略高，不像单纯由 response norm 极端异常造成。
- bird 的 output-trajectory 错配更强，并有 bottom-right 区域集中迹象；head 的错配较分散。head final PSNR 高，进一步说明 failure 不只是 fitting quality。
- patch crop grid 目检显示，bird/head 的最坏 output-trajectory 匹配常见“小 patch 色块/边缘相似，但上下文位置和内容结构明显不同”的情况。这支持当前局部 descriptor 缺少上下文或结构约束的解释，但还不是 aliasing/语义机制的证明。

更新后的阶段判断：

> 现在可以更具体地说：bird/head 失败主要表现为局部外观近邻与 fitting response 近邻脱钩；可视化提示局部 descriptor 缺少上下文或结构约束。近邻候选接近现象本身是普遍的，不能单独解释 failure；response norm 异常也不像主因。下一步应优先检查上下文增强或结构化 patch matching、aliasing/scale-aware descriptor 或 response object 定义，而不是继续扩大图像/seed 或引入等变性比较。

---

## 11. 2026-05-08 追加：上下文增强 descriptor 对照

新增 `rgb_grad_context` 作为最小只读方法审计：

- 原 `rgb_grad` 使用 7x7 patch 的 mean-centered RGB + gradient；
- `rgb_grad_context` 拼接一个 3 倍边长上下文 crop 的同类 descriptor；
- context factor 固定为 3，不做超参数搜索；
- 目标是检查 patch crop 诊断中暴露的“局部外观相似但上下文结构不同”是否能被简单上下文补充缓解。

命令：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_failure_audit.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_bird_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_head_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_baby_sr4_seed123 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_woman_sr4_seed42 \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad_context \
  --response_modes trajectory_delta,feature_trajectory_delta,coord_jacobian_final_delta \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0 \
  --patch_failure_diagnostics \
  --patch_failure_top_n 10 \
  --format table \
  --output_json results/FittingDynamics_StageC_diagnostics/context_descriptor_audit_2026-05-08.json
```

关键结果：

| run | response | rgb_grad effect | context effect | label change |
|---|---|---:|---:|---|
| bird | output trajectory | -0.0341 | -0.0071 | negative -> negative |
| bird | feature trajectory | 0.0082 | 0.0952 | tiny -> support |
| head | output trajectory | -0.0063 | 0.0297 | negative -> tiny |
| head | feature trajectory | 0.0206 | 0.0622 | tiny -> support |
| baby seed123 | output trajectory | 0.1318 | 0.1140 | support -> support |
| baby seed123 | feature trajectory | 0.0838 | 0.1268 | support -> support |
| woman seed42 | output trajectory | 0.0649 | 0.1037 | support -> support |
| woman seed42 | feature trajectory | 0.0458 | 0.1201 | support -> support |

patch mismatch 变化：

| run | response | top1 pct mean | worst frac | top2 close frac |
|---|---|---:|---:|---:|
| bird output | rgb_grad -> context | 0.527 -> 0.490 | 0.281 -> 0.231 | 0.306 -> 0.661 |
| head output | rgb_grad -> context | 0.456 -> 0.412 | 0.190 -> 0.124 | 0.281 -> 0.620 |
| baby output | rgb_grad -> context | 0.252 -> 0.310 | 0.041 -> 0.116 | 0.314 -> 0.496 |
| woman output | rgb_grad -> context | 0.337 -> 0.345 | 0.066 -> 0.025 | 0.248 -> 0.413 |

审稿式解释：

- 上下文确实是有用线索，尤其 feature trajectory：bird/head 从 tiny 变成 support，baby/woman 的 feature effect 也增强。
- output trajectory 没有被解决：bird 仍 negative，head 只是 tiny。
- coord-Jacobian 仍是 tiny，不能作为强机制证据。
- top2 close frac 显著升高，说明上下文 descriptor 让近邻候选更拥挤，可能引入新的多义风险。
- baby output patch mismatch 变差，说明上下文不是无害修复。

更新后的阶段判断：

> Stage C 当前最合理的方向不是继续堆图像，而是修复 matching/descriptor。`rgb_grad_context` 证明“上下文信息”值得继续检查，但它只改善 feature response，不足以修复 output response failure。下一步应做更结构化的 matching 诊断，例如上下文相似性门控、多候选 response oracle 或边缘方向一致性约束。

---

## 12. 2026-05-08 追加评审：top-k oracle 和可用 rerank 的分歧

新增证据：

- `results/FittingDynamics_StageC_diagnostics/topk_oracle_rgb_grad_2026-05-08.json`
- `results/FittingDynamics_StageC_diagnostics/topk_oracle_rgb_grad_context_2026-05-08.json`
- `results/FittingDynamics_StageC_diagnostics/topk_rerank_rgb_grad_to_context_2026-05-08.json`

核心结果：

| diagnostic | bird output | head output | baby seed123 output | woman output |
|---|---:|---:|---:|---:|
| `rgb_grad` top1 response pct mean | 0.527 | 0.456 | 0.252 | 0.337 |
| top-k response oracle pct mean | 0.175 | 0.148 | 0.076 | 0.124 |
| oracle gain | 0.352 | 0.308 | 0.176 | 0.213 |
| context rerank pct mean | 0.528 | 0.469 | 0.294 | 0.346 |
| context rerank gain | -0.000 | -0.013 | -0.042 | -0.010 |
| oracle is context-best frac | 0.198 | 0.207 | 0.264 | 0.240 |
| oracle farther than top1 in context frac | 0.512 | 0.463 | 0.438 | 0.430 |

评审判断：

- oracle 显示 `rgb_grad` top-5 候选中常有 response 上更合适的 patch，特别是 bird/head output。因而不能把 failure 简化成“descriptor 完全找不到候选”。
- 但 oracle 是 response-label 上界，不能作为可用方法或科学 claim。
- 真正 response-blind 的 context rerank 没有吃到 output trajectory 的 oracle headroom；bird/head output 基本不变或变差，baby/woman output 也未改善。
- 候选级解释显示，bird/head output 的 oracle 候选多数情况下不是 context descriptor 最近者，且约半数情况下在 context 空间比原 top1 更远。这说明失败不是简单“加上下文再排一次”能解决的问题。
- context 对 feature trajectory 仍有一定帮助，但该现象不能外推到 output response。

更新后的下一步建议：

1. 暂停扩大样本或引入等变性。当前瓶颈不是证据量，而是 matching 规则不能解释 failure。
2. 继续做 candidate-level、response-blind 的结构化诊断：
   - 边缘方向/structure tensor 一致性；
   - 低频上下文布局相似性；
   - top-k 内 geometry distance、context distance、content rank 的受控组合；
   - 检查 oracle 选中的候选与 top1 候选在这些解释性属性上的差异。
3. 成功标准应是：bird/head output `rerank_response_percentile_mean` 和 `worst_response_frac` 明显下降，同时 baby/woman 不明显变差，并能用 patch-level 属性解释。

当前结论边界：

> 现在可以说“top-k 候选集合存在 response headroom”，但不能说“上下文或当前 descriptor 已经能预测 response”。Stage C 仍处于 probe/matching 诊断阶段，不应升级为跨图像规律、等变性机制或训练加速主张。

---

## 13. 2026-05-08 追加评审：冻结 Stage C stop/go 门槛

由于最近工作已经暴露“oracle 有上界、context rerank 吃不到上界”的结构性问题，继续开放式尝试 descriptor 会增加 p-hacking 风险。本轮新增独立决策门槛文档：

- [`doc/StageC_decision_gate_2026-05-08.md`](StageC_decision_gate_2026-05-08.md)

冻结后的下一节点只有一个核心问题：

> 在 `rgb_grad` top-5 candidate 集合内，是否存在一个 response-blind、预注册的 matching/rerank 规则，能稳定利用 bird/head output trajectory 的 oracle headroom，同时不破坏 baby/woman 正例？

冻结协议：

- failure gate：`LIIF_reduced_bird_sr4_seed42`、`LIIF_reduced_head_sr4_seed42`；
- positive guardrail：`LIIF_reduced_baby_sr4_seed123`、`LIIF_reduced_woman_sr4_seed42`；
- primary response：`trajectory_delta`；
- geometry source：`lr_up`；
- baseline candidate set：`rgb_grad` top-5；
- `patch_size=7`、`stride=4`、`k=5`、`min_spatial_distance=8`、`n_shuffles=200`、`seed=0`。

成功标准摘要：

- bird/head output 均达到 `rerank_response_percentile_gain_mean >= 0.10`；
- 两者均至少恢复各自 oracle gain 的约 30%；
- bird `rerank_response_percentile_mean < 0.43`，head `< 0.36`；
- baby123 / woman42 不明显变差，即 gain `>= -0.02`；
- 同一规则适用于四个 run。

失败后降级：

> 当前 scratch reduced LIIF 中，某些图像和高内容强度 patch 存在局部 geometry-response 线索，但当前 patch matching / descriptor 协议不能形成跨图像可靠机制。

该门槛通过前，不启动完整 Stage C、等变性比较、pretrained LIIF/LIIF-EQ 或训练加速主张。

### 13.1 Gate 执行结论

本轮已执行冻结 gate，产物：

- `results/FittingDynamics_StageC_diagnostics/candidate_attribute_gate_2026-05-08.json`

primary response 为 `trajectory_delta`，结果：

| run | oracle gain | context rerank gain | recovered oracle gain |
|---|---:|---:|---:|
| bird output | 0.352 | -0.000 | -0.001 |
| head output | 0.308 | -0.013 | -0.041 |
| baby123 output | 0.176 | -0.042 | -0.237 |
| woman42 output | 0.213 | -0.010 | -0.045 |

判定：gate 未通过。bird/head 的 output rerank gain 均 `<0.05`，baby123 guardrail 也明显变差。

候选级属性审计没有找到足够稳定的 response-blind 区分：edge orientation、context low-frequency 和 context descriptor distance 对 bird/head output 通常不支持 oracle candidate；patch variance absdiff 只有小幅正向，不能构成可靠 rule。

更新后的审稿判断：

> 当前 Stage C 不应继续走开放式 descriptor/rerank 修补路线。更准确的结论是：scratch reduced LIIF 中有部分局部 geometry-response 线索，但当前 patch matching / descriptor 协议不能形成跨图像可靠机制。下一步若继续，应重新定义 response object 或 patch matching 单元；否则应将贡献降级为 Stage B 平台和 Stage C failure audit 框架。

### 13.2 Response-object audit 执行结论

candidate-level gate 失败后，本轮进一步检查“当前 output response object 是否因为幅值、归一化或时间聚合而掩盖规律”。该检查使用新的只读 CLI：

- `experiments/Phase1_FittingDynamics/analyze_stage_c_response_object_audit.py`

产物：

- `results/FittingDynamics_StageC_diagnostics/response_object_audit_2026-05-08.json`

固定协议仍为：

- runs：bird/head seed42 作为 failure gate，baby seed123 / woman seed42 作为 positive guardrail；
- geometry：`lr_up + rgb_grad`；
- patch：`patch_size=7`、`stride=4`、`k=5`、`min_spatial_distance=8`；
- controls：`n_shuffles=200`、`seed=0`；
- 只审查 output response object，不引入 pretrained、等变性、新训练或新 descriptor。

预注册变体：

- full trajectory：`trajectory_unit`、`trajectory_raw`、`trajectory_norm`；
- endpoint final delta：`final_unit`、`final_raw`、`final_norm`；
- temporal split：`trajectory_early_unit`、`trajectory_mid_unit`、`trajectory_late_unit`。

关键结果：

| variant | failure effect frac mean | failure patch pct mean | guardrail effect frac mean | guardrail patch pct mean | verdict |
|---|---:|---:|---:|---:|---|
| trajectory_unit | -0.0184 | 0.4918 | 0.0786 | 0.2943 | does_not_resolve_failure |
| trajectory_raw | -0.0159 | 0.4860 | 0.0752 | 0.3040 | does_not_resolve_failure |
| trajectory_norm | 0.0243 | 0.4909 | 0.0310 | 0.4651 | does_not_resolve_failure |
| trajectory_early_unit | -0.0201 | 0.4919 | 0.0788 | 0.2947 | does_not_resolve_failure |
| trajectory_mid_unit | -0.0020 | 0.4800 | 0.0200 | 0.4104 | does_not_resolve_failure |
| trajectory_late_unit | 0.0002 | 0.4903 | -0.0002 | 0.4958 | does_not_resolve_failure |

temporal consistency：

| group | segment pair Spearman mean | directed percentile std mean |
|---|---:|---:|
| bird/head failure | 0.1392 | 0.1987 |
| baby/woman guardrail | 0.1511 | 0.2009 |

审稿式解释：

- 主 response `trajectory_unit` 仍呈现原始格局：bird/head negative，baby/woman positive。
- raw trajectory 保留幅值后没有恢复 bird/head；norm-only 对 bird 是 tiny、对 head 是 negative，并且 guardrail 也被削弱。
- early/mid/late 固定时间段没有共同修复 bird/head；late 还会破坏 baby/woman guardrail。
- temporal consistency 在 failure 与 guardrail 之间接近，不能支持“full trajectory 聚合混掉阶段性结构”的解释。

结论边界：

> 当前预注册的 output response-object 分解没有解释 bird/head failure。简单幅值保留、log-norm-only、endpoint final delta 和固定三段时间切分都不能同时恢复 bird/head 且保护 baby/woman。

这不是“所有 response object 都失败”的证明；它只排除了当前最直接、预注册的 output response 修补路线。下一步若继续 Stage C 方法研究，应转向 patch matching 单元定义，而不是继续修补 descriptor/rerank 或简单 response object。

### 13.3 Patch matching 单元 gate 执行结论

为避免过早降级，本轮进一步测试一个最小 LIIF-aware patch matching 单元：

- 脚本：`experiments/Phase1_FittingDynamics/analyze_stage_c_patch_unit_gate.py`
- 产物：`results/FittingDynamics_StageC_diagnostics/patch_unit_gate_lr_cell_2026-05-08.json`
- 测试：`tests/test_stage_c_patch_unit_gate.py`

冻结 candidate：

> `lr_cell_query_support`：每个 LR feature cell 作为 matching unit；x4 SR 下对应 HR `4x4` output response block；geometry 使用该 LR cell 周围固定 `3x3` LR support 的 mean-centered RGB + grayscale gradient；response 固定为 output `trajectory_delta`。

该 gate 同时报告旧 HR7 baseline、coordinate-only、content-intensity-only 和 geometry-shuffled controls。

主结果：

| run | HR7 geometry | LR-cell geometry | LR-cell coordinate-only | LR-cell content-only |
|---|---|---|---|---|
| bird | negative / -0.0319 / 0.5275 | tiny / 0.0257 / 0.4507 | support / 0.1494 / 0.3540 | support / 0.1177 / 0.3791 |
| head | negative / -0.0048 / 0.4562 | negative / -0.0057 / 0.4669 | tiny / 0.0094 / 0.4729 | support / 0.0811 / 0.3909 |
| baby123 | support / 0.1070 / 0.2518 | support / 0.1156 / 0.3278 | negative / -0.0697 / 0.6064 | support / 0.1944 / 0.2874 |
| woman42 | support / 0.0502 / 0.3367 | local_only / 0.0595 / 0.3697 | support / 0.0238 / 0.4583 | support / 0.1554 / 0.3210 |

表格中每格格式为：`label / effect_vs_shuffle_frac / patch_top1_response_percentile_mean`。

Gate verdict：`fail`，24 个检查中 15 个失败。

审稿式解释：

- LR-cell unit 对 bird 有轻微改善，但只到 tiny，patch pct 仍为 `0.4507`，未达到 failure gate。
- head 没有改善，LR-cell geometry 仍为 negative。
- 更重要的是，LR-cell geometry 没有优于弱 controls：bird 中 coordinate-only / content-only 更强，head 和 baby/woman 中 content-only 也更强。
- 这说明当前 signal 很可能含有明显空间位置或内容强度结构；但这不能被写成 geometry-response 机制。

更新后的判断：

> 当前不只是 HR 7x7 patch 单元有问题；简单 LIIF-aware LR-cell/query-support 单元也不能形成可靠主证据。若继续挽救 Stage C，必须进入更重的 model-internal unit 定义，例如显式 LIIF decoder input / encoder feature trajectory 单元；否则应把 Stage C 定位为 failure audit / negative diagnostic，而不是主证明。
