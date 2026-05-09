# 阶段 A 研究问题与可证伪假设

## 0. 目的

本文档用于在当前研究方向校准后，明确项目的核心科学问题、研究对象、可证伪假设、非目标、负对照和未来论文 claim 的最低证据标准。

所有表述均为待验证假设或计划问题，除非明确说明已由当前仓库证据支持。

---

## 1. 中心研究问题

在任意尺度 INR 超分中，图像局部几何是否能够预测模型参数空间或响应空间中的一部分适配变化？等变性是否会让这种关系更结构化、更可迁移，或更适合用于减少训练？

该问题分为三个递进层级：

1. **观测层**：参数/响应变化能否被可靠测量？
2. **对应层**：局部几何相似性是否对应 response/update similarity？
3. **预测层**：是否存在一部分适配可以由 geometry 和 scale 推断，而不是从头训练？

---

## 2. 研究对象

### 2.1 图像侧对象

候选局部几何描述符：

- raw patch similarity；
- local frequency spectrum；
- edge orientation；
- gradient histogram；
- structure tensor；
- texture periodicity；
- cross-scale recurrence；
- learned patch embedding（只能作为受控辅助描述符，不能替代可解释几何定义）。

### 2.2 模型侧对象

候选 response/update 描述符：

- full parameter update；
- encoder-only update；
- decoder-only update；
- aligned decoder update；
- local gradient contribution；
- decoder output response；
- decoder Jacobian with respect to coordinate/cell/feature；
- feature response；
- future adapter/modulation update。

项目不能预设 full flattened parameter vector 是最佳研究对象。

---

## 3. 可证伪假设

## H1：局部几何对应

**假设**：几何相似的 patch pair 相比随机 patch pair，具有更相似的 INR response/update descriptor。

**可能支持证据**：

- geometry similarity 与 response/update similarity 存在统计显著相关；
- response space 的 nearest neighbors 能以高于随机的概率恢复 geometry-neighbor pairs；
- geometry cluster 内 response variance 低于 cluster 间 variance。

**可能证伪结果**：

- geometry-similar pairs 与 random pairs 的 response/update similarity 无差异；
- 相关性在多 seed、跨图像或跨 scale 下消失；
- 信号只存在于手选样例，整体统计不成立。

## H2：尺度条件结构

**假设**：邻近 scale 或相关 scale 之间的 response/update direction 存在结构化、部分可复用模式。

**可能支持证据**：

- 相邻 scale 的 update/response direction 相似性高于 random scale-label control；
- low-rank 或简单 scale-conditioned 预测在 unseen scale 上优于 trivial baseline；
- geometry-matched patch groups 的 scale-conditioned residual 更小。

**可能证伪结果**：

- scale-conditioned changes 主要由图像内容或优化噪声主导；
- linear/low-rank scale prediction 不优于 random 或 constant baseline；
- 结果只在单图或窄 scale range 中成立。

## H3：等变性正则化对应关系

**假设**：等变架构使 local geometry-to-response/update correspondence 更稳定、更低维或更可迁移。

**可能支持证据**：

- matched LIIF-EQ 的 geometry-response correlation 高于 LIIF；
- geometry-conditioned prediction residual 更低；
- cross-scale 或 cross-image transfer 更好；
- 即便 full parameter space 信号弱，response/feature space 中仍出现更清晰规律。

**可能证伪结果**：

- LIIF-EQ 提升输出质量，但不提升 response/update predictability；
- 差异可由参数量、buffer policy、架构不匹配或 alignment artifact 解释；
- 非等变模型同样存在该规律。

## H4：部分更新可预测

**假设**：适配更新中的某个子集可由 local geometry 和 target scale 预测，从而降低 full training 成本。

**可能支持证据**：

- predicted update + residual training 比 full fine-tuning 或 decoder-only fine-tuning 更快达到 target PSNR；
- predicted component 泛化到 unseen image/scale；
- residual update norm 稳定降低。

**可能证伪结果**：

- predicted update 不减少训练步数或计算量；
- 提升在公平 baseline 下消失；
- 预测只记住 seen image/scale pair。

---

## 4. 实验必须回答的问题

### Stage B：观测可靠性

1. 现有 trajectory recording 是否跨 seed 可复现？
2. PC1/effective dimension 对 snapshot count 和 alignment 有多敏感？
3. 现有脚本是否记录足够 metadata 支持复现？
4. 当前 LIIF-EQ state_dict / buffer policy 下，snapshot 是否可比？
5. [`aggregate_results.py`](../../../experiments/Phase1_FittingDynamics/aggregate_results.py) 是否能正确揭示 PC1 相对 random baseline 的强弱？

### Stage C：局部几何对应

1. 哪类 geometry descriptor 最能预测 response/update similarity？
2. 关系出现在 parameters、responses、features 还是 Jacobians？
3. 关系是 image-specific 还是 cross-image？
4. 关系是否跨 scale 保持？
5. 等变性是否在 matched controls 下增强这种关系？

### Stage D：预测更新

1. 哪个参数子集或 response object 足够可预测？
2. low-rank、adapter、bias、modulation prediction 是否优于简单 baseline？
3. 预测是否降低 unseen images/scales 上的适配成本？
4. 哪些纹理或几何类型失败？

---

## 5. 非目标与反声明

项目不应声称：

- PCA 图看起来有结构就说明参数轨迹有深层意义；
- spectral bias 是本项目新发现；
- 没有受控实验时等变性已经解释参数空间；
- 群变换是研究中心；
- 在直接测试前已经实现训练加速。

---

## 6. 必须负对照

每个主要实验应尽可能包含：

- random patch pairs；
- shuffled patch labels；
- shuffled scale labels；
- 同 snapshot 数量的 random trajectories；
- no-alignment vs alignment；
- same architecture different seed；
- architecture-matched non-equivariant baseline；
- self-reconstruction vs SR task distinction；
- feature-only vs parameter-only analysis。

---

## 7. 未来论文 claim 的最低证据标准

一个 claim 至少应满足：

1. 不是已经熟知的 generic phenomenon；
2. 支持来自多个 image 或 seed；
3. 包含负对照；
4. 明确 effect 出现在 parameter、response、feature 还是 update space；
5. 至少击败一个公平 baseline；
6. 报告 failure cases。

---

## 8. 当前阶段 A 结论

当前最有潜力的科学主张不是：

> INR 参数轨迹是低维的。

更有潜力但尚未证明的主张是：

> 图像局部几何自相似性在任意尺度 INR 超分中诱导可测量、部分可复用的 response/update 结构；等变性可能正则化这种结构，并使部分适配可预测。

该主张仍未被证明，是下一阶段实验的核心目标。
