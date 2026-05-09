# 阶段 A 文献基线：研究问题重建

## 0. 目的

本文档用于建立当前项目的阶段 A 文献基线。项目主题是：面向任意尺度 INR 超分辨率，研究图像局部几何相似性是否会诱导可复用、低复杂度或可预测的参数/响应变化，以及等变性是否能够使这些变化更结构化、更稳定或更可利用。

本文档不是结论文档，而是区分：

- 当前仓库中已经存在或可核查的起点；
- 与项目相关的研究缺口；
- 必须比较的 baseline；
- 尚不确定、需要进一步查证的文献信息。

核心问题：

> 在任意尺度 INR 超分中，图像局部几何相似性是否对应可复用、低复杂度或可预测的参数/响应变化？等变性是否让这种变化更结构化或更可利用？

---

## 0.1 2026-05-07 文献核查补记

本轮重新审视后，需要把阶段 A 的文献状态说得更硬一些：核心方向已经成立，但文献表还没有达到论文引用标准。以下条目已用公开论文页或作者/项目页做了第一轮核查，可作为后续正式 bib 修正的依据：

| 条目 | 核查状态 | 与本项目关系 |
|---|---|---|
| LIIF | 已核查：Yinbo Chen, Sifei Liu, Xiaolong Wang, CVPR 2021, pp. 8628-8638, <https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html> | 任意尺度连续图像表示的核心 baseline |
| LTE | 已核查：Jaewon Lee, Kyong Hwan Jin, CVPR 2022, pp. 1929-1938, <https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Local_Texture_Estimator_for_Implicit_Representation_Function_CVPR_2022_paper.html> | 局部频率/纹理 descriptor 与 response-space probe 的重要参照 |
| Meta-SR | 已核查 arXiv:1903.00875，<https://arxiv.org/abs/1903.00875> | 任意尺度 SR 的早期 scale-conditioned baseline |
| ArbSR | 已核查：Learning A Single Network for Scale-Arbitrary Super-Resolution, ICCV 2021，项目页/代码页记录 <https://replicate.com/longguangwang/arbsr> | scale-arbitrary SR baseline，不是 INR response 解释工作 |
| CiaoSR | 已核查：CVPR 2023/arXiv:2212.04362，<https://arxiv.org/abs/2212.04362> | implicit attention / non-local feature ensemble baseline |
| SRNO | 已核查：CVPR 2023 poster，<https://cvpr.thecvf.com/virtual/2023/poster/21373> | neural-operator 视角的 arbitrary-scale SR baseline |
| UltraSR | 已核查 arXiv:2103.12716，<https://arxiv.org/abs/2103.12716> | spatial / periodic encoding 对 implicit image function 的影响 |
| A-LIIF | 已核查 arXiv:2208.04318，<https://arxiv.org/abs/2208.04318> | 用多个 local implicit functions 处理边缘/ringing，可作为 LIIF 局部函数局限的相关工作 |
| LINF | 已核查 arXiv:2303.05156，<https://arxiv.org/abs/2303.05156> | arbitrary-scale SR 中的 flow / perceptual-quality 方向，不直接解释 fitting dynamics |
| CLIT / LIT | 已核查 arXiv:2303.16513，<https://arxiv.org/abs/2303.16513> | local implicit transformer / cross-scale local attention 方向，提示 attention baseline 需要覆盖 |
| DIIF | 已核查 arXiv:2306.12321，<https://arxiv.org/abs/2306.12321> | 加速 implicit arbitrary-scale decoding 的相关工作，和本项目未来“减少适配成本”目标相关 |
| Thera | 已核查 arXiv:2311.17643 / 项目页，<https://therasr.github.io/> | anti-aliasing / neural heat field 方向，提醒 Stage C 后续要区分几何对应与 aliasing artifact |
| Continuous Optical Zooming | 已核查：CVPR 2024, pp. 3035-3044，<https://openaccess.thecvf.com/content/CVPR2024/html/Fu_Continuous_Optical_Zooming_A_Benchmark_for_Arbitrary-Scale_Image_Super-Resolution_in_CVPR_2024_paper.html> | real-world arbitrary-scale benchmark，影响后续数据集与退化模型选择 |
| DIIN | 已核查：IJCAI 2025, pp. 855-863，<https://www.ijcai.org/proceedings/2025/96> | diffusion/iterative implicit network 方向，属于较新的 arbitrary-scale SR baseline 候选 |
| Scale-Equivariance Pursuit / EQSR | 已核查：CVPR 2023, pp. 1786-1795，<https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Deep_Arbitrary-Scale_Image_Super-Resolution_via_Scale-Equivariance_Pursuit_CVPR_2023_paper.html> | 等变性相关 SR baseline，但不能直接等同于当前仓库的 LIIF-EQ wrapper |
| ZSSR | 已核查：Shocher, Cohen, Irani, CVPR 2018，<https://openaccess.thecvf.com/content_cvpr_2018/html/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.html> | 内部图像统计 / test-time image-specific learning 的关键参照 |
| PatchMatch | 已核查：ACM TOG / SIGGRAPH 2009，<https://research.adobe.com/publication/patchmatch-a-randomized-correspondence-algorithm-for-structural-image-editing/> | patch matching / internal correspondence 背景 |
| Single-image internal recurrence | 已核查作者页：<https://www.wisdom.weizmann.ac.il/~vision/SingleImageSR.html> | “自然图像 patch recurrence” 是本项目 geometry hypothesis 的经典背景 |

需要明确纠正或继续核查：

- 当前 [`paper/refs/main.bib`](../../../paper/refs/main.bib) 中 `li2021learning` 的作者信息已在本轮修正；任意尺度 SR 新增条目仍需在论文写作前做一次完整 bib 风格和字段核查；
- 当前 pretrained LIIF-EQ / SE-INR wrapper 的精确论文来源、checkpoint 训练细节和 state/buffer policy 仍未核清，不能作为阶段 B/C 的解释性证据；
- `EquiGen` 与 `anonymous2024dataweight` 条目仍需独立核验，不应在正文里作为已确认文献使用；
- 本轮核查只建立最小文献地图，尚未完成系统性 related work 表格。

---

## 1. 文献类别

## 1.1 任意尺度超分与连续图像表示

### 已知起点

1. **LIIF：Learning Continuous Image Representation with Local Implicit Image Function**
   - 仓库引用：[`paper/refs/main.bib`](../../../paper/refs/main.bib) 中包含 `li2021learning`。
   - 相关性：LIIF 是任意尺度超分的核心 baseline，通过 local feature、relative coordinate 和 cell 信息预测连续坐标 RGB。
   - 项目角色：主要非等变 conditional INR baseline。

2. **LTE：Local Texture Estimator for Implicit Representation Function**
   - 仓库实现：[`src/models/lte.py`](../../../src/models/lte.py)。
   - 相关性：LTE 显式建模局部纹理/频率，可用于测试局部频率结构是否比 raw weights 更清晰地体现在 response space。
   - 项目角色：重要独立架构 baseline。

3. **MetaSR：Meta-learning based arbitrary-scale SR**
   - 仓库引用：[`paper/refs/main.bib`](../../../paper/refs/main.bib) 中已新增 `hu2019metasr`。
   - 相关性：早期任意尺度 SR 方法，使用尺度条件预测机制。
   - 状态：已完成第一轮 bibliographic verification，论文写作前仍需统一 bib 风格。

4. **其他任意尺度 SR 方法**
   - 第一层候选包括 CiaoSR、ArbSR、SRNO、UltraSR、A-LIIF、LINF、CLIT/LIT、DIIF、Thera、COZ、DIIN 等。
   - 状态：已完成第一轮 bibliographic verification；其中部分已补入 [`paper/refs/main.bib`](../../../paper/refs/main.bib)，其余先作为 Stage A 待整理条目，不直接影响 Stage B/C 当前 scratch reduced LIIF 证据链。

### 已知但需细化查证的背景事实

- 任意尺度 SR 通常通过 scale、coordinate、cell size 或 local feature 条件化输出，而不是为每个 scale 单独训练模型。
- LIIF 类模型跨尺度共享 decoder 参数，因此适合研究 scale-conditioned response 是否可复用。
- 大多数 SR 论文主要关注重建质量，而非参数更新几何或响应空间结构。

### 研究缺口

已有任意尺度 SR 工作通常回答“如何重建任意尺度图像”，但不一定回答：

- 局部几何如何影响参数或响应变化；
- scale adaptation direction 是否可预测；
- 等变性是否正则化 parameter/response dynamics；
- 适配中的一部分是否可由几何和尺度推演而非训练。

### 必须 baseline

- LIIF / pretrained LIIF；
- LTE；
- LIIF-EQ 或当前可用的等变 LIIF；
- bicubic / no-adaptation；
- full fine-tuning 与 decoder-only fine-tuning。

---

## 1.2 INR 与神经场表示

### 已知起点

1. **SIREN**
   - 仓库引用：[`paper/refs/main.bib`](../../../paper/refs/main.bib) 中包含 `sitzmann2020siren`。
   - 仓库实现：[`src/siren.py`](../../../src/siren.py)。
   - 相关性：干净的 non-conditional INR baseline，适合观测参数动态。

2. **NeRF**
   - 仓库引用：[`paper/refs/main.bib`](../../../paper/refs/main.bib) 中包含 `mildenhall2020nerf`。
   - 相关性：坐标神经场经典工作，不是 SR 主 baseline，但属于 INR 背景。

3. **COIN / COIN++**
   - 仓库引用：`chen2022coin`、`dupont2022coinpp`。
   - 相关性：说明 weights 或 modulations 可以作为信号表示。
   - 项目角色：概念背景；除非后续实现 modulation/adaptation，否则暂不作为直接实验 baseline。

4. **Functa**
   - 仓库引用：`dupont2022functa`。
   - 相关性：把神经表示或 modulation 作为数据对象，与参数空间表示相关。

### 已知但需细化查证的背景事实

- INR weights/modulations 可以编码 signal identity。
- SIREN 与 coordinate MLP 有频谱行为，但 spectral bias 本身不是新贡献。
- 由于参数非唯一性，modulation 可能比 full weights 更适合作为稳定分析对象。

### 研究缺口

现有 INR 表示工作尚未充分解释：任意尺度超分中，局部图像几何是否映射到局部 parameter/response changes。

### 必须 baseline

- SIREN fitting dynamics 作为简单观测 baseline；
- COIN/COIN++ 作为概念 baseline；
- 若后续实现 modulation/adaptation，再纳入直接实验比较。

---

## 1.3 等变性与几何深度学习

### 当前仓库起点

仓库中已有等变 LIIF 封装：[`src/models/pretrained_liif_eq.py`](../../../src/models/pretrained_liif_eq.py)，以及等变组件：[`src/models/eq/`](../../../src/models/eq)。但其精确论文来源、模型谱系和 checkpoint 训练细节仍需进一步查证。

### 需查证的通用起点

1. Group Equivariant CNNs；
2. Steerable CNNs / E2CNN；
3. Gauge Equivariant CNNs；
4. Equivariant SR、Rot-E、SE-INR、ASISR 相关工作。

### 已知但需细化查证的背景事实

- 等变性对几何变换下的 feature transformation 加结构约束。
- 在 SR 中，等变性可能改善旋转/反射一致性或样本效率，但不自动说明参数空间更可预测。

### 研究缺口

本项目关注的不是等变性是否提高输出 PSNR，而是它是否：

- 增强 local geometry-to-response correspondence；
- 降低受控条件下的有效更新维度；
- 改善 predicted update 在 scale/image 间的迁移；
- 改变几何信息承载位置：weights、features、responses 或 buffers。

### 必须 baseline

- matched condition 下的 LIIF vs LIIF-EQ；
- encoder-only、decoder-only、full-state 比较；
- feature/response-space probe，因为等变性可能不直接体现在 full flattened weights。

---

## 1.4 图像内部自相似与 patch recurrence

### 需查证起点

1. Non-local means；
2. PatchMatch；
3. natural images 中的 internal patch recurrence；
4. ZSSR：利用内部统计的 zero-shot SR；
5. Non-local neural networks / attention-based SR，作为相关但不等价文献。

### 已知但需细化查证的背景事实

- 自然图像中常有重复 patch 与跨尺度 recurrence。
- 内部自相似长期被用于去噪与超分。
- 深度 SR 中的 non-local/attention 也利用相似性，但不必然形成参数空间理论。

### 研究缺口

关键缺口是：internal local geometric similarity 是否映射到 INR parameter/response similarity。

该方向是项目核心，因为它把研究锚定在图像内部局部几何，而不是抽象群变换。

### 必须 baseline

- random patch pair；
- same-image vs cross-image patch pair；
- appearance similar but geometry different 控制（如可行）；
- geometry similar but color/texture different 控制（如可行）。

---

## 1.5 参数高效适配、Meta-learning 与 Hypernetwork

### 需查证起点

1. LoRA；
2. Adapter；
3. Hypernetwork；
4. MAML；
5. Test-time adaptation；
6. coordinate representation 相关 meta-learning。

### 已知但需细化查证的背景事实

- 许多适配方法将更新限制到低维或结构化参数子集。
- MAML/hypernetwork 能产生快速适配，但不一定解释 local geometry-to-parameter correspondence。
- 若 Stage C 发现结构化 update 子空间，low-rank update 是合理候选参数化。

### 研究缺口

现有参数高效适配方法不一定回答：INR SR 适配中哪些部分可由局部图像几何和尺度预测。

### Stage D 可能 baseline

- full fine-tuning；
- decoder-only fine-tuning；
- bias-only 或 adapter-only fine-tuning；
- random low-rank update；
- learned scale-conditioned hypernetwork（若实现）；
- MAML/meta-init（若实现并验证）。

---

## 1.6 优化动力学与权重空间几何

### 仓库相关起点

1. 频谱偏置 / NTK 相关文献
   - 仓库中有 `basri2020neural` 引用，但相关性和准确引用需进一步查证。
   - 仓库中有 [`src/spectral.py`](../../../src/spectral.py) 用于频谱诊断。

2. 权重空间对称性与 alignment
   - 代码：[`src/alignment.py`](../../../src/alignment.py)。
   - 测试：[`tests/test_alignment.py`](../../../tests/test_alignment.py)。

3. EquiGen 类 horizontal weight-space symmetry
   - 仓库中有 `peignier2023equigen`，但 bibliographic 信息和状态需核查。

### 已知但需细化查证的背景事实

- 神经网络参数空间有 permutation、sign、scale 等非唯一性。
- 平滑优化轨迹和少量 snapshot 的 PCA 很容易产生伪低维。
- 比较多个网络参数向量前必须处理 alignment 或明确报告未对齐。

### 研究缺口

本项目必须区分 image-induced structure 与 generic optimizer-induced structure。

### 必须控制

- 同 snapshot 数量的 random trajectory；
- shuffled snapshot order；
- no-alignment vs alignment；
- same architecture different seed；
- randomized patch/scale labels。

---

## 2. 当前仓库证据与未验证 claim

### 2.1 当前仓库支持的事实

- 已实现 SIREN、LIIF、LTE、pretrained LIIF、pretrained LIIF-EQ；
- 已有脚本记录 fitting/fine-tuning parameter trajectories；
- 已有参数 alignment 工具，并有 SIREN/LIIF decoder synthetic recovery 测试；
- 已有 trajectory visualization pipeline；
- 已有 result aggregator 计算 PC1 和 random baseline ratio；
- 已新增 trajectory/schema 非侵入式校验工具。

### 2.2 当前仓库尚不支持的结论

当前仓库尚未证明：

- local geometry similarity 映射到 parameter/update similarity；
- equivariance 使参数更新更可预测；
- scale adaptation update 可复用或可推演；
- 任意训练加速结果；
- 任何关于 PC1、closure、symmetry error、speedup 的具体强结论。

这些只能作为待验证假设。

---

## 3. 阶段 A 结论

1. 最有潜力的研究路径不是 generic trajectory PCA，而是 local geometry-to-response/update correspondence。
2. 等变性应作为可能的 regularizer/exposer 来研究，而不是预设解释。
3. full parameters 可能过粗，response/Jacobian/feature/modulation space 很可能必要。
4. 文献基线必须继续扩展和核查后才能支撑论文 claim。
5. 当前代码支持 Stage B observation，但 Stage C 需要新增 geometry 与 response probing 模块。

---

## 4. 不确定项与后续查证清单

以下内容不得在未查证前正式引用或作为结论：

- 任意尺度 SR 文献已完成第一轮核查，但 `paper/refs/main.bib` 还不是最终论文级 related-work bibliography；
- 当前 pretrained LIIF-EQ / SE-INR 实现的精确论文来源与 checkpoint 训练细节；
- EquiGen 条目的准确 venue、年份与 highlight 状态；
- [`paper/refs/main.bib`](../../../paper/refs/main.bib) 中 `anonymous2024dataweight` 的真实性和准确性；
- 是否已有工作直接研究 patch geometry 到 INR parameter/update similarity；
- 是否已有 equivariant INR 文献提供参数空间解释。

---

## 5. 阶段 A 交付清单

- [x] 重写 [`RESEARCH_PLAN.md`](../../../RESEARCH_PLAN.md)；
- [x] 创建本文献基线文档；
- [x] 创建 [`doc/StageA_research_questions.md`](StageA_research_questions.md)；
- [x] 创建 [`doc/Refactor_plan.md`](../../Refactor_plan.md)；
- [ ] 查证全部候选文献 bibliographic 信息；
- [ ] 决定哪些现有脚本冻结为 baseline；
- [ ] 定义 Stage B/C 最小实验阶梯。
