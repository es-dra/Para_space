# Stage B 正式观测记录：scratch fitting dynamics

## 0. 目的

本记录对应 Stage B 的第一轮正式观测实验。目标是验证当前 fitting-dynamics 平台在真实 scratch 训练下能否稳定生成、检查、聚合轨迹输出。

本轮不证明 Stage C 的局部几何对应假设，也不使用任何预训练 LIIF/LIIF-EQ checkpoint。

---

## 1. 执行原则

- 只运行 scratch 模型；
- 不运行 `run_finetune.py`；
- 不使用 pretrained LIIF 或 LIIF-EQ；
- LIIF 使用 `LIIF_CONFIG_REDUCED`，避免 full LIIF 过参数化设置；
- 输出全部经过 `check_outputs.py` 的 schema、shape、summary consistency 与 analysis readiness 检查；
- PCA 数字只作为 Stage B 观测平台诊断，不作为科学结论。

---

## 2. 实验矩阵

结果目录：

```text
results/FittingDynamics_StageB/
```

未保留专用启动脚本。正式记录采用下列直接命令模板复现实验；此前用于批量提交的一次性 launcher 已删除，以保持工作区干净。

SIREN scratch 命令模板：

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/Phase1_FittingDynamics/run_siren.py \
  --image Data/Set5/HR/{image}.png \
  --seed {seed} \
  --device cuda \
  --save_dir results/FittingDynamics_StageB/SIREN_{image}_seed{seed}
```

LIIF reduced scratch SR x4 命令模板：

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/Phase1_FittingDynamics/run.py \
  --model liif \
  --image Data/Set5/HR/{image}.png \
  --sr 4 \
  --seed {seed} \
  --device cuda \
  --save_dir results/FittingDynamics_StageB/LIIF_reduced_{image}_sr4_seed{seed}
```

运行环境：

- 使用 `CUDA_VISIBLE_DEVICES=2`；
- 物理 GPU 为 RTX 4090；
- 其余 GPU 在本轮开始时已有较高占用；
- 本轮共 10 个 job，全部成功。

### 2.1 SIREN scratch

| image | seed | steps | snapshots |
|---|---:|---:|---:|
| baby.png | 42 | 5000 | 12 |
| baby.png | 123 | 5000 | 12 |
| baby.png | 456 | 5000 | 12 |
| bird.png | 42 | 5000 | 12 |
| butterfly.png | 42 | 5000 | 12 |

### 2.2 LIIF reduced scratch SR x4

| image | seed | steps | snapshots | params | status |
|---|---:|---:|---:|---:|---|
| baby.png | 42 | 25000 | 11 | 138531 | original Stage B |
| baby.png | 123 | 25000 | 11 | 138531 | original Stage B |
| baby.png | 456 | 25000 | 11 | 138531 | original Stage B |
| bird.png | 42 | 25000 | 11 | 138531 | original Stage B |
| butterfly.png | 42 | 25000 | 11 | 138531 | original Stage B |
| head.png | 42 | 25000 | 11 | 138531 | Stage C pilot extension |
| woman.png | 42 | 25000 | 11 | 138531 | Stage C pilot extension |

所有 LIIF summary 均记录：

```text
model_config_name = LIIF_CONFIG_REDUCED
sr_scale = 4
```

head/woman 是在 response-object pilot 阶段补齐 Set5 覆盖的正式 scratch reduced LIIF 输出；它们没有参与原始 10-run Stage B PCA/control 表，因此 Stage B PCA 统计仍按原始 10 个 run 解释。

---

## 3. 输出检查

原始 10 个输出目录以及后续补充的 head/woman LIIF reduced 输出均通过：

- `trajectory schema: OK`
- `trajectory shapes: OK`
- `summary schema: OK`
- `analysis readiness: OK`
- `summary consistency: OK`

检查命令：

```bash
for d in results/FittingDynamics_StageB/*; do
  [ -d "$d" ] || continue
  [ "$(basename "$d")" = logs ] && continue
  python experiments/Phase1_FittingDynamics/check_outputs.py "$d"
done
```

---

## 4. 聚合结果

聚合命令：

```bash
python experiments/Phase1_FittingDynamics/aggregate_results.py \
  --results_dir results/FittingDynamics_StageB
```

| run | final PSNR | PC1 % | PC1 / random baseline |
|---|---:|---:|---:|
| LIIF baby seed123 | 30.7 | 94.8 | 9.5x |
| LIIF baby seed42 | 28.5 | 93.4 | 9.3x |
| LIIF baby seed456 | 30.6 | 92.0 | 9.2x |
| LIIF bird seed42 | 21.5 | 94.2 | 9.4x |
| LIIF butterfly seed42 | 26.9 | 96.3 | 9.6x |
| SIREN baby seed123 | 100.0 | 83.4 | 9.2x |
| SIREN baby seed42 | 99.0 | 82.2 | 9.0x |
| SIREN baby seed456 | 100.0 | 83.0 | 9.1x |
| SIREN bird seed42 | 100.0 | 84.6 | 9.3x |
| SIREN butterfly seed42 | 100.0 | 85.1 | 9.4x |

Alignment diagnostic:

- SIREN 使用 `full_snapshots_aligned`；
- LIIF 使用 `dec_snapshots_aligned`；
- 本轮 raw vs aligned PC1 差异为 0.00 percentage point；
- 这只说明这些具体轨迹中 alignment 没有改变 PC1 诊断，不说明 alignment 可省略。

### 4.1 控制分析

控制分析命令：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_b_controls.py \
  --results_dir results/FittingDynamics_StageB \
  --n_controls 100 \
  --seed 0
```

该脚本只读已有输出，不提交训练 job，不修改结果目录。每个 run 使用 100 个 Monte Carlo controls：

- norm-matched random walk：保留真实 step norm schedule，但随机化 update direction；
- iid Gaussian snapshots：匹配 snapshot 数量和参数维度；
- permuted update trajectory：保留真实 update vectors 与端点，但打乱 update 顺序。

核心结果：

| group | observed PC1 | norm-matched random-walk PC1 mean | permuted-update PC1 mean | observed step cosine | permuted step cosine mean |
|---|---:|---:|---:|---:|---:|
| LIIF baby seed123 | 94.78 | 72.08 | 95.76 | 0.679 | 0.405 |
| LIIF baby seed42 | 93.38 | 72.10 | 94.71 | 0.662 | 0.368 |
| LIIF baby seed456 | 92.02 | 73.51 | 93.27 | 0.635 | 0.330 |
| LIIF bird seed42 | 94.22 | 73.75 | 94.28 | 0.497 | 0.239 |
| LIIF butterfly seed42 | 96.35 | 74.59 | 96.74 | 0.662 | 0.404 |
| SIREN baby seed123 | 83.44 | 73.11 | 80.37 | 0.426 | 0.114 |
| SIREN baby seed42 | 82.22 | 71.99 | 79.56 | 0.456 | 0.126 |
| SIREN baby seed456 | 82.99 | 72.15 | 80.43 | 0.423 | 0.132 |
| SIREN bird seed42 | 84.59 | 74.84 | 80.31 | 0.312 | 0.096 |
| SIREN butterfly seed42 | 85.07 | 76.57 | 79.61 | 0.364 | 0.099 |

解释：

- 所有 observed PC1 均高于 norm-matched random-walk control，说明轨迹不是简单随机方向游走；
- 但 LIIF observed PC1 与 permuted-update PC1 非常接近，说明 LIIF 的高 PC1 很大程度可能由 update vector 集合、step norm schedule 或端点方向决定，而不是训练时间顺序中的额外结构；
- SIREN observed PC1 高于 permuted-update PC1，但差距仍不足以支持主线 claim；
- observed step cosine 明显高于 permuted update control，说明真实训练顺序存在连续方向性；
- 这些控制支持 Stage B 平台可用于后续研究，但也进一步证明 PC1 本身不应作为 Stage C/H1 的核心证据。

---

## 5. 当前观测解释边界

可以说：

- Stage B 最小正式 scratch 管线已经跑通；
- SIREN 与 reduced LIIF SR x4 均能稳定输出可检查的 trajectory；
- 多 seed baby 实验已存在，可用于下一步 seed reproducibility 诊断；
- Set5 中 baby/bird/butterfly 的图像变化实验已存在；
- 当前 aggregator 能扫描正式输出并拒绝明显不可分析的轨迹。
- 控制分析显示真实轨迹强于 norm-matched random walk，并存在训练顺序方向连续性。

不能说：

- PC1 高就证明参数空间存在有意义低维结构；
- LIIF 的 high PC1 已经说明局部几何到 update/response 的对应；
- alignment 不重要；
- 等变性机制已被解释；
- 当前结果支持部分更新推演。
- LIIF 的 PC1 结构显著强于 update-order permutation control。

这些都需要 Stage B 后续负对照与 Stage C response/geometry probe。

---

## 6. 下一步建议

### 6.1 补充诊断

基于当前 10 个正式输出目录的只读二次统计：

| group | metric | mean | std | n |
|---|---|---:|---:|---:|
| SIREN baby seeds | final PSNR | 99.65 | 0.56 | 3 |
| SIREN baby seeds | aligned PC1 % | 82.88 | 0.62 | 3 |
| SIREN baby seeds | trajectory straightness | 0.517 | 0.0068 | 3 |
| SIREN baby seeds | consecutive update cosine | 0.435 | 0.018 | 3 |
| LIIF baby seeds | final PSNR | 29.95 | 1.24 | 3 |
| LIIF baby seeds | aligned PC1 % | 93.39 | 1.38 | 3 |
| LIIF baby seeds | trajectory straightness | 0.764 | 0.027 | 3 |
| LIIF baby seeds | consecutive update cosine | 0.659 | 0.022 | 3 |

LIIF reduced 的 total encoder update norm 明显小于 decoder update norm：

| run group | encoder/decoder total update norm |
|---|---:|
| baby seed123 | 0.138 |
| baby seed42 | 0.129 |
| baby seed456 | 0.130 |
| bird seed42 | 0.158 |
| butterfly seed42 | 0.122 |

这提示当前 reduced LIIF scratch SR x4 的可观测适配变化主要集中在 decoder 侧，但这仍只是轨迹统计，不等于局部几何对应关系已经成立。

### 6.2 阶段判断

Stage B 的 **平台有效性目标已经达到最低正式标准**：

- scratch SIREN 与 reduced LIIF 均完成多 seed / 多图像正式运行；
- 输出 schema、shape、summary consistency 与 analysis readiness 全部通过；
- 聚合器可以扫描正式输出，并正确跳过非实验目录；
- summary 已记录 reduced LIIF 的配置名，避免与 full LIIF 或 pretrained LIIF 混淆；
- 未使用 pretrained LIIF/LIIF-EQ，规避了当前阶段的过参数化与 checkpoint confound。

但 Stage B 的 **科学可靠性诊断尚未完全闭合**。控制分析已经排除“完全随机方向游走”这一过弱解释，但当前高 PC1 与较高 update coherence 仍可能来自以下因素：

- snapshot 数量少，且 snapshot 按训练时间顺序平滑采样；
- optimizer trajectory 本身天然连续，可能让任意训练轨迹呈现高 PC1；
- LIIF 的 PC1 与 permuted-update control 接近，提示其低维性可能主要来自 update vector 集合或端点方向；
- raw vs aligned PC1 当前相同，只能说明这些 runs 的 PC1 不受 alignment 影响，不能说明后续 parameter comparison 可跳过 alignment；
- SIREN self-reconstruction 已接近饱和 PSNR，不能代表 SR 场景中的 response/update 规律；
- LIIF reduced 的 decoder 主导现象需要进一步拆分到 response、feature 与 Jacobian，而不能只看 flattened parameter update。

因此，Stage B 可以标记为：

> 最小正式观测完成；第一版轻量控制分析已补齐，足以进入 Stage C pilot，但不足以支撑完整 Stage C 或论文级 claim。

### 6.3 建议的下一步

Stage B 控制分析层已完成第一版。若继续补强，可优先做：

1. 把 no-alignment vs alignment 指标加入控制分析 CSV/JSON 输出；
2. 增加同端点、同 cumulative path length 的 Brownian bridge control；
3. 对 LIIF reduced 进一步输出 encoder/decoder update norm 与 response sensitivity 的关系；
4. 将控制分析的 `--format csv/json` 输出用于后续 report/table，而不手工复制表格；
5. 基于小规模 Stage C pilot 的结果决定是否扩展 descriptor 与数据覆盖。

Stage C 不建议现在全量开启。合理的推进方式是：

1. 使用当前 Stage B 控制分析作为进入 Stage C pilot 的质量门；
2. 设计 Stage C 的最小 pilot：固定 reduced LIIF、baby/bird、seed 42，优先分析 decoder response 或 feature response，而不是 full parameter vector；
3. 只有当 Stage B 控制分析显示当前轨迹结构强于 temporal smoothness / random-walk control 时，才把 Stage C 扩展到多 descriptor、多 image、多 seed 和等变性比较。

---

## 7. Stage C pilot 初步结果

新增只读 pilot：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_geometry_response.py \
  --results_dir results/FittingDynamics_StageB \
  --device cpu \
  --geometry_source lr_up \
  --response_mode trajectory_delta \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0
```

方法边界：

- 只分析 Stage B 已有 reduced LIIF scratch 输出；
- 使用 raw `full_snapshots` 重建真实 LIIF 函数响应；
- 不使用 `dec_snapshots_aligned` 做 response probe，因为当前 LIIF decoder 是 ReLU MLP，sign-flip alignment 不是一般意义下的函数等价变换；
- 默认 geometry descriptor 使用 patch RGB residual + gradient descriptor；
- response descriptor 使用输出图像在 snapshot trajectory 上的 patch-level delta；
- 对照为 shuffled response-label control 与 random eligible pair control。

重建 sanity check：

- 所有 LIIF run 的 recomputed final PSNR 与 summary final PSNR 绝对误差为 `0.00000`；
- 说明 raw full snapshot → model state → response 的还原链路可用于 pilot 分析。

LR-up geometry + trajectory-delta response 的初步结果：

| run | neighbor response dist | shuffled mean | effect vs shuffle | shuffled p<= | Spearman rho |
|---|---:|---:|---:|---:|---:|
| LIIF baby seed123 | 1.0967 | 1.2285 | 0.1318 | 0.0050 | 0.1583 |
| LIIF baby seed42 | 1.2311 | 1.3184 | 0.0873 | 0.0050 | 0.1269 |
| LIIF baby seed456 | 1.0223 | 1.0984 | 0.0761 | 0.0050 | 0.0588 |
| LIIF bird seed42 | 1.1021 | 1.0669 | -0.0352 | 0.9950 | -0.0015 |
| LIIF butterfly seed42 | 1.2352 | 1.2740 | 0.0388 | 0.0050 | 0.0405 |

HR geometry 对照显示 baby 的 effect 更强，bird 仍然接近失败；final-delta response 对照也保持同样趋势。

当前解释：

- baby 三个 seed 都出现 geometry-neighbor response 更近的初步信号；
- butterfly 有较弱正信号；
- bird 是明显 failure / near-null case；
- Spearman rho 整体很低，说明信号不是强单调全局相关，更可能是局部 nearest-neighbor 或内容类别相关现象；
- 该 pilot 支持继续研究 response-space 几何对应，但不支持直接扩展为“跨图像稳定规律”的结论。

下一步 Stage C 应优先：

1. 做第一版 geometry descriptor ablation，检查初步信号是否只是 descriptor 选择造成；
2. 对每张图记录 failure case，尤其分析 bird 为什么破坏信号；
3. 将 response object 拆成 final output delta、trajectory output delta、feature response、decoder Jacobian；
4. 扩展到 Set5 全部图像和 baby 多 seed后，再判断是否值得引入 LIIF-EQ matched control；
5. 在等变性比较前，先修正或隔离 decoder alignment 的函数等价性问题，避免把 ReLU sign-flip alignment 的参数现象误解释为响应现象。

## 8. Stage C descriptor ablation 与 bird failure 诊断

新增只读 ablation 入口仍在同一个正式分析工具中：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_geometry_response.py \
  --results_dir results/FittingDynamics_StageB \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor all \
  --response_mode trajectory_delta \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0
```

`--geometry_descriptor all` 依次评估：

- `rgb_grad`：原始 Stage C pilot descriptor，mean-centered RGB + grayscale gradient；
- `rgb`：mean-centered RGB patch；
- `gradient`：grayscale dx/dy；
- `structure_tensor`：grayscale dx^2、dx*dy、dy^2；
- `local_spectrum`：centered grayscale patch 的 phase-free FFT magnitude。

LR-up geometry + trajectory-delta response ablation 结果：

| run | rgb_grad | rgb | gradient | structure tensor | local spectrum |
|---|---:|---:|---:|---:|---:|
| LIIF baby seed123 | 0.1318 | 0.1310 | 0.1419 | 0.0172 | 0.0047 |
| LIIF baby seed42 | 0.0873 | 0.0816 | 0.0904 | 0.0155 | -0.0177 |
| LIIF baby seed456 | 0.0761 | 0.0752 | 0.0791 | 0.0176 | -0.0057 |
| LIIF bird seed42 | -0.0352 | -0.0295 | -0.0123 | -0.0115 | -0.0309 |
| LIIF butterfly seed42 | 0.0388 | 0.0417 | 0.0147 | 0.0071 | -0.0006 |

表中数字为 `effect_vs_shuffle = shuffled_response_dist_mean - neighbor_response_dist`。正值表示 geometry-neighbor patch 的 response descriptor 比 shuffled response-label control 更近。

主要判断：

- baby 三个 seed 的正信号主要由 RGB / gradient / rgb_grad 承载；
- structure tensor 与 local spectrum 在当前 patch_size=7、stride=4 设置下明显较弱，不能作为更强证据；
- butterfly 的正信号较弱，且主要来自 RGB 类 descriptor；
- bird 在所有五种 descriptor 下都没有正效应；gradient 和 structure tensor 只把负效应减弱，不能把 failure 反转为支持性结果；
- 因此 bird failure 不是原始 `rgb_grad` descriptor 单点选择导致的。

bird seed42 的进一步诊断：

- raw full snapshot 重建 final PSNR 误差仍为 `0.00000`，排除 response 重建链路错误；
- response descriptor norm 不接近零：`response_norm_mean = 8.795`，`response_norm_cv = 0.134`；
- 五种 descriptor 的 `geometry_near_zero_frac` 均为 0，排除大量零 geometry descriptor；
- geometry-neighbor 的平均空间距离约 23.6-24.4 pixels，不是由近邻空间重叠直接造成；
- Spearman rho 仍接近 0，说明全局 pairwise geometry distance 与 response distance 没有稳定单调关系。

当前最合理解释是：bird 的局部外观/边缘 descriptor 可以找到几何近邻，但这些近邻没有共享相似的 fitting response。这可能来自内容语义、非局部结构、SR aliasing、或者 reduced LIIF 对该图的低 PSNR 适配失败；现有证据还不能区分这些机制。

更新后的 Stage C 判断：

- 可以继续研究 response-space geometry correspondence；
- 但第一版证据更支持“图像/内容依赖的局部现象”，不支持“跨图像稳定规律”；
- 下一步不应直接开启等变性比较或完整 Stage C，而应先拆 response object：final output delta、trajectory output delta、feature response、decoder Jacobian；
- 扩展数据时应优先补 Set5 的 head/woman 和更多 seed，并记录每张图的 failure mode；
- LIIF-EQ matched control 仍应等 scratch reduced LIIF 的 response-object 分析更清楚后再引入。

## 9. Stage C response-object 拆分初步结果

新增只读 response-object probe 仍复用同一分析工具：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_geometry_response.py \
  --results_dir results/FittingDynamics_StageB \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad \
  --response_mode {mode} \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0
```

其中 `{mode}` 本轮比较：

- `trajectory_delta`：输出图像全轨迹 patch delta；
- `final_delta`：最终输出相对初始化的 patch delta；
- `feature_trajectory_delta`：LIIF encoder feature map 上采样到 HR grid 后的全轨迹 patch delta；
- `coord_jacobian_final_delta`：输出 RGB 对 normalized coordinate 的有限差分 Jacobian，最终相对初始化的 patch delta。

重要边界：

- 所有 response object 仍由 raw `full_snapshots` 重建，未使用 `dec_snapshots_aligned`；
- feature response 是 encoder feature response，不是 decoder hidden activation；
- coordinate-Jacobian 使用有限差分，是函数敏感性 probe，不是参数空间 Jacobian；
- 本轮未跑 `coord_jacobian_trajectory_delta`，因为计算量较高，先用 final-delta Jacobian 做保守检查。

LR-up `rgb_grad` geometry 下的 `effect_vs_shuffle`：

| run | output trajectory | output final | feature trajectory | coord-Jacobian final |
|---|---:|---:|---:|---:|
| LIIF baby seed123 | 0.1318 | 0.0950 | 0.0854 | 0.0385 |
| LIIF baby seed42 | 0.0873 | 0.0848 | 0.0558 | 0.0288 |
| LIIF baby seed456 | 0.0761 | 0.0371 | 0.0546 | 0.0178 |
| LIIF bird seed42 | -0.0352 | -0.0140 | 0.0088 | 0.0098 |
| LIIF butterfly seed42 | 0.0388 | 0.0387 | 0.0302 | 0.0331 |

对应 `shuffled_response_p_le`：

| run | output trajectory | output final | feature trajectory | coord-Jacobian final |
|---|---:|---:|---:|---:|
| LIIF baby seed123 | 0.0050 | 0.0050 | 0.0050 | 0.0050 |
| LIIF baby seed42 | 0.0050 | 0.0050 | 0.0050 | 0.0050 |
| LIIF baby seed456 | 0.0050 | 0.0050 | 0.0050 | 0.0050 |
| LIIF bird seed42 | 0.9950 | 0.8607 | 0.2886 | 0.0149 |
| LIIF butterfly seed42 | 0.0050 | 0.0050 | 0.0050 | 0.0050 |

当前解释：

- baby 三个 seed 在四种 response object 中都保持正效应；
- output trajectory 仍是 baby 上最强的对象；
- feature trajectory 保留 baby 和 butterfly 的正信号，并把 bird 从负效应推到近零小正效应，但 bird 仍不显著；
- coordinate-Jacobian final delta 在所有五个 run 上均为正，bird 也为正，但 effect 很小，应视为弱函数敏感性信号；
- bird failure 主要出现在 output response object；feature/Jacobian object 可能更接近局部几何，但当前效应不足以支撑强 claim。

更新后的推进判断：

- response-object 拆分支持继续做 Stage C pilot；
- 最值得优先扩展的是 output trajectory、feature trajectory、coord-Jacobian final 三个对象；
- 下一步应先扩展 Set5 head/woman 或更多 seed，而不是立即引入等变性；
- 若 coord-Jacobian 小正效应在更多图像/seed 上稳定，才值得投入更昂贵的 full trajectory Jacobian 或 decoder-internal Jacobian。

## 10. Stage C Set5 head/woman response-object 扩展

本轮按相同 scratch reduced LIIF 协议补齐 Set5 中剩余两张图：

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/Phase1_FittingDynamics/run.py \
  --model liif \
  --image Data/Set5/HR/{head,woman}.png \
  --sr 4 \
  --seed 42 \
  --device cuda \
  --save_dir results/FittingDynamics_StageB/LIIF_reduced_{head,woman}_sr4_seed42
```

两张图均为：

- `LIIF_CONFIG_REDUCED`;
- `n_params = 138531`;
- scratch training;
- 无 pretrained LIIF/LIIF-EQ checkpoint。

输出检查：

```bash
python experiments/Phase1_FittingDynamics/check_outputs.py \
  results/FittingDynamics_StageB/LIIF_reduced_head_sr4_seed42

python experiments/Phase1_FittingDynamics/check_outputs.py \
  results/FittingDynamics_StageB/LIIF_reduced_woman_sr4_seed42
```

两者均通过 trajectory schema、trajectory shapes、summary schema、analysis readiness、summary consistency 检查。

训练质量：

| run | final PSNR | final loss |
|---|---:|---:|
| LIIF head seed42 | 31.59 | 0.000693 |
| LIIF woman seed42 | 29.72 | 0.001066 |

Stage C probe 设置与前一节一致：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_geometry_response.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_{image}_sr4_seed42 \
  --device cpu \
  --geometry_source lr_up \
  --geometry_descriptor rgb_grad \
  --response_mode {trajectory_delta,feature_trajectory_delta,coord_jacobian_final_delta} \
  --patch_size 7 \
  --stride 4 \
  --k 5 \
  --min_spatial_distance 8 \
  --n_shuffles 200 \
  --seed 0 \
  --format table
```

结果：

| run | output trajectory | feature trajectory | coord-Jacobian final |
|---|---:|---:|---:|
| LIIF head seed42 | -0.0067 | 0.0234 | 0.0049 |
| LIIF woman seed42 | 0.0676 | 0.0463 | 0.0229 |

对应 `shuffled_response_p_le`：

| run | output trajectory | feature trajectory | coord-Jacobian final |
|---|---:|---:|---:|
| LIIF head seed42 | 0.6965 | 0.0448 | 0.0647 |
| LIIF woman seed42 | 0.0050 | 0.0050 | 0.0050 |

所有 probe 的 raw full snapshot 重建 final PSNR 误差仍为 `0.00000`。

更新后的 Set5 seed42 图像层面判断：

- `baby` 多 seed 是当前最稳定的正信号来源；
- `woman` 在三个 response object 上均有正效应，是新的支持性图像；
- `butterfly` 保持弱正信号；
- `head` 的 output trajectory 不支持正效应，但 feature trajectory 和 coord-Jacobian final 给出弱正信号；
- `bird` 仍是最明确 failure case，尤其在 output response object 上。

这说明 Stage C pilot 的支持证据比 baby-only 更丰富，但并没有达到“扎实进入完整 Stage C”的标准。按审稿人口径，woman 是支持性图像，baby 多 seed 仍是主要正证据；head/bird failure 说明当前 descriptor、response object、patch matching 或 fitting quality 至少有一环还不充分。当前不能把失败解释成“规律强弱不同”，更不能声称跨图像稳定规律。

## 11. Stage C failure audit：审稿人口径复核

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

该工具只读已有 `trajectory.npz` 和 summary，不训练、不写结果目录。它把每个 effect 按保守审稿标准标为：

- `support`：效应量、fractional gain、shuffle p 值和 Spearman 均达到最低阈值；
- `local_only`：nearest-neighbor 有效但 Spearman 近零，只能说明局部现象；
- `tiny`：正效应太小，不能支撑 claim；
- `negative`：方向相反；
- `weak_control`：没有通过 shuffle control。

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

关键复核意见：

- 只有 `baby seed123` 在三个 response object 上达到一致支持；
- baby 其他 seed、woman、butterfly 都是 mixed/weak support；
- bird/head 是明确 failure 或 unresolved；
- bird 的 final PSNR 低，存在 fitting-quality confound；
- head 的 final PSNR 高但 output response 仍失败，说明失败不能只归因于拟合质量；
- head 的 patch gradient/variance 较低，当前 descriptor 可能无法稳定刻画弱结构/平滑图像；
- coord-Jacobian 的小正效应经常被判为 `tiny`，不能作为强证据。

更新后的阶段判断：

> Stage B 作为平台完成；Stage C pilot 只有线索，不足以直接扩展为主实验。下一步应先做方法审计和 failure-case 解释，而不是继续堆更多图像、seed 或等变性对照。

## 12. Stage C 内容分层 failure audit

为检查 positive / failure 是否只是由平滑区、纹理区或边缘区混在一起造成，本轮在 `analyze_stage_c_failure_audit.py` 中加入只读内容分层诊断：

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

方法边界：

- 仍只读已有 scratch reduced LIIF 输出，不训练、不写结果目录；
- 按 LR-up patch 的平均梯度强度和 patch 方差分别分成 low / mid / high rank tercile；
- 每个 stratum 内重复同一 geometry-neighbor vs shuffled-response control；
- `min_stratum_patches=12`，样本不足时显式标为 insufficiency；
- 该分层是 failure diagnosis，不是新的结论搜索器，也不能用来事后挑选最有利 stratum。

内容分层摘要：

| image | seed | high-gradient verdict | high-gradient mean effect | high-variance verdict | high-variance mean effect |
|---|---:|---|---:|---|---:|
| baby | 42 | consistent stratum support | 0.0734 | consistent stratum support | 0.0620 |
| baby | 123 | consistent stratum support | 0.0913 | consistent stratum support | 0.0878 |
| baby | 456 | consistent stratum support | 0.0683 | consistent stratum support | 0.0577 |
| woman | 42 | consistent stratum support | 0.0611 | mixed stratum support | 0.0695 |
| butterfly | 42 | mixed stratum support | 0.0190 | mixed stratum support | 0.0319 |
| bird | 42 | no support or negative | -0.0213 | no support or negative | -0.0178 |
| head | 42 | no support or negative | 0.0026 | no support | 0.0056 |

关键判断：

- baby 三个 seed 和 woman 的正信号主要集中在 high-gradient / high-variance patch；
- 这说明当前 `rgb_grad` geometry-response probe 测到的不是均匀全图规律，更像局部高内容强度区域中的 correspondence；
- butterfly 的分层结果仍弱，不能被包装成稳定支持；
- bird 在 high-gradient / high-variance 分层中也没有恢复支持，说明 bird failure 不是简单由平滑 patch 稀释造成；
- head 的高梯度和高方差分层仍不支持，且 head final PSNR 高，因此 failure 也不能只归因于拟合质量；
- 低/中梯度或低/中方差区域普遍更弱或出现 negative/object-dependent，提示当前 descriptor 对弱结构区域不足。

更新后的方法判断：

> 内容分层提高了对正例的理解：baby/woman 的信号集中在高内容强度 patch。但它没有解释或救回 bird/head failure，因此不能作为进入完整 Stage C 或等变性比较的通行证。下一步应优先改进或替换 geometry/response 定义，并做更严格的 patch-level failure 可视化与非唯一匹配诊断。

## 13. Stage C bird/head patch-level failure 定位

为定位 bird/head 的 failure，本轮继续扩展同一个正式只读审计工具，而不是新增临时 launcher：

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

同时保存一个正例参考诊断，便于判断 failure 指标是否只是所有图像都会出现的基线现象：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_failure_audit.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_baby_sr4_seed123 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_woman_sr4_seed42 \
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
  --output_json results/FittingDynamics_StageC_diagnostics/baby123_woman42_patch_failure_reference_2026-05-07.json
```

产物位置：

- `results/FittingDynamics_StageC_diagnostics/bird_head_patch_failure_diagnostic_2026-05-07.json`
- `results/FittingDynamics_StageC_diagnostics/baby123_woman42_patch_failure_reference_2026-05-07.json`
- `results/FittingDynamics_StageC_diagnostics/patch_crops_2026-05-07/`

这些 JSON 和 PNG 是 diagnostic 产物，不是论文图表；其中 `patch_failure_summary` 是响应模式级摘要，`worst_patch_rows` 是最坏 patch-level 表格行，`patch_crops_2026-05-07/` 保存 anchor/neighbor patch crop grid。

指标定义：

- `top1_response_percentile_mean`：每个 anchor patch 的 geometry top-1 近邻，在该 anchor 所有 eligible candidate response distance 中的分位均值；越高表示 geometry top-1 往往在 response 上不近。
- `top1_response_percentile_p90`：上述分位的 90th percentile，用于看少数严重错配是否很重。
- `worst_response_frac`：`top1_response_percentile >= 0.8` 的 anchor patch 比例。
- `nonunique_geometry_frac`：top-k geometry candidates 中至少两个候选与 top-1 的 geometry distance 在 1% 内的 anchor 比例；这是 strict near-tie 诊断。
- `top2_nonunique_geometry_frac`：top-2 与 top-1 的 geometry distance 比值 `<= 1.05` 的 anchor 比例；这是更宽松的 near-tie 诊断。
- `worst_and_nonunique_frac`：严重 response 错配且 strict near-tie 的比例。
- `worst_response_norm_rank_mean`：严重错配 patch 的 raw response descriptor norm 分位均值，用于检查是否只是 response norm 异常区。

bird/head 诊断摘要：

| run | response | top1 pct mean | top1 pct p90 | worst frac | strict ties frac | top2 close frac | worst+strict ties | worst norm rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bird | output trajectory | 0.527 | 0.912 | 0.281 | 0.066 | 0.306 | 0.008 | 0.552 |
| bird | feature trajectory | 0.401 | 0.847 | 0.132 | 0.066 | 0.306 | 0.008 | 0.460 |
| bird | coord-Jacobian final | 0.402 | 0.833 | 0.132 | 0.066 | 0.306 | 0.008 | 0.362 |
| head | output trajectory | 0.456 | 0.865 | 0.190 | 0.107 | 0.281 | 0.025 | 0.536 |
| head | feature trajectory | 0.415 | 0.856 | 0.149 | 0.107 | 0.281 | 0.017 | 0.493 |
| head | coord-Jacobian final | 0.444 | 0.807 | 0.107 | 0.107 | 0.281 | 0.025 | 0.490 |

正例参考：

| run | response | top1 pct mean | top1 pct p90 | worst frac | strict ties frac | top2 close frac |
|---|---|---:|---:|---:|---:|---:|
| baby seed123 | output trajectory | 0.252 | 0.613 | 0.041 | 0.083 | 0.314 |
| baby seed123 | feature trajectory | 0.221 | 0.694 | 0.058 | 0.083 | 0.314 |
| baby seed123 | coord-Jacobian final | 0.302 | 0.820 | 0.107 | 0.083 | 0.314 |
| woman seed42 | output trajectory | 0.337 | 0.728 | 0.066 | 0.058 | 0.248 |
| woman seed42 | feature trajectory | 0.290 | 0.658 | 0.074 | 0.058 | 0.248 |
| woman seed42 | coord-Jacobian final | 0.365 | 0.757 | 0.074 | 0.058 | 0.248 |

最坏 output-trajectory patch 示例：

| image | anchor center | neighbor center | response pct | response dist | geometry dist | grad rank | var rank | response norm rank |
|---|---|---|---:|---:|---:|---:|---:|---:|
| bird | (35, 35) | (7, 35) | 1.000 | 1.7099 | 0.7123 | 0.642 | 0.625 | 0.917 |
| bird | (39, 15) | (7, 35) | 0.991 | 1.8159 | 0.5013 | 0.833 | 0.417 | 0.142 |
| bird | (7, 35) | (39, 15) | 0.982 | 1.8159 | 0.5013 | 0.692 | 1.000 | 0.633 |
| head | (31, 23) | (7, 35) | 0.991 | 1.5633 | 0.5364 | 0.942 | 0.950 | 0.208 |
| head | (7, 43) | (35, 39) | 0.974 | 1.4951 | 0.6600 | 0.308 | 0.542 | 0.100 |
| head | (11, 39) | (31, 39) | 0.955 | 1.5020 | 0.7124 | 0.483 | 0.350 | 0.225 |

审稿式解释：

- bird/head 的 failure 确实能在 patch 层定位到：geometry top-1 近邻经常是 response 上相对远的 candidate，尤其 bird output trajectory 的 `worst_response_frac=0.281`，而 baby seed123 output trajectory 只有 `0.041`。
- 对上一版诊断的修正：此前把 `nonunique_geometry_frac` 解释成 `top-k/top-1 <= 1.05`，这个定义过严且会低估近邻多义风险；现在拆成 strict 1% ties 和更宽松的 top2/top1<=1.05。
- strict 1% ties 在 bird/head 中不高，但 top2 close frac 约 `0.28-0.31`，且 baby/woman 正例也有 `0.25-0.31`。因此“近邻候选接近”是普遍现象，不能单独解释 bird/head failure，也不能被完全排除。
- 严重 response 错配与 strict ties 的重叠很小，说明最强错配不主要来自“多个几何候选几乎完全等距”；但这不排除语义层面的非唯一、aliasing 或重复纹理。
- 严重错配 patch 的 response norm rank 接近中位或略高，不像单纯由 response norm 极端异常驱动。
- bird output trajectory 的错配在 bottom-right 区域更集中；head 的错配更分散，且 high-gradient/high-variance 分层已不能救回 head。这进一步说明 failure 不能只归因于拟合质量或平滑区域稀释。
- patch crop grid 的目检显示，bird/head 的最坏 output-trajectory 匹配常见“小 patch 色块/边缘相似，但上下文位置和内容结构明显不同”的情况；这更支持“当前局部 descriptor 缺少上下文或结构约束”的方法问题，而不是一个已经闭合的机制解释。

更新后的方法判断：

> bird/head failure 更像是当前局部 `rgb_grad` descriptor 找到的外观近邻没有共享相似 fitting response，且最坏 patch 可视化提示上下文/结构约束不足。近邻候选接近现象普遍存在，不能作为单独解释；response norm 异常也不像主因。下一步应检查上下文增强或结构化 patch matching、scale/aliasing-aware geometry descriptor，或重新定义 response object；但 coord-Jacobian 目前仍是 tiny 级别，不能直接升级为 claim。

## 14. Stage C 上下文增强 descriptor 最小对照

基于 patch crop 诊断，本轮新增一个只读方法审计对照：`rgb_grad_context`。它不是新主 claim，而是检查“局部 7x7 descriptor 缺少上下文”是否能解释 bird/head failure。

定义：

- 原 `rgb_grad`：7x7 mean-centered RGB + grayscale gradient；
- 新 `rgb_grad_context`：原 7x7 `rgb_grad` 拼接一个以 patch center 为中心、边长为 3 倍 patch size 的上下文 crop；上下文 crop 用 nearest resize 回 7x7 后提取同样的 `rgb_grad`；
- 没有调参搜索，context factor 固定为 3。

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

产物：

- `results/FittingDynamics_StageC_diagnostics/context_descriptor_audit_2026-05-08.json`

相对原 `rgb_grad` 的 response-level 变化：

| run | response | rgb_grad effect | context effect | delta | label change |
|---|---|---:|---:|---:|---|
| bird | output trajectory | -0.0341 | -0.0071 | +0.0271 | negative -> negative |
| bird | feature trajectory | 0.0082 | 0.0952 | +0.0870 | tiny -> support |
| bird | coord-Jacobian final | 0.0101 | 0.0137 | +0.0036 | tiny -> tiny |
| head | output trajectory | -0.0063 | 0.0297 | +0.0360 | negative -> tiny |
| head | feature trajectory | 0.0206 | 0.0622 | +0.0416 | tiny -> support |
| head | coord-Jacobian final | 0.0050 | 0.0075 | +0.0025 | tiny -> tiny |
| baby seed123 | output trajectory | 0.1318 | 0.1140 | -0.0178 | support -> support |
| baby seed123 | feature trajectory | 0.0838 | 0.1268 | +0.0430 | support -> support |
| baby seed123 | coord-Jacobian final | 0.0382 | 0.0333 | -0.0049 | support -> support |
| woman seed42 | output trajectory | 0.0649 | 0.1037 | +0.0387 | support -> support |
| woman seed42 | feature trajectory | 0.0458 | 0.1201 | +0.0744 | support -> support |
| woman seed42 | coord-Jacobian final | 0.0233 | 0.0207 | -0.0026 | tiny -> tiny |

patch mismatch 变化：

| run | response | top1 pct mean | worst frac | top2 close frac |
|---|---|---:|---:|---:|
| bird output | rgb_grad -> context | 0.527 -> 0.490 | 0.281 -> 0.231 | 0.306 -> 0.661 |
| bird feature | rgb_grad -> context | 0.401 -> 0.291 | 0.132 -> 0.099 | 0.306 -> 0.661 |
| head output | rgb_grad -> context | 0.456 -> 0.412 | 0.190 -> 0.124 | 0.281 -> 0.620 |
| head feature | rgb_grad -> context | 0.415 -> 0.380 | 0.149 -> 0.182 | 0.281 -> 0.620 |
| baby output | rgb_grad -> context | 0.252 -> 0.310 | 0.041 -> 0.116 | 0.314 -> 0.496 |
| woman output | rgb_grad -> context | 0.337 -> 0.345 | 0.066 -> 0.025 | 0.248 -> 0.413 |

审稿式解释：

- 上下文增强对 feature trajectory 明显有帮助：bird/head 从 tiny 到 support，baby/woman 仍保持 support 且 effect 变大。
- 对 output trajectory 只是部分改善：bird 从更负变为接近 0 但仍 negative，head 变为 tiny；这不能说 output response failure 被解决。
- coord-Jacobian 仍是 tiny，不能升级为强证据。
- context descriptor 会显著增加 top2 close frac，说明加入上下文后 descriptor 空间里近邻候选更拥挤；这可能带来新的匹配多义风险。
- baby output trajectory 的 patch mismatch 反而变差，说明上下文并非无害修复。

更新后的方法判断：

> `rgb_grad_context` 支持“上下文缺失是当前 descriptor 问题之一”这个方向，尤其对 feature response 有用；但它不是完整修复。下一步不应直接扩大 Stage C，而应比较更受控的结构化 matching：例如在局部 descriptor 外显式加入上下文相似性门控、空间/边缘方向一致性或多候选 response oracle diagnostic，判断 failure 是 descriptor 排序问题还是 response object 本身不稳定。

---

## 15. 2026-05-08 追加：top-k oracle 与 response-blind context rerank

本轮继续沿用同一个只读 `analyze_stage_c_failure_audit.py` 入口，目标是区分两种 failure：

- geometry top-k 候选集合里根本没有 response 近邻；
- top-k 里有较好 response 候选，但当前 descriptor 或排序规则选不出来。

新增诊断字段：

- `oracle_*`：在 geometry top-k 候选中用真实 response distance 选择最优候选。这是 response-label oracle，只能作为上界和定位工具，不能作为可部署 predictor。
- `rerank_*`：先用 `--geometry_descriptor` 选择 top-k，再用 `--patch_rerank_descriptor` 在 top-k 内选择 descriptor distance 最小者。该 rerank 不看 response，可用于评估一个真实可用的重排规则。

相关产物：

- `results/FittingDynamics_StageC_diagnostics/topk_oracle_rgb_grad_2026-05-08.json`
- `results/FittingDynamics_StageC_diagnostics/topk_oracle_rgb_grad_context_2026-05-08.json`
- `results/FittingDynamics_StageC_diagnostics/topk_rerank_rgb_grad_to_context_2026-05-08.json`

`rgb_grad` top-k oracle 命令：

```bash
python experiments/Phase1_FittingDynamics/analyze_stage_c_failure_audit.py \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_bird_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_head_sr4_seed42 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_baby_sr4_seed123 \
  --result_dir results/FittingDynamics_StageB/LIIF_reduced_woman_sr4_seed42 \
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
  --output_json results/FittingDynamics_StageC_diagnostics/topk_oracle_rgb_grad_2026-05-08.json
```

`rgb_grad` top-k + `rgb_grad_context` response-blind rerank 命令：

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
  --output_json results/FittingDynamics_StageC_diagnostics/topk_rerank_rgb_grad_to_context_2026-05-08.json
```

`rgb_grad` top-k oracle 摘要：

| run | response | top1 pct mean | oracle pct mean | oracle gain | oracle fixes worst | worst frac |
|---|---|---:|---:|---:|---:|---:|
| bird | output trajectory | 0.527 | 0.175 | 0.352 | 0.281 | 0.281 |
| bird | feature trajectory | 0.401 | 0.143 | 0.258 | 0.132 | 0.132 |
| head | output trajectory | 0.456 | 0.148 | 0.308 | 0.190 | 0.190 |
| head | feature trajectory | 0.415 | 0.146 | 0.269 | 0.149 | 0.149 |
| baby seed123 | output trajectory | 0.252 | 0.076 | 0.176 | 0.041 | 0.041 |
| baby seed123 | feature trajectory | 0.221 | 0.076 | 0.145 | 0.041 | 0.058 |
| woman seed42 | output trajectory | 0.337 | 0.124 | 0.213 | 0.066 | 0.066 |
| woman seed42 | feature trajectory | 0.290 | 0.096 | 0.195 | 0.074 | 0.074 |

解释：

- bird/head 的 top-k oracle 明显降低 response percentile，尤其 output trajectory：bird `0.527 -> 0.175`，head `0.456 -> 0.148`。
- 这说明在 `rgb_grad` top-5 候选内，经常存在 response 上更近的候选；failure 不是简单的“top-k 候选集合完全没有好匹配”。
- 但 oracle 使用真实 response 距离，不能被写成方法收益，也不能用于证明局部几何已经能预测 response。它只说明 reranking 有理论 headroom。

`rgb_grad` top-k + `rgb_grad_context` rerank 摘要：

| run | response | top1 pct mean | context rerank pct mean | rerank gain | oracle pct mean | oracle gain |
|---|---|---:|---:|---:|---:|---:|
| bird | output trajectory | 0.527 | 0.528 | -0.000 | 0.175 | 0.352 |
| bird | feature trajectory | 0.401 | 0.326 | 0.075 | 0.143 | 0.258 |
| head | output trajectory | 0.456 | 0.469 | -0.013 | 0.148 | 0.308 |
| head | feature trajectory | 0.415 | 0.414 | 0.000 | 0.146 | 0.269 |
| baby seed123 | output trajectory | 0.252 | 0.294 | -0.042 | 0.076 | 0.176 |
| baby seed123 | feature trajectory | 0.221 | 0.189 | 0.032 | 0.076 | 0.145 |
| woman seed42 | output trajectory | 0.337 | 0.346 | -0.010 | 0.124 | 0.213 |
| woman seed42 | feature trajectory | 0.290 | 0.225 | 0.066 | 0.096 | 0.195 |

候选级 context-distance 解释：

| run | response | oracle is context-best frac | oracle farther than top1 in context frac | rerank-oracle pct gap | rerank worsens frac |
|---|---|---:|---:|---:|---:|
| bird | output trajectory | 0.198 | 0.512 | 0.353 | 0.264 |
| bird | feature trajectory | 0.397 | 0.298 | 0.183 | 0.190 |
| head | output trajectory | 0.207 | 0.463 | 0.320 | 0.314 |
| head | feature trajectory | 0.314 | 0.438 | 0.268 | 0.298 |
| baby seed123 | output trajectory | 0.264 | 0.438 | 0.218 | 0.306 |
| baby seed123 | feature trajectory | 0.488 | 0.273 | 0.113 | 0.223 |
| woman seed42 | output trajectory | 0.240 | 0.430 | 0.222 | 0.322 |
| woman seed42 | feature trajectory | 0.380 | 0.322 | 0.129 | 0.198 |

审稿式解释：

- 简单 context rerank 没有解决 output trajectory：bird/head output 反而基本不变或略差，baby/woman output 也没有改善。
- context rerank 对 feature trajectory 有局部帮助，尤其 bird 和 woman，但仍远低于 oracle 上界；head feature 几乎没有改善。
- 对 bird/head output，oracle 候选很少也是 context descriptor 的最优候选，并且约半数 anchor 中 oracle 候选在 context 空间比原 top1 更远；这解释了为什么 context rerank 不能吃到 oracle headroom。
- 这比单独的 `rgb_grad_context` top-1 结果更严格：即使把 context 用作 top-k 内的二阶段重排，它仍无法选出 oracle 指示的 output 好候选。
- 因此下一步不应声称“上下文修复了 matching”。更准确的判断是：top-k 内存在 response headroom，但简单上下文距离不是足够的 response-blind 选择规则。

更新后的方法判断：

> bird/head failure 目前更像“候选集合内有潜在 response 近邻，但当前局部外观 descriptor 和简单上下文重排都不能可靠选中”。这支持继续研究结构化、可解释的 candidate-level 诊断，例如边缘方向一致性、低频上下文布局、位置/尺度 aliasing-aware 条件或 response object 的重新定义；但不支持扩展 full Stage C、等变性比较或训练加速 claim。

---

## 16. 2026-05-08 追加：Stage C stop/go 决策门槛

最近的 oracle / rerank 诊断说明，继续开放式尝试 descriptor 已经不再是稳健推进。为避免指标漂移和事后挑选，本轮冻结下一节点协议，详见：

- [`doc/StageC_decision_gate_2026-05-08.md`](StageC_decision_gate_2026-05-08.md)

冻结问题：

> 在 `rgb_grad` top-5 candidate 集合内，是否存在一个 response-blind、预注册的 matching/rerank 规则，能稳定利用 bird/head output trajectory 的 oracle headroom，同时不破坏 baby/woman 正例？

冻结主指标：

- primary response：`trajectory_delta`；
- failure gate：bird/head seed42；
- guardrail：baby seed123 / woman seed42；
- candidate set：`rgb_grad` top-5；
- key metrics：`rerank_response_percentile_gain_mean`、`rerank_response_percentile_mean`、`worst_response_frac`，并报告 oracle 上界和 guardrail 退化。

成功标准摘要：

| condition | threshold |
|---|---:|
| bird output rerank gain | >= 0.10 |
| head output rerank gain | >= 0.10 |
| recovered oracle gain | >= 30% for both bird/head |
| bird rerank pct mean | < 0.43 |
| head rerank pct mean | < 0.36 |
| baby/woman gain | >= -0.02 |

失败判定：

- bird 或 head 任一 output gain `< 0.05`；
- 改善只出现在 feature/Jacobian；
- baby/woman 明显变差；
- 需要 response label、oracle 信息或事后挑选 stratum 才成立；
- oracle candidate 与 top1 candidate 在预注册 response-blind 属性上不可分。

该 gate 通过前，Stage C 不进入 full expansion，不引入 LIIF-EQ / pretrained checkpoint，不写训练加速 claim。

### 16.1 Gate 执行结果

产物：

- `results/FittingDynamics_StageC_diagnostics/candidate_attribute_gate_2026-05-08.json`

本次只使用冻结 primary response：`trajectory_delta`。

| run | top1 pct | oracle pct | oracle gain | context rerank pct | rerank gain | recovered oracle gain |
|---|---:|---:|---:|---:|---:|---:|
| bird output | 0.527 | 0.175 | 0.352 | 0.528 | -0.000 | -0.001 |
| head output | 0.456 | 0.148 | 0.308 | 0.469 | -0.013 | -0.041 |
| baby123 output | 0.252 | 0.076 | 0.176 | 0.294 | -0.042 | -0.237 |
| woman42 output | 0.337 | 0.124 | 0.213 | 0.346 | -0.010 | -0.045 |

判定：gate 未通过。

原因：

- bird/head output 的 rerank gain 均 `< 0.05`，触发失败标准；
- baby123 guardrail 明显变差，`rerank gain=-0.042 < -0.02`；
- 预注册 response-blind 属性没有给出稳定可用的区分：
  - bird/head 的 oracle candidate 在 edge orientation、context low-frequency 和 context descriptor distance 上通常不优于 top1；
  - patch variance absdiff 对 bird/head 有小幅正向，但效应很小，且不足以构成可靠 rerank 规则；
  - same spatial region 不稳定，且不能独立解释 oracle candidate。

更新后的 Stage C 判断：

> 当前可以保留“局部线索存在”的弱结论，但应停止把 patch-level rerank/descriptor 当作即将修复的路线。`rgb_grad` top-5 中有 oracle headroom，但预注册 response-blind 属性无法稳定选出 oracle candidate。Stage C 主线应降级为方法诊断 / failure audit 阶段，不应进入完整 Stage C、等变性比较或训练加速。

## 17. 2026-05-08 追加：response-object audit 未解释 bird/head failure

candidate-level gate 失败后，本轮执行只读 response-object audit，产物：

- `experiments/Phase1_FittingDynamics/analyze_stage_c_response_object_audit.py`
- `results/FittingDynamics_StageC_diagnostics/response_object_audit_2026-05-08.json`

固定协议仍使用 bird/head seed42 作为 failure gate，baby seed123 / woman seed42 作为 guardrail；geometry 固定为 `lr_up + rgb_grad`，`patch_size=7`、`stride=4`、`k=5`、`min_spatial_distance=8`、`n_shuffles=200`、`seed=0`。

关键 summary：

| variant | failure effect frac mean | failure patch pct mean | guardrail effect frac mean | guardrail patch pct mean | verdict |
|---|---:|---:|---:|---:|---|
| trajectory_unit | -0.0184 | 0.4918 | 0.0786 | 0.2943 | does_not_resolve_failure |
| trajectory_raw | -0.0159 | 0.4860 | 0.0752 | 0.3040 | does_not_resolve_failure |
| trajectory_norm | 0.0243 | 0.4909 | 0.0310 | 0.4651 | does_not_resolve_failure |
| trajectory_early_unit | -0.0201 | 0.4919 | 0.0788 | 0.2947 | does_not_resolve_failure |
| trajectory_mid_unit | -0.0020 | 0.4800 | 0.0200 | 0.4104 | does_not_resolve_failure |
| trajectory_late_unit | 0.0002 | 0.4903 | -0.0002 | 0.4958 | does_not_resolve_failure |

temporal consistency 没有把 failure 与 guardrail 分开：bird/head segment pair Spearman mean `0.1392`，baby/woman 为 `0.1511`；directed percentile std mean 分别为 `0.1987` 和 `0.2009`。

审稿式解释：

- 现有主 response 已经是 L2-normalized trajectory descriptor，因此 failure 不是简单 raw amplitude 支配。
- raw trajectory、log-norm-only、endpoint final delta 和 early/mid/late 固定时间切分都没有同时恢复 bird/head 且保护 baby/woman。
- norm gap 相关性不能作为主解释，因为 norm-only 本身不解决 failure，且 guardrail 也出现类似模式。

更新后的判断：

> 当前预注册 output response-object 修补路线未解释 bird/head failure。下一步不应继续修补 descriptor/rerank 或简单 response object；若继续 Stage C 方法研究，应转向 patch matching 单元定义，或将贡献降级为 Stage B 平台与 Stage C failure audit 框架。

## 18. 2026-05-08 追加：LIIF-aware LR-cell patch-unit gate

为继续尝试但避免开放式 descriptor 搜索，本轮执行一个固定 patch matching 单元 gate：

- 新脚本：`experiments/Phase1_FittingDynamics/analyze_stage_c_patch_unit_gate.py`
- 新测试：`tests/test_stage_c_patch_unit_gate.py`
- 产物：`results/FittingDynamics_StageC_diagnostics/patch_unit_gate_lr_cell_2026-05-08.json`

冻结 candidate：

- `lr_cell_query_support`；
- 每个 LR feature cell 作为一个 matching unit；
- x4 SR 下对应 HR `4x4` output response block；
- geometry 使用固定 `3x3` LR support 的 mean-centered RGB + grayscale gradient；
- response 固定为 output `trajectory_delta`；
- 不扫 support size，不换 response object，不做 rerank，不使用 oracle。

运行命令：

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

关键结果：

| run | analysis | label | effect frac | patch pct | worst frac |
|---|---|---|---:|---:|---:|
| bird | HR7 geometry | negative | -0.0319 | 0.5275 | 0.2810 |
| bird | LR-cell geometry | tiny | 0.0257 | 0.4507 | 0.1875 |
| bird | LR-cell coordinate-only | support | 0.1494 | 0.3540 | 0.0833 |
| bird | LR-cell content-only | support | 0.1177 | 0.3791 | 0.1042 |
| head | HR7 geometry | negative | -0.0048 | 0.4562 | 0.1901 |
| head | LR-cell geometry | negative | -0.0057 | 0.4669 | 0.1806 |
| head | LR-cell content-only | support | 0.0811 | 0.3909 | 0.1042 |
| baby123 | HR7 geometry | support | 0.1070 | 0.2518 | 0.0413 |
| baby123 | LR-cell geometry | support | 0.1156 | 0.3278 | 0.1389 |
| baby123 | LR-cell content-only | support | 0.1944 | 0.2874 | 0.0972 |
| woman42 | HR7 geometry | support | 0.0502 | 0.3367 | 0.0661 |
| woman42 | LR-cell geometry | local_only | 0.0595 | 0.3697 | 0.0764 |
| woman42 | LR-cell content-only | support | 0.1554 | 0.3210 | 0.0903 |

Gate verdict：`fail`，24 个检查中 15 个失败。

审稿式解释：

- LR-cell unit 没有共同修复 bird/head：bird 只到 tiny，head 仍 negative。
- LR-cell geometry 不强于弱 controls。特别是 bird 的 coordinate-only / content-only 明显强于 geometry，head 和 baby/woman 中 content-only 也更强。
- 这说明 response 相似性里可能有空间位置和内容强度结构，但当前 LR-cell geometry descriptor 没有提供独立、稳定、跨图像的机制证据。

更新后的路线判断：

> 当前 HR7 patch、descriptor/context/rerank、简单 output response-object、以及最小 LIIF-aware LR-cell patch unit 都未通过 gate。若继续 Stage C，只能进入更重的 model-internal unit 重新定义；否则应降级为 Stage B 平台 + Stage C failure audit / negative diagnostic。
