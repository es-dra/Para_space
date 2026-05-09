# 阶段 A 剩余风险与阶段 B 依赖关系记录

## 0. 目的

本文档记录阶段 A 尚未完全补齐、但不应阻塞阶段 B 最小观测实验的事项。目标是避免遗漏隐含假设，同时防止项目因过度文献整理或工程准备而停滞。

当前判断：阶段 A 已足以支撑进入阶段 B。剩余事项应持续推进，但不应阻止最小 fitting dynamics smoke run、输出 schema 检查和观测平台可靠性验证。

---

## 1. 必须现在处理的事项

以下事项与阶段 B 最小观测直接相关，必须优先保证：

1. **输出目录可检查**
   - 需要能只读检查 `trajectory.npz` 与 `dynamics_summary.json`。
   - 已由 [`experiments/Phase1_FittingDynamics/check_outputs.py`](../../../experiments/Phase1_FittingDynamics/check_outputs.py) 初步支持。

2. **trajectory key 与 shape 一致性**
   - 后续 PCA、update norm、response probe 都依赖 snapshot 维度正确。
   - 已由 [`src/trajectory_schema.py`](../../../src/trajectory_schema.py) 与 [`tests/test_trajectory_schema.py`](../../../tests/test_trajectory_schema.py) 初步覆盖。

3. **summary 与 trajectory 的基本一致性**
   - `n_snapshots`、`snapshot_steps`、`final_psnr` 等字段若与轨迹不一致，会影响复现和聚合。
   - 本轮应在输出检查工具中补强。

4. **数据缺失要明确报告**
   - 当前仓库未包含 `Data/`，阶段 B smoke run 可能因数据缺失无法执行。
   - 这不是研究失败，但必须在协议中记录为环境阻塞。

---

## 2. 不阻塞阶段 B 的剩余事项

以下事项重要，但不应阻塞阶段 B：

1. **完整文献查证**
   - MetaSR、CiaoSR、ArbSR、SRNO、UltraSR 等仍需核查准确 bibliographic 信息。
   - 不影响先运行 SIREN/LIIF smoke run。

2. **LIIF-EQ 来源与 checkpoint 细节**
   - 当前等变模型封装可作为工程对象存在，但论文引用和实验解释需要后续核查。
   - 不阻塞先跑非等变 SIREN/LIIF smoke run。

3. **Stage C 几何 descriptor 细化**
   - 局部几何定义是核心科学问题，但阶段 B 只验证观测平台，不要求先实现 descriptor。

4. **论文结构与相关工作表格**
   - 写作准备可以延后到 Stage B 观测稳定、Stage C 问题更具体后。

---

## 3. 当前隐含假设

1. **当前 trajectory schema 能代表后续主要 fitting dynamics 输出**
   - 风险：未来新增 response/Jacobian/probe 输出后 schema 会扩展。
   - 处理：当前 schema 工具允许 extra keys，避免阻塞扩展。

2. **SIREN 与 conditional model 可用同一检查入口**
   - 风险：pretrained LIIF-EQ 的 buffers/state_dict 可能带来额外 key 或 shape 特例。
   - 处理：先用 family 推断与可选 `--model_family` 覆盖，暂不强制更细分类。

3. **summary 缺失应视为阶段 B smoke run 失败**
   - 理由：阶段 B 关注可复现性，缺少 summary 会削弱实验记录。
   - 处理：检查工具默认要求 summary，但允许后续增加宽松模式。

4. **metric 长度允许为 snapshot 数或 snapshot 数减一**
   - 理由：现有脚本通常包含 step 0 snapshot，但 PSNR/freq_ratios 从训练后 snapshot 开始记录。
   - 风险：这是一种兼容策略，不代表理想规范。

---

## 4. 阶段 B 进入实质观测前的风险清单

| 风险 | 是否阻塞 B1/B2 | 处理方式 |
|---|---|---|
| `Data/` 不存在 | 阻塞真实 smoke run | 先报告数据缺失；可由用户提供图像路径 |
| summary 字段不一致 | 阻塞输出可信性 | 检查工具报错 |
| LIIF-EQ buffer policy 不清晰 | 不阻塞 SIREN/LIIF | 暂缓 LIIF-EQ 解释，只做记录 |
| snapshot schedule 与 small steps 不匹配 | 可能影响 smoke run | 先记录，不立即改训练脚本 |
| 文献条目未完全核查 | 不阻塞 B | 继续作为阶段 A 延伸任务 |

---

## 5. 当前结论

阶段 A 已能支撑阶段 B。现阶段最重要的是从“文档与工具准备”推进到“最小真实观测输出”，但在运行真实实验前，应先确保输出检查工具能捕获 summary 与 trajectory 的基本不一致。