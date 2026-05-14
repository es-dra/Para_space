# Project History Summary

This replaces the old long daily logs for the whiteboard restart.

## Retained Facts

- The project studies INR / arbitrary-scale SR fitting or adaptation dynamics.
- Stage A remains useful as a pilot problem definition: it fixed the broad
  question, hypothesis ladder, non-goals, and basic evidence standard.
- Scratch SIREN and scratch LIIF training entrypoints remain usable engineering
  assets.
- Old exploratory routes did not establish a positive mechanism claim.

## Active Entry

Future ARIS work should start from:

- `CLAUDE.md`
- `doc/aris_project_brief.md`
- `doc/research_contract.md`
- `doc/stages/stage_a/closeout.md`
- `experiments/Phase1_FittingDynamics/README.md`

## 2026-05-14 Cleanup Note

完成范围：按白板重启方案清理旧编号节点、旧阶段叙事、旧证据索引、旧分析脚本、旧测试、运行态目录、旧结果目录和外部模型资产；新增 ARIS 白板入口和短历史负面摘要。

改动文件或产物：新增 `CLAUDE.md`、`doc/aris_project_brief.md`、`doc/history_negative_summary.md`、`memory/project_history_summary.md`；重写文档入口、研究合同、Stage A closeout、训练入口 README、工程边界和 paper README；删除旧活跃证据链与旧分析代码。

运行过的命令/测试/分析及结果：`pytest tests` 通过，剩余 23 个基础测试全部通过；`python experiments/Phase1_FittingDynamics/run_siren.py --help`、`run.py --help`、`check_outputs.py --help` 均可打开；本地链接检查通过；`git diff --check` 通过；ARIS read-only smoke 返回 `BRIEF_OK`，实际只读取了 `CLAUDE.md` 和 `doc/aris_project_brief.md`；旧引用搜索只剩 `doc/aris_project_brief.md` 的禁止继承段落和 `doc/history_negative_summary.md` 的短负面摘要。

证据支持的结论：当前只完成项目入口清理，不产生新的科研结论。

未解决风险和下一步：后续正式研究必须从新的短节点开始，不得直接继承旧描述符或旧失败路线；第一步应让 ARIS 基于白板 brief 锁定一个小问题和停止条件。

专家评审：清理本身可靠，因为它只改变默认入口和活跃代码面；它不证明任何参数空间规律。下一步合理，但要防止 ARIS 把“旧路线失败”误读成“相反路线成立”，也要防止为了自动化而过早扩大实验范围。

## 2026-05-14 Alignment Note

完成范围：新增 `doc/aris_research_alignment.md`，把交给 ARIS 前需要确认的研究方向、目的、暂不做事项、文献边界和推荐首个 ARIS 节点写成短文档；更新 `CLAUDE.md` 和 `doc/README.md`，让该对齐文档成为干净入口的一部分。

改动文件或产物：`doc/aris_research_alignment.md`、`CLAUDE.md`、`doc/README.md`、`memory/project_history_summary.md`。

运行过的命令/测试/分析及结果：读取了当前白板入口、研究合同、Stage A closeout 和项目历史摘要；通过 web 核查了 LIIF、Meta-SR、LTE、SIREN、coordinate MLP spectral bias、cross-scale internal recurrence 和 weight-space alignment 相关主文献入口。

证据支持的结论：文献支持把项目定位为 INR/SR fitting dynamics probing study，支持关注 local feature、coordinate、scale/cell、frequency 和 training dynamics；文献不支持直接宣称存在可复用参数空间规律。

未解决风险和下一步：需要项目负责人确认这份对齐稿是否准确；确认后再把建议的 start request 提交给 ARIS，让 ARIS 先产出首个节点合同，而不是直接开跑实验。

专家评审：本次只完成方向和文献边界对齐，没有产生实验结论。下一步先让 ARIS 设计节点是合理的，因为它能防止自动化一上来就变成盲跑；需要防止 ARIS 把文献中的“局部隐式表示/频谱偏置/内部相似性”误读成当前项目已经证明的机制。
