# 研究状态快照（RESEARCH_STATE）

> 自动生成: 2026-08-08 19:57:19 | 生成器: scripts/save_research_state.py | 铁律七
> 本文件是研究状态的**单一事实源**——会话启动/恢复时优先读取，1 分钟重建全部状态。
> 详细过程记录见 `V3研究方向与实验研究记录.md`；教训/规则见 memory/。

## 一、各方向实验状态

| 方向 | 状态 | 进度 | 关键结果 | 日志尾部 |
|------|------|------|---------|---------|
| 方向一 LGBMRanker | 已完成 | 70/70 个月 | 70 个月明细 | (无日志) |
| 方向二 DLinear | 已完成 | 70/70 个月 | 70 个月明细 | (无日志) |
| 方向三 RankIC-LSTM | 已完成 | 70/70 个月 | 70 个月明细 | (无日志) |
| 方向四 Kronos | 未启动/调研完成 | — | — | — |
| 方向五 PatchTST | 未启动/调研完成 | — | — | — |

## 二、运行进程

无相关进程（实验/监督/管线均未运行）

## 三、监督链状态

监督日志尾部: 已保存: /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/output/backtest_v2/expe | [supervisor] ✅ 分析完成 2026年 08月 07日 星期五 15:38:08 CST — 结果见 /public/home/hpc/zhulei/superman/quant/code | [supervisor] ═══ 监督链结束 2026年 08月 07日 星期五 15:38:08 CST ═══

## 四、待办（详见 V3 文档 §14）

- 方向三 RankIC-LSTM 70 个月运行中（完成 → 自动 analyze → 填充 §8.8）
- 方向四 Kronos 3a 零样本基线（待用户确认启动）
- 方向三完成后：融合矩阵实验 / 回测验证
