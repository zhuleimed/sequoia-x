# 研究状态快照（RESEARCH_STATE）

> 自动生成: 2026-08-18 11:28:39 | 生成器: scripts/save_research_state.py | 铁律七
> 本文件是研究状态的**单一事实源**——会话启动/恢复时优先读取，1 分钟重建全部状态。
> 详细过程记录见 `V3研究方向与实验研究记录.md`；教训/规则见 memory/。

## 一、各方向实验状态

| 方向 | 状态 | 进度 | 关键结果 | 日志尾部 |
|------|------|------|---------|---------|
| 方向一 LGBMRanker |  证伪（ADR V3-03） | 70/70 个月 | 70 个月明细 | (无日志) |
| 方向二 DLinear |  证伪（ADR V3-05） | 70/70 个月 | 70 个月明细 | (无日志) |
| 方向三 RankIC-LSTM |  证伪（ADR V3-15） | 70/70 个月 | 70 个月明细 | (无日志) |
| 方向四 Kronos |  3b 证伪（ADR V3-20）, 方向四收尾 | — | — | — |
| 方向五 PatchTST |  终止（ADR V3-22, 不启动） | — | — | — |

## 二、运行进程

### pctChg 全量补写（2026-08-18 进行中）
- 进程: `scripts/backfill_pctchg.py`（setsid 脱离会话, PPID=1, 退出 Claude Code 不受影响）
- 目的: stock_daily.pctChg 全历史补写（baostock adjustflag=3 标准口径）
- 进度: scripts/tmp/pctchg_backfill_progress.json（completed 数）；日志 logs/pctchg_backfill_run.log
- 断点续跑: 同命令恢复（已完成自动跳过）
- 完成后验证: 全库 pctChg 覆盖率 + 除权日抽样（600519/688167/000001 已测基准值）
- 今日已完成: pctChg 增量口径修正（sync.py 腾讯快照 close_yest + NULL 兜底）、
  模拟盘推送分组序号（【LLM 1】【V2 3/3】）、LLM 格式漂移修复（回退推荐+max_tokens 16384+限长）
  （git: fbceac6/6315d1c/8401bba/ac3dcfe/93112dc）

## 三、监督链状态

监督日志尾部: 已保存: /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x/output/backtest_v2/expe | [supervisor] ✅ 分析完成 2026年 08月 07日 星期五 15:38:08 CST — 结果见 /public/home/hpc/zhulei/superman/quant/code | [supervisor] ═══ 监督链结束 2026年 08月 07日 星期五 15:38:08 CST ═══

## 四、待办（详见 V3 文档 §14）

- 方向一/二/三/四均已完结（前三完成 70 个月, 方向四 3a+3b 证伪）
- 后续: 融合矩阵实验 / 72 组回测验证 / 2026-07 月补测
