# CLAUDE.md — Sequoia-X V2

本文件为 004_sequoia-x 项目的 Claude Code 工作指南。父级规则见 `../CLAUDE.md`。

## 当前状态（2026-08-01 晚 21:20 更新）

**数据已扩展到 2020-01 起（1594 交易日）**，缓存已重建（80维=`62cf234c5440`，88维=`13132147f8e8`，40 万样本）。

**进行中的工作（60 个月扩展回测流水线，预计今晚全部完成）**：
1. 🟢 T2/T1/T3 预测缓存构建（**24 进程并行**，PID 见 ps，v7 日志 `logs/pred_cache_t2t1t3_20260801_v7.log`；预计 **~00:00-00:30** 完成 36 个月）
2. 🟢 T4 71 个月并行训练（`launch_t4_parallel.sh`，**12 并发**，已完成 48/71，预计 **~23:00-23:30** 完成）
3. ⏳ 完成后：`bash scripts/launch_t4_parallel.sh --merge` 合并 t4 → 60 个月回测 → **逐月 T2/T4 Rank IC 分析**（见 BACKTEST_PLAN §25 优化方案）

**⚠️ 今晚关键修复（勿回退）**：
- **KMP_AFFINITY 锁核**（CLAUDE.md 铁律 + BACKTEST_PLAN §26）：T4/build 均需 `env -u KMP_AFFINITY` + 脚本内 `pop("KMP_AFFINITY")` + OMP 硬设 1（build 已修复，`build_prediction_cache.py` 顶部）
- **T4/build 并行时序**：T4 worker 缓存检查已放宽（不在缓存存 `.t4_tmp` 快照），merge 只更新 t4 字段
- 并发配置：build 24 worker（n_jobs=1）、T4 12 worker（TF 2/1 + OMP=2）

**已废弃**：`pipeline_t4_auto_start.sh` 监督脚本已 kill（T4 已并行启动，不再需要）；旧 11 个月（2025-08~2026-06）已在 20:33 build 重启时自动重建。

**后台任务查询**：`ps -eo pid,etime,pcpu,args | grep -E "build_prediction_cache|t4_monthly"`；
**T4 状态**：`bash scripts/launch_t4_parallel.sh --status`

## ⚠️ 首次读取指引

如果你是新启动的 Claude 会话，**请务必先阅读以下文件**:

1. **`V2_OPERATION_GUIDE.md`** ← **V2 框架完整指南（V3.0，~2.8 万字）**！背景/验证/训练/回测/模拟盘/教训/操作全流程，后续只需读此文档
2. **`BACKTEST_PLAN.md`** ← 项目全景 + 回测计划 + 数据扩展记录（§24），约 1,530 行
2. **`CLAUDE.md`**（本文件）← 快速参考
3. **记忆文件目录** `memory/` ← 历史决策和教训（含最终回测结果、铁律、进程管理规则）
4. **本节"当前状态"** ← 正在进行的 60 个月扩展回测流水线

## 关键发现（2026-07-29）

### T4 纯 LSTM 有效（L2=0, num_transformers=0）

| Fold | 测试期 | T4 Rank IC |
|------|--------|-----------|
| 3 | 2025 全年 | **+0.0712** |
| 4 | 2025 Q2-Q4 | **+0.1007** |
| 5 | 2026 H1 | **-0.2584** ← 市场风格切换 |
| 6 | 2026 Q2 | **-0.0909** ← 正在恢复 |

**根因链**: L2=1e-4 杀死 LSTM kernel → Transformer 层稀释信号 → 预测退化。
**修复**: `lstm_l2_reg=0.0`, `num_transformers=0`, KMP_AFFINITY 清除。

### 2026 年市场风格切换

2025 年 y2 均值 +1~2%，2026 H1 y2 均值 -10.8%，Q2 -21.7%。仅 20% 股票跑赢沪深 300。
滚动窗口测试证明：缩短训练窗口 + 纳入近期数据可逐步改善 IC（-0.258→-0.156→-0.093）。

### 待执行改进（详见 memory/v2-postmortem-improvements.md）

1. 月度 Walk-Forward（12月滚动→1月测试）
2. 市场状态特征（80→88维，新增 8 个大盘环境特征）
3. 双模型集成（短周期 6月+长周期 2年）
4. 数据同步 + Fold 7 扩展

## 关键配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `window` | 120 | 时序窗口 |
| `lstm_units` | 128 | LSTM 单元数（最佳参数） |
| `lstm_num_transformers` | 0 | Transformer 层数（已移除） |
| `lstm_l2_reg` | 0.0 | L2 正则（关键：1e-4会杀死kernel） |
| `lstm_dropout_rate` | 0.285 | Dropout |
| `lstm_learning_rate` | 0.0096 | 学习率 |
| `lstm_optuna_n_trials` | 18 | T4 Optuna 搜索 trial 数 |
| `lstm_optuna_timeout` | 86400 | 24h 超时 |
| `optuna_n_trials` | 50 | 树模型 Optuna trials |
| `optuna_timeout` | 7200 | 树模型 2h 超时 |
| `n_jobs` | 8 | 树模型内部线程 |
| `lstm_tf_intraop_threads` | 16 | TF 单 op 并行 |
| `lstm_tf_interop_threads` | 8 | TF op 间并行 |
| `lstm_omp_num_threads` | 10 | BLAS/MKL 线程 |
| `sample_end` | 2026-07-28 | 数据截止日期 |

## 重要：成交量单位规则（2026-08-02 实测固化）

**Tencent 接口返回的成交量单位是"手"（lots），入库前必须 ×100 转"股"**：
- 已固化：`tencent_source.py::_tencent_kline` 的 `df["volume"] = df["volume"] * 100`（**不得移除**）
- baostock/Sina 返回"股"（无需转换）
- **存量实测已统一**（2026-08-02）：全量 4956 只 + 100 只三段抽查 0 只混入
- 任何新增数据源/拉取路径：先确认 volume 单位再入库（手→股 / 股不变）

## 重要：研究状态管理（铁律七，2026-08-07）

**背景**：长会话 /resume 会耗尽上下文（1M 上限），研究过程无法依赖对话历史 100% 还原。
**方案**：研究状态用结构化文件精确保存（不受上下文压缩损失影响），恢复不依赖对话。

**四层信息架构（所有新建/大改项目必须按此执行）**：
1. 详细过程记录 .md（如 `V3研究方向与实验研究记录.md`）——人类可读
2. **`RESEARCH_STATE.md`（单一事实源）**——机器可读状态快照，脚本自动生成
3. memory/ 记忆文件——教训/规则/决策
4. 知识图谱（codebase-memory-mcp）——代码结构定位

**自动化机制（无需人工提醒）**：
- `scripts/save_research_state.py` 自动提取：实验进度（.tmp 计数）、进程状态、结果 CSV、日志尾部 → 生成 RESEARCH_STATE.md
- **SessionStart hook 已配置**（.claude/settings.json）：每次会话启动自动运行状态脚本，输出注入会话开头
- 会话启动流程：读 RESEARCH_STATE.md（1 分钟恢复全部状态）→ 按需读详细文档 → 开始工作
- 上下文剩余 <30% 时：用 /compact 或新会话 + 状态文件，而非继续 /resume

**git checkpoint**：每个实验里程碑（完成/证伪/修复）提交一次——代码+文档+结果版本化。

## 重要：Python 环境（铁律六，2026-08-02）

**所有运行（验证/训练/回测/模拟盘/分析/绘图）必须用生产环境 py312**：
`/home/zhulei/anaconda3/envs/zhulei_py312/bin/python`（**禁止裸 `python3`**——base 环境的 numpy/scipy 版本差异导致回测结果漂移 25%，详见 BACKTEST_PLAN §4.4 铁律六）。

## 重要：环境变量问题

**KMP_AFFINITY** 在 `.bashrc` 中设置，会锁定线程到特定核心，导致 TF 无法充分利用 CPU。
启动任何 TensorFlow 脚本前必须 `env -u KMP_AFFINITY -u OMP_NUM_THREADS`。

**TF 线程配置**：`deep_lstm.py` 模块顶部显式调用 `tf.config.threading.set_*()`，
确保 `get_*()` 返回实际值（而非误导性的 0）。

## 已知 Bug 与修复

### t4_pending 断点续跑 Bug（已修复 2026-07-26）
**位置**: `evaluate.py:100` | **修复**: 过滤 `t4_pending=true` 的 Fold

### T4 Optuna trials 截断（已修复 2026-07-26）
`lstm_optuna_n_trials` 60 → 18

### L2 正则化杀死 LSTM kernel（已修复 2026-07-28）
`lstm_l2_reg` 1e-4 → 0.0，kernel norm 从 0.00 恢复到 15.78

### Transformer 层信号退化（已修复 2026-07-28）
`num_transformers` 2 → 0，纯 LSTM pred_std 从 0 → 0.022

### KMP_AFFINITY 锁核（已修复 2026-07-28）
修复: 启动命令加 `env -u KMP_AFFINITY -u OMP_NUM_THREADS`

### TF 线程数显示为 0（已修复 2026-07-29）
`tf.config.threading.get_*()` 返回 0 是因为 env var 设置但未通过 Python API 调用。
修复: 显式调用 `tf.config.threading.set_*()`。

## 特征维度

| 版本 | 维度 | 说明 |
|------|------|------|
| v1 (旧) | 80 | 原始特征，padding 到 80 |
| v2 (新) | 88 | 新增 8 维市场状态特征（大盘涨跌/波动/回撤/均线/上涨占比），padding 到 88 |

缓存路径: `data/cache/v2_dataset/<hash>/`，特征版本变更自动重建（`feature_version` 在 hash key 中）。

## 断点续跑机制

- **数据缓存**: `data/cache/v2_dataset/<hash>/`，mmap 秒级加载
- **Fold 级 Checkpoint**: 每 Fold 完成后存 `walk_forward_results.json`
- **Optuna 复用**: 树模型 Study 跨 Fold 共享（`load_if_exists=True`），skip 优化
- **T4 best_params**: `best_params_t4_lstm.json` 存在则跳过 Optuna Phase 1

## 扩展维度数据工程（2026-08-07 起）

**架构**：交易日照旧（OHLCV 日频同步不变）；扩展维度（资金流/财务/股东/研报/新闻/分红）**仅在 V2 月度重训前补齐**（`v2_monthly_retrain.py` Step0，每月 1 日 **03:00**——2026-08-10 用户决策：月末链 19:00 起约 8h 至 02:00，03:00 重训零重叠，每月普适）。

**V3 修订二：合成序列增强（2026-08-10 定稿）**：
- Kronos 合成完整序列（300 天滚动生成）注入训练 = 真·数据增强（特征+标签自洽），
  失效月保护最强（06 月 ΔIC +0.634，均值 Δ+0.308）；**121 维保持**（种子基本面快照广播，不降级）
- 启用方式：`cfg.synth_series_dir` 或月末链 marker（`output/backtest_v2/.synth_series_marker`，
  与 .stock_pool.json 同模式；删 marker 即关闭）；合成是增强非必需——生成失败告警跳过，不阻断月度流程
- 月末链集成：`month_end_pull.py` 1.5 步骤（生成 96 只 → 检验 → 写 marker，串行在缓存重建前）

**V3 vs V2 关系（用户确认 2026-08-10）**：V3 = V2 + Kronos 两点增量（①模型融合：指数择时
信号作 T2/T4 仓位开关，待接入 ②数据合成：已集成）；cron 中无独立 V3 任务，
`v2_monthly_retrain` 为 v2/v3 共用生产重训。

**代码归属（用户明确要求）**：020_TDX 为探索验证项目；**一切拉取/清洗/重训相关生产代码必须在本项目**。同步规则见 `V3研究方向与实验研究记录.md §16`。

**生产代码位置**：
- 采集器：`scripts/collect_extra_features.py`（全市场 5206 只，断点续跑 + failed 清单 + refresh-days）
- DDE 资金流自算：`scripts/dde_calculator.py`（mootdx 逐笔，实盘当日资金流）
- mootdx 客户端：`scripts/mootdx_client.py`（**必须指定服务器**，内置列表已失效）
- 特征工程（规划）：`sequoia_x/features_extra/`（88 维 → 88+N）
- 数据落盘：`data/extra_features/{subset}/{code}.parquet` + failed 清单 + manifest.json
- 股票列表：`scripts/all_a_codes.txt`（全市场沪深 A 股，与池子规则解耦）

**数据源**：
| 数据 | 主源 | 特点 |
|------|------|------|
| fund_flow | 东财 push2his **直连**（间隔≥1s 限流）| 120 天五档；主力=大单+超大单 |
| finance | 同花顺 akshare | 102 期全历史（1998 起），混合列已清洗 |
| holders/reports/news | 东财各子域**直连**（datacenter-web/reportapi/search-api）| 限频靠重试+failed 清单补偿；reportapi 间歇封禁→串行 |
| xdxr | mootdx（通达信直连）| 全历史分红，独立数据面 |
| forecast | baostock | 业绩预告事件（类型/增幅/发布日），单进程强制 |
| news_cls（备源） | 财联社 api3.cls.cn | 快讯流+关联股票，签名无盐，rn≤5，并入月末拉取 |
| 实盘资金流 | mootdx 逐笔 DDE 自算 | 复刻东财口径，摆脱东财依赖 |

**mootdx 服务器修复**：内置 HQ_HOSTS K线已失效 → 指定 `('180.153.18.170', 7709)`；TDXSource 已修复（`tencent_source.py`，含 offset=5 bug）。

## 目录速查

| 路径 | 用途 |
|------|------|
| `V2_OPERATION_GUIDE.md` | **V2 框架完整指南（V3.0，2.8 万字，首选）** |
| `BACKTEST_PLAN.md` | **综合回测计划书（最重要）** |
| `data/sequoia_v2.db` | 主 SQLite 数据库 |
| `data/cache/v2_dataset/` | 数据集磁盘缓存（80维/88维） |
| `data/models/v2_selection/` | 模型文件 + Walk-Forward 结果 |
| `logs/` | 运行日志 |
| `scripts/` | 测试/验证/回测脚本 |
| `memory/` | 项目记忆文件 |
