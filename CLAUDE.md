# CLAUDE.md — Sequoia-X V2

本文件为 004_sequoia-x 项目的 Claude Code 工作指南。父级规则见 `../CLAUDE.md`。

## 当前状态（2026-08-17 更新）

**V2 模拟盘月末清仓模式 A 定稿**：月末最后交易日标记全部持仓"月末清仓（换仓）" → 下月首日
开盘先卖后买 → 满仓新 TOP10（与回测 M4+TOP_N=10 口径一致）。A/B 回测实证（70 个月）：
A +988.1% vs B（不清仓补空位）+953.7%，A 略优 3.5%——详见 BACKTEST_PLAN §29。

**60 个月扩展回测已完成**，V2/V3 体系稳定运行中。

**新增数据资产（2026-08-12）**：
- ✅ **mootdx 通达信财务数据全量下载**：119 期（1990-2026），585 字段/期，398 MB
- 位置：`data/extra_features/mootdx_finance/gpcwYYYYMMDD.parquet`
- 脚本：`scripts/download_mootdx_finance.py`（断点续跑，同命令恢复）
- 远超现有 finance（同花顺 10 维）+ holders（东财 2 维），一源覆盖全部基本面
- **暂不接入特征工程**（用户指令），后续接入时筛选 585→N 维

**后台任务查询**：`ps -eo pid,etime,pcpu,args | grep python`

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

## 重要：复权口径铁律（2026-08-10 实测固化）

**全库统一【不复权】实际成交价（除权日有完整跳空），复权只在计算层做**：
- 实测确认（2026-08-10）：全库 100% 匹配腾讯 bfq / baostock adjustflag=3；79028 个除权事件验证一致
- **所有拉取一律不复权**：baostock `adjustflag="3"`（sync.py 6 处）、腾讯 fq 参数留空读 `day`（tencent_source.py，2026-08-10 修复，原 qfq 会污染多行写库路径）、新浪 `getKLineData` 天然不复权
- **禁止写前复权/后复权**：前复权基准=拉取日，历史随除权漂移，增量/补拉会产生假断层
- **复权在计算层**：特征/标签用 `model_selection_v2/adjust.py`（后复权模块，因子来自 xdxr 分红表），`feature_version=3` 为后复权版本
- 执行层（模拟盘/回测）必须用实际成交价，不走复权
- 任何新增数据源/拉取路径：先确认复权口径再入库（全部要求不复权）

## 重要：模拟盘推送分组序号（2026-08-18）

**背景**：LLM/V2 模拟盘消息在微信端偶发乱序（wxpusher 连发 2-3 秒内多消息，
微信展示顺序不保证，出现 A1→B1→B2→A2）。pipeline 本身顺序执行（多日日志验证 LLM→V2→汇总）。
**方案（用户选择）**：消息加分组序号前缀，乱序也能识别归属：
- 卖出/清仓报告逐笔：【LLM 1】【LLM 2】...（`engine.py::_push_trade_report`，`push_tag` 实例参数）
- 日报（组内最后一条，带总数）：【LLM 3/3】【V2 3/3】（engine `_push_daily_summary` /
  `v2_simulation_daily.py` 脚本日报，序号 = 卖出报告数+1；**买入不推消息，序号只算 sold**）
- 实例化：main.py `SimEngine(push_tag="LLM")`；v2_simulation_daily.py `push_tag="V2"`；
  LSTM 等未启用场景 push_tag="" 无前缀（行为不变）
- 策略汇总（strategy_summary）最后推送，不加序号

## 重要：硬止损双轨触发 + 模拟盘三层解耦架构（2026-08-12 定稿）

**硬止损双轨触发（收窄 T+1 跳空亏损）**：
- 硬止损（-8%）判定价 = **min(当日开盘价, 收盘价)**——跳空低开跌破止损线当天即标记卖出，次日开盘执行（T+1 时点不变）
- 规则实现在 `rules.py::_check_hard_stop(entry_price, current_price, day_open)`，**仅 S 硬止损用双轨**，其余规则（移动止盈/均线/夏普/相对弱势）仍用收盘价
- 触发原因标注来源：`硬止损(开盘触发): 成本30.21×0.92=27.79, 开盘26.00`
- 卖出执行前的跌停/停牌检查不变：开盘跌停 → 取消标记 → 当晚重新评估 → 次日再试（自动重试循环）

**三层解耦架构（模拟盘）**：
- 策略层（LLM/V2/LSTM 模拟盘，只产信号）→ 执行层（`SimEngine`，唯一入口，统一传 day_open）→ 规则层（`rules.py` 唯一实现）
- **新增策略必须复用 SimEngine** 才能自动继承全部卖出规则；绕开 SimEngine 自建执行引擎 = 丢失规则，代码审查必须检查
- 回测引擎（`model_selection*/backtest/*` 三处）2026-08-12 已同步双轨（传 T-1 开盘价），与实盘模拟口径一致；改造执行时点需同步传 day_open
- ETF 择时模拟盘（`scripts/sim_etf_timing.py`）为独立体系，不走 SimEngine、无个股硬止损概念

**月末清仓（模式 A，2026-08-17 定稿，V2 模拟盘专用）**：
- **清仓**：月末**最后交易日**（`is_last_trading_day_of_month`，akshare 交易日历优先，非固定 30/31 日）
  以**当日收盘价**卖出全部持仓（`SimEngine.liquidate_all_at_close`，T+1 保护：today_opened=1 跳过；
  清仓后重写日结）→ 满仓空位 10
- **买入**：重训日（1 日 03:00）若为交易日 → **当天晚上以当日开盘价**买入新 TOP10（信号凌晨已就绪，
  `allow_same_day=True` 机制）；非交易日顺延首个交易日——与回测"次月首日开盘买入"口径一致
- 实现：`scripts/v2_simulation_daily.py` 月末分支；`engine.py` 新增 `liquidate_all_at_close` +
  `allow_same_day` 参数；`models.py get_pending_signals` 新增 `allow_same_day`（默认 False=LLM 严格 T+1 不变）
- 回测引擎 `monthly_engine.py` 新增 `keep_survivors` 参数（True=模式 B 实验用，默认 False=模式 A 不变）；
  A/B 对比实验见 `experiments/compare_eom_modes.py` + BACKTEST_PLAN §29（A +988.1% vs B +953.7%）
- LLM 模拟盘不受影响（月末清仓与 allow_same_day 仅 V2 模拟盘启用）

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

### LLM 格式漂移致推送无推荐 + 信号丢失（已修复 2026-08-18）
**现象**: AI 综合研判消息底部不再显示两个最终推荐（8/14、8/17）；
连带 sim_buy_signals 无信号写入（日志"无推荐股票，跳过"）。
**根因**: DeepSeek 输出不再含 RECOMMEND 行（格式漂移，非持仓上限）——推送的是
report 原文所以底部无推荐；save_llm_recommendations 二次解析失败即丢信号。
**修复**: ① `analyst.py::analyze` 解析失败回退频率推荐后，把
`📌 最终推荐: ...` + `RECOMMEND: ...` 附加到报告末尾（消息始终显示最终推荐，
与能否买入无关）；② `save_llm_recommendations` 新增 `recommended` 参数，
main.py 直接传 analyze 的返回值（避免二次解析不一致）。

### pctChg 全历史缺失 + 增量口径修正（已修复 2026-08-18）
**背景**: stock_daily.pctChg 仅最近 11 天有值（8/3 起），全历史 NULL。
**修复**: 新增 `scripts/backfill_pctchg.py`（baostock adjustflag=3 全量补写，断点续跑，
同命令恢复）——与腾讯 bfq / 库已有 11 天数据逐日 100% 一致。
**增量口径**（sync.py 4 处修改）: 腾讯/新浪主路径 pctChg 改用 **腾讯快照 close_yest
（标准昨收，除权日=除权基准价）** 计算，与 baostock 口径一致（旧逻辑用实际前收自算，
除权日必错，如 688167 得 -35.9% vs 标准 -7.13%）；`_write_to_db` 的 pctChg 不再
ffill/fillna(0.0)——**pctChg 缺失一律写 NULL**（0.0 是假数据，消费者自行计算才是兜底）；
`_fill_ohlcv_gaps` 首行 NaN 同样落 NULL。用户决策：**数据源失败写空值（NULL）**，程序
运算时自行 `close.pct_change()` 计算。

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

**V3 修订二：合成序列增强（2026-08-10 定稿, V3 体系专用）**：
- Kronos 合成完整序列（300 天滚动生成）注入训练 = 真·数据增强（特征+标签自洽），
  失效月保护最强（06 月 ΔIC +0.634，均值 Δ+0.308）；**121 维保持**（种子基本面快照广播，不降级）
- **不纳入 V2 生产 cron**（2026-08-10 用户明确：V3 体系功能不在 V2 生产体系中实现；
  将来 V3 替代 V2 时再集成）；实验/验证用 `build_prediction_cache --synth-series`
  与 `experiments/kronos/synth_full_series.py`，V2 重训/月末链保持纯真实数据

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
| `scripts/` | 测试/验证/回测脚本（含 `download_mootdx_finance.py` 财务数据下载） |
| `data/extra_features/mootdx_finance/` | mootdx 通达信财务数据（119 期 parquet，585 字段，398 MB） |
| `memory/` | 项目记忆文件 |
