# LSTM-Transformer 模型选股策略 — 设计文档

> 版本: v1.0 | 日期: 2026-07-20 | 状态: 已确认

## 一、项目目标

在 004_sequoia-x 项目基础上，新增 LSTM-Transformer 深度学习模型选股策略，包含完整的：模型训练 → 每日预测 → 回测验证 → 模拟盘运行 → 收益报告。与现有 LLM 选股策略完全独立运行、独立账户、互不干扰。

**参考项目**：
- `DoubleColorBall/` — LSTM-Transformer 模型架构、Optuna 搜索、训练管线
- `019_etf_daily_sync_and_backtest/` — 回测框架、多策略收益汇总报告

---

## 二、核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 数据时间分割 | 滚动窗口 70/15/15 | 保持时间顺序，模拟真实场景 |
| 月度完整训练 | 每月 15 日 00:00 | 与双色球月末训练错开 12 天 |
| 每日增量学习 | 每日 18:10 管线中 | ~5min，纳入最新市场信息 |
| 每周刷新 | 每周六 00:00 | 兜底防止增量学习累积偏差 |
| 股票池筛选 | 不做策略预筛选 | 2000 只全量预测 ~6-8min，简单公平 |
| 评估指标 | IC + 分层回测 Q1-Q5 | 业界标准，衡量排序能力 |
| 特征来源 | 纯 stock_daily 表计算 | 零外部依赖，速度快 |
| 参数量 | Optuna 自动搜索（30万~1500万） | 让数据决定最佳参数量 |
| 模拟盘账户 | 独立 sim_lstm.db | 与 LLM 策略完全隔离 |
| 回测重训频率 | 每月末扩展窗口重训 | 模拟真实月度重训节奏 |

---

## 三、目录结构

```
004_sequoia-x/
├── pipeline/pipeline.py                 # [修改] STEPS 新增 3 项
│
├── sequoia_x/model_selection/           # [新增] 模型选股子模块
│   ├── __init__.py
│   ├── config.py                        # 所有可配置参数
│   ├── features.py                      # 股票时序特征工程 (~55维)
│   ├── model.py                         # LSTM-Transformer 回归模型
│   ├── train.py                         # 训练 CLI (full/incremental/weekly)
│   ├── predict.py                       # 每日预测 CLI
│   │
│   ├── backtest/                        # 回测（独立调用，不进管线）
│   │   ├── __init__.py
│   │   ├── config.py                    # 回测参数
│   │   ├── data.py                      # 数据加载与时间切分
│   │   ├── engine.py                    # 逐日回测引擎
│   │   ├── reporter.py                  # 报告 + 图表
│   │   └── run.py                       # CLI 入口
│   │
│   └── simulation/                      # 模拟盘适配层
│       ├── __init__.py
│       ├── daily.py                     # 每日信号生成 + 推送
│       └── reporter.py                  # LSTM 策略日报
│
├── sequoia_x/simulation/
│   └── strategy_summary.py              # [新增] 多策略收益汇总日报
│
├── data/
│   ├── sim_lstm.db                      # [新增] LSTM 独立模拟盘 DB
│   └── models/lstm_selection/           # [新增] 模型文件
│       └── v{timestamp}/
│           ├── model.keras
│           └── params.json
│
└── output/
    ├── backtest_lstm/                   # [新增] 回测输出
    └── sim_lstm/                        # [新增] 模拟盘状态
```

---

## 四、特征工程

### 输入输出
- 输入: 单只股票 symbol + 截止日期 ref_date + DB 连接
- 输出: `(window=60, n_features≈55)` 三维张量
- 标签 y: 未来 5 日收益率 = `close[T+5] / close[T] - 1`

### 特征分组（全部从 stock_daily 表计算）

| 分组 | 维度 | 指标 |
|------|:---:|------|
| 价格收益 | 8 | 1/5/10/20日收益率、开盘缺口、最高最低比、涨跌幅 |
| 均线偏离 | 6 | Close/MA5/10/20/60/120 偏离百分比、MA5-MA20 差距 |
| 量能 | 8 | 量比(当日量/20日均量)、5日量变化率、换手率、成交额/流通市值、近10日量价相关系数、5日均量/20日均量 |
| 技术指标 | 14 | RSI(14)、MACD(DIF/DEA/Hist)、BOLL位置+宽度、ATR(14)/Close、KDJ-K/D、OBV变化率、ADX(14) |
| 波动率 | 4 | 5/10/20日年化波动率、波动率变化率 |
| 大盘关联 | 8 | 指数收益率、20日Beta、相对强度、指数MA20/MA60位置、近5日跑赢/跑输幅度 |
| 价格形态 | 7 | 涨跌停标记、连续涨跌天数、N日新高/新低标记、实体/影线比、振幅 |
| **合计** | **~55** | |

### 关键约束
- 严格避免 look-ahead：第 T 日特征只用 T 日及之前的数据
- 归一化：每个特征序列独立做 Z-score 标准化（rolling basis）
- 缺失值：停牌日沿用前值（与数据同步的停牌保全策略一致）

---

## 五、模型架构

```
Input: (batch, 60, ~55)
  │
  ├─ LSTM(lstm_units, return_sequences=True)
  │     units ∈ [64, 320]  由 Optuna 搜索
  ├─ Dropout(dropout_rate)
  │
  ├─ TransformerBlock × num_transformers
  │    embed_dim = lstm_units
  │    num_heads ∈ [2, 12]    由 Optuna 搜索
  │    key_dim = embed_dim（与双色球一致）
  │    ff_dim ∈ [64, 768]     由 Optuna 搜索
  │    每块: MHA → Add&Norm → FFN → Add&Norm
  │
  ├─ LSTM(lstm_units2, return_sequences=False)
  │     units ∈ [32, 192]    由 Optuna 搜索
  ├─ Dropout
  │
  ├─ Dense(dense_units, relu)
  │     units ∈ [32, 384]    由 Optuna 搜索
  ├─ Dropout
  │
  └─ Dense(1, linear)  → 预测 5 日收益率

参数量: 30万 ~ 1500万（由 Optuna 自动决定最优值）
Loss: MSE
早停: patience=20, min_delta=1e-4
学习率调度: ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
```

---

## 六、训练流程

### 6.1 三种模式

| 模式 | CLI | 触发 | 耗时 | 内容 |
|------|-----|------|:---:|------|
| 完整训练 | `--full` | 每月15日 00:00 | 50-80h | Optuna 50 trials + 最终训练 200 epochs |
| 增量学习 | `--incremental` | 每日管线 18:10 | ~5min | 加载模型 → 近60日数据微调10轮 |
| 每周刷新 | `--weekly` | 每周六 00:00 | 2-3h | 最佳参数 → 近252日数据训练100轮 |

### 6.2 完整训练 (`--full`)

```
Phase 1: Optuna 搜索
  - 50 trials × 3-fold TimeSeriesSplit
  - MedianPruner 剪枝
  - 目标: 最小化 val_loss (MSE)
  - 并行: n_jobs=6

Phase 2: 最终训练
  - 用最佳参数
  - 200 epochs + EarlyStopping(patience=20)
  - 训练/验证/测试: 70/15/15 按时间切分
  - 评估: IC + Rank IC + Q1-Q5 分层收益
```

### 6.3 增量学习 (`--incremental`)

```
1. 加载 data/models/lstm_selection/ 下最新模型
2. 抽样 200 只代表性股票（市值分层）
3. 构建近 60 个交易日特征
4. Adam(lr=1e-5), epochs=10, batch_size=64
5. 保存（覆盖原文件，不做版本管理）
```

### 6.4 每周刷新 (`--weekly`)

```
1. 加载最佳超参数（从 params.json）
2. 抽样 200 只代表性股票
3. 构建近 252 个交易日特征
4. 训练 100 epochs + EarlyStopping
5. 新建版本目录保存
```

---

## 七、每日预测

```
流程:
  1. 从 stock_daily 读取所有活跃股票
  2. 基础过滤（复用 DataEngine.get_base_stock_pool）
     → ~2000 只
  3. 并行构建特征 + 模型推理（multiprocessing, 28 workers）
     → ~6-8 分钟
  4. 输出 [(symbol, pred_return), ...] 按收益率降序
  5. 过滤 pred_return < 1%（没信心的不买）
  6. 取前 2 只（TOP_N_BUY_PER_DAY）

关键约束:
  - 使用 T-1 日及之前的数据构建特征
  - 模型用当日增量学习后的最新版本
```

---

## 八、回测

### 8.1 调用方式

```bash
# 独立调用，不在管线中
python -m sequoia_x.model_selection.backtest.run
python -m sequoia_x.model_selection.backtest.run --period 2024
python -m sequoia_x.model_selection.backtest.run --start 2024-01-01 --end 2025-12-31
```

### 8.2 回测引擎

```
逐日循环:
  for each 交易日 T (2024-01-01 ~ 2026-07-20):
    1. 信号计算（用 T-1 日 CLOSE 数据）
       a. 基础过滤 → ~2000 只
       b. 模型预测收益率 → 排序
    2. 生成交易信号
       a. 取 top-2（排除已持仓、预测收益 < 1%）
       b. 检查持仓上限（10只）
    3. 执行买卖（用 T 日 OPEN 价）
       a. 先卖后买
       b. 滑点万1、佣金万2.5、印花税千1（卖出）
    4. 收盘估值 + 止损止盈
    5. 每月末: 用扩展窗口重训模型（--quick 模式）

初始资金: 50万 | 每只: 5万 | 最大持仓: 10只
```

### 8.3 输出报告

仿 ETF 项目 `STRATEGY_COMPARISON.md` 格式：

```
四期对比:
  2024全年 (震荡市, HS300 +1.71%)
  2025全年 (大牛市, HS300 +34.94%)
  2026至今 (快牛, HS300 +24.25%)
  2024-2026 全周期

指标: 累计收益、年化收益、最大回撤、夏普比率、日胜率、交易次数

输出文件:
  output/backtest_lstm/daily_records.csv     — 逐日净值
  output/backtest_lstm/trade_records.csv     — 交易明细
  output/backtest_lstm/metrics.json          — 绩效指标
  output/backtest_lstm/equity_curve.png      — 净值曲线
  output/backtest_lstm/drawdown.png          — 回撤曲线
```

---

## 九、模拟盘

### 9.1 账户参数

| 参数 | 值 | 说明 |
|------|-----|------|
| INITIAL_CAPITAL | 500,000 | 初始 50 万 |
| PER_STOCK_BUDGET | 50,000 | 每只 5 万 |
| MAX_POSITIONS | 10 | 最大持仓 |
| TOP_N_BUY_PER_DAY | 2 | 每天最多买入 |
| MIN_PRED_RETURN | 0.01 | 最低预测收益率阈值 |
| DB_PATH | data/sim_lstm.db | 独立数据库 |

### 9.2 每日流程

```
管线 Step: LSTM 模拟盘更新
  1. 增量学习（纳入昨日收盘数据）
  2. 预测 2000 只 → 取 top-2 信号
  3. 写入 sim_buy_signals 表 (status='pending', buy_date=今日)
  4. SimEngine.run_daily(db_path='data/sim_lstm.db')
     ├── 执行待卖出（昨日标记的 pending_sell）
     ├── 执行待买入（昨日写入的 pending 信号）
     ├── 估值更新
     ├── 多因子卖出评分（复用 13 条规则）
     └── 日结写入
  5. 推送 LSTM 策略日报（微信）
```

### 9.3 与 LLM 策略隔离

```
LLM 策略:  data/sequoia_v2.db  → sim_buy_signals/sim_positions/sim_closed_trades/sim_account_daily
LSTM 策略: data/sim_lstm.db     → 同上表结构，独立 DB 文件

各自独立核算，互不干扰。
```

---

## 十、管线集成

### pipeline/pipeline.py STEPS 修改

```python
# 在 simulation 步骤之前插入:
{
    "id": "lstm_incremental",
    "name": "LSTM 增量学习",
    "cmd": ["-m", "sequoia_x.model_selection.train", "--incremental"],
    "cwd": str(PROJECT_DIR),
    "python": PY312,
    "required": False,
    "timeout": 600,  # 10min
},
{
    "id": "lstm_predict",
    "name": "LSTM 预测+信号写入",
    "cmd": ["-m", "sequoia_x.model_selection.simulation.daily"],
    "cwd": str(PROJECT_DIR),
    "python": PY312,
    "required": False,
    "timeout": 900,  # 15min (2000只预测 ~8min + 缓冲)
},
```

### 管线结束后推送

在 pipeline 末尾新增多策略汇总推送：
```
📊 Sequoia-X 策略汇总 | 07-20
════════════════════════════════════════
❶ LLM选股
  累计+3.2% | 持仓2只 | 今日买入: 000757
❷ LSTM-Transformer选股
  累计+0.5% | 持仓1只 | 今日买入: 600519
```

---

## 十一、Cron 调度

```cron
# 每日管线（已有，不变）
10 18 * * 1-5 cd ...004_sequoia-x && python pipeline/pipeline.py >> logs/pipeline_$(date +\%Y\%m\%d).log 2>&1

# LSTM 月度完整训练（新增，每月15日 00:00）
0 0 15 * * cd ...004_sequoia-x && python -m sequoia_x.model_selection.train --full >> logs/lstm_retrain_$(date +\%Y\%m).log 2>&1

# LSTM 每周刷新（新增，每周六 00:00）
0 0 * * 6 cd ...004_sequoia-x && python -m sequoia_x.model_selection.train --weekly >> logs/lstm_weekly_$(date +\%Y\%m\%d).log 2>&1

# 双色球月度训练（已有，不变）
0 0 27-31 * * [ "$(date -d '+2 day' +\%d)" = "01" ] && cd ...DoubleColorBall && python -m ssq_model.train --full >> logs/retrain_$(date +\%Y\%m).log 2>&1
```

---

## 十二、开发阶段

| 阶段 | 内容 | 验证方式 |
|:----:|------|----------|
| Phase 1 | config.py + features.py + model.py | 单只股票跑通特征构建→模型训练→预测 |
| Phase 2 | train.py (full/incremental/weekly) | 小规模训练成功，IC 可计算 |
| Phase 3 | predict.py | 2000 只全量预测 < 10min |
| Phase 4 | backtest/ | 回测跑通，输出 4 期对比报告 |
| Phase 5 | simulation/ + pipeline 集成 | 模拟盘正常运转，日报推送成功 |
| Phase 6 | strategy_summary.py | 双策略汇总推送 |
| Phase 7 | 全量代码审计 | 检查 look-ahead bias、边界情况、错误处理 |

---

## 十三、风险与注意事项

1. **Look-ahead bias** — 所有特征和标签必须严格按时间切分
2. **幸存者偏差** — 回测中的 2000 只股票必须是当时实际存在的股票
3. **停牌处理** — 停牌日沿用前值，成交量=0 的股票跳过
4. **过拟合** — 用 Optuna 的验证集 IC 而非训练集 IC 选最佳参数
5. **双色球训练冲突** — 15 日 vs 27-31 日，间隔 12 天确保错开
