# Sequoia-X V2 模型选股系统 — 操作说明指南

> **版本**: V2.0 | **最后更新**: 2026-07-24 | **维护者**: zhulei
>
> 本文档是 V2 多任务树模型 + LSTM-Transformer 选股系统的完整操作指南。
> 持续更新中，每次功能迭代后追加新内容。

---

## 目录

1. [系统总览](#1-系统总览)
2. [整体架构](#2-整体架构)
3. [数据管线](#3-数据管线)
4. [模型体系](#4-模型体系)
5. [核心工作流](#5-核心工作流)
6. [当前运行任务](#6-当前运行任务)
7. [运维指南](#7-运维指南)
8. [结果解读](#8-结果解读)

---

## 1. 系统总览

### 1.1 V2 是什么

V2（`model_selection_v2`）是一个 **A 股多模型 Walk-Forward 选股系统**，目标是从全市场 ~3000 只股票中选出未来 20 个交易日超额收益最高的股票。

### 1.2 核心思想

```
历史日线数据 → 80维特征 → 4个模型 → 预测信号 → 回测/模拟盘
```

- **多模型互补**：树模型学截面规律（"当前 PE 低 + 放量 = 好"），LSTM 学时序规律（"PE 在加速下降 + 量能萎缩后爆发 = 好"）
- **Walk-Forward**：滚动时间窗口训练→测试，严格模拟实盘环境，杜绝 look-ahead bias
- **多任务标签**：同时预测方向（涨跌）、超额收益（跑赢大盘多少）、波动率（风险）

### 1.3 与 V1 的关系

| | V1 (model_selection) | V2 (model_selection_v2) |
|---|---|---|
| 模型 | 单 LSTM-Transformer | T1+T2+T3 树模型 + T4 LSTM |
| 评估 | 简单 train/test | Purged Walk-Forward 6-Fold |
| 标签 | 1 个（收益率） | 3 个（方向+超额收益+波动率） |
| 配置 | 独立 LSTMConfig | 统一 V2Config |
| 状态 | 稳定运行中 | 开发测试中 |

---

## 2. 整体架构

### 2.1 目录结构

```
sequoia_x/
├── core/                          # 全局基础设施
│   ├── config.py                  # Settings: 数据库路径、API Key 等
│   └── logger.py                  # 统一日志（structlog）
├── data/
│   ├── engine.py                  # DataEngine: SQLite 读写、股票池管理
│   ├── sync.py                    # 日线数据同步（腾讯+新浪+baostock）
│   └── tencent_source.py          # 腾讯行情 API 适配
│
├── model_selection/               # V1: LSTM-Transformer（参考，维护模式）
│   ├── model.py                   # LSTM+Transformer 模型定义
│   ├── train.py                   # 训练入口（full/incremental/weekly）
│   └── ...
│
└── model_selection_v2/            # ★ V2: 多任务选股系统（主力开发）
    ├── config.py                  # V2Config：全部可配置参数
    ├── features.py                # 80维时序特征提取（12组）
    ├── labels.py                  # 多任务标签构建（y1/y2/y3）
    ├── train.py                   # 批量训练入口
    ├── evaluate.py                # Walk-Forward 评估入口 ★
    │
    ├── models/                    # 模型实现
    │   ├── tree_cls.py            # T1: XGBoost 分类（涨跌方向）
    │   ├── tree_reg.py            # T2: LightGBM 回归（超额收益）
    │   ├── tree_vol.py            # T3: CatBoost 回归（波动率）
    │   └── deep_lstm.py           # T4: LSTM-Transformer 回归（超额收益）★新增
    │
    ├── deep/                      # 深度学习模块（占位 → 已激活）
    │   └── __init__.py
    │
    ├── backtest/                   # 回测引擎
    │   ├── config.py              # 回测参数（资金、仓位、费率）
    │   ├── engine.py              # V2BacktestEngine：逐日模拟
    │   └── reporter.py            # 回测报告生成
    │
    └── simulation/                 # 模拟盘（待完善）
        └── __init__.py
```

### 2.2 数据流全景

```
                    ┌─────────────┐
                    │  SQLite DB  │  ← 腾讯/新浪/baostock 三轨每日同步
                    │sequoia_v2.db│
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        features.py   features.py   features.py
        (逐日特征)    (逐日特征)    (逐日特征)
              │            │            │
              ▼            ▼            ▼
        labels.py     labels.py    labels.py
        y1=方向       y2=超额收益   y3=波动率
              │            │            │
              └────────────┼────────────┘
                           ▼
                  build_training_dataset()
                  X: (n, 120, 80)  ← 3D 时序张量
                  y1, y2, y3
                           │
              ┌────────────┼────────────────────┐
              ▼            ▼            ▼       ▼
           T1:XGBoost  T2:LightGBM  T3:CatBoost  T4:LSTM
           (分类)       (回归)       (回归)      (回归)
              │            │            │       │
              └────────────┼────────────┘       │
                           ▼                     ▼
                    Walk-Forward 评估    Walk-Forward 评估
                    (T1/T2/T3)          (包含 T4)
                           │
                           ▼
                    回测/模拟盘信号
```

---

## 3. 数据管线

### 3.1 数据源

SQLite 数据库 `data/sequoia_v2.db`，核心表：

| 表 | 内容 | 字段 |
|----|------|------|
| `stock_daily` | 个股日线 | OHLCV + peTTM/pbMRQ/psTTM/pcfNcfTTM（TDX/mootdx 主力，腾讯实时行情补充，baostock 后备） |
| `index_daily` | 指数日线 | 同上（沪深300等） |
| `stock_list` | 股票列表 | symbol + name |
| `sync_log` | 同步日志 | 每日同步状态、覆盖率 |

数据由 `sequoia_x/data/sync.py` 每日自动同步（腾讯+新浪+baostock 三轨制 OHLCV，TDX 估值）。
> **2026-07-24**: 估值数据源切换为 **TDX/mootdx**（快 50 倍，稳定 100%），baostock 代码保留备用。

### 3.2 基础股票池

`DataEngine.get_base_stock_pool()` 三步过滤：

1. **板块剔除**：去掉科创板(688/689)、创业板(300/301)、北交所(4xx/8xx)
2. **质量剔除**：去掉 ST/\*ST/退市、上市不满 1 年的次新股
3. **价格剔除**：去掉最新收盘价 < 2 元的低价股

当前（2026-07-24）基础池约 **3012 只**。

### 3.3 特征工程（features.py）

从 OHLCV 日线数据提取 **80 维**时序特征，严格避免 look-ahead bias（第 T 日特征仅用 T 日及之前数据）：

| 分组 | 维度 | 典型特征 |
|------|------|----------|
| 价格收益 | 8维 | 1/5/10/20日收益率、跳空缺口、日内振幅 |
| 均线偏离 | 6维 | 5/10/20/60/120日均线偏离、MA5/MA20比 |
| 量能 | 8维 | 量比、量变化、换手率、量价相关、放量标记 |
| 技术指标 | 11维 | RSI、MACD(DIF/DEA/Hist)、布林带、KDJ、OBV、ADX |
| 波动率 | 4维 | 5/10/20日年化波动率、波动率变化 |
| 大盘关联 | 8维 | 指数收益、Beta、超额收益、指数均线偏离 |
| 价格形态 | 7维 | 涨跌停标记、连涨/连跌天数、新高/新低、K线实体比 |
| 最大回撤 | 3维 | 20日/60日回撤、恢复率 |
| 收益分布 | 4维 | 偏度、峰度、上涨天数比、非对称波动 |
| 时间日历 | 4维 | 星期(sin/cos)、月中日、季末标记 |
| 价格位置 | 3维 | K线收盘位置、跳空均值、振幅变化 |
| 估值指标 | 4维 | peTTM + pbMRQ + 各自60日分位 |

> **2026-07-24 更新**: 移除 psTTM/pcfNcfTTM（腾讯/新浪不提供），新增4组14维纯OHLCV派生特征（最大回撤、收益分布、时间日历、价格位置）。

所有特征沿时间轴 **Z-score 归一化**。最终输出形状 `(n_samples, window=120, n_features=80)`。

### 3.4 标签构建（labels.py）

每个采样日期的每只股票计算 **3 个标签**：

| 标签 | 含义 | 计算方式 | 类型 |
|------|------|----------|------|
| y1 | 5日涨跌方向 | 未来第5个交易日收盘 vs 当日收盘 | 二分类(0/1) |
| y2 | 20日超额收益率 | 个股20日收益 - 沪深300的20日收益 | 回归（裁剪±50%） |
| y3 | 20日波动率 | 未来20日日收益率的年化标准差 | 回归（上限200%） |

**采样策略**：每月取 2 天（月初第5个交易日 + 月中第15个交易日），跳过前 150 个交易日（window=120+30 缓冲）。

**数据处理**：多进程并行（16 workers），每 200 只股票一个 chunk，管道直传无磁盘中转。SQLite 热缓存下数据构建约 12 分钟。

---

## 4. 模型体系

### 4.1 四模型一览

| 模型 | 代号 | 算法 | 目标 | 输入格式 |
|------|------|------|------|----------|
| T1 | tree_cls | XGBoost 分类 | y1（5日涨跌方向） | flat (9600,) |
| T2 | tree_reg | LightGBM 回归 | y2（20日超额收益） | flat (9600,) |
| T3 | tree_vol | CatBoost 回归 | y3（20日波动率） | flat (9600,) |
| T4 | deep_lstm | LSTM-Transformer | y2（20日超额收益） | 3D (120, 80) |

### 4.2 为什么树模型 + LSTM 互补

```
树模型（T1/T2/T3）                   LSTM-Transformer（T4）
─────────────────────                 ─────────────────────
看到：9600 维扁平向量                  看到：(120, 80) 时序张量
学到：截面规律                         学到：时序演化规律
     "PE低+放量+RSI超卖 → 涨"            "RSI从超卖区回升+量能逐日放大 → 涨"
盲区：趋势方向、加速度、模式切换         盲区：静态阈值、简单组合
```

T2 和 T4 预测同一目标（y2=20日超额收益），信号互补。最终 ensemble：

```
最终信号 = α × T2 + (1-α) × T4
```

### 4.3 T4 LSTM-Transformer 架构

```
Input (120天, 80特征)
   │
   ▼
LSTM(1) ──→ 全序列输出 (120, lstm_units)
   │
   ▼
Dropout
   │
   ▼
TransformerBlock × N  ──→ 自注意力捕捉跨时间步依赖
   │                      (LayerNorm → MultiHeadAttention → Residual
   │                       → LayerNorm → FFN → Residual)
   ▼
LSTM(2) ──→ 压缩为向量 (lstm_units2,)
   │
   ▼
Dense(relu) → Dropout → Dense(1, linear)
   │
   ▼
预测 20日超额收益率
```

**超参搜索**：6 个核心参数（lstm_units, num_transformers, dropout_rate, learning_rate, l2_reg, batch_size），其余推导或固定。使用 HyperbandPruner 剪枝。

**训练环境**：CPU-only，TF 内部 16 线程并行（intra_op=16, interop=8, OMP=10），Huber loss（对极端行情鲁棒）。

### 4.4 Optuna 超参搜索策略

每个模型在 Walk-Forward 的每个 Fold 中先跑 Optuna 搜索（树模型 50 trials / 2h timeout，LSTM 18 trials / 24h timeout），找到最佳参数后全量训练。

**T4 trials 选择说明**：LSTM 每个 trial 耗时 0.5-3.3h（取决于模型大小），60 trials 理论上需 45-90h，远超 24h timeout。实测仅完成 8-15 个（13-25%），Hyperband 资源分配被 timeout 截断。18 trials 确保在 ~15-20h 内完整运行，TPE sampler 可均匀探索 6 维搜索空间，实际效果优于被截断的 60 trials。

**多核并行策略**（36核机器，保留3核）：
- **数据构建**：16 workers 多进程并行
- **树模型**：Fold 内 T1∥T2∥T3 并行训练（`ThreadPoolExecutor(3)`），各 8 线程 = 24 核
- **LSTM**：T4 单独训练，TF 16 线程独占
- **峰值**：~33 核

**关键设计**：
- **SQLite 持久化**：搜索结果存 `data/models/v2_selection/optuna/*.db`，跨 Fold 复用
- **HyperbandPruner**（T4）：差的 trial 早期终止，节省时间
- **MedianPruner**（T1/T2/T3）：树模型 trial 轻量，用中位数剪枝
- **跨 Fold 共享 Study**：`load_if_exists=True`，Fold 3 完整搜索后 Fold 4-6 秒完成

---

## 5. 核心工作流

### 5.1 Walk-Forward 评估（evaluate.py）★ 主要入口

```bash
python -m sequoia_x.model_selection_v2.evaluate
```

**6 个滚动 Fold**：

| Fold | 训练期 | 测试期 | 目的 |
|------|--------|--------|------|
| 1 | 2020-2023 | 2024全年 | 基准评估 |
| 2 | 2020-2024Q1 | 2024Q2-Q4 | 扩展窗口 |
| 3 | 2020-2024 | 2025全年 | 扩展窗口 |
| 4 | 2020-2025Q1 | 2025Q2-Q4 | 扩展窗口 |
| 5 | 2020-2025 | 2026H1 | 扩展窗口 |
| 6 | 2020-2026Q1 | 2026Q2 | 最新评估 |

每个 Fold 的流程：
```
1. 切分 train/test（按日期 mask）
2. T1∥T2∥T3 并行训练（ThreadPool, 24核）
   ├ T1: XGBoost Optuna → 全量训练 → AUC + 准确率
   ├ T2: LightGBM Optuna → 全量训练 → Rank IC + RMSE
   └ T3: CatBoost Optuna → 全量训练 → RMSE
3. T4 单独训练（TF 16核独占）
   └ LSTM-Transformer Optuna → 全量训练 → Rank IC + RMSE
4. 汇总 Fold 结果 + ETA 预估
```

**输出**：
- 日志：`logs/v2_evaluate_optuna_YYYYMMDD_HHMM.log`
- 结果 JSON：`data/models/v2_selection/walk_forward_results.json`
- Optuna DB：`data/models/v2_selection/optuna/*.db`

### 5.2 批量训练（train.py）

```bash
python -m sequoia_x.model_selection_v2.train
```

一次性训练 T1+T2+T3（不含 Walk-Forward），用于快速获得可用模型：
- 抽样 20000 样本跑 Optuna → 全量重训
- 保存模型到 `data/models/v2_selection/`（xgb.json / lgbm.txt / cat.cbm）

### 5.3 回测（backtest/）

```python
from sequoia_x.model_selection_v2.backtest.engine import V2BacktestEngine
engine = V2BacktestEngine(data_engine, model_t1, model_t2, model_t3)
result = engine.run("2025-01-01", "2025-12-31")
```

逐日循环：T-1 日收盘特征 → 3 模型预测 → T 日开盘执行交易。

回测参数：
- 初始资金：50万
- 单只股票仓位上限：5万
- 最大持仓：10只
- 佣金 0.025% + 印花税 0.1% + 滑点 0.01%
- 买入阈值：T1 预测概率 > 0.55

### 5.4 模拟盘（simulation/）

目前为占位模块，计划接入 T4 后统一更新。

---

## 6. 当前运行任务

### 6.1 任务：V2 Walk-Forward 评估（运行中）

| 项目 | 详情 |
|------|------|
| **命令** | `python -m sequoia_x.model_selection_v2.evaluate` |
| **PID** | 3541766 |
| **启动时间** | 2026-07-26 18:11 |
| **日志** | `logs/v2_evaluate_20260726_1811.log` |
| **预计完成** | 2026-07-28 晚间（~50-60h） |

**配置**：
- 特征：80 维（12组，含PE/PB + 14维新派生特征）
- 并行：T1∥T2∥T3（3×8核）+ T4（16核）per Fold
- T4 Optuna：**18 trials**（从 60 降低，避免 24h 超时截断）
- 数据：磁盘缓存 mmap 秒级加载（~18s），无需重建

> **2026-07-26 变更日志**:
> - **Bug 修复**: `evaluate.py:100` — `t4_pending` 断点续跑逻辑修复（之前含 t4_pending 的 Fold 会被错误跳过）
> - **T4 trials 优化**: `lstm_optuna_n_trials` 60 → 18（实测 8-15 个即被 timeout 截断，18 个完整搜索质量更优）
> - 数据缓存: 首次构建后 mmap 加载 (~18s)，避免 12min 重复构建
> - 磁盘缓存目录: `data/cache/v2_dataset/` (4.9GB)

> **2026-07-24 变更日志**:
> - 特征 69→80 维（移除 psTTM/pcfNcfTTM，+最大回撤/收益分布/时间日历/价格位置）
> - 腾讯 API 接入 PE(字段39) 和 PB(字段46)，替代不可用的 baostock 估值
> - Fold 内 T1+T2+T3 并行训练
> - TF 线程提升：intra_op 8→16, interop 4→8, OMP 6→10
> - 数据构建 8→16 workers

---

## 7. 运维指南

### 7.1 常用命令

```bash
# 查看运行中的进程
ps aux | grep -E "evaluate|train" | grep -v grep

# 查看内存/CPU
htop -p $(pgrep -f evaluate)

# 查看日志尾部
tail -100 logs/v2_evaluate_optuna_*.log

# 查看 Optuna 搜索进度
python3 -c "
import sqlite3, glob, os
base = 'data/models/v2_selection/optuna'
for db in sorted(glob.glob(f'{base}/*.db')):
    name = os.path.basename(db)
    conn = sqlite3.connect(db)
    total = conn.execute('SELECT COUNT(*) FROM trials').fetchone()[0]
    done = conn.execute(\"SELECT COUNT(*) FROM trials WHERE state='COMPLETE'\").fetchone()[0]
    print(f'{name}: {done}/{total} complete')
    conn.close()
"

# 启动 Walk-Forward 评估
LOGFILE="logs/v2_evaluate_optuna_$(date +%Y%m%d_%H%M).log"
nohup python -m sequoia_x.model_selection_v2.evaluate > "$LOGFILE" 2>&1 &
echo "PID: $! | 日志: $LOGFILE"

# 跳过 Optuna（快速模式）
python -m sequoia_x.model_selection_v2.evaluate --no-optuna

# T4 小规模验证
python -m sequoia_x.model_selection_v2.models.deep_lstm
```

### 7.2 日志位置

| 日志 | 路径 | 内容 |
|------|------|------|
| Walk-Forward 评估 | `logs/v2_evaluate_optuna_*.log` | 完整 Fold 进度 + 结果 |
| 批量训练 | `logs/v2_train_*.log` | T1/T2/T3 训练过程 |
| T4 vs T2 测试 | `logs/t4_vs_t2_test.log` | 对比测试输出 |
| 数据同步 | `logs/pipeline_catchup.log` | 日线同步状态 |

### 7.3 配置调优

`V2Config` 关键参数（`sequoia_x/model_selection_v2/config.py`）：

```python
# 时间窗口
window: int = 120              # 时序窗口（交易日）
predict_horizon_t1: int = 5    # T1: 5日涨跌方向
predict_horizon_t2: int = 20   # T2/T4: 20日超额收益
predict_horizon_t3: int = 20   # T3: 20日波动率

# 采样
sample_start: str = "2020-01-01"
sample_end: str = "2026-07-20"
samples_per_month: int = 2     # 每月2天

# Optuna（树模型）
optuna_n_trials: int = 50      # 每 Fold 搜索次数
optuna_timeout: int = 7200     # 超时 2h

# Optuna（LSTM）
lstm_optuna_n_trials: int = 18   # 搜索 6 个核心参数（18: Hyperband完整分配，20h内完成）
lstm_optuna_timeout: int = 86400 # 超时 24h

# 多核并行
n_jobs: int = 8                    # 树模型内部线程（×3并行=24核）
lstm_tf_intraop_threads: int = 16  # TF 单 op 并行
lstm_tf_interop_threads: int = 8   # TF op 间并行
lstm_omp_num_threads: int = 10     # BLAS/MKL 线程

# LSTM 架构（默认值，Optuna 覆盖核心参数）
lstm_units: int = 128
lstm_num_transformers: int = 2
lstm_dropout_rate: float = 0.3
lstm_learning_rate: float = 0.001
lstm_l2_reg: float = 1e-4
lstm_batch_size: int = 64

# Walk-Forward
purge_gap: int = 22            # 训练/测试间隔（交易日）

# 回测
initial_capital: float = 500_000.0
max_positions: int = 10
min_buy_prob: float = 0.55     # T1 买入概率阈值
```

### 7.4 目录速查

| 目录 | 用途 |
|------|------|
| `data/sequoia_v2.db` | 主数据库 |
| `data/cache/v2_dataset/` | 数据集磁盘缓存（mmap 秒级加载） |
| `data/models/v2_selection/` | 模型文件 + Walk-Forward 结果 |
| `data/models/v2_selection/optuna/` | 树模型 Optuna 搜索记录（SQLite） |
| `data/models/v2_selection/optuna_t4_lstm.db` | T4 LSTM Optuna 搜索记录 |
| `logs/` | 所有运行日志 |
| `output/backtest_v2/` | V2 回测结果输出 |
| `output/backtest_lstm/` | V1 LSTM 回测输出（历史参考） |
| `output/sim_lstm/` | V1 LSTM 模拟盘输出（历史参考） |

---

## 8. 结果解读

### 8.1 关键指标

| 指标 | 含义 | 好/坏判断 |
|------|------|-----------|
| **Rank IC** | 预测排名与实际收益的 Spearman 相关 | >0.03 可用，>0.05 好，<0 差 |
| **RMSE** | 预测误差均方根 | 越小越好，需结合基准 |
| **AUC** | 二分类区分能力 | >0.55 可用，>0.6 好，0.5=随机 |
| **方向胜率** | 买入信号正确率 | >52% 有正期望 |
| **>0 比例** | Rank IC 为正的 Fold 占比 | >60% 稳定 |

### 8.2 历史结果（第一轮：2026-07-23）

> ⚠️ 此轮因估值数据（psTTM/pcfNcfTTM）在近期大量缺失导致特征噪声，已于 07-24 停止。结果仅供参考。

| 指标 | 值 | 问题诊断 |
|------|-----|----------|
| T1 AUC | 0.485 | 测试集远低于 CV 的 0.744 |
| T2 Rank IC | -0.026 | 负值，树模型时序过拟合 |
| 方向胜率 | 51.86% | 微弱正期望 |

**根因**：90天前估值 100% 完整（baostock），最近30天仅 12%（腾讯/新浪不提供），特征分布断崖 → 模型失效。

**修复方案**：
1. 移除 psTTM/pcfNcfTTM，只保留腾讯可提供的 peTTM + pbMRQ
2. 新增 14 维纯 OHLCV 派生特征（最大回撤/收益分布/时间日历/价格位置）
3. 特征总维度：69 → 80

### 8.3 第二轮结果（2026-07-24 启动，运行中）

> 待 Fold 3 完成后填入。

---

> **文档维护规则**：每次 V2 功能迭代后，在对应章节追加新内容。重大变更在目录中标注日期。
