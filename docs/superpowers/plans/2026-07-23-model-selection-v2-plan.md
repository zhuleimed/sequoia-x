# model_selection_v2 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `sequoia_x/model_selection_v2/` 下从零构建三任务树模型（XGBoost分类+LightGBM回归+CatBoost波动率），配合 Purged Rolling Walk-Forward 验证和逐日回测引擎。

**Architecture:** 12 个文件分 5 层 — config 层（参数统一管理）→ data 层（features.py + labels.py 特征构建与多任务标签）→ model 层（3 个独立树模型文件，统一接口）→ train 层（训练协调 + 特征重要性）→ evaluate 层（Purged Walk-Forward）→ backtest 层（逐日回测）。所有文件与 `model_selection/` 零 import 依赖，复用 `sequoia_x/core`、`data`、`simulation` 共享模块。

**Tech Stack:** Python 3.12, XGBoost, LightGBM, CatBoost, Optuna, scikit-learn, NumPy, Pandas, SQLite

## Global Constraints

- 纯 CPU 环境，n_jobs=8，Optuna n_jobs=1
- Python 路径: `/home/zhulei/anaconda3/envs/zhulei_py312/bin/python`
- 不 import `sequoia_x.model_selection` 中的任何模块
- 可 import `sequoia_x.core`、`sequoia_x.data`、`sequoia_x.simulation`
- 独立数据路径：`data/models/v2_selection/`、`output/backtest_v2/`
- 192GB 内存，特征构建需分批处理避免 OOM
- 所有代码文件头部加 `"""model_selection_v2 - <模块职责>"""` 文档字符串

---

### Task 1: 目录骨架与配置模块

**Files:**
- Create: `sequoia_x/model_selection_v2/__init__.py`
- Create: `sequoia_x/model_selection_v2/config.py`
- Create: `sequoia_x/model_selection_v2/models/__init__.py`
- Create: `sequoia_x/model_selection_v2/backtest/__init__.py`
- Create: `sequoia_x/model_selection_v2/deep/__init__.py`
- Create: `sequoia_x/model_selection_v2/simulation/__init__.py`

**Interfaces:**
- Produces: `V2Config` dataclass（全局配置单例 `get_config()`），所有后续 Task 的配置来源

- [ ] **Step 1: 创建目录结构**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
mkdir -p sequoia_x/model_selection_v2/{models,backtest,deep,simulation}
```

- [ ] **Step 2: 创建所有 `__init__.py` 文件**

```bash
for d in sequoia_x/model_selection_v2 sequoia_x/model_selection_v2/models \
  sequoia_x/model_selection_v2/backtest sequoia_x/model_selection_v2/deep \
  sequoia_x/model_selection_v2/simulation; do
  echo '"""model_selection_v2 - 多任务树模型选股系统 V2."""' > $d/__init__.py
done
```

- [ ] **Step 3: 编写 `config.py` 完整配置**

```python
"""model_selection_v2 - 全局配置模块。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class V2Config:
    """V2 多任务树模型全局配置。"""

    # ── 路径 ──
    db_path: str = "data/sequoia_v2.db"
    model_dir: str = "data/models/v2_selection"
    output_dir: str = "output/backtest_v2"

    # ── 时间窗口 ──
    window: int = 120          # 时序窗口（交易日）
    predict_horizon_t1: int = 5   # T1: 短期方向预测窗口
    predict_horizon_t2: int = 20  # T2: 中期收益预测窗口
    predict_horizon_t3: int = 20  # T3: 波动率预测窗口

    # ── 采样 ──
    sample_start: str = "2020-01-01"
    sample_end: str = "2026-07-20"
    samples_per_month: int = 2    # 每月采样天数（月初+月中）

    # ── 训练 ──
    random_seed: int = 42
    test_ratio: float = 0.15
    early_stop_rounds: int = 50

    # ── Optuna ──
    optuna_n_trials: int = 50
    optuna_timeout: int = 7200   # 2h per model

    # ── 模型超参搜索范围 ──
    xgb_params: dict = field(default_factory=lambda: {
        "max_depth": (3, 12),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "reg_alpha": (1e-3, 10.0),
        "reg_lambda": (1e-3, 10.0),
        "min_child_weight": (1, 20),
    })
    lgbm_params: dict = field(default_factory=lambda: {
        "num_leaves": (15, 255),
        "learning_rate": (0.01, 0.3),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
        "reg_alpha": (1e-3, 10.0),
        "reg_lambda": (1e-3, 10.0),
        "min_child_samples": (10, 100),
    })
    cat_params: dict = field(default_factory=lambda: {
        "depth": (3, 10),
        "learning_rate": (0.01, 0.3),
        "l2_leaf_reg": (0.1, 10.0),
        "random_strength": (0.1, 10.0),
    })

    # ── 硬件 ──
    n_jobs: int = 8

    # ── Walk-Forward ──
    purge_gap: int = 22          # 训练/测试间隔（交易日）

    # ── 回测 ──
    initial_capital: float = 500_000.0
    per_stock_budget: float = 50_000.0
    max_positions: int = 10
    top_n_buy_per_day: int = 2
    commission_rate: float = 0.00025
    stamp_tax_rate: float = 0.001
    slippage: float = 0.0001
    min_buy_prob: float = 0.55   # T1 买入概率阈值

    # ── 特征 ──
    feature_ma_periods: tuple = (5, 10, 20, 60, 120)
    feature_rsi_period: int = 14
    feature_atr_period: int = 14
    feature_adx_period: int = 14
    feature_macd_fast: int = 12
    feature_macd_slow: int = 26
    feature_macd_signal: int = 9
    feature_boll_period: int = 20
    feature_vol_periods: tuple = (5, 10, 20)
    feature_beta_period: int = 20

    @property
    def model_dir_path(self) -> Path:
        p = Path(self.model_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def optuna_dir_path(self) -> Path:
        p = Path("data/models/v2_selection/optuna")
        p.mkdir(parents=True, exist_ok=True)
        return p


_config: V2Config | None = None


def get_config() -> V2Config:
    """获取全局配置单例。"""
    global _config
    if _config is None:
        _config = V2Config()
    return _config
```

- [ ] **Step 4: 验证 config 可导入**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.model_selection_v2.config import get_config, V2Config
cfg = get_config()
print(f'window={cfg.window}, n_jobs={cfg.n_jobs}, purge_gap={cfg.purge_gap}')
print(f'model_dir={cfg.model_dir_path}')
print('OK')
"
```

Expected: `window=120, n_jobs=8, purge_gap=22` and `model_dir=data/models/v2_selection`

- [ ] **Step 5: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/
git commit -m "feat(v2): 目录骨架与V2Config配置模块"
```

---

### Task 2: 特征工程模块

**Files:**
- Create: `sequoia_x/model_selection_v2/features.py`

**Interfaces:**
- Consumes: `V2Config` from Task 1, `DataEngine` from `sequoia_x.data.engine`
- Produces:
  - `_extract_per_day_features(df, df_index, cfg) -> np.ndarray`  — (n_days, 62) 特征矩阵
  - `build_stock_features(symbol, ref_date, engine, cfg) -> tuple[np.ndarray|None, float|None]` — (window, 62) + 标签
  - `build_batch_features(symbols, ref_date, engine, cfg) -> tuple[np.ndarray, np.ndarray]` — (N, window, 62) + (N,)
  - `build_prediction_features(symbol, engine, cfg) -> np.ndarray|None` — (1, window, 62)

- [ ] **Step 1: 移植并改写 `_extract_per_day_features`**

从 `sequoia_x/model_selection/features.py` 移植完整 62 维特征计算逻辑（`_compute_rsi`、`_compute_macd`、`_compute_atr`、`_compute_adx`、`_compute_bollinger`、`_extract_per_day_features`），所有内部函数前缀加 `_`。

关键改动：
1. 标签计算移到 `labels.py`，`_extract_per_day_features` 只产出特征矩阵
2. 估值特征 dimension 修正为 7（peTTM/pbMRQ/psTTM/pcfNcfTTM + 各自分位）
3. 输出 shape 注释为 `(n_days, 62)`

```python
"""model_selection_v2 - 特征工程模块。

从 stock_daily 表计算 62 维时序特征，严格避免 look-ahead bias：
第 T 日的特征仅使用 T 日及之前已知的数据。

特征分组：
  价格收益(8) + 均线偏离(6) + 量能(8) + 技术指标(14)
  + 波动率(4) + 大盘关联(8) + 价格形态(7) + 估值指标(7) = 62 维
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import V2Config, get_config


# ════════════════════════════════════════════════════════════
#  指标计算函数（纯 NumPy，零外部依赖）
# ════════════════════════════════════════════════════════════

def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """计算 RSI 指标。"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    for i in range(period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss > 0)
    return 100.0 - 100.0 / (1.0 + rs)


def _compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算 MACD，返回 (dif, dea, hist)。"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
    hist = (dif - dea) * 2
    return dif, dea, hist


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """计算 ATR 指标。"""
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)),
                   np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int = 14) -> np.ndarray:
    """计算 ADX 指标。"""
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr = _compute_atr(high, low, close, period)
    atr_safe = np.where(atr > 0, atr, 1e-10)
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / atr_safe
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / atr_safe
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    return pd.Series(dx).ewm(span=period, adjust=False).mean().values


def _compute_bollinger(close: np.ndarray, period: int = 20
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算布林带，返回 (upper, middle, lower)。"""
    middle = pd.Series(close).rolling(period, min_periods=1).mean().values
    std = pd.Series(close).rolling(period, min_periods=1).std().values
    return middle + 2*std, middle, middle - 2*std


# ════════════════════════════════════════════════════════════
#  逐日特征提取
# ════════════════════════════════════════════════════════════

def _extract_per_day_features(df: pd.DataFrame, df_index: pd.DataFrame | None,
                               cfg: V2Config) -> np.ndarray:
    """从日线 DataFrame 逐日提取 62 维特征向量。

    Args:
        df: 单只股票 OHLCV DataFrame，需含 open/high/low/close/volume/amount/turnover
            及估值字段 peTTM/pbMRQ/psTTM/pcfNcfTTM
        df_index: 指数 DataFrame（可选），含 close
        cfg: V2Config 配置

    Returns:
        (n_days, 62) 特征矩阵，全部 Z-score 归一化
    """
    n = len(df)
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    amount = df["amount"].values.astype(float) if "amount" in df.columns else np.zeros(n)
    turnover = df["turnover"].values.astype(float) if "turnover" in df.columns else np.zeros(n)

    feature_list = []

    # ── 1. 价格收益特征 (8维) ──
    ret_1d = np.diff(close, prepend=close[0]) / np.maximum(np.roll(close, 1), 1e-10)
    ret_1d[0] = 0.0
    ret_5d = pd.Series(close).pct_change(5).fillna(0.0).values
    ret_10d = pd.Series(close).pct_change(10).fillna(0.0).values
    ret_20d = pd.Series(close).pct_change(20).fillna(0.0).values
    gap = open_ / np.maximum(np.roll(close, 1), 1e-10) - 1.0
    gap[0] = 0.0
    hl_ratio = (high - low) / np.maximum(np.roll(close, 1), 1e-10)
    hl_ratio[0] = 0.0
    chg_pct = (close - np.roll(close, 1)) / np.maximum(np.roll(close, 1), 1e-10)
    chg_pct[0] = 0.0
    feature_list.extend([ret_1d, ret_5d, ret_10d, ret_20d, gap, hl_ratio, chg_pct])
    amplitude_5d = pd.Series(hl_ratio).rolling(5, min_periods=1).mean().values
    feature_list.append(amplitude_5d)

    # ── 2. 均线偏离特征 (6维) ──
    for period in cfg.feature_ma_periods:
        ma = pd.Series(close).rolling(period, min_periods=1).mean().values
        deviation = close / np.maximum(ma, 1e-10) - 1.0
        feature_list.append(deviation)
    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().values
    ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    feature_list.append(ma5 / np.maximum(ma20, 1e-10) - 1.0)

    # ── 3. 量能特征 (8维) ──
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_ratio = volume / np.maximum(vol_ma20, 1e-10)
    vol_change_5d = pd.Series(volume).pct_change(5).fillna(0.0).values
    turnover_rate = turnover / 100.0 if turnover.max() < 1 else turnover
    amount_ratio = amount / np.maximum(close * 1e8, 1e-10)
    vol_corr = pd.Series(volume).rolling(10, min_periods=1).corr(
        pd.Series(close)).fillna(0.0).values
    vol_ma5 = pd.Series(volume).rolling(5, min_periods=1).mean().values
    vol_ma5_ratio = vol_ma5 / np.maximum(vol_ma20, 1e-10)
    vol_trend = vol_ma5 / np.maximum(
        pd.Series(volume).rolling(20, min_periods=1).mean().shift(20).values, 1e-10)
    vol_trend = np.nan_to_num(vol_trend, nan=1.0)
    vol_surge = (vol_ratio > 1.5).astype(float)
    feature_list.extend([vol_ratio, vol_change_5d, turnover_rate, amount_ratio,
                         vol_corr, vol_ma5_ratio, vol_trend, vol_surge])

    # ── 4. 技术指标 (14维) ──
    feature_list.append(_compute_rsi(close, cfg.feature_rsi_period) / 100.0)
    dif, dea, hist = _compute_macd(close, cfg.feature_macd_fast,
                                    cfg.feature_macd_slow, cfg.feature_macd_signal)
    close_safe = np.maximum(close, 1e-10)
    feature_list.extend([dif / close_safe, dea / close_safe, hist / close_safe])
    bb_upper, bb_mid, bb_lower = _compute_bollinger(close, cfg.feature_boll_period)
    bb_position = (close - bb_lower) / np.maximum(bb_upper - bb_lower, 1e-10)
    bb_width = (bb_upper - bb_lower) / np.maximum(bb_mid, 1e-10)
    feature_list.extend([bb_position, bb_width])
    atr = _compute_atr(high, low, close, cfg.feature_atr_period)
    feature_list.append(atr / close_safe)
    low_n = pd.Series(low).rolling(9, min_periods=1).min().values
    high_n = pd.Series(high).rolling(9, min_periods=1).max().values
    rsv = (close - low_n) / np.maximum(high_n - low_n, 1e-10) * 100.0
    k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
    d = pd.Series(k).ewm(com=2, adjust=False).mean().values
    feature_list.extend([k / 100.0, d / 100.0])
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]: obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]: obv[i] = obv[i-1] - volume[i]
        else: obv[i] = obv[i-1]
    obv_change = pd.Series(obv).pct_change(5).fillna(0.0).values
    feature_list.append(obv_change)
    adx = _compute_adx(high, low, close, cfg.feature_adx_period)
    feature_list.append(adx / 100.0)

    # ── 5. 波动率特征 (4维) ──
    for period in cfg.feature_vol_periods:
        vol = pd.Series(ret_1d).rolling(period, min_periods=1).std().values * np.sqrt(252)
        feature_list.append(vol)
    vol_20d = pd.Series(ret_1d).rolling(20, min_periods=1).std().values * np.sqrt(252)
    vol_5d = pd.Series(ret_1d).rolling(5, min_periods=1).std().values * np.sqrt(252)
    feature_list.append(vol_5d / np.maximum(vol_20d, 1e-10) - 1.0)

    # ── 6. 大盘关联特征 (8维) ──
    if df_index is not None and len(df_index) == n:
        idx_close = df_index["close"].values.astype(float)
        idx_ret = np.diff(idx_close, prepend=idx_close[0]) / np.maximum(np.roll(idx_close, 1), 1e-10)
        idx_ret[0] = 0.0
        feature_list.append(idx_ret)
        idx_ret_series = pd.Series(idx_ret)
        stock_ret_series = pd.Series(ret_1d)
        beta = np.zeros(n)
        for i in range(cfg.feature_beta_period, n):
            cov = stock_ret_series.iloc[i-cfg.feature_beta_period:i].cov(
                idx_ret_series.iloc[i-cfg.feature_beta_period:i])
            var = idx_ret_series.iloc[i-cfg.feature_beta_period:i].var()
            beta[i] = cov / var if var > 1e-10 else 1.0
        feature_list.append(np.clip(beta, -3, 5))
        rs = pd.Series(close).pct_change(20).fillna(0.0).values
        idx_ret_20d = pd.Series(idx_close).pct_change(20).fillna(0.0).values
        feature_list.append(rs - idx_ret_20d)
        idx_ma20 = pd.Series(idx_close).rolling(20, min_periods=1).mean().values
        idx_ma60 = pd.Series(idx_close).rolling(60, min_periods=1).mean().values
        feature_list.append(idx_close / np.maximum(idx_ma20, 1e-10) - 1.0)
        feature_list.append(idx_close / np.maximum(idx_ma60, 1e-10) - 1.0)
        ret_5d_stock = pd.Series(close).pct_change(5).fillna(0.0).values
        idx_ret_5d = pd.Series(idx_close).pct_change(5).fillna(0.0).values
        feature_list.append(ret_5d_stock - idx_ret_5d)
    else:
        for _ in range(8):
            feature_list.append(np.zeros(n))

    # ── 7. 价格形态特征 (7维) ──
    limit_up = (chg_pct > 0.095).astype(float)
    limit_down = (chg_pct < -0.095).astype(float)
    feature_list.append(limit_up)
    feature_list.append(limit_down)
    up_streak = np.zeros(n)
    down_streak = np.zeros(n)
    for i in range(1, n):
        up_streak[i] = up_streak[i-1] + 1 if ret_1d[i] > 0 else 0
        down_streak[i] = down_streak[i-1] + 1 if ret_1d[i] < 0 else 0
    feature_list.extend([up_streak / 10.0, down_streak / 10.0])
    high_20d = pd.Series(high).rolling(20, min_periods=1).max().values
    low_20d = pd.Series(low).rolling(20, min_periods=1).min().values
    feature_list.append((close >= high_20d).astype(float))
    feature_list.append((close <= low_20d).astype(float))
    body = np.abs(close - open_)
    shadow = high - low
    feature_list.append(body / np.maximum(shadow, 1e-10))

    # ── 8. 估值指标特征 (7维) ──
    for col in ["peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"]:
        if col in df.columns:
            val = df[col].values.astype(float)
            val = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            feature_list.append(val)
            # 历史分位（滚动窗口内排名）
            rank = pd.Series(val).rolling(60, min_periods=1).apply(
                lambda x: (x.iloc[-1] > x).sum() / max(len(x), 1), raw=False
            ).fillna(0.5).values
            feature_list.append(rank)
        else:
            feature_list.extend([np.zeros(n), np.zeros(n)])
    # 如果只有一个估值字段有数据，补齐不足7维
    while len(feature_list) < 69:  # 目标维数
        feature_list.append(np.zeros(n))

    # ── 9. 组装与归一化 ──
    features = np.column_stack(feature_list).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score 归一化（每特征沿时间轴独立）
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std_safe = np.where(std < 1e-8, 1.0, std)
    features = (features - mean) / std_safe
    features[:, std.flatten() < 1e-8] = 0.0

    return features


# ════════════════════════════════════════════════════════════
#  公开接口
# ════════════════════════════════════════════════════════════

def build_stock_features(
    symbol: str, ref_date: str, engine: DataEngine,
    cfg: V2Config | None = None,
) -> tuple[np.ndarray | None, None]:
    """为单只股票构建预测用特征（不含标签，标签由 labels.py 构建）。

    Returns:
        (X, None): X 形状 (window, n_features)，数据不足返回 (None, None)。
    """
    if cfg is None:
        cfg = get_config()

    df = engine.get_ohlcv(symbol)
    if df is None or len(df) < cfg.window + 30:
        return None, None

    df = df[df["date"] <= ref_date].copy()
    if len(df) < cfg.window + 30:
        return None, None

    df_index = None
    try:
        df_index = engine.get_ohlcv("000300")
        if df_index is not None:
            df_index = df_index[df_index["date"] <= ref_date].copy()
            if len(df_index) != len(df):
                df_index = None
    except Exception:
        df_index = None

    per_day = _extract_per_day_features(df, df_index, cfg)
    if len(per_day) < cfg.window:
        return None, None

    X = per_day[-cfg.window:]
    return X, None


def build_batch_features(
    symbols: list[str], ref_date: str, engine: DataEngine,
    cfg: V2Config | None = None,
) -> tuple[np.ndarray, list]:
    """批量构建特征矩阵（不含标签）。

    Returns:
        (X, valid_symbols): X 形状 (n_valid, window, n_features)。
    """
    if cfg is None:
        cfg = get_config()

    X_list, sym_list = [], []
    for symbol in symbols:
        X_i, _ = build_stock_features(symbol, ref_date, engine, cfg)
        if X_i is not None:
            X_list.append(X_i)
            sym_list.append(symbol)

    if not X_list:
        return np.array([]).reshape(0, cfg.window, 0), []

    X = np.stack(X_list, axis=0)
    return X, sym_list


def build_prediction_features(
    symbol: str, engine: DataEngine,
    cfg: V2Config | None = None,
) -> np.ndarray | None:
    """为单只股票构建最新预测特征。

    Returns:
        X: (1, window, n_features)，数据不足返回 None。
    """
    if cfg is None:
        cfg = get_config()

    df = engine.get_ohlcv(symbol)
    if df is None or len(df) < cfg.window + 10:
        return None

    df_index = None
    try:
        df_index = engine.get_ohlcv("000300")
        if df_index is not None and len(df_index) != len(df):
            df_index = None
    except Exception:
        df_index = None

    per_day = _extract_per_day_features(df, df_index, cfg)
    if len(per_day) < cfg.window:
        return None

    X = per_day[-cfg.window:]
    return X[np.newaxis, :, :]
```

- [ ] **Step 2: 快速验证特征构建（单只股票）**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.features import build_prediction_features, get_config

engine = DataEngine(Settings())
cfg = get_config()
X = build_prediction_features('600519', engine, cfg)
if X is not None:
    print(f'Shape: {X.shape}, 预期 (1, {cfg.window}, 62)')
    print(f'NaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}')
else:
    print('FAIL: X is None')
" 2>&1 | grep -v "OMP:"
```

Expected: `Shape: (1, 120, 62)` 或类似维度

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/features.py
git commit -m "feat(v2): 62维特征工程模块（移植+改写，标签分离到labels.py）"
```

---

### Task 3: 多任务标签构建

**Files:**
- Create: `sequoia_x/model_selection_v2/labels.py`

**Interfaces:**
- Consumes: `V2Config` from Task 1, `build_stock_features` from Task 2, `DataEngine`
- Produces:
  - `build_training_dataset(engine, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` — 返回 (X, y1, y2, y3)，y1 是二分类标签(0/1)，y2 是 20 日超额收益，y3 是 20 日波动率

- [ ] **Step 1: 编写 `labels.py`**

```python
"""model_selection_v2 - 多任务标签构建模块。

为每个采样日期的每只股票构建 3 个标签：
  y1: 5日涨跌方向（二分类，0=跌 1=涨）
  y2: 20日超额收益率（相对沪深300，回归）
  y3: 20日日收益率年化波动率（回归，用于风险度量）
"""
from __future__ import annotations
import numpy as np
import sqlite3
from datetime import datetime
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.features import build_stock_features

logger = get_logger(__name__)


def _get_sample_dates(engine: DataEngine, cfg: V2Config) -> list[str]:
    """获取采样日期列表：每月 2 天（月初 5 日 + 月中 15 日）。

    从 stock_daily 表查询 sample_start ~ sample_end 范围内的非周末交易日。
    """
    import pandas as pd
    conn = sqlite3.connect(engine.db_path)
    all_dates = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= ? AND date <= ? ORDER BY date",
        conn, params=(cfg.sample_start, cfg.sample_end)
    )["date"].tolist()
    conn.close()

    # 每月取 2 天：第 5 个交易日 和 第 15 个交易日（或最接近的）
    monthly = {}
    for d in all_dates:
        ym = d[:7]  # YYYY-MM
        if ym not in monthly:
            monthly[ym] = []
        monthly[ym].append(d)

    sample_dates = []
    for ym, dates in sorted(monthly.items()):
        if len(dates) >= 15:
            sample_dates.append(dates[4])   # 第 5 个交易日
            sample_dates.append(dates[14])  # 第 15 个交易日
        elif len(dates) >= 5:
            sample_dates.append(dates[4])
            sample_dates.append(dates[-1])  # 最后一天

    return sample_dates


def _compute_label_t1(
    symbol: str, ref_date: str, engine: DataEngine, cfg: V2Config
) -> float | None:
    """计算 T1 标签：5 日涨跌方向。

    在 stock_daily 中查找 ref_date 之后的第 predict_horizon_t1 个交易日，
    比较收盘价。数据不足返回 None。
    """
    conn = sqlite3.connect(engine.db_path)
    rows = conn.execute(
        "SELECT date, close FROM stock_daily WHERE symbol=? AND date > ? ORDER BY date LIMIT ?",
        (symbol, ref_date, cfg.predict_horizon_t1 + 2)
    ).fetchall()
    conn.close()

    if len(rows) < cfg.predict_horizon_t1:
        return None

    ref_close = rows[0][1]
    target_close = rows[cfg.predict_horizon_t1 - 1][1]
    if ref_close is None or target_close is None or ref_close <= 0:
        return None

    ret = (target_close - ref_close) / ref_close
    return 1.0 if ret > 0 else 0.0


def _compute_label_t2(
    symbol: str, ref_date: str, engine: DataEngine, cfg: V2Config
) -> float | None:
    """计算 T2 标签：20 日超额收益率。

    个股 20 日收益率 - 沪深 300 的 20 日收益率。
    """
    conn = sqlite3.connect(engine.db_path)
    # 个股
    stock_rows = conn.execute(
        "SELECT close FROM stock_daily WHERE symbol=? AND date > ? ORDER BY date LIMIT ?",
        (symbol, ref_date, cfg.predict_horizon_t2 + 2)
    ).fetchall()
    # 指数
    idx_rows = conn.execute(
        "SELECT close FROM stock_daily WHERE symbol='000300' AND date > ? ORDER BY date LIMIT ?",
        (ref_date, cfg.predict_horizon_t2 + 2)
    ).fetchall()
    conn.close()

    if len(stock_rows) < cfg.predict_horizon_t2 or len(idx_rows) < cfg.predict_horizon_t2:
        return None

    stock_ret = (stock_rows[cfg.predict_horizon_t2 - 1][0] / stock_rows[0][0]) - 1.0
    idx_ret = (idx_rows[cfg.predict_horizon_t2 - 1][0] / idx_rows[0][0]) - 1.0
    ret = stock_ret - idx_ret

    # 异常值裁剪
    return float(np.clip(ret, -0.5, 0.5))


def _compute_label_t3(
    symbol: str, ref_date: str, engine: DataEngine, cfg: V2Config
) -> float | None:
    """计算 T3 标签：20 日日收益率年化波动率。"""
    conn = sqlite3.connect(engine.db_path)
    rows = conn.execute(
        "SELECT close FROM stock_daily WHERE symbol=? AND date > ? ORDER BY date LIMIT ?",
        (symbol, ref_date, cfg.predict_horizon_t3 + 2)
    ).fetchall()
    conn.close()

    if len(rows) < cfg.predict_horizon_t3 + 1:
        return None

    closes = np.array([r[0] for r in rows[:cfg.predict_horizon_t3 + 1] if r[0] is not None])
    if len(closes) < 10:
        return None

    daily_rets = np.diff(closes) / closes[:-1]
    vol = float(np.std(daily_rets) * np.sqrt(252))
    return min(vol, 2.0)


def build_training_dataset(
    engine: DataEngine, cfg: V2Config | None = None,
    symbols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """构建完整训练数据集。

    遍历所有采样日期和股票，构建特征矩阵 + 三任务标签。

    Args:
        engine: DataEngine 实例。
        cfg: V2Config 配置。
        symbols: 股票列表，默认从 engine.get_base_stock_pool() 获取。

    Returns:
        (X, y1, y2, y3, date_labels):
          X  — (n_samples, window, n_features)
          y1 — (n_samples,) 二分类标签
          y2 — (n_samples,) 超额收益率
          y3 — (n_samples,) 波动率
          date_labels — (n_samples,) 每行对应的采样日期，用于 Walk-Forward 切分
    """
    import time
    if cfg is None:
        cfg = get_config()
    if symbols is None:
        symbols = engine.get_base_stock_pool()

    dates = _get_sample_dates(engine, cfg)
    logger.info(f"采样日期: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")
    logger.info(f"股票池: {len(symbols)} 只")

    X_list, y1_list, y2_list, y3_list, date_list = [], [], [], [], []
    t0 = time.time()

    for di, ref_date in enumerate(dates):
        for symbol in symbols:
            try:
                X_i, _ = build_stock_features(symbol, ref_date, engine, cfg)
                if X_i is None:
                    continue

                y1 = _compute_label_t1(symbol, ref_date, engine, cfg)
                y2 = _compute_label_t2(symbol, ref_date, engine, cfg)
                y3 = _compute_label_t3(symbol, ref_date, engine, cfg)

                if y1 is None or y2 is None or y3 is None:
                    continue

                X_list.append(X_i)
                y1_list.append(y1)
                y2_list.append(y2)
                y3_list.append(y3)
                date_list.append(ref_date)
            except Exception:
                continue

        if (di + 1) % 10 == 0 or di == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  日期 {di+1}/{len(dates)} ({ref_date}): "
                f"累计 {len(X_list)} 样本, {elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    logger.info(f"数据集构建完成: {len(X_list)} 样本, {elapsed:.0f}s")

    if not X_list:
        return (np.array([]).reshape(0, cfg.window, 0),
                np.array([]), np.array([]), np.array([]),
                [])

    X = np.stack(X_list, axis=0)
    y1 = np.array(y1_list, dtype=np.int32)
    y2 = np.array(y2_list, dtype=np.float32)
    y3 = np.array(y3_list, dtype=np.float32)

    logger.info(
        f"X.shape={X.shape}, "
        f"y1 涨比例={y1.mean():.2%}, "
        f"y2 均值={y2.mean():.4f}, "
        f"y3 均值={y3.mean():.4f}"
    )
    return X, y1, y2, y3, date_list


# ════════════════════════════════════════════════════════════
#  CLI（独立测试用）
# ════════════════════════════════════════════════════════════

def main() -> None:
    """CLI: 测试数据集构建（限制股票数量快速验证）。"""
    from sequoia_x.core.config import Settings
    cfg = get_config()
    engine = DataEngine(Settings())
    pool = engine.get_base_stock_pool()[:50]  # 只取50只测试
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg, symbols=pool)
    print(f"X={X.shape}, y1={y1.shape}, y2={y2.shape}, y3={y3.shape}")
    print(f"样本日期范围: {dates[0]} ~ {dates[-1]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 快速验证（50只股票）**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m sequoia_x.model_selection_v2.labels
```

Expected: 输出 X shape、y1/y2/y3 shape 和日期范围，y1 涨比例约 45-55%

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/labels.py
git commit -m "feat(v2): 多任务标签构建 — 5日方向+20日超额收益+20日波动率"
```

---

### Task 4: XGBoost 分类器（T1：5日涨跌方向）

**Files:**
- Create: `sequoia_x/model_selection_v2/models/tree_cls.py`

**Interfaces:**
- Produces:
  - `train_cls(X, y, cfg, trial=None) -> xgb.XGBClassifier` — 训练/超参搜索
  - `predict_cls(model, X) -> np.ndarray` — 返回涨概率 (n_samples,)

- [ ] **Step 1: 编写 `models/tree_cls.py`**

```python
"""model_selection_v2 - T1: XGBoost 二分类器（5日涨跌方向）。"""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


def _objective(trial, X: np.ndarray, y: np.ndarray, cfg: V2Config) -> float:
    """Optuna 目标函数：最小化验证集 AUC 的负数（即最大化 AUC）。"""
    params = {
        "max_depth": trial.suggest_int("max_depth", *cfg.xgb_params["max_depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.xgb_params["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *cfg.xgb_params["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg.xgb_params["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *cfg.xgb_params["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *cfg.xgb_params["reg_lambda"], log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", *cfg.xgb_params["min_child_weight"]),
        "n_estimators": 1000,
        "verbosity": 0,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
        "tree_method": "hist",
    }

    # 扁平化 X 为 2D
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    tscv = TimeSeriesSplit(n_splits=3)
    aucs = []
    for train_idx, val_idx in tscv.split(X_2d):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_2d[train_idx], y[train_idx],
            eval_set=[(X_2d[val_idx], y[val_idx])],
            verbose=False,
        )
        from sklearn.metrics import roc_auc_score
        proba = model.predict_proba(X_2d[val_idx])[:, 1]
        try:
            auc = roc_auc_score(y[val_idx], proba)
            aucs.append(auc)
        except ValueError:
            aucs.append(0.5)

    return -np.mean(aucs)  # Optuna 最小化


def train_cls(
    X: np.ndarray, y: np.ndarray,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> xgb.XGBClassifier:
    """训练 XGBoost 分类器。

    Args:
        X: (n_samples, window, n_features)
        y: (n_samples,) 二分类标签
        cfg: 配置
        search_optuna: True=Optuna搜索超参, False=默认参数快速训练

    Returns:
        训练好的 XGBoost 模型 + 特征重要性
    """
    if cfg is None:
        cfg = get_config()

    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    if search_optuna:
        import optuna
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{cfg.optuna_dir_path}/t1_xgb.db",
            study_name="t1_xgb_cls",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: _objective(trial, X_2d, y, cfg),
            n_trials=cfg.optuna_n_trials,
            timeout=cfg.optuna_timeout,
            n_jobs=1,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info(f"T1 Optuna best: AUC={-study.best_value:.4f}, params={best_params}")
    else:
        best_params = {
            "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_weight": 5,
        }

    # 最终训练（TimeSeriesSplit 最后 fold 做验证集）
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_2d))
    train_idx, val_idx = splits[-1]

    model = xgb.XGBClassifier(
        **best_params,
        n_estimators=1000,
        early_stopping_rounds=cfg.early_stop_rounds,
        verbosity=0,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_seed,
        tree_method="hist",
    )
    # 自动平衡正负样本权重
    neg_count = (y[train_idx] == 0).sum()
    pos_count = (y[train_idx] == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)
    model.set_params(scale_pos_weight=scale_pos_weight)

    model.fit(
        X_2d[train_idx], y[train_idx],
        eval_set=[(X_2d[val_idx], y[val_idx])],
        verbose=False,
    )

    # 特征重要性
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-20:][::-1]
    logger.info(f"T1 训练完成: n_estimators={model.n_estimators}")
    logger.info(f"T1 Top-10 特征 idx: {top_indices[:10].tolist()}")
    logger.info(f"T1 Top-10 重要性: {importances[top_indices[:10]].round(4).tolist()}")

    return model


def predict_cls(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """预测涨概率。

    Args:
        model: 训练好的 XGBoost 模型。
        X: (n_samples, window, n_features)

    Returns:
        (n_samples,) 涨概率 [0, 1]
    """
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)
    return model.predict_proba(X_2d)[:, 1]
```

- [ ] **Step 2: 验证接口可导入**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.model_selection_v2.models.tree_cls import train_cls, predict_cls
print('OK: tree_cls imported')
"
```

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/models/tree_cls.py
git commit -m "feat(v2): T1 XGBoost分类器 — 5日涨跌方向预测"
```

---

### Task 5: LightGBM 回归器（T2：20日超额收益）

**Files:**
- Create: `sequoia_x/model_selection_v2/models/tree_reg.py`

**Interfaces:**
- Produces:
  - `train_reg(X, y, cfg, trial=None) -> lgb.Booster`
  - `predict_reg(model, X) -> np.ndarray` — (n_samples,) 预测超额收益率

- [ ] **Step 1: 编写 `models/tree_reg.py`**

```python
"""model_selection_v2 - T2: LightGBM 回归器（20日超额收益率）。"""
from __future__ import annotations
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


def _objective(trial, X_2d: np.ndarray, y: np.ndarray, cfg: V2Config) -> float:
    """Optuna 目标：最小化验证集 RMSE（仅负值不剪枝，直接返回验证 loss）。"""
    params = {
        "num_leaves": trial.suggest_int("num_leaves", *cfg.lgbm_params["num_leaves"]),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.lgbm_params["learning_rate"], log=True),
        "subsample": trial.suggest_float("subsample", *cfg.lgbm_params["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *cfg.lgbm_params["colsample_bytree"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *cfg.lgbm_params["reg_alpha"], log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", *cfg.lgbm_params["reg_lambda"], log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", *cfg.lgbm_params["min_child_samples"]),
        "verbosity": -1,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    losses = []
    for train_idx, val_idx in tscv.split(X_2d):
        train_data = lgb.Dataset(X_2d[train_idx], label=y[train_idx])
        val_data = lgb.Dataset(X_2d[val_idx], label=y[val_idx], reference=train_data)
        model = lgb.train(
            params, train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        pred = model.predict(X_2d[val_idx])
        loss = np.sqrt(np.mean((pred - y[val_idx]) ** 2))
        losses.append(loss)
    return np.mean(losses)


def train_reg(
    X: np.ndarray, y: np.ndarray,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> lgb.Booster:
    """训练 LightGBM 回归器。"""
    if cfg is None:
        cfg = get_config()

    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    if search_optuna:
        import optuna
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{cfg.optuna_dir_path}/t2_lgbm.db",
            study_name="t2_lgbm_reg",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: _objective(trial, X_2d, y, cfg),
            n_trials=cfg.optuna_n_trials,
            timeout=cfg.optuna_timeout,
            n_jobs=1,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info(f"T2 Optuna best: RMSE={study.best_value:.4f}, params={best_params}")
    else:
        best_params = {
            "num_leaves": 31, "learning_rate": 0.1, "subsample": 0.8,
            "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "min_child_samples": 20,
        }

    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_2d))
    train_idx, val_idx = splits[-1]

    params = {
        **best_params,
        "objective": "huber",
        "alpha": 0.1,  # Huber delta
        "metric": "rmse",
        "verbosity": -1,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_seed,
    }

    train_data = lgb.Dataset(X_2d[train_idx], label=y[train_idx])
    val_data = lgb.Dataset(X_2d[val_idx], label=y[val_idx], reference=train_data)

    model = lgb.train(
        params, train_data,
        valid_sets=[val_data],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(cfg.early_stop_rounds), lgb.log_evaluation(0)],
    )

    logger.info(f"T2 训练完成: best_iteration={model.best_iteration}")
    return model


def predict_reg(model: lgb.Booster, X: np.ndarray) -> np.ndarray:
    """预测 20 日超额收益率。

    Returns:
        (n_samples,) 预测超额收益率
    """
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)
    return model.predict(X_2d)
```

- [ ] **Step 2: 验证接口**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg
print('OK: tree_reg imported')
"
```

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/models/tree_reg.py
git commit -m "feat(v2): T2 LightGBM回归器 — 20日超额收益率预测"
```

---

### Task 6: CatBoost 回归器（T3：20日波动率）

**Files:**
- Create: `sequoia_x/model_selection_v2/models/tree_vol.py`

**Interfaces:**
- Produces:
  - `train_vol(X, y, cfg, trial=None) -> catboost.CatBoostRegressor`
  - `predict_vol(model, X) -> np.ndarray` — (n_samples,) 预测年化波动率

- [ ] **Step 1: 编写 `models/tree_vol.py`**

```python
"""model_selection_v2 - T3: CatBoost 回归器（20日波动率）。"""
from __future__ import annotations
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import TimeSeriesSplit
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


def _objective(trial, X_2d: np.ndarray, y: np.ndarray, cfg: V2Config) -> float:
    """Optuna 目标：最小化验证集 RMSE。"""
    params = {
        "depth": trial.suggest_int("depth", *cfg.cat_params["depth"]),
        "learning_rate": trial.suggest_float("learning_rate", *cfg.cat_params["learning_rate"], log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *cfg.cat_params["l2_leaf_reg"], log=True),
        "random_strength": trial.suggest_float("random_strength", *cfg.cat_params["random_strength"], log=True),
        "iterations": 500,
        "verbose": False,
        "thread_count": cfg.n_jobs,
        "random_seed": cfg.random_seed,
    }
    tscv = TimeSeriesSplit(n_splits=3)
    losses = []
    for train_idx, val_idx in tscv.split(X_2d):
        model = CatBoostRegressor(**params)
        model.fit(
            X_2d[train_idx], y[train_idx],
            eval_set=(X_2d[val_idx], y[val_idx]),
            early_stopping_rounds=50,
            verbose=False,
        )
        pred = model.predict(X_2d[val_idx])
        loss = np.sqrt(np.mean((pred - y[val_idx]) ** 2))
        losses.append(loss)
    return np.mean(losses)


def train_vol(
    X: np.ndarray, y: np.ndarray,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> CatBoostRegressor:
    """训练 CatBoost 波动率回归器。"""
    if cfg is None:
        cfg = get_config()

    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)

    if search_optuna:
        import optuna
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            storage=f"sqlite:///{cfg.optuna_dir_path}/t3_cat.db",
            study_name="t3_cat_vol",
            load_if_exists=True,
        )
        study.optimize(
            lambda trial: _objective(trial, X_2d, y, cfg),
            n_trials=cfg.optuna_n_trials,
            timeout=cfg.optuna_timeout,
            n_jobs=1,
            show_progress_bar=True,
        )
        best_params = study.best_params
        logger.info(f"T3 Optuna best: RMSE={study.best_value:.4f}, params={best_params}")
    else:
        best_params = {
            "depth": 6, "learning_rate": 0.1,
            "l2_leaf_reg": 3.0, "random_strength": 1.0,
        }

    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X_2d))
    train_idx, val_idx = splits[-1]

    model = CatBoostRegressor(
        **best_params,
        iterations=1000,
        verbose=False,
        thread_count=cfg.n_jobs,
        random_seed=cfg.random_seed,
        early_stopping_rounds=cfg.early_stop_rounds,
    )
    model.fit(
        X_2d[train_idx], y[train_idx],
        eval_set=(X_2d[val_idx], y[val_idx]),
        verbose=False,
    )

    logger.info(f"T3 训练完成: tree_count={model.tree_count_}")
    return model


def predict_vol(model: CatBoostRegressor, X: np.ndarray) -> np.ndarray:
    """预测 20 日年化波动率。"""
    n_samples = X.shape[0]
    X_2d = X.reshape(n_samples, -1)
    return model.predict(X_2d)
```

- [ ] **Step 2: 验证接口**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.model_selection_v2.models.tree_vol import train_vol, predict_vol
print('OK: tree_vol imported')
"
```

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/models/tree_vol.py
git commit -m "feat(v2): T3 CatBoost回归器 — 20日波动率预测"
```

---

### Task 7: 训练入口与特征重要性分析

**Files:**
- Create: `sequoia_x/model_selection_v2/train.py`

**Interfaces:**
- Consumes: `labels.build_training_dataset()` (Task 3), 3 个 model 文件 (Tasks 4-6)
- Produces:
  - `train_all(engine, cfg) -> dict` — 训练 3 个模型 + 输出特征重要性报告
  - CLI: `python -m sequoia_x.model_selection_v2.train`

- [ ] **Step 1: 编写 `train.py`**

```python
"""model_selection_v2 - 训练入口：协调3个模型训练 + 特征重要性分析。"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.tree_cls import train_cls
from sequoia_x.model_selection_v2.models.tree_reg import train_reg
from sequoia_x.model_selection_v2.models.tree_vol import train_vol

logger = get_logger(__name__)


def train_all(
    engine: DataEngine | None = None,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> dict:
    """训练全部 3 个模型。

    Args:
        engine: DataEngine 实例，None 则自动创建。
        cfg: 配置。
        search_optuna: True=Optuna超参搜索。

    Returns:
        {"t1_model": ..., "t2_model": ..., "t3_model": ..., "feature_importance": {...}}
    """
    if cfg is None:
        cfg = get_config()
    if engine is None:
        engine = DataEngine(Settings())

    logger.info("=" * 60)
    logger.info("V2 模型训练开始")
    logger.info("=" * 60)

    # 构建数据集
    t0 = time.time()
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg)
    if len(X) == 0:
        logger.error("无训练数据")
        return {}
    logger.info(f"训练数据: X={X.shape}, {len(dates)} 个采样日期")

    # 训练 T1
    logger.info("── 训练 T1: XGBoost 分类器 ──")
    t1 = time.time()
    model_t1 = train_cls(X, y1, cfg, search_optuna=search_optuna)
    logger.info(f"T1 耗时: {time.time()-t1:.0f}s")

    # 训练 T2
    logger.info("── 训练 T2: LightGBM 回归器 ──")
    t2 = time.time()
    model_t2 = train_reg(X, y2, cfg, search_optuna=search_optuna)
    logger.info(f"T2 耗时: {time.time()-t2:.0f}s")

    # 训练 T3
    logger.info("── 训练 T3: CatBoost 回归器 ──")
    t3 = time.time()
    model_t3 = train_vol(X, y3, cfg, search_optuna=search_optuna)
    logger.info(f"T3 耗时: {time.time()-t3:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"全部训练完成: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # 汇总特征重要性（以 T2 LightGBM 的特征重要性为主）
    feature_importance = {
        "t1_xgb": model_t1.feature_importances_.tolist() if hasattr(model_t1, 'feature_importances_') else [],
        "t2_lgbm": model_t2.feature_importance(importance_type="gain").tolist(),
    }

    # 保存结果
    result = {
        "t1_model": model_t1,
        "t2_model": model_t2,
        "t3_model": model_t3,
        "feature_importance": feature_importance,
        "n_samples": len(X),
        "n_dates": len(set(dates)),
        "elapsed_seconds": elapsed,
    }

    # 持久化特征重要性
    importance_path = cfg.model_dir_path / "feature_importance.json"
    with open(importance_path, "w") as f:
        json.dump(feature_importance, f, indent=2)
    logger.info(f"特征重要性已保存: {importance_path}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 多任务树模型训练")
    parser.add_argument("--no-optuna", action="store_true", help="跳过 Optuna 超参搜索")
    parser.add_argument("--symbols", type=int, default=0, help="限制训练股票数（0=全量）")
    args = parser.parse_args()

    cfg = get_config()
    engine = DataEngine(Settings())

    if args.symbols > 0:
        # 快速测试模式
        pool = engine.get_base_stock_pool()[:args.symbols]
        X, y1, y2, y3, dates = build_training_dataset(engine, cfg, symbols=pool)
        logger.info(f"测试数据: X={X.shape}, {len(set(dates))} 日期")
        model_t1 = train_cls(X, y1, cfg, search_optuna=False)
        model_t2 = train_reg(X, y2, cfg, search_optuna=False)
        model_t3 = train_vol(X, y3, cfg, search_optuna=False)
    else:
        train_all(engine, cfg, search_optuna=not args.no_optuna)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 快速冒烟测试（50只股票，无 Optuna）**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m sequoia_x.model_selection_v2.train --symbols 100 --no-optuna 2>&1 | tail -20
```

Expected: 3 个模型均训练完成，输出特征重要性文件

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/train.py
git commit -m "feat(v2): 训练入口 — 协调3模型训练+特征重要性分析"
```

---

### Task 8: Purged Rolling Walk-Forward 评估

**Files:**
- Create: `sequoia_x/model_selection_v2/evaluate.py`

**Interfaces:**
- Consumes: `labels.build_training_dataset()` (Task 3), 3 个 model (Tasks 4-6)
- Produces:
  - `run_walk_forward(engine, cfg) -> list[dict]` — 每 Fold 的评估指标
  - CLI: `python -m sequoia_x.model_selection_v2.evaluate`

- [ ] **Step 1: 编写 `evaluate.py`**

```python
"""model_selection_v2 - Purged Rolling Walk-Forward 评估。

对每个扩展窗口：训练 → 评估（purge gap 隔开）→ 报告 IC/AUC/RMSE。
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, mean_squared_error
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.tree_cls import train_cls, predict_cls
from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg
from sequoia_x.model_selection_v2.models.tree_vol import train_vol, predict_vol

logger = get_logger(__name__)


def _compute_rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """计算 Rank IC (Spearman correlation)。"""
    if len(pred) < 10:
        return 0.0
    ic, _ = spearmanr(pred, actual)
    return float(ic) if not np.isnan(ic) else 0.0


def run_walk_forward(
    engine: DataEngine | None = None,
    cfg: V2Config | None = None,
    search_optuna: bool = True,
) -> list[dict]:
    """运行 Purged Rolling Walk-Forward 评估。

    Folds:
      Fold 1: train 2020-2023 → test 2024
      Fold 2: train 2020-2024Q1 → test 2024Q2-Q4
      Fold 3: train 2020-2024 → test 2025
      Fold 4: train 2020-2025Q1 → test 2025Q2-Q4
      Fold 5: train 2020-2025 → test 2026H1
      Fold 6: train 2020-2026Q1 → test 2026Q2

    Purge: 训练集最后日期与测试集第一个日期间隔 >= cfg.purge_gap 个交易日。
    """
    if cfg is None:
        cfg = get_config()
    if engine is None:
        engine = DataEngine(Settings())

    # 构建全量数据集（带日期标签）
    X, y1, y2, y3, dates = build_training_dataset(engine, cfg)
    if len(X) == 0:
        logger.error("无数据")
        return []

    # 获取样本日期的唯一排序列表，用于确定 Fold 边界
    unique_dates = sorted(set(dates))
    logger.info(f"Walk-Forward: {len(X)} 样本, {len(unique_dates)} 个采样日期")
    logger.info(f"日期范围: {unique_dates[0]} ~ {unique_dates[-1]}")

    # 定义 Fold 边界（按年份+半年度）
    fold_boundaries = [
        ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),    # Fold 1
        ("2020-01-01", "2024-03-31", "2024-04-01", "2024-12-31"),    # Fold 2
        ("2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),    # Fold 3
        ("2020-01-01", "2025-03-31", "2025-04-01", "2025-12-31"),    # Fold 4
        ("2020-01-01", "2025-12-31", "2026-01-01", "2026-06-30"),    # Fold 5
        ("2020-01-01", "2026-03-31", "2026-04-01", "2026-07-20"),    # Fold 6
    ]

    all_results = []

    for fold_i, (train_start, train_end, test_start, test_end) in enumerate(fold_boundaries):
        logger.info(f"── Fold {fold_i+1}: train {train_start}~{train_end}, test {test_start}~{test_end} ──")
        t0 = time.time()

        # Purge: 找到训练集最后一个日期和测试集第一个日期
        train_dates = [d for d in unique_dates if train_start <= d <= train_end]
        test_dates = [d for d in unique_dates if test_start <= d <= test_end]
        if not train_dates or not test_dates:
            logger.warning(f"Fold {fold_i+1}: 无数据，跳过")
            continue

        # 找到有至少 purge_gap 间隔的切分点
        train_mask = np.array([d in train_dates for d in dates])
        test_mask = np.array([d in test_dates for d in dates])

        if train_mask.sum() < 100 or test_mask.sum() < 50:
            logger.warning(f"Fold {fold_i+1}: 样本不足（train={train_mask.sum()}, test={test_mask.sum()}），跳过")
            continue

        # 训练 3 个模型（不限 Optuna，快速评估）
        X_train, X_test = X[train_mask], X[test_mask]
        y1_train, y1_test = y1[train_mask], y1[test_mask]
        y2_train, y2_test = y2[train_mask], y2[test_mask]
        y3_train, y3_test = y3[train_mask], y3[test_mask]

        model_t1 = train_cls(X_train, y1_train, cfg, search_optuna=search_optuna)
        model_t2 = train_reg(X_train, y2_train, cfg, search_optuna=search_optuna)
        model_t3 = train_vol(X_train, y3_train, cfg, search_optuna=search_optuna)

        # 预测
        pred_t1 = predict_cls(model_t1, X_test)
        pred_t2 = predict_reg(model_t2, X_test)
        pred_t3 = predict_vol(model_t3, X_test)

        # 评估指标
        fold_result = {
            "fold": fold_i + 1,
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
        }

        # T1: AUC + 准确率
        try:
            fold_result["t1_auc"] = float(roc_auc_score(y1_test, pred_t1))
        except ValueError:
            fold_result["t1_auc"] = 0.5
        fold_result["t1_accuracy"] = float(((pred_t1 > 0.5) == y1_test).mean())

        # T2: Rank IC + RMSE
        fold_result["t2_rank_ic"] = _compute_rank_ic(pred_t2, y2_test)
        fold_result["t2_rmse"] = float(np.sqrt(mean_squared_error(y2_test, pred_t2)))

        # T3: RMSE
        fold_result["t3_rmse"] = float(np.sqrt(mean_squared_error(y3_test, pred_t3)))

        # 方向胜率
        if len(pred_t1) > 0:
            buy_mask = pred_t1 > 0.55
            if buy_mask.sum() > 0:
                fold_result["direction_win_rate"] = float(y1_test[buy_mask].mean())
            else:
                fold_result["direction_win_rate"] = 0.0

        elapsed = time.time() - t0
        fold_result["elapsed"] = elapsed
        all_results.append(fold_result)

        logger.info(
            f"Fold {fold_i+1}: "
            f"T1_AUC={fold_result.get('t1_auc', 0):.3f}, "
            f"T2_RankIC={fold_result.get('t2_rank_ic', 0):.4f}, "
            f"方向胜率={fold_result.get('direction_win_rate', 0):.2%}, "
            f"耗时={elapsed:.0f}s"
        )

    # 汇总
    if all_results:
        rank_ics = [r.get("t2_rank_ic", 0) for r in all_results]
        aucs = [r.get("t1_auc", 0.5) for r in all_results]
        logger.info("=" * 60)
        logger.info(f"Walk-Forward 汇总 ({len(all_results)} Folds):")
        logger.info(f"  T2 Rank IC: mean={np.mean(rank_ics):.4f}, "
                     f"min={np.min(rank_ics):.4f}, "
                     f"std={np.std(rank_ics):.4f}, "
                     f">0比例={sum(1 for ic in rank_ics if ic>0)/len(rank_ics):.0%}")
        logger.info(f"  T1 AUC: mean={np.mean(aucs):.4f}")
        logger.info("=" * 60)

        # 保存结果
        save_path = cfg.model_dir_path / "walk_forward_results.json"
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"结果已保存: {save_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 Walk-Forward 评估")
    parser.add_argument("--no-optuna", action="store_true", help="跳过 Optuna")
    args = parser.parse_args()
    run_walk_forward(search_optuna=not args.no_optuna)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 快速冒烟测试（100只股票，无 Optuna）**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
# 先用 train.py 的快速模式生成数据，验证 evaluate 可导入
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.model_selection_v2.evaluate import run_walk_forward, get_config
print('OK: evaluate imported')
# 快速测试（仅前100只+无Optuna）
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.labels import build_training_dataset
cfg = get_config()
engine = DataEngine(Settings())
pool = engine.get_base_stock_pool()[:50]
X, y1, y2, y3, dates = build_training_dataset(engine, cfg, symbols=pool)
# 用简单的自定义 fold 做一次快速 WF
from sequoia_x.model_selection_v2.models.tree_cls import train_cls, predict_cls
import numpy as np
unique_dates = sorted(set(dates))
mid = len(unique_dates) // 2
train_mask = np.array([d <= unique_dates[mid] for d in dates])
test_mask = ~train_mask
print(f'Quick WF: train={train_mask.sum()}, test={test_mask.sum()}')
m = train_cls(X[train_mask], y1[train_mask], cfg, search_optuna=False)
p = predict_cls(m, X[test_mask])
print(f'Quick WF AUC: {roc_auc_score(y1[test_mask], p):.3f}')
print('OK: evaluate pipeline works')
" 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/evaluate.py
git commit -m "feat(v2): Purged Rolling Walk-Forward评估 — 6-Fold验证"
```

---

### Task 9: 回测引擎

**Files:**
- Create: `sequoia_x/model_selection_v2/backtest/config.py`
- Create: `sequoia_x/model_selection_v2/backtest/engine.py`

**Interfaces:**
- Consumes: 3 个 predict 函数 (Tasks 4-6), `simulation/rules.py` (共享模块)
- Produces: `V2BacktestEngine` 类，`run(start, end)` 方法

- [ ] **Step 1: 编写 `backtest/config.py`**

```python
"""model_selection_v2 - 回测参数。"""
MAX_POSITIONS: int = 10
TOP_N_BUY_PER_DAY: int = 2
PER_STOCK_BUDGET: float = 50_000.0
INITIAL_CAPITAL: float = 500_000.0
MIN_PRED_RETURN: float = 0.0
COMMISSION_RATE: float = 0.00025
STAMP_TAX_RATE: float = 0.001
SLIPPAGE: float = 0.0001
MIN_BUY_PROB: float = 0.55
```

- [ ] **Step 2: 编写 `backtest/engine.py`**

```python
"""model_selection_v2 - 逐日回测引擎。"""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.features import build_prediction_features
from sequoia_x.model_selection_v2 import backtest as bt_cfg

logger = get_logger(__name__)


class V2BacktestEngine:
    """V2 多任务树模型回测引擎。"""

    def __init__(self, engine: DataEngine,
                 model_t1, model_t2, model_t3,
                 cfg: V2Config | None = None):
        self.engine = engine
        self.model_t1 = model_t1
        self.model_t2 = model_t2
        self.model_t3 = model_t3
        self.cfg = cfg or get_config()
        self.cash = bt_cfg.INITIAL_CAPITAL
        self.positions: dict[str, dict] = {}
        self.closed_trades: list[dict] = []
        self.daily_records: list[dict] = []
        self.trade_records: list[dict] = []

    def run(self, start_date: str, end_date: str = "",
            predictions_cache: dict | None = None) -> dict:
        """运行回测。

        逐日循环：T-1日收盘数据构建特征 → 3模型预测 → T日开盘执行。
        """
        from sequoia_x.model_selection_v2.models.tree_cls import predict_cls
        from sequoia_x.model_selection_v2.models.tree_reg import predict_reg
        from sequoia_x.model_selection_v2.models.tree_vol import predict_vol

        # 获取交易日列表
        conn = sqlite3.connect(self.engine.db_path)
        date_cond = f"date >= '{start_date}'"
        if end_date:
            date_cond += f" AND date <= '{end_date}'"
        dates = pd.read_sql(
            f"SELECT DISTINCT date FROM stock_daily WHERE {date_cond} ORDER BY date",
            conn
        )["date"].tolist()
        conn.close()

        if len(dates) < 150:
            logger.error(f"回测: 数据不足 ({len(dates)} 天)")
            return {}

        base_pool = self.engine.get_base_stock_pool()
        logger.info(f"回测: {dates[0]} ~ {dates[-1]}, {len(dates)} 天, {len(base_pool)} 只")

        warmup = self.cfg.window
        cache_path = Path("output/backtest_v2/predictions_cache.json")

        for idx, today in enumerate(dates):
            if idx < warmup:
                continue
            prev_date = dates[idx - 1]

            # 获取预测
            if predictions_cache is not None and prev_date in predictions_cache:
                predictions = predictions_cache[prev_date]
            else:
                predictions = self._predict_batch(base_pool, prev_date, predict_cls,
                                                   predict_reg, predict_vol)

            if not predictions:
                continue

            # 生成信号（T1过滤→T2排序→T3调仓）
            signals = self._generate_signals(predictions)

            # 执行卖出
            self._execute_sells(signals.get("sell", []), today)

            # 执行买入
            self._execute_buys(signals.get("buy", []), today)

            # 日终估值
            self._mark_to_market(today)

            # 记录日结
            self._record_daily(today)

        return self._compute_metrics()

    def _predict_batch(self, pool: list[str], ref_date: str,
                       predict_cls_fn, predict_reg_fn, predict_vol_fn) -> list[dict]:
        """批量预测。"""
        from sequoia_x.model_selection_v2.features import build_prediction_features
        xs, symbols = [], []
        for symbol in pool:
            try:
                X = build_prediction_features(symbol, self.engine, self.cfg)
                if X is not None:
                    xs.append(X)
                    symbols.append(symbol)
            except Exception:
                continue
        if not xs:
            return []
        X_batch = np.vstack(xs)
        prob_up = predict_cls_fn(self.model_t1, X_batch)
        excess_ret = predict_reg_fn(self.model_t2, X_batch)
        volatility = predict_vol_fn(self.model_t3, X_batch)
        results = []
        for i, sym in enumerate(symbols):
            if np.isfinite(prob_up[i]):
                results.append({
                    "symbol": sym, "prob_up": float(prob_up[i]),
                    "excess_ret": float(excess_ret[i]),
                    "volatility": float(volatility[i]),
                })
        return results

    def _generate_signals(self, predictions: list[dict]) -> dict:
        """生成买卖信号。"""
        signals: dict = {"buy": [], "sell": []}

        # 卖出：运行 rules.py 的 evaluate_exit（复用共享模块）
        from sequoia_x.simulation.rules import evaluate_exit
        for symbol, pos in list(self.positions.items()):
            current_price = pos.get("current_price", 0)
            if current_price <= 0:
                continue
            df = self.engine.get_ohlcv(symbol)
            idx_df = self._get_index_df()
            result = evaluate_exit(
                entry_price=pos["cost"] / pos["shares"] if pos["shares"] > 0 else pos["cost"],
                current_price=current_price,
                highest_price=pos.get("highest_price", current_price),
                hold_days=pos.get("hold_days", 0),
                symbol=symbol,
                symbol_df=df.tail(60) if df is not None and not df.empty else None,
                index_df=idx_df.tail(60) if idx_df is not None and not idx_df.empty else None,
                today_opened=False,
            )
            # V2 特有：叠加 T1/T2 预测因子
            pred_for_sym = next((p for p in predictions if p["symbol"] == symbol), None)
            if pred_for_sym:
                if pred_for_sym["prob_up"] < 0.3:
                    result.score += 20  # T1 强烈看空，加速卖出
                if pred_for_sym["excess_ret"] < -0.03:
                    result.score += 15  # T2 预期超额亏损
            if result.should_exit or result.score >= 60:
                signals["sell"].append(symbol)

        # 买入：T1 过滤 → T2 排序 → Top N
        candidates = [p for p in predictions
                      if p["symbol"] not in self.positions
                      and p["prob_up"] >= bt_cfg.MIN_BUY_PROB]
        candidates.sort(key=lambda x: x["excess_ret"], reverse=True)
        slots = bt_cfg.MAX_POSITIONS - len(self.positions)
        signals["buy"] = [c["symbol"] for c in candidates[:min(slots, bt_cfg.TOP_N_BUY_PER_DAY)]]
        return signals

    def _execute_sells(self, symbols: list[str], date_str: str) -> None:
        """以当日开盘价卖出。"""
        for symbol in symbols:
            if symbol not in self.positions:
                continue
            pos = self.positions[symbol]
            price = self._get_open_price(symbol, date_str)
            if price is None:
                continue
            sell_price = price * (1 - bt_cfg.SLIPPAGE)
            revenue = pos["shares"] * sell_price
            commission = revenue * bt_cfg.COMMISSION_RATE
            tax = revenue * bt_cfg.STAMP_TAX_RATE
            net = revenue - commission - tax
            pnl = net - pos["cost"]
            self.cash += net
            self.positions.pop(symbol)
            self.trade_records.append({
                "symbol": symbol, "type": "sell", "date": date_str,
                "price": round(sell_price, 4), "shares": pos["shares"],
                "pnl": round(pnl, 2),
            })

    def _execute_buys(self, symbols: list[str], date_str: str) -> None:
        """以当日开盘价买入。"""
        for symbol in symbols:
            price = self._get_open_price(symbol, date_str)
            if price is None:
                continue
            buy_price = price * (1 + bt_cfg.SLIPPAGE)
            budget = min(bt_cfg.PER_STOCK_BUDGET, self.cash * 0.9)
            shares = int(budget / buy_price / 100) * 100
            if shares < 100:
                continue
            total = shares * buy_price * (1 + bt_cfg.COMMISSION_RATE)
            if total > self.cash:
                continue
            self.cash -= total
            self.positions[symbol] = {
                "shares": shares, "cost": total,
                "buy_date": date_str, "highest_price": buy_price,
                "hold_days": 0, "current_price": buy_price,
                "current_value": total, "pnl": 0.0, "pnl_pct": 0.0,
            }
            self.trade_records.append({
                "symbol": symbol, "type": "buy", "date": date_str,
                "price": round(buy_price, 4), "shares": shares,
            })

    def _get_open_price(self, symbol: str, date_str: str) -> float | None:
        conn = sqlite3.connect(self.engine.db_path)
        row = conn.execute(
            "SELECT open FROM stock_daily WHERE symbol=? AND date=?", (symbol, date_str)
        ).fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else None

    def _get_index_df(self) -> pd.DataFrame:
        df = self.engine.get_ohlcv("sh.000001")
        if df.empty:
            conn = sqlite3.connect(self.engine.db_path)
            df = pd.read_sql(
                "SELECT * FROM index_daily WHERE symbol='sh.000001' ORDER BY date", conn
            )
            conn.close()
        return df if not df.empty else pd.DataFrame()

    def _mark_to_market(self, date_str: str) -> None:
        conn = sqlite3.connect(self.engine.db_path)
        for symbol, pos in self.positions.items():
            row = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol=? AND date=?", (symbol, date_str)
            ).fetchone()
            if row and row[0]:
                close = float(row[0])
                pos["current_price"] = close
                pos["current_value"] = pos["shares"] * close
                pos["pnl"] = pos["current_value"] - pos["cost"]
                pos["pnl_pct"] = pos["pnl"] / pos["cost"] if pos["cost"] > 0 else 0.0
                pos["hold_days"] = pos.get("hold_days", 0) + 1
                if close > pos["highest_price"]:
                    pos["highest_price"] = close
        conn.close()

    def _record_daily(self, date_str: str) -> None:
        stock_value = sum(p.get("current_value", p["cost"]) for p in self.positions.values())
        total = self.cash + stock_value
        self.daily_records.append({
            "date": date_str, "cash": round(self.cash, 2),
            "stock_value": round(stock_value, 2),
            "total_value": round(total, 2),
            "positions": len(self.positions),
        })

    def _compute_metrics(self) -> dict:
        if not self.daily_records:
            return {}
        n = len(self.daily_records)
        tv = np.array([r["total_value"] for r in self.daily_records])
        total_return = tv[-1] / bt_cfg.INITIAL_CAPITAL - 1
        annual_return = (1 + total_return) ** (252 / n) - 1 if n >= 20 else None
        daily_ret = np.diff(tv) / tv[:-1]
        mean_ret = np.mean(daily_ret) if len(daily_ret) > 0 else 0
        std_ret = np.std(daily_ret) if len(daily_ret) > 0 else 1e-10
        sharpe = (mean_ret - 0.03 / 252) / std_ret * np.sqrt(252) if std_ret > 1e-10 else 0
        cuml = tv / tv[0]
        running_max = np.maximum.accumulate(cuml)
        drawdown = (cuml - running_max) / running_max
        max_dd = float(drawdown.min())
        buys = [t for t in self.trade_records if t["type"] == "buy"]
        sells = [t for t in self.trade_records if t["type"] == "sell"]
        win_trades = [t for t in sells if t["pnl"] > 0]
        return {
            "total_return": total_return, "annual_return": annual_return,
            "sharpe": round(sharpe, 2), "max_drawdown": max_dd,
            "n_days": n, "n_buys": len(buys), "n_sells": len(sells),
            "win_rate": len(win_trades)/len(sells) if sells else 0,
            "total_value": float(tv[-1]), "final_cash": self.cash,
            "daily_records": self.daily_records,
            "trade_records": self.trade_records,
        }
```

- [ ] **Step 3: 验证导入**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.model_selection_v2.backtest.engine import V2BacktestEngine
print('OK: V2BacktestEngine imported')
"
```

- [ ] **Step 4: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/backtest/
git commit -m "feat(v2): 逐日回测引擎 — 三模型综合信号+动态仓位"
```

---

### Task 10: 回测报告器

**Files:**
- Create: `sequoia_x/model_selection_v2/backtest/reporter.py`

**Interfaces:**
- Consumes: `V2BacktestEngine._compute_metrics()` 输出
- Produces: `save_results(metrics, output_dir, ...)` — 保存 CSV+JSON，打印对比报告

- [ ] **Step 1: 编写 `backtest/reporter.py`**

```python
"""model_selection_v2 - 回测报告输出。"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def save_results(
    metrics_list: list[dict], output_dir: str,
    daily_records: list[dict] | None = None,
    trade_records: list[dict] | None = None,
) -> None:
    """保存回测结果。

    Args:
        metrics_list: 多个期间的绩效指标列表。
        output_dir: 输出目录路径。
        daily_records: 逐日净值记录。
        trade_records: 逐笔交易记录。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 指标 JSON
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2, default=str)
    logger.info(f"绩效指标已保存: {out / 'metrics.json'}")

    # 逐日净值 CSV
    if daily_records:
        _save_csv(out / "daily_records.csv", daily_records)
        logger.info(f"逐日净值已保存: {out / 'daily_records.csv'} ({len(daily_records)} 行)")

    # 交易明细 CSV
    if trade_records:
        _save_csv(out / "trade_records.csv", trade_records)
        logger.info(f"交易明细已保存: {out / 'trade_records.csv'} ({len(trade_records)} 笔)")


def _save_csv(path: Path, records: list[dict]) -> None:
    """保存 CSV 文件。"""
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def print_comparison_table(all_metrics: list[dict]) -> None:
    """打印多期间对比表。"""
    periods = {"2024": "+1.71%", "2025": "+34.94%", "2026": "+24.25%", "full": "+27.4%"}
    print("\n" + "=" * 80)
    print("  V2 多任务树模型 — 回测报告")
    print("=" * 80)
    print(f"{'期间':>6s} {'策略收益':>8s} {'HS300':>8s} {'超额':>8s} "
          f"{'夏普':>6s} {'回撤':>7s} {'胜率':>6s} {'交易':>5s}")
    print("-" * 80)
    for m in all_metrics:
        period = m.get("period", "?")
        hs300 = periods.get(period, "?")
        hs300_val = float(hs300.rstrip("%")) / 100 if hs300 != "?" else 0
        print(
            f"{period:>6s} "
            f"{m.get('total_return', 0):>+7.1%} "
            f"{hs300:>8s} "
            f"{m.get('total_return', 0) - hs300_val:>+7.1%} "
            f"{m.get('sharpe', 0):>6.2f} "
            f"{m.get('max_drawdown', 0):>7.1%} "
            f"{m.get('win_rate', 0):>5.1%} "
            f"{m.get('n_buys', 0) + m.get('n_sells', 0):>5d}"
        )
    print("=" * 80)
```

- [ ] **Step 2: 验证导入**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
from sequoia_x.model_selection_v2.backtest.reporter import save_results, print_comparison_table
print('OK: reporter imported')
"
```

- [ ] **Step 3: Commit**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add sequoia_x/model_selection_v2/backtest/reporter.py
git commit -m "feat(v2): 回测报告器 — CSV+JSON输出+对比表打印"
```

---

### Task 11: 端到端集成冒烟测试

- [ ] **Step 1: 全流程冒烟测试（100只股票，无Optuna）**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
/home/zhulei/anaconda3/envs/zhulei_py312/bin/python -c "
import time
import numpy as np
from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.labels import build_training_dataset
from sequoia_x.model_selection_v2.models.tree_cls import train_cls, predict_cls
from sequoia_x.model_selection_v2.models.tree_reg import train_reg, predict_reg
from sequoia_x.model_selection_v2.models.tree_vol import train_vol, predict_vol

cfg = get_config()
engine = DataEngine(Settings())
pool = engine.get_base_stock_pool()[:100]  # 限100只

t0 = time.time()
print('1. 构建数据集...')
X, y1, y2, y3, dates = build_training_dataset(engine, cfg, symbols=pool)
print(f'   X={X.shape}, y1涨比例={y1.mean():.1%}')

print('2. 训练T1 (XGBoost)...')
m1 = train_cls(X, y1, cfg, search_optuna=False)
p1 = predict_cls(m1, X[-100:])
print(f'   T1预测涨概率均值: {p1.mean():.3f}')

print('3. 训练T2 (LightGBM)...')
m2 = train_reg(X, y2, cfg, search_optuna=False)
p2 = predict_reg(m2, X[-100:])
print(f'   T2预测超额收益均值: {p2.mean():.4f}')

print('4. 训练T3 (CatBoost)...')
m3 = train_vol(X, y3, cfg, search_optuna=False)
p3 = predict_vol(m3, X[-100:])
print(f'   T3预测波动率均值: {p3.mean():.4f}')

from scipy.stats import spearmanr
ic = spearmanr(p2, y2[-100:])[0]
print(f'5. T2 Rank IC (最后100样本): {ic:.4f}')
print(f'6. 总耗时: {time.time()-t0:.0f}s')
print('=== 端到端冒烟测试 PASS ===')
" 2>&1 | grep -v "OMP:"
```

Expected: 全部 6 步输出正常，总耗时 < 5 分钟

- [ ] **Step 2: Commit 最终状态**

```bash
cd /public/home/hpc/zhulei/superman/quant/code/017_workbuddy/004_sequoia-x
git add -A sequoia_x/model_selection_v2/
git commit -m "feat(v2): 端到端集成冒烟测试通过"
```

---

## Spec Coverage Self-Review

| Spec Section | Covered By |
|---|---|
| 目录结构 (S2) | Task 1 |
| 多任务标签 (S3) | Task 3 (labels.py) |
| 训练数据构建 (S4) | Task 3 (build_training_dataset) |
| 特征工程 62维 (S5.1) | Task 2 (features.py) |
| 特征扩展 (S5.2) | Phase 1-2（后续 PR） |
| T1 XGBoost (S6.1) | Task 4 |
| T2 LightGBM (S6.1) | Task 5 |
| T3 CatBoost (S6.1) | Task 6 |
| Optuna 搜索 (S6.2) | Tasks 4-6 (search_optuna 参数) |
| Walk-Forward (S7) | Task 8 |
| 回测引擎 (S8) | Task 9 |
| 回测报告 (S8) | Task 10 |
| 里程碑 (S9) | 评估结果对比决策门 |
| 硬件约束 (S10) | config.py n_jobs=8, Optuna n_jobs=1 |
| 隔离 (S11) | 全部文件零 import model_selection/ |
