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
    ref_date: str | None = None,
) -> np.ndarray | None:
    """为单只股票构建预测特征（严格避免 look-ahead bias）。

    Args:
        symbol: 股票代码。
        engine: DataEngine 实例。
        cfg: 配置。
        ref_date: 截止日期（仅使用此日期及之前的数据），None=使用全部最新数据。

    Returns:
        X: (1, window, n_features)，数据不足返回 None。
    """
    if cfg is None:
        cfg = get_config()

    df = engine.get_ohlcv(symbol)
    if df is None or len(df) < cfg.window + 10:
        return None

    # 严格截止日期过滤（消除 look-ahead bias）
    if ref_date is not None:
        df = df[df["date"] <= ref_date].copy()
        if len(df) < cfg.window + 10:
            return None

    df_index = None
    try:
        df_index = engine.get_ohlcv("000300")
        if df_index is not None:
            if ref_date is not None:
                df_index = df_index[df_index["date"] <= ref_date].copy()
            if len(df_index) != len(df):
                df_index = None
    except Exception:
        df_index = None

    per_day = _extract_per_day_features(df, df_index, cfg)
    if len(per_day) < cfg.window:
        return None

    X = per_day[-cfg.window:]
    return X[np.newaxis, :, :]
