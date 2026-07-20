"""股票 LSTM-Transformer 特征工程模块。

为每只股票构建 (window, n_features) 的时序特征矩阵。
全部特征从 stock_daily 表计算，零外部依赖。

严格避免 look-ahead bias：
  第 T 日的特征只使用第 T 日及之前已知的数据。
  标签使用 T+predict_horizon 日的数据（未来）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection.config import LSTMConfig, get_config


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
    """计算 MACD 指标。返回 (dif, dea, hist)。"""
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
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr


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
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    return adx


def _compute_bollinger(close: np.ndarray, period: int = 20
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算布林带。返回 (upper, middle, lower)。"""
    middle = pd.Series(close).rolling(period, min_periods=1).mean().values
    std = pd.Series(close).rolling(period, min_periods=1).std().values
    return middle + 2*std, middle, middle - 2*std


def _extract_per_day_features(df: pd.DataFrame, df_index: pd.DataFrame | None,
                               cfg: LSTMConfig) -> np.ndarray:
    """从日线 DataFrame 逐日提取特征向量。

    Args:
        df: 单只股票的 OHLCV DataFrame，含 open/high/low/close/volume/amount/turnover
        df_index: 指数 DataFrame（可选），含 close
        cfg: 配置对象

    Returns:
        features: (n_days, n_features) 的特征矩阵
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
    # 第8维：5日振幅均值
    amplitude_5d = pd.Series(hl_ratio).rolling(5, min_periods=1).mean().values
    feature_list.append(amplitude_5d)

    # ── 2. 均线偏离特征 (6维) ──
    for period in cfg.feature_ma_periods:
        ma = pd.Series(close).rolling(period, min_periods=1).mean().values
        deviation = close / np.maximum(ma, 1e-10) - 1.0
        feature_list.append(deviation)
    # MA5-MA20 差距
    ma5 = pd.Series(close).rolling(5, min_periods=1).mean().values
    ma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
    feature_list.append(ma5 / np.maximum(ma20, 1e-10) - 1.0)

    # ── 3. 量能特征 (8维) ──
    vol_ma20 = pd.Series(volume).rolling(20, min_periods=1).mean().values
    vol_ratio = volume / np.maximum(vol_ma20, 1e-10)
    vol_change_5d = pd.Series(volume).pct_change(5).fillna(0.0).values
    # 换手率（已有字段，缺失时用 amount/流通市值 近似）
    turnover_rate = turnover / 100.0 if turnover.max() < 1 else turnover  # 标准化
    # 成交额/流通市值近似（用成交量×价格作为代理）
    amount_ratio = amount / np.maximum(close * 1e8, 1e-10)  # 近似
    # 量价相关
    vol_corr = pd.Series(volume).rolling(10, min_periods=1).corr(
        pd.Series(close)
    ).fillna(0.0).values
    # 5日均量/20日均量
    vol_ma5 = pd.Series(volume).rolling(5, min_periods=1).mean().values
    vol_ma5_ratio = vol_ma5 / np.maximum(vol_ma20, 1e-10)
    # 成交量趋势
    vol_trend = vol_ma5 / np.maximum(pd.Series(volume).rolling(20, min_periods=1).mean().shift(20).values, 1e-10)
    vol_trend = np.nan_to_num(vol_trend, nan=1.0)
    feature_list.extend([vol_ratio, vol_change_5d, turnover_rate, amount_ratio,
                         vol_corr, vol_ma5_ratio, vol_trend])
    # 第8维：放量/缩量标记
    vol_surge = (vol_ratio > 1.5).astype(float)
    feature_list.append(vol_surge)

    # ── 4. 技术指标 (14维) ──
    feature_list.append(_compute_rsi(close, cfg.feature_rsi_period) / 100.0)
    dif, dea, hist = _compute_macd(close, cfg.feature_macd_fast,
                                    cfg.feature_macd_slow, cfg.feature_macd_signal)
    # MACD 值除以收盘价做归一化
    close_safe = np.maximum(close, 1e-10)
    feature_list.extend([dif / close_safe, dea / close_safe, hist / close_safe])
    bb_upper, bb_mid, bb_lower = _compute_bollinger(close, cfg.feature_boll_period)
    bb_position = (close - bb_lower) / np.maximum(bb_upper - bb_lower, 1e-10)
    bb_width = (bb_upper - bb_lower) / np.maximum(bb_mid, 1e-10)
    feature_list.extend([bb_position, bb_width])
    atr = _compute_atr(high, low, close, cfg.feature_atr_period)
    feature_list.append(atr / close_safe)
    # KDJ
    low_n = pd.Series(low).rolling(9, min_periods=1).min().values
    high_n = pd.Series(high).rolling(9, min_periods=1).max().values
    rsv = (close - low_n) / np.maximum(high_n - low_n, 1e-10) * 100.0
    k = pd.Series(rsv).ewm(com=2, adjust=False).mean().values
    d = pd.Series(k).ewm(com=2, adjust=False).mean().values
    feature_list.extend([k / 100.0, d / 100.0])
    # OBV 变化率
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    obv_change = pd.Series(obv).pct_change(5).fillna(0.0).values
    feature_list.append(obv_change)
    # ADX
    adx = _compute_adx(high, low, close, cfg.feature_adx_period)
    feature_list.append(adx / 100.0)

    # ── 5. 波动率特征 (4维) ──
    for period in cfg.feature_vol_periods:
        vol = pd.Series(ret_1d).rolling(period, min_periods=1).std().values * np.sqrt(252)
        feature_list.append(vol)
    vol_20d = pd.Series(ret_1d).rolling(20, min_periods=1).std().values * np.sqrt(252)
    vol_5d = pd.Series(ret_1d).rolling(5, min_periods=1).std().values * np.sqrt(252)
    vol_change = vol_5d / np.maximum(vol_20d, 1e-10) - 1.0
    feature_list.append(vol_change)

    # ── 6. 大盘关联特征 (8维) ──
    if df_index is not None and len(df_index) == n:
        index_close = df_index["close"].values.astype(float)
        idx_ret = np.diff(index_close, prepend=index_close[0]) / np.maximum(
            np.roll(index_close, 1), 1e-10)
        idx_ret[0] = 0.0
        feature_list.append(idx_ret)  # 1维: 指数收益
    else:
        feature_list.append(np.zeros(n))  # 占位

    # Beta (20日滚动)
    if df_index is not None and len(df_index) == n:
        idx_ret_series = pd.Series(np.diff(index_close, prepend=index_close[0])
                                   / np.maximum(np.roll(index_close, 1), 1e-10))
        idx_ret_series.iloc[0] = 0.0
        stock_ret_series = pd.Series(ret_1d)
        beta = np.zeros(n)
        for i in range(cfg.feature_beta_period, n):
            cov = stock_ret_series.iloc[i-cfg.feature_beta_period:i].cov(
                idx_ret_series.iloc[i-cfg.feature_beta_period:i])
            var = idx_ret_series.iloc[i-cfg.feature_beta_period:i].var()
            beta[i] = cov / var if var > 1e-10 else 1.0
    else:
        beta = np.ones(n)
    feature_list.append(np.clip(beta, -3, 5))

    # 相对强度
    rs = pd.Series(close).pct_change(20).fillna(0.0).values
    if df_index is not None and len(df_index) == n:
        idx_ret_20d = pd.Series(index_close).pct_change(20).fillna(0.0).values
        feature_list.append(rs - idx_ret_20d)
    else:
        feature_list.append(rs)

    # 指数 MA 位置
    if df_index is not None and len(df_index) == n:
        idx_ma20 = pd.Series(index_close).rolling(20, min_periods=1).mean().values
        idx_ma60 = pd.Series(index_close).rolling(60, min_periods=1).mean().values
        feature_list.append(index_close / np.maximum(idx_ma20, 1e-10) - 1.0)
        feature_list.append(index_close / np.maximum(idx_ma60, 1e-10) - 1.0)
    else:
        feature_list.extend([np.zeros(n), np.zeros(n)])

    # 跑赢/跑输指数幅度
    ret_5d_stock = pd.Series(close).pct_change(5).fillna(0.0).values
    if df_index is not None and len(df_index) == n:
        idx_ret_5d = pd.Series(index_close).pct_change(5).fillna(0.0).values
        feature_list.append(ret_5d_stock - idx_ret_5d)
    else:
        feature_list.append(ret_5d_stock)
    # 剩余维度补零
    while len(feature_list) < 55:
        feature_list.append(np.zeros(n))

    # ── 7. 价格形态特征 (7维) ──
    # 涨跌停标记
    limit_up = (chg_pct > 0.095).astype(float)
    limit_down = (chg_pct < -0.095).astype(float)
    feature_list.append(limit_up)
    feature_list.append(limit_down)
    # 连续涨跌天数
    up_streak = np.zeros(n)
    down_streak = np.zeros(n)
    for i in range(1, n):
        up_streak[i] = up_streak[i-1] + 1 if ret_1d[i] > 0 else 0
        down_streak[i] = down_streak[i-1] + 1 if ret_1d[i] < 0 else 0
    feature_list.extend([up_streak / 10.0, down_streak / 10.0])
    # N日新高/新低
    high_20d = pd.Series(high).rolling(20, min_periods=1).max().values
    low_20d = pd.Series(low).rolling(20, min_periods=1).min().values
    feature_list.append((close >= high_20d).astype(float))
    feature_list.append((close <= low_20d).astype(float))
    # 实体/影线比
    body = np.abs(close - open_)
    shadow = high - low
    feature_list.append(body / np.maximum(shadow, 1e-10))

    # ── 组装 ──
    features = np.column_stack(feature_list).astype(np.float32)
    # 处理 NaN 和 Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def build_stock_features(
    symbol: str,
    ref_date: str,
    engine: DataEngine,
    cfg: LSTMConfig | None = None,
) -> tuple[np.ndarray | None, float | None]:
    """为单只股票构建训练/验证用的时序特征。

    Args:
        symbol: 股票代码。
        ref_date: 参考日期（YYYY-MM-DD），构建该日期之前 window 天的特征。
        engine: DataEngine 实例。
        cfg: 配置对象。

    Returns:
        (X, y): X 形状 (window, n_features)，y 是未来 predict_horizon 日收益率。
                如果数据不足，返回 (None, None)。
    """
    if cfg is None:
        cfg = get_config()

    df = engine.get_ohlcv(symbol)
    if df is None or len(df) < cfg.window + cfg.predict_horizon + 10:
        return None, None

    # 截取 ref_date 之前的数据（含 ref_date 当日）
    df = df[df["date"] <= ref_date].copy()
    if len(df) < cfg.window + cfg.predict_horizon + 10:
        return None, None

    # 加载指数数据
    df_index = None
    try:
        df_index = engine.get_ohlcv("000300")  # 沪深300
        if df_index is not None:
            df_index = df_index[df_index["date"] <= ref_date].copy()
            # 对齐长度
            if len(df_index) != len(df):
                df_index = None
    except Exception:
        df_index = None

    # 提取逐日特征
    per_day = _extract_per_day_features(df, df_index, cfg)

    # 构建滑动窗口和标签
    n = len(per_day)
    # 取最后一个窗口
    end_idx = n - cfg.predict_horizon
    start_idx = end_idx - cfg.window
    if start_idx < 0:
        return None, None

    X = per_day[start_idx:end_idx]  # (window, n_features)

    # 标签：未来 predict_horizon 日的收益率
    future_close = df["close"].values.astype(float)[end_idx + cfg.predict_horizon - 1]
    current_close = df["close"].values.astype(float)[end_idx - 1]
    y = (future_close / current_close) - 1.0

    return X, y


def build_batch_features(
    symbols: list[str],
    ref_date: str,
    engine: DataEngine,
    cfg: LSTMConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """批量构建特征矩阵。

    Args:
        symbols: 股票代码列表。
        ref_date: 参考日期。
        engine: DataEngine 实例。
        cfg: 配置对象。

    Returns:
        (X, y): X 形状 (n_valid, window, n_features), y 形状 (n_valid,)。
    """
    if cfg is None:
        cfg = get_config()

    X_list, y_list = [], []
    for symbol in symbols:
        X_i, y_i = build_stock_features(symbol, ref_date, engine, cfg)
        if X_i is not None and y_i is not None:
            X_list.append(X_i)
            y_list.append(y_i)

    if not X_list:
        return np.array([]).reshape(0, cfg.window, 0), np.array([])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32)
    return X, y


def build_prediction_features(
    symbol: str,
    engine: DataEngine,
    cfg: LSTMConfig | None = None,
) -> np.ndarray | None:
    """为单只股票构建预测用特征（使用最新 window 天数据，不含标签）。

    Args:
        symbol: 股票代码。
        engine: DataEngine 实例。
        cfg: 配置对象。

    Returns:
        X: 形状 (1, window, n_features)，数据不足返回 None。
    """
    if cfg is None:
        cfg = get_config()

    df = engine.get_ohlcv(symbol)
    if df is None or len(df) < cfg.window + 10:
        return None

    # 加载指数数据
    df_index = None
    try:
        df_index = engine.get_ohlcv("000300")
        if df_index is not None:
            if len(df_index) != len(df):
                # 不对齐则不用指数特征
                df_index = None
    except Exception:
        df_index = None

    per_day = _extract_per_day_features(df, df_index, cfg)

    if len(per_day) < cfg.window:
        return None

    X = per_day[-cfg.window:]  # 最新 window 天
    return X[np.newaxis, :, :]  # (1, window, n_features)
