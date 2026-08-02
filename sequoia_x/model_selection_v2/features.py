"""model_selection_v2 - 特征工程模块。

从 stock_daily 表计算 78 维时序特征，严格避免 look-ahead bias：
第 T 日的特征仅使用 T 日及之前已知的数据。

特征分组：
  价格收益(8) + 均线偏离(6) + 量能(8) + 技术指标(11)
  + 波动率(4) + 大盘关联(8) + 市场状态(8)
  + 价格形态(7) + 最大回撤(3) + 收益分布(4) + 时间日历(4)
  + 价格位置(3) + 估值指标(4: peTTM+pbMRQ+分位) = 78 维
  padding 到 88 维（预留 10 维扩展空间）
  (padding 到 80)
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
                               cfg: V2Config,
                               include_market_state: bool = True) -> np.ndarray:
    """从日线 DataFrame 逐日提取特征向量。

    Args:
        df: 单只股票 OHLCV DataFrame，需含 open/high/low/close/volume/amount/turnover
            及估值字段 peTTM/pbMRQ/psTTM/pcfNcfTTM
        df_index: 指数 DataFrame（可选），含 close
        cfg: V2Config 配置
        include_market_state: True=88维(含8维市场状态), False=80维(T4 LSTM用)。
                              树模型需要显式市场特征，LSTM能自学时序中的市场模式。

    Returns:
        (n_days, 80) 或 (n_days, 88) 特征矩阵，全部 Z-score 归一化
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

    # ── 6b. 市场状态特征 (8维，仅树模型使用) ──
    # 描述整体市场环境（牛市/熊市/震荡市/高波动/低波动），
    # 帮助模型学习「不同市场状态下因子方向不同」的规律。
    # T4 LSTM 可通过 120 步时序隐式推断市场状态，不需要显式特征。
    if include_market_state:
        if df_index is not None and len(df_index) == n:
            # 指数中期涨跌幅
            idx_ret_20d_raw = pd.Series(idx_close).pct_change(20).fillna(0.0).values
            idx_ret_60d_raw = pd.Series(idx_close).pct_change(60).fillna(0.0).values
            feature_list.append(idx_ret_20d_raw)            # 大盘近1月方向
            feature_list.append(idx_ret_60d_raw)            # 大盘近1季方向
            # 指数波动率（年化）
            idx_vol_20d_raw = pd.Series(idx_ret).rolling(20, min_periods=5).std().values * np.sqrt(252)
            idx_vol_60d_raw = pd.Series(idx_ret).rolling(60, min_periods=10).std().values * np.sqrt(252)
            feature_list.append(np.clip(idx_vol_20d_raw, 0.0, 1.0))                # 大盘短期波动
            feature_list.append(idx_vol_20d_raw / np.maximum(idx_vol_60d_raw, 1e-6) - 1.0)  # 波动率加速度
            # 指数回撤（从高点跌了多少）
            idx_high_20d_raw = pd.Series(idx_close).rolling(20, min_periods=1).max().values
            idx_high_60d_raw = pd.Series(idx_close).rolling(60, min_periods=1).max().values
            feature_list.append(idx_close / np.maximum(idx_high_20d_raw, 1e-10) - 1.0)  # 大盘20日回撤
            feature_list.append(idx_close / np.maximum(idx_high_60d_raw, 1e-10) - 1.0)  # 大盘60日回撤
            # 短期均线vs中期均线（趋势方向信号）
            idx_ma5_new = pd.Series(idx_close).rolling(5, min_periods=1).mean().values
            feature_list.append(idx_ma5_new / np.maximum(idx_ma20, 1e-10) - 1.0)  # 大盘5日vs20日均线
            # 指数上涨天数占比（市场广度代理变量）
            idx_up = (idx_ret > 0).astype(float)
            feature_list.append(pd.Series(idx_up).rolling(20, min_periods=1).mean().values)  # 近20日上涨占比
        else:
            for _ in range(8):
                feature_list.append(np.zeros(n))

    # 确定目标维度
    target_dim = 88 if include_market_state else 80

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

    # ── 7b. 最大回撤特征 (3维) ──
    # 20日滚动最大回撤：从20日高点跌了多少
    high_20d_roll = pd.Series(high).rolling(20, min_periods=1).max().values
    drawdown_20d = close / np.maximum(high_20d_roll, 1e-10) - 1.0  # <=0 的值
    feature_list.append(drawdown_20d)
    # 60日滚动最大回撤
    high_60d_roll = pd.Series(high).rolling(60, min_periods=1).max().values
    drawdown_60d = close / np.maximum(high_60d_roll, 1e-10) - 1.0
    feature_list.append(drawdown_60d)
    # 恢复率：从60日最低点恢复了多少（0=还在底部，1=回到顶部）
    low_60d_roll = pd.Series(low).rolling(60, min_periods=1).min().values
    recovery = (close - low_60d_roll) / np.maximum(high_60d_roll - low_60d_roll, 1e-10)
    recovery = np.clip(recovery, 0.0, 1.0)
    feature_list.append(recovery)

    # ── 7c. 收益分布特征 (4维) ──
    # 20日收益率偏度：正偏=稳步上涨，负偏=暴涨暴跌
    ret_skew = pd.Series(ret_1d).rolling(20, min_periods=5).skew().fillna(0.0).values
    feature_list.append(np.clip(ret_skew, -3.0, 3.0))
    # 20日收益率峰度（超额峰度）：高=极端行情，低=平稳
    ret_kurt = pd.Series(ret_1d).rolling(20, min_periods=5).kurt().fillna(0.0).values
    feature_list.append(np.clip(ret_kurt, -3.0, 10.0))
    # 20日内上涨天数占比
    up_days = (ret_1d > 0).astype(float)
    up_ratio = pd.Series(up_days).rolling(20, min_periods=1).mean().values
    feature_list.append(up_ratio)
    # 非对称波动：下跌日波动 / 全样本波动（>1=下跌波动更大）
    ret_neg = np.where(ret_1d < 0, ret_1d, 0.0)
    neg_vol = pd.Series(ret_neg).rolling(20, min_periods=5).std().fillna(0.0).values
    all_vol = pd.Series(ret_1d).rolling(20, min_periods=5).std().fillna(0.0).values
    asym_vol = neg_vol / np.maximum(all_vol, 1e-6)
    feature_list.append(np.clip(asym_vol, 0.0, 3.0))

    # ── 7d. 时间日历特征 (4维) ──
    # 从 date 列提取时间信息
    if "date" in df.columns:
        dates_pd = pd.to_datetime(df["date"])
        weekday = dates_pd.dt.dayofweek.values.astype(float)  # 0=周一
        day_of_month = dates_pd.dt.day.values.astype(float)
        month = dates_pd.dt.month.values.astype(float)
        # 季末标记：3/6/9/12月的最后5个交易日
        is_quarter_end = np.zeros(n)
        for qm in [3, 6, 9, 12]:
            qm_mask = month == qm
            if qm_mask.sum() > 0:
                qm_days = day_of_month[qm_mask]
                threshold = np.sort(qm_days)[-5] if len(qm_days) >= 5 else qm_days[0]
                is_quarter_end[qm_mask & (day_of_month >= threshold)] = 1.0
    else:
        weekday = np.zeros(n)
        day_of_month = np.zeros(n)
        month = np.zeros(n)
        is_quarter_end = np.zeros(n)
    # sin/cos 编码保证周期性（周一和周五在圆的同一侧）
    feature_list.append(np.sin(2 * np.pi * weekday / 5.0))
    feature_list.append(np.cos(2 * np.pi * weekday / 5.0))
    feature_list.append(day_of_month / 31.0)        # 归一化到 [0,1]
    feature_list.append(is_quarter_end)              # 0/1 布尔

    # ── 7e. 价格位置特征 (3维) ──
    # 收盘在今日波幅中的位置（0=最低，1=最高）
    close_position = (close - low) / np.maximum(high - low, 1e-10)
    feature_list.append(close_position)
    # 5日平均绝对跳空缺口：反映近期开盘情绪的强度
    avg_gap_5d = pd.Series(np.abs(gap)).rolling(5, min_periods=1).mean().values
    feature_list.append(avg_gap_5d)
    # 5日平均振幅 vs 20日平均振幅（波动强度变化）
    range_5d = pd.Series(hl_ratio).rolling(5, min_periods=1).mean().values
    range_20d = pd.Series(hl_ratio).rolling(20, min_periods=1).mean().values
    feature_list.append(range_5d / np.maximum(range_20d, 1e-10))

    # ── 8. 估值指标特征 (4维: peTTM + pbMRQ + 各自60日分位) ──
    # psTTM/pcfNcfTTM 因数据源不稳定已移除，仅保留腾讯实时行情可获取的PE/PB
    for col in ["peTTM", "pbMRQ"]:
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
    # padding 到目标维度（80维=无市场状态, 88维=含市场状态）
    while len(feature_list) < target_dim:
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
    include_market_state: bool = True,
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

    per_day = _extract_per_day_features(df, df_index, cfg, include_market_state)
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
    include_market_state: bool = True,
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

    per_day = _extract_per_day_features(df, df_index, cfg, include_market_state)
    if len(per_day) < cfg.window:
        return None

    X = per_day[-cfg.window:]
    return X[np.newaxis, :, :]
