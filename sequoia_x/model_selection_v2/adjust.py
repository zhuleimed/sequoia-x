"""复权模块（2026-08-10 新增，补丁②）——后复权价计算。

背景（实测确认 2026-08-10）：
- sequoia_v2.db 全库为【不复权】价（实际成交价，除权日有完整跳空；抽样 30 只全历史
  100% 匹配腾讯 bfq / baostock adjustflag=3；全库 79028 个除权事件验证一致）。
- DB 存不复权是**正确**的（模拟盘/回测执行层必须用实际成交价）。
- 但特征/标签计算用不复权价会在除权日产生假断层（分红 1-3%、送转 10-50% 假跌）——
  因此特征层与标签层必须用【后复权】价。

为什么用后复权而非前复权：
- 后复权基准固定在上市日，历史价格永不漂移——任何时点计算同一历史日的后复权价都相同，
  写入/缓存一次永久有效；前复权基准=拉取日，每次除权后全历史重算，缓存会持续失效。
- 后复权价连续（除权日无跳空），收益率/均线/动量等特征全部正确。

实现：
- 复权因子来自 extra_features/xdxr/{code}.parquet（分红/送转/配股事件记录）。
- 单事件复权因子 f = 前收盘价 / 除权参考价
    A 股除权参考价 = (前收盘 − 每股现金红利 + 配股价×配股比例) / (1 + 送转比例 + 配股比例)
- 累计后复权因子 F(t) = ∏(f_i, 事件日 ≤ t)；后复权价 = 实际价 × F(t)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sequoia_x.features_extra.build_extra_features import _load  # 复用 lru_cache 读取

_EVENT_COLS = ["year", "month", "day", "fenhong", "songzhuangu", "peigu", "peigujia"]


def _load_events(symbol: str) -> pd.DataFrame | None:
    """读取并规范化除权事件（每10股单位 → 每股）。"""
    df = _load("xdxr", symbol)
    if df is None or len(df) == 0:
        return None
    try:
        ev = pd.DataFrame({
            "avail": pd.to_datetime(
                df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str),
                errors="coerce"),
            "div_ps": pd.to_numeric(df["fenhong"], errors="coerce").fillna(0.0) / 10.0,       # 每股派息(元)
            "szg_ps": pd.to_numeric(df["songzhuangu"], errors="coerce").fillna(0.0) / 10.0,   # 每股送转(股)
            "peigu_ps": pd.to_numeric(df["peigu"], errors="coerce").fillna(0.0) / 10.0,       # 每股配股(股)
            "peigujia": pd.to_numeric(df["peigujia"], errors="coerce").fillna(0.0),           # 配股价(元/股)
        })
    except Exception:
        return None
    ev = ev.dropna(subset=["avail"])
    # 无实际影响的"事件"（无分红无送转无配股）直接丢弃
    ev = ev[(ev["div_ps"] != 0) | (ev["szg_ps"] != 0) | (ev["peigu_ps"] != 0)]
    ev = ev.drop_duplicates(subset="avail", keep="last").sort_values("avail").reset_index(drop=True)
    return ev if len(ev) else None


def build_adjust_factors(symbol: str, prices: pd.Series) -> pd.Series:
    """计算每交易日的累计后复权因子 F(t)。

    Args:
        symbol: 股票代码（与 extra_features/xdxr/{code}.parquet 同名）。
        prices: 按日期升序的收盘价序列（index=日期, 值=不复权价），仅用于取"除权日前收盘价"。

    Returns:
        Series: 与 prices 同 index 的累计后复权因子 F(t)（≥1）。
    """
    n = len(prices)
    factors = np.ones(n)
    ev = _load_events(symbol)
    if ev is None or n == 0:
        return pd.Series(factors, index=prices.index)

    dates = pd.to_datetime(prices.index)
    prev_close = prices.values.astype(float)
    # 对每个事件：找除权日前一交易日的收盘价（用于计算参考价）
    for _, r in ev.iterrows():
        d = r["avail"]
        # 除权日前最近的交易日（严格 < d）
        mask = dates < d
        if not mask.any():
            continue
        p_prev = prev_close[np.where(mask)[0][-1]]
        if p_prev is None or not np.isfinite(p_prev) or p_prev <= 0:
            continue
        # 除权参考价 = (P_prev − 每股派息 + 配股价×配股比例) / (1 + 送转比例 + 配股比例)
        ref = (p_prev - r["div_ps"] + r["peigujia"] * r["peigu_ps"]) / (1.0 + r["szg_ps"] + r["peigu_ps"])
        if ref <= 0:
            continue
        f = p_prev / ref
        if not np.isfinite(f) or f <= 0:
            continue
        # 事件日（含）之后的所有交易日都乘上该因子
        idx_start = np.where(mask)[0][-1] + 1  # 事件日起（除权日当天价格已按新基准）
        factors[idx_start:] *= f
    return pd.Series(factors, index=prices.index)


def apply_adjust(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """对 OHLCV DataFrame 做后复权（原地修改 open/high/low/close 列）。

    Args:
        df: 含 date/open/high/low/close 列，date 为 YYYY-MM-DD 字符串。
        symbol: 股票代码。
    """
    if df is None or len(df) == 0 or "close" not in df.columns:
        return df
    close_s = pd.Series(df["close"].values.astype(float), index=pd.to_datetime(df["date"]))
    F = build_adjust_factors(symbol, close_s)
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = (df[col].values.astype(float) * F.values)
    return df


def adjusted_close(symbol: str, ref_date: str, engine, horizon: int = 0) -> pd.Series:
    """取某股票某日后的后复权收盘序列（labels.py 用）。"""
    df = engine.get_ohlcv(symbol)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    df = df[df["date"] <= ref_date].copy()
    apply_adjust(df, symbol)
    return df["close"]
