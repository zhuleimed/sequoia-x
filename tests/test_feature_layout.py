"""特征列布局一致性回归测试（2026-09-25 新增）。

背景
----
`features.py` 的 §6 大盘关联段曾在"指数可用 / 不可用"两条分支下追加**不同列数**
（`if` 6 维 vs `else` 8 维），使 §7 之后**所有列**在两套布局下相差 2 位。
而"指数是否可用"是**逐股**判定的（`len(df_index) == n`），
于是一次重建会让**同一个 X 里同时存在两套列布局** ——
即"同一列在不同样本里含义不同"，且**不会报错**，只会静默污染训练。

本测试锁死该不变量：**无论指数是否可用，列数与关键区段位置都必须一致。**

列布局（base=68，含市场状态时 +8 = 76，padding 到 88）：
    §6  大盘关联   : 37..42   (6)
    §6b 市场状态   : 43..50   (8)
    §7  价格形态   : 51..57   (7)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.features import (
    BASE_FEATURE_DIM,
    _extract_per_day_features,
)

# 关键区段起始位置（与 BASE_FEATURE_DIM 同源，改布局时必须同步改这里）
SEC6_START = 37
SEC6_END = 43          # exclusive
SEC6B_START = 43
SEC6B_END = 51


def _make_ohlcv(n: int = 260, seed: int = 7) -> pd.DataFrame:
    """构造一段合成日线（字段与真实 stock_daily 对齐）。"""
    rng = np.random.default_rng(seed)
    close = 10.0 * np.cumprod(1 + rng.normal(0, 0.02, n))
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "open": close * 0.99,
        "high": close * 1.02,
        "low": close * 0.98,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        "amount": close * 2_000_000,
        "turnover": 1.5,
        "peTTM": 20.0,
        "pbMRQ": 2.0,
    })


def _make_index(df: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """构造与 df 等长的指数序列（走 §6/§6b 的 `if` 分支）。"""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "date": df["date"],
        "close": 3000.0 * np.cumprod(1 + rng.normal(0, 0.01, len(df))),
    })


def test_base_dim_arithmetic() -> None:
    """BASE_FEATURE_DIM 必须等于各段维数之和（8+6+8+11+4+6+7+3+4+4+3+4）。"""
    assert BASE_FEATURE_DIM == 8 + 6 + 8 + 11 + 4 + 6 + 7 + 3 + 4 + 4 + 3 + 4


def test_layout_identical_with_and_without_index() -> None:
    """核心不变量：指数可用与否，列数必须一致（这正是原 bug 违反的）。"""
    cfg = get_config()
    df = _make_ohlcv()
    idx = _make_index(df)

    x_with = _extract_per_day_features(df, idx, cfg, include_market_state=True)
    x_without = _extract_per_day_features(df, None, cfg, include_market_state=True)

    assert x_with.shape == x_without.shape, (
        f"指数可用/不可用两种布局列数不一致：{x_with.shape} vs {x_without.shape}"
    )
    # 76 基础 + 12 padding = 88
    assert x_with.shape[1] == 88


def test_index_unavailable_zeroes_only_sec6_and_sec6b() -> None:
    """指数不可用时，应**只有** §6/§6b 两段为 0（其余段照常有值）。"""
    cfg = get_config()
    df = _make_ohlcv()
    x = _extract_per_day_features(df, None, cfg, include_market_state=True)

    assert np.allclose(x[:, SEC6_START:SEC6_END], 0.0), "§6 应全 0"
    assert np.allclose(x[:, SEC6B_START:SEC6B_END], 0.0), "§6b 应全 0"
    # §1 价格收益段必须有信息（证明不是整体没算）
    assert x[:, 0:8].std(axis=0).max() > 0.0
    # §7 价格形态段（紧随 §6b 之后）也必须有信息 —— 若布局错位，这里会退化
    assert x[:, SEC6B_END:SEC6B_END + 7].std(axis=0).max() > 0.0


def test_index_available_populates_sec6_and_sec6b() -> None:
    """指数可用时，§6/§6b 必须有真实信息（这正是"推理时恒 0"要修的点）。"""
    cfg = get_config()
    df = _make_ohlcv()
    idx = _make_index(df)
    x = _extract_per_day_features(df, idx, cfg, include_market_state=True)

    assert x[:, SEC6_START:SEC6_END].std(axis=0).max() > 0.0, "§6 不应全 0"
    assert x[:, SEC6B_START:SEC6B_END].std(axis=0).max() > 0.0, "§6b 不应全 0"


def test_market_state_toggle_shifts_layout_by_8() -> None:
    """include_market_state=False（T4 LSTM）时应少 8 维，且 §7 起点前移 8。"""
    cfg = get_config()
    df = _make_ohlcv()
    idx = _make_index(df)

    x_on = _extract_per_day_features(df, idx, cfg, include_market_state=True)
    x_off = _extract_per_day_features(df, idx, cfg, include_market_state=False)
    assert x_on.shape[1] - x_off.shape[1] == 8
    assert x_off.shape[1] == 80


def test_extra_matrix_appends_without_shifting_base() -> None:
    """拼接扩展特征只应**追加**列，不应改变基础段位置。"""
    cfg = get_config()
    df = _make_ohlcv()
    idx = _make_index(df)
    extra = np.random.default_rng(3).normal(size=(len(df), 41)).astype(np.float32)

    x = _extract_per_day_features(df, idx, cfg, include_market_state=True,
                                  extra_matrix=extra)
    x_base = _extract_per_day_features(df, idx, cfg, include_market_state=True)
    assert x.shape[1] == 88 + 41

    # 基础段 = 前 BASE_FEATURE_DIM+8 (=76) 列，逐列一致。
    # ⚠️ 注意不能拿 `:88` 来比 —— 拼了扩展特征后 88 里已经含了扩展列，
    #    而不拼时 88 的尾部是 padding 0，两者本就应该不同。
    n_base = BASE_FEATURE_DIM + 8
    assert np.allclose(x[:, :n_base], x_base[:, :n_base], atol=1e-5)
    # 扩展列应落在 [n_base, n_base+41)
    assert np.abs(x[:, n_base:n_base + 41]).max() > 0.0
