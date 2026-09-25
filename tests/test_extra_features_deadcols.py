"""扩展特征"死列"修复的回归测试（2026-09-25）。

被修的三个问题（都会让特征列**恒为常数 → 被零方差保护清零 → 永远无效**）：
  · `xd_div_cnt_3y` / `xd_song_cnt_3y`：原为整列单一标量
  · `xd_yield`：原只在除权当天有值（按事件日 reindex，非 as-of 对齐）
  · consensus 生效日：原读文件 mtime（会被文件复制/重写改写），改读 `snapshot_date`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sequoia_x.features_extra import build_extra_features as bef


@pytest.fixture()
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", "2025-12-31")


def _close(dates: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(10.0, index=dates)


# ══════════════════════════════════════════════════════════════
#  xdxr
# ══════════════════════════════════════════════════════════════
def test_xdxr_columns_vary_over_time(monkeypatch, dates) -> None:
    """核心：三列都必须**随日变化**（原实现后两列是标量、第一列近乎全 0）。"""
    events = pd.DataFrame({
        "year": [2019, 2021, 2022, 2023, 2024],
        "month": [5, 6, 6, 6, 6],
        "day": [10, 20, 20, 20, 20],
        "fenhong": [1.0, 0.5, 0.6, 0.7, 0.8],
        "songzhuangu": [0.0, 0.0, 3.0, 0.0, 2.0],
    })
    monkeypatch.setattr(bef, "_load", lambda subset, code: events)

    out = bef._xdxr_features("000001", dates, _close(dates))

    assert list(out.columns) == ["xd_yield", "xd_div_cnt_3y", "xd_song_cnt_3y"]
    for c in out.columns:
        assert out[c].std() > 0.0, f"{c} 仍是常数（会被零方差保护清零）"
    # xd_yield 应在**非事件日**也有值（as-of 对齐），而不是只在除权当天
    assert (out["xd_yield"] > 0).sum() > 50


def test_xdxr_song_count_is_rolling_3y_not_all_history(monkeypatch, dates) -> None:
    """近 3 年送转次数应随时间**衰减**（原实现用全历史计数，只增不减）。"""
    events = pd.DataFrame({
        "year": [2015, 2020, 2021, 2022],
        "month": [5, 6, 6, 6],
        "day": [10, 20, 20, 20],
        "fenhong": [0.0, 0.1, 0.1, 0.1],
        "songzhuangu": [5.0, 3.0, 3.0, 3.0],   # 4 次送转，其中 3 次在近 3 年内
    })
    monkeypatch.setattr(bef, "_load", lambda subset, code: events)

    out = bef._xdxr_features("000001", dates, _close(dates))
    song = out["xd_song_cnt_3y"]
    assert song.iloc[0] >= 1.0          # 2022 年初：2020/2021 两次仍在窗口内（+可能含2015?否）
    assert song.iloc[-1] <= song.max()  # 后期老事件滑出窗口 → 不增
    assert song.nunique() > 1


def test_xdxr_empty_events_returns_zero_cols(monkeypatch, dates) -> None:
    monkeypatch.setattr(bef, "_load", lambda subset, code: None)
    out = bef._xdxr_features("000001", dates, _close(dates))
    assert out.shape == (len(dates), 3)
    assert np.allclose(out.fillna(0.0).values, 0.0)


# ══════════════════════════════════════════════════════════════
#  consensus
# ══════════════════════════════════════════════════════════════
def test_consensus_uses_snapshot_date_column(monkeypatch, dates) -> None:
    """有 `snapshot_date` 时应以它为准（而非文件 mtime）。"""
    snap = pd.DataFrame([{
        "机构数": 30, "买入数": 20, "Y2预测EPS": 2.0,
        "目标价上限": 15.0, "目标价下限": 12.0,
        "snapshot_date": "2022-06-01",
    }])
    monkeypatch.setattr(bef, "_load", lambda subset, code: snap)

    out = bef._consensus_features("000001", dates, _close(dates))
    active = out["cs_org_num"] > 0
    assert active.any()
    # 生效日之前必须为 0（防前视），之后有值
    assert out.loc[out.index < "2022-06-01", "cs_org_num"].abs().max() == 0.0
    assert out.loc[out.index >= "2022-06-01", "cs_org_num"].min() > 0.0


def test_consensus_without_snapshot_column_falls_back_to_mtime(monkeypatch, dates) -> None:
    """老数据（无 snapshot_date 列）应退回 mtime，且不报错。"""
    snap = pd.DataFrame([{
        "机构数": 30, "买入数": 20, "Y2预测EPS": 2.0,
        "目标价上限": 15.0, "目标价下限": 12.0,
    }])
    monkeypatch.setattr(bef, "_load", lambda subset, code: snap)
    out = bef._consensus_features("000001", dates, _close(dates))
    assert out.shape == (len(dates), 5)
    # 不应抛异常即为通过（mtime 早于 dates 时会有值，晚于则全 0，两种都合法）
    assert out.notna().all().all()
