"""追高过滤（2026-09-25）单元测试。

覆盖 `sequoia_x.simulation.signals` 的：
  - signal_day_return()  —— 取信号日单日涨幅
  - filter_chase_risk()  —— 按阈值剔除信号日大涨的标的

背景见 docs/2026-09_LLM模拟盘买入即跌归因诊断.md：
信号日涨幅与买入后收益单调负相关（0~6% 为正超额，>6% 转负）。
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from sequoia_x.simulation.signals import (
    CHASE_RET_LIMIT,
    filter_chase_risk,
    signal_day_return,
)


@pytest.fixture()
def db() -> str:
    """构造一个只含 stock_daily 的临时库。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "t.db")
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE stock_daily (symbol TEXT, date TEXT, close REAL, "
                "UNIQUE(symbol, date))"
            )
            rows = [
                # AAA：信号日 +10%（涨停）→ 应被剔除
                ("000001", "2026-09-01", 10.00),
                ("000001", "2026-09-02", 11.00),
                # BBB：信号日 +3% → 保留
                ("000002", "2026-09-01", 10.00),
                ("000002", "2026-09-02", 10.30),
                # CCC：信号日 -2% → 保留
                ("000003", "2026-09-01", 10.00),
                ("000003", "2026-09-02", 9.80),
                # DDD：恰好等于阈值 6% → 保留（判据是严格大于）
                ("000004", "2026-09-01", 10.00),
                ("000004", "2026-09-02", 10.60),
                # EEE：信号日无数据 → 保留（fail-open）
                ("000005", "2026-09-01", 10.00),
            ]
            conn.executemany("INSERT INTO stock_daily VALUES (?,?,?)", rows)
            conn.commit()
        yield path


def test_signal_day_return_basic(db: str) -> None:
    assert signal_day_return(db, "000001", "2026-09-02") == pytest.approx(0.10)
    assert signal_day_return(db, "000002", "2026-09-02") == pytest.approx(0.03)
    assert signal_day_return(db, "000003", "2026-09-02") == pytest.approx(-0.02)


def test_signal_day_return_missing_returns_none(db: str) -> None:
    """信号日当天无数据 / 无前收 / 股票不存在 → None。"""
    assert signal_day_return(db, "000005", "2026-09-02") is None   # 当天无行
    assert signal_day_return(db, "000001", "2026-09-01") is None   # 无前一交易日
    assert signal_day_return(db, "999999", "2026-09-02") is None   # 股票不存在


def test_filter_drops_limit_up(db: str) -> None:
    kept, dropped = filter_chase_risk(
        db, ["000001", "000002", "000003"], "2026-09-02")
    assert kept == ["000002", "000003"]
    assert [s for s, _ in dropped] == ["000001"]
    assert dropped[0][1] == pytest.approx(0.10)


def test_filter_keeps_at_threshold(db: str) -> None:
    """恰好 6% 应保留（判据是 > limit，不是 >=）。"""
    kept, dropped = filter_chase_risk(db, ["000004"], "2026-09-02")
    assert kept == ["000004"]
    assert dropped == []
    assert CHASE_RET_LIMIT == pytest.approx(0.06)


def test_filter_fail_open_on_missing_data(db: str) -> None:
    """取不到当日涨幅时保留原标的（过滤依据是"确认涨太多"，不是"无法判断"）。"""
    kept, dropped = filter_chase_risk(db, ["000005"], "2026-09-02")
    assert kept == ["000005"]
    assert dropped == []


def test_filter_custom_limit(db: str) -> None:
    """自定义阈值应生效：阈值 2% 时 +3% 的标的被剔除。"""
    kept, dropped = filter_chase_risk(db, ["000002"], "2026-09-02", limit=0.02)
    assert kept == []
    assert [s for s, _ in dropped] == ["000002"]


def test_filter_preserves_order(db: str) -> None:
    """保留项顺序应与输入一致（下游可能依赖顺序）。"""
    kept, _ = filter_chase_risk(
        db, ["000003", "000002", "000004"], "2026-09-02")
    assert kept == ["000003", "000002", "000004"]


def test_signal_day_return_raises_on_wrong_db() -> None:
    """传递台账库（无 stock_daily）必须立刻报错，不许静默放行。

    2026-10-09 回归：LLM-HS300（024）台账库独立后，此处曾静默 fail-open
    （只留一行警告）⇒ 追高过滤整条失效。表不存在 = 传错库，属配置错误。
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "ledger.db")
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE sim_positions (id INTEGER PRIMARY KEY)")
            conn.commit()
        with pytest.raises(RuntimeError, match="stock_daily"):
            signal_day_return(path, "000001", "2026-09-02")


def test_save_llm_recommendations_with_split_dbs() -> None:
    """台账库 / 行情库分离时（024 的真实拓扑），追高过滤仍必须生效。

    行情库：AAA 信号日 +10%（应被剔除）、BBB +3%（应写入台账库）。
    """
    from datetime import date, timedelta

    from sequoia_x.simulation.signals import save_llm_recommendations

    today = date.today().isoformat()
    prev = (date.today() - timedelta(days=1)).isoformat()

    with tempfile.TemporaryDirectory() as tmp:
        ledger = str(Path(tmp) / "sim_ledger.db")
        prices = str(Path(tmp) / "prices.db")
        with sqlite3.connect(prices) as conn:
            conn.execute("CREATE TABLE stock_daily (symbol TEXT, date TEXT, "
                         "close REAL, UNIQUE(symbol, date))")
            conn.executemany("INSERT INTO stock_daily VALUES (?,?,?)", [
                ("AAA", prev, 10.00), ("AAA", today, 11.00),
                ("BBB", prev, 10.00), ("BBB", today, 10.30),
            ])
            conn.commit()

        n = save_llm_recommendations(
            db_path=ledger,
            price_db_path=prices,                      # ← 关键：显式传行情库
            strategies_results={"tag": ["AAA", "BBB"]},
            top_n=2,
            recommended=["AAA", "BBB"],
        )
        assert n == 1
        with sqlite3.connect(ledger) as conn:
            wrote = [r[0] for r in conn.execute("SELECT symbol FROM sim_buy_signals")]
        assert wrote == ["BBB"]
