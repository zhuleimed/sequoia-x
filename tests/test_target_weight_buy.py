#!/usr/bin/env python
"""验证：SimEngine 按 target_weight 分配买入预算（2026-10-03，P0 方向一）。

覆盖三件事（对应生产侧接入的三条承诺）：
  1. **权重生效**：带 target_weight 的信号，买入金额 ≈ 基准预算 × 权重（整手取整内）
  2. **向后兼容**：target_weight = NULL（老信号）→ 按等权处理（权重 1.0）
  3. **回滚开关**：SimEngine(use_weights=False) → 忽略权重，全部按等权

做法：临时 sim 库 + **真实行情主库**（只读），把"今天"patch 成主库最后一个交易日，
直接跑 `_execute_pending_buys`（生产买入路径本身，不是简化复刻）。

运行（铁律六：py312）：
    /home/zhulei/anaconda3/envs/zhulei_py312/bin/python -m pytest tests/test_target_weight_buy.py -q
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

MAIN_DB = PROJ / "data" / "sequoia_v2.db"

pytestmark = pytest.mark.skipif(not MAIN_DB.exists(), reason="需要主库（行情数据）")

PER_BUDGET = 50_000.0
COMMISSION = 0.00025   # 与 simulation/config.COMMISSION_RATE 一致（仅用于校验上限）


def _last_trade_date() -> str:
    with sqlite3.connect(MAIN_DB) as c:
        return c.execute("SELECT MAX(date) FROM stock_daily").fetchone()[0]


def _pick_symbols(date: str, n: int = 4) -> list[str]:
    """挑几只该日有行情、开盘价 > 2 元的股票（避免退市/低价股干扰）。"""
    with sqlite3.connect(MAIN_DB) as c:
        rows = c.execute(
            "SELECT symbol FROM stock_daily WHERE date=? AND open>2 AND volume>0 "
            "ORDER BY symbol LIMIT ?", (date, n * 4)).fetchall()
    syms, seen = [], set()
    for (s,) in rows:
        if s not in seen:
            seen.add(s)
            syms.append(s)
        if len(syms) >= n:
            break
    return syms


def _open_price(symbol: str, date: str) -> float:
    with sqlite3.connect(MAIN_DB) as c:
        r = c.execute("SELECT open FROM stock_daily WHERE symbol=? AND date=?",
                      (symbol, date)).fetchone()
    return float(r[0]) if r and r[0] else 0.0


def _run_arm(tmp_path: Path, monkeypatch, symbols: list[str], weights: list,
             use_weights: bool, buy_date: str) -> dict[str, dict]:
    """跑一臂：建临时库 → 写信号 → 执行买入 → 返回 {symbol: 买入记录}。"""
    from sequoia_x.core.config import get_settings
    from sequoia_x.simulation import models
    from sequoia_x.simulation.engine import SimEngine
    from sequoia_x.simulation.models import init_sim_tables

    class _FakeDate(dt.date):
        @classmethod
        def today(cls):
            return dt.date.fromisoformat(buy_date)

    # 只 patch models 模块里的 date（get_pending_signals 用它取"今天"）
    monkeypatch.setattr(models, "date", _FakeDate)

    db = tmp_path / f"sim_weight_{'on' if use_weights else 'off'}.db"
    init_sim_tables(str(db))
    with sqlite3.connect(db) as c:
        for sym, w in zip(symbols, weights):
            c.execute(
                "INSERT INTO sim_buy_signals "
                "(symbol, strategy_from, llm_score, target_weight, buy_date, status) "
                "VALUES (?, '测试', NULL, ?, ?, 'pending')",
                (sym, w, buy_date),   # buy_date == 今天 → T+1 不成立，故下面用更早的日期
            )
        c.commit()

    # 信号 buy_date 必须 < 执行日（严格 T+1），改为前一自然日
    prev_day = (dt.date.fromisoformat(buy_date) - dt.timedelta(days=3)).isoformat()
    with sqlite3.connect(db) as c:
        c.execute("UPDATE sim_buy_signals SET buy_date=?", (prev_day,))
        c.commit()

    settings = get_settings()
    engine = SimEngine(settings, db_path=str(db), max_positions=20,
                       per_stock_budget=PER_BUDGET, use_weights=use_weights)
    bought, _cancelled = engine._execute_pending_buys(buy_date)
    return {b["symbol"]: b for b in bought}


@pytest.mark.parametrize("use_weights", [True, False])
def test_target_weight_buy(tmp_path, monkeypatch, use_weights):
    """权重生效（use_weights=True）且回滚开关有效（False）。"""
    buy_date = _last_trade_date()
    symbols = _pick_symbols(buy_date, n=4)
    if len(symbols) < 4:
        pytest.skip("该交易日行情不足，跳过")

    weights = [1.5, 1.0, 0.5, None]     # 最后一个验证"NULL=等权"
    got = _run_arm(tmp_path, monkeypatch, symbols, weights, use_weights, buy_date)

    # 被涨跌停/停牌取消的标的会不在 got 里 —— 只校验真正买到的
    checked = 0
    for sym, w in zip(symbols, weights):
        if sym not in got:
            continue
        expect_w = 1.0 if (w is None or not use_weights) else w
        price = _open_price(sym, buy_date) * (1 + 0.0001)   # 含滑点（与引擎一致）
        expect_shares = int((PER_BUDGET * expect_w) // price // 100) * 100
        assert got[sym]["shares"] == expect_shares, (
            f"{sym}: 期望 {expect_shares} 股（权重 {expect_w}），实际 {got[sym]['shares']} 股")
        checked += 1

    assert checked >= 2, f"可校验样本太少（{checked} 只，可能都被涨跌停取消）"
