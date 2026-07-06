"""SQLite 数据模型：模拟盘相关表的建表与 CRUD 操作。"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Optional

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# ════════════════════════════════════════════════════════════
#  建表 SQL（CREATE TABLE IF NOT EXISTS）
# ════════════════════════════════════════════════════════════

SQL_BUY_SIGNALS = """
CREATE TABLE IF NOT EXISTS sim_buy_signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    strategy_from TEXT    NOT NULL DEFAULT '',
    llm_score     REAL,
    buy_date      TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending','executed','cancelled')),
    executed_at   TEXT,
    buy_price     REAL,
    shares        INTEGER,
    cancel_reason TEXT,
    created_at    TEXT    DEFAULT (datetime('now','localtime'))
);
"""

SQL_POSITIONS = """
CREATE TABLE IF NOT EXISTS sim_positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    strategy_from   TEXT    NOT NULL DEFAULT '',
    buy_date        TEXT    NOT NULL,
    buy_price       REAL    NOT NULL,
    shares          INTEGER NOT NULL,
    total_cost      REAL    NOT NULL,
    highest_price   REAL,
    highest_value   REAL,
    current_price   REAL,
    current_value   REAL,
    pnl             REAL,
    pnl_pct         REAL,
    hold_days       INTEGER DEFAULT 0,
    today_opened    INTEGER DEFAULT 0,      -- 今日开仓（T+1 保护）
    signal_id       INTEGER,                -- 关联的买入信号 id
    pending_sell_reason TEXT,               -- 非空=待卖出（T+1待执行订单）
    llm_override_count  INTEGER DEFAULT 0,  -- LLM覆盖卖出次数（≥3强制卖出）
    UNIQUE(symbol, buy_date)
);
"""

SQL_CLOSED_TRADES = """
CREATE TABLE IF NOT EXISTS sim_closed_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    name            TEXT,
    strategy_from   TEXT    NOT NULL DEFAULT '',
    buy_date        TEXT    NOT NULL,
    sell_date       TEXT    NOT NULL,
    hold_days       INTEGER NOT NULL,
    buy_price       REAL    NOT NULL,
    sell_price      REAL    NOT NULL,
    shares          INTEGER NOT NULL,
    total_cost      REAL    NOT NULL,
    total_revenue   REAL    NOT NULL,
    commission      REAL    DEFAULT 0,
    stamp_tax       REAL    DEFAULT 0,
    pnl             REAL    NOT NULL,
    pnl_pct         REAL    NOT NULL,
    max_drawdown    REAL,
    sharpe_ratio    REAL,
    exit_reason     TEXT    NOT NULL,
    report_pushed   INTEGER DEFAULT 0,
    created_at      TEXT    DEFAULT (datetime('now','localtime'))
);
"""

SQL_ACCOUNT_DAILY = """
CREATE TABLE IF NOT EXISTS sim_account_daily (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL UNIQUE,
    cash            REAL    NOT NULL,
    stock_value     REAL    NOT NULL,
    total_value     REAL    NOT NULL,
    daily_pnl       REAL,
    daily_pnl_pct   REAL,
    total_pnl       REAL,
    total_return    REAL,
    position_count  INTEGER DEFAULT 0,
    created_at      TEXT    DEFAULT (datetime('now','localtime'))
);
"""

SQL_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_sim_signals_status ON sim_buy_signals (status, buy_date);
CREATE INDEX IF NOT EXISTS idx_sim_positions_symbol ON sim_positions (symbol);
CREATE INDEX IF NOT EXISTS idx_sim_closed_sell ON sim_closed_trades (sell_date);
CREATE INDEX IF NOT EXISTS idx_sim_account_date ON sim_account_daily (date);
"""


# ════════════════════════════════════════════════════════════
#  初始化
# ════════════════════════════════════════════════════════════


def init_sim_tables(db_path: str) -> None:
    """确保模拟盘相关的四张表存在（幂等）。"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        for sql in [SQL_BUY_SIGNALS, SQL_POSITIONS, SQL_CLOSED_TRADES, SQL_ACCOUNT_DAILY]:
            conn.execute(sql)
        # 迁移：为已有数据库补充新列
        for col_sql in [
            "ALTER TABLE sim_positions ADD COLUMN pending_sell_reason TEXT",
            "ALTER TABLE sim_positions ADD COLUMN llm_override_count INTEGER DEFAULT 0",
        ]:
            try:
                conn.execute(col_sql)
            except sqlite3.OperationalError:
                pass
        # 单独执行索引
        for stmt in SQL_INDEXES.strip().split("\n"):
            s = stmt.strip()
            if s:
                try:
                    conn.execute(s)
                except sqlite3.OperationalError:
                    pass
        conn.commit()
    logger.debug("模拟盘数据表初始化完成")


# ════════════════════════════════════════════════════════════
#  买入信号 CRUD
# ════════════════════════════════════════════════════════════


def insert_buy_signal(db_path: str, symbol: str, strategy_from: str,
                      llm_score: Optional[float] = None) -> Optional[int]:
    """写入一条买入信号，返回 id。"""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO sim_buy_signals (symbol, strategy_from, llm_score, buy_date, status) "
            "VALUES (?, ?, ?, ?, 'pending')",
            (symbol, strategy_from, llm_score, date.today().isoformat()),
        )
        conn.commit()
        return cur.lastrowid


def insert_buy_signals_batch(db_path: str, signals: list[dict]) -> int:
    """批量写入买入信号。

    Args:
        signals: [{"symbol": "600519", "strategy_from": "...", "llm_score": ...}]

    Returns:
        写入数量。
    """
    today_str = date.today().isoformat()
    with sqlite3.connect(db_path) as conn:
        count = 0
        for s in signals:
            try:
                conn.execute(
                    "INSERT INTO sim_buy_signals (symbol, strategy_from, llm_score, buy_date, status) "
                    "VALUES (?, ?, ?, ?, 'pending')",
                    (s["symbol"], s.get("strategy_from", ""), s.get("llm_score"), today_str),
                )
                count += 1
            except sqlite3.IntegrityError:
                continue
        conn.commit()
    return count


def get_pending_signals(db_path: str, target_date: Optional[str] = None) -> list[dict]:
    """获取待执行的买入信号（仅限 target_date 之前写入的），按 LLM 评分降序排列。
    今天写入的信号不会在今天执行，符合 T+1 模型。

    Args:
        target_date: 要执行的日期（默认为今天），信号 buy_date < target_date。

    Returns:
        [{"id": 1, "symbol": "600519", "strategy_from": "...", "llm_score": 4.5}, ...]
    """
    if target_date is None:
        target_date = date.today().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, symbol, strategy_from, llm_score, buy_date FROM sim_buy_signals "
            "WHERE status = 'pending' AND buy_date < ? "
            "ORDER BY llm_score IS NOT NULL DESC, llm_score DESC, buy_date ASC, id ASC",
            (target_date,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_signal_executed(db_path: str, signal_id: int, buy_price: float,
                         shares: int, executed_at: str) -> None:
    """标记买入信号已执行。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sim_buy_signals SET status='executed', buy_price=?, shares=?, executed_at=? "
            "WHERE id=?",
            (buy_price, shares, executed_at, signal_id),
        )
        conn.commit()


def mark_signal_cancelled(db_path: str, signal_id: int, reason: str) -> None:
    """标记买入信号已取消。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sim_buy_signals SET status='cancelled', cancel_reason=? WHERE id=?",
            (reason, signal_id),
        )
        conn.commit()


def get_today_recommended_symbols(db_path: str, today_str: str) -> set[str]:
    """获取今日 LLM 推荐的股票代码集合（pending 状态的今日信号）。

    用于卖出规则覆盖：若某股触发卖出但同日 LLM 也推荐，则跳过卖出。

    Returns:
        {"600519", "000858", ...}
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM sim_buy_signals "
            "WHERE buy_date = ? AND status = 'pending'",
            (today_str,),
        ).fetchall()
    return {r[0] for r in rows}


# ════════════════════════════════════════════════════════════
#  持仓 CRUD
# ════════════════════════════════════════════════════════════


def insert_position(db_path: str, symbol: str, strategy_from: str,
                    buy_date: str, buy_price: float, shares: int,
                    total_cost: float, signal_id: Optional[int] = None) -> Optional[int]:
    """写入一条持仓记录。"""
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO sim_positions "
            "(symbol, strategy_from, buy_date, buy_price, shares, total_cost, "
            " highest_price, highest_value, current_price, current_value, "
            " pnl, pnl_pct, hold_days, today_opened, signal_id, llm_override_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
            (symbol, strategy_from, buy_date, buy_price, shares, total_cost,
             buy_price, buy_price * shares, buy_price, buy_price * shares,
             0.0, 0.0, 1, 1, signal_id),
        )
        conn.commit()
        return cur.lastrowid


def get_all_positions(db_path: str) -> list[dict]:
    """获取当前所有持仓。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sim_positions ORDER BY buy_date ASC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_position_valuation(db_path: str, pos_id: int, current_price: float,
                              shares: int, total_cost: float) -> None:
    """更新单只持仓的每日估值。"""
    current_value = round(shares * current_price, 2)
    pnl = round(current_value - total_cost, 2)
    pnl_pct = round(pnl / total_cost, 6) if total_cost > 0 else 0.0

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT highest_price, hold_days FROM sim_positions WHERE id=?",
            (pos_id,),
        ).fetchone()
        highest = max(row[0] or 0, current_price) if row else current_price
        hold_days = row[1] if row else 0  # 保持当前值，不递增

        conn.execute(
            "UPDATE sim_positions SET current_price=?, current_value=?, "
            "pnl=?, pnl_pct=?, highest_price=?, highest_value=?, "
            "hold_days=? WHERE id=?",
            (current_price, current_value, pnl, pnl_pct,
             highest, max(highest * shares, current_value),
             hold_days, pos_id),
        )
        conn.commit()


def remove_position(db_path: str, pos_id: int) -> Optional[dict]:
    """删除持仓并返回其记录（用于卖出后转入 closed_trades）。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM sim_positions WHERE id=?", (pos_id,)).fetchone()
        if row is None:
            return None
        pos = dict(row)
        conn.execute("DELETE FROM sim_positions WHERE id=?", (pos_id,))
        conn.commit()
    return pos


# ── 待卖出订单（T+1 待执行卖出） ──


def mark_position_for_sell(db_path: str, pos_id: int, reason: str) -> None:
    """标记持仓为待卖出（创建 T+1 待执行卖出订单）。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sim_positions SET pending_sell_reason=? WHERE id=?",
            (reason, pos_id),
        )
        conn.commit()


def get_positions_pending_sell(db_path: str) -> list[dict]:
    """获取所有标记为待卖出的持仓。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM sim_positions WHERE pending_sell_reason IS NOT NULL"
        ).fetchall()
    return [dict(r) for r in rows]


def clear_pending_sell(db_path: str, pos_id: int) -> None:
    """清除待卖出标记（卖出取消时用）。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sim_positions SET pending_sell_reason=NULL WHERE id=?",
            (pos_id,),
        )
        conn.commit()


def increment_llm_override(db_path: str, pos_id: int) -> int:
    """LLM 覆盖卖出次数 +1，返回更新后的次数。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sim_positions SET llm_override_count = llm_override_count + 1 "
            "WHERE id=?",
            (pos_id,),
        )
        conn.commit()
        row = conn.execute(
            "SELECT llm_override_count FROM sim_positions WHERE id=?",
            (pos_id,),
        ).fetchone()
    return row[0] if row else 0


def reset_llm_override(db_path: str, pos_id: int) -> None:
    """重置 LLM 覆盖计数器（卖出正常执行时调用）。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE sim_positions SET llm_override_count=0 WHERE id=?",
            (pos_id,),
        )
        conn.commit()


# ════════════════════════════════════════════════════════════
#  已完成交易 CRUD
# ════════════════════════════════════════════════════════════


def insert_closed_trade(db_path: str, trade: dict) -> None:
    """写入一条已完成交易。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sim_closed_trades "
            "(symbol, name, strategy_from, buy_date, sell_date, hold_days, "
            " buy_price, sell_price, shares, total_cost, total_revenue, "
            " commission, stamp_tax, pnl, pnl_pct, max_drawdown, sharpe_ratio, exit_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (trade["symbol"], trade.get("name"), trade.get("strategy_from", ""),
             trade["buy_date"], trade["sell_date"], trade["hold_days"],
             trade["buy_price"], trade["sell_price"], trade["shares"],
             trade["total_cost"], trade["total_revenue"],
             trade.get("commission", 0), trade.get("stamp_tax", 0),
             trade["pnl"], trade["pnl_pct"],
             trade.get("max_drawdown"), trade.get("sharpe_ratio"),
             trade["exit_reason"]),
        )
        conn.commit()


# ════════════════════════════════════════════════════════════
#  账户总览 CRUD
# ════════════════════════════════════════════════════════════


def upsert_account_daily(db_path: str, record: dict) -> None:
    """写入或更新当日账户总览。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sim_account_daily "
            "(date, cash, stock_value, total_value, daily_pnl, daily_pnl_pct, "
            " total_pnl, total_return, position_count) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (record["date"], record["cash"], record["stock_value"],
             record["total_value"], record.get("daily_pnl"), record.get("daily_pnl_pct"),
             record.get("total_pnl"), record.get("total_return"), record.get("position_count", 0)),
        )
        conn.commit()


def get_account_summary(db_path: str, date_str: str) -> Optional[dict]:
    """获取指定日期的账户总览。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sim_account_daily WHERE date=?",
            (date_str,),
        ).fetchone()
    return dict(row) if row else None


def get_recent_account_days(db_path: str, days: int = 30) -> list[dict]:
    """获取最近 N 天的账户记录（用于计算滚动夏普）。"""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT date, total_value, daily_pnl_pct FROM sim_account_daily "
            "ORDER BY date DESC LIMIT ?",
            (days,),
        ).fetchall()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════


def get_cash_balance(db_path: str) -> float:
    """从账户总览获取最新现金余额，若无记录则返回初始资金。"""
    from sequoia_x.simulation.config import INITIAL_CAPITAL
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT cash FROM sim_account_daily ORDER BY date DESC LIMIT 1"
        ).fetchone()
    return row[0] if row else INITIAL_CAPITAL
