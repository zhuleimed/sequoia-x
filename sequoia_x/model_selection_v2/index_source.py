"""指数取数统一入口（2026-09-25 新增）。

背景（为什么需要这个模块）
--------------------------
训练侧（`labels.py`）与推理侧（`features.py`）**取指数用的是两张不同的表**：

  · `labels.py:137-144` —— 先查 `index_daily`（symbol=`sh.000300`），查不到再回退
    `stock_daily`（symbol=`000300`）。**这是对的。**
  · `features.py` 的两处调用点 —— 直接用 `engine.get_ohlcv("000300")`，而
    `DataEngine.get_ohlcv`（`sequoia_x/data/engine.py:167-174`）只查 `stock_daily`，
    该表里根本没有 `000300` 这行（指数在 `index_daily`）→ **恒返回空** →
    `df_index=None` → §6 大盘关联(6维) + §6b 市场状态(8维) **共 14 维恒为 0**。

这类差异**不会报错**，只会静默产生恒 0 特征（本次实测：生产缓存里 37~52 列零方差）。
故抽出本模块，让训练与推理走同一条取数路径。

用法
----
    from sequoia_x.model_selection_v2.index_source import load_index_close
    df_index = load_index_close(engine, ref_date="2026-09-24")   # ref_date 可选
"""

from __future__ import annotations

import sqlite3

import pandas as pd

# 指数在两个表里的代码写法不同，别写混
INDEX_SYMBOL_INDEX_DAILY: str = "sh.000300"
INDEX_SYMBOL_STOCK_DAILY: str = "000300"   # 旧路径/历史遗留，仅作兜底


def load_index_close(engine, ref_date: str | None = None) -> pd.DataFrame | None:
    """加载沪深300指数收盘序列（按日期升序）。

    取数优先级：`index_daily`（`sh.000300`）→ `stock_daily`（`000300`）兜底。

    Args:
        engine: DataEngine（只用到其 db_path）。
        ref_date: 截止日期（含），None 表示不限制。

    Returns:
        含 `date` / `close` 两列的 DataFrame（升序）；两处都取不到时返回 None。
    """
    where = " AND date <= ?" if ref_date else ""
    params_tail: tuple = (ref_date,) if ref_date else ()

    conn = sqlite3.connect(engine.db_path)
    try:
        rows = conn.execute(
            f"SELECT date, close FROM index_daily WHERE symbol = ?{where} ORDER BY date",
            (INDEX_SYMBOL_INDEX_DAILY,) + params_tail,
        ).fetchall()
        if not rows:
            rows = conn.execute(
                f"SELECT date, close FROM stock_daily WHERE symbol = ?{where} ORDER BY date",
                (INDEX_SYMBOL_STOCK_DAILY,) + params_tail,
            ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df


def load_index_close_forward(conn: sqlite3.Connection, ref_date: str,
                             limit: int) -> list[tuple]:
    """加载 ref_date **之后**的指数收盘序列（用于计算未来超额收益标签）。

    与 `load_index_close` 的区别：标签侧需要的是"未来 N 日"的指数走势，
    故是**前视**查询（`date > ref_date LIMIT ?`）——这不是 look-ahead bias，
    而是标签的定义本身。放在同一模块是为了让"取哪张表、什么代码"只有一处定义。

    Args:
        conn: 已打开的 sqlite 连接。
        ref_date: 基准日（不含）。
        limit: 最多取多少条。

    Returns:
        [(close,), ...]，按日期升序；取不到时返回空列表。
    """
    rows = conn.execute(
        "SELECT close FROM index_daily WHERE symbol = ? AND date > ? ORDER BY date LIMIT ?",
        (INDEX_SYMBOL_INDEX_DAILY, ref_date, limit),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT close FROM stock_daily WHERE symbol = ? AND date > ? ORDER BY date LIMIT ?",
            (INDEX_SYMBOL_STOCK_DAILY, ref_date, limit),
        ).fetchall()
    return rows
