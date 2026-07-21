"""回测数据加载与时间切分。"""

from __future__ import annotations

import sqlite3

from sequoia_x.data.engine import DataEngine


def get_trade_dates(
    engine: DataEngine, start_date: str, end_date: str = ""
) -> list[str]:
    """获取回测期间的所有交易日。

    Args:
        engine: DataEngine 实例。
        start_date: 起始日期 YYYY-MM-DD。
        end_date: 结束日期（空=最新）。

    Returns:
        交易日日期列表，按时间升序。
    """
    conn = sqlite3.connect(engine.db_path)
    query = "SELECT DISTINCT date FROM stock_daily WHERE date >= ?"
    params = [start_date]
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    query += " ORDER BY date ASC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_monthly_boundaries(dates: list[str]) -> list[int]:
    """找到每月最后一个交易日的索引。

    Returns:
        [idx1, idx2, ...] 每月末的索引。
    """
    boundaries = []
    current_month = ""
    for i, d in enumerate(dates):
        month = d[:7]  # YYYY-MM
        if month != current_month:
            if current_month:
                boundaries.append(i - 1)  # 上月最后一个
            current_month = month
    boundaries.append(len(dates) - 1)  # 最后一个月
    return boundaries
