"""LSTM 策略日报推送模块。

与 sequoia_x/simulation/reporter.py 类似，但专用于 LSTM-Transformer 策略。
提供：
  - build_lstm_daily_text()  — LSTM 策略日报文本
  - push_lstm_daily_report() — 推送日报到微信
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection.config import get_config

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  辅助
# ════════════════════════════════════════════════════════════


# 股票名称缓存（baostock 查询较慢，同一次运行中缓存结果）
_name_cache: dict[str, str] = {}


def _get_stock_names_batch(symbols: list[str]) -> dict[str, str]:
    """批量获取股票名称（一次 baostock 连接查询全部）。

    避免逐只 login/logout 导致的速率限制和性能问题。
    结果写入模块级缓存 _name_cache，后续调用直接返回。
    """
    uncached = [s for s in symbols if s not in _name_cache]
    if not uncached:
        return {s: _name_cache.get(s, "") for s in symbols}

    try:
        import baostock as bs

        bs.login()
        try:
            rs = bs.query_stock_basic()
            if rs.error_code == "0":
                while rs.next():
                    row = rs.get_row_data()
                    code = row[0].split(".")[1] if "." in row[0] else row[0]
                    name = row[1]
                    if code in uncached:
                        _name_cache[code] = name
        finally:
            bs.logout()
    except Exception:
        pass

    # 未命中的返回空字符串
    for s in uncached:
        if s not in _name_cache:
            _name_cache[s] = ""

    return {s: _name_cache.get(s, "") for s in symbols}


def _get_stock_name(symbol: str) -> str:
    """获取单只股票名称（优先查缓存，缓存未命中时批量查询）。"""
    if symbol in _name_cache:
        return _name_cache[symbol]
    # 单个未命中时走批量查询（最小开销）
    _get_stock_names_batch([symbol])
    return _name_cache.get(symbol, "")


def _query_dict(conn: sqlite3.Connection, sql: str,
                params: tuple = ()) -> Optional[dict]:
    """查询单行返回 dict。"""
    conn.row_factory = sqlite3.Row
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def _query_all(conn: sqlite3.Connection, sql: str,
               params: tuple = ()) -> list[dict]:
    """查询多行返回 dict 列表。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ════════════════════════════════════════════════════════════
#  日报构建
# ════════════════════════════════════════════════════════════


def build_lstm_daily_text(
    today_str: str,
    db_path: str,
) -> str:
    """生成 LSTM-Transformer 策略每日模拟盘日报。

    数据来源：LSTM 模拟盘 DB（sim_lstm.db）中的 sim_account_daily、
    sim_positions、sim_buy_signals、sim_closed_trades 表。

    Args:
        today_str: 日期 "YYYY-MM-DD"。
        db_path: LSTM 模拟盘数据库路径。

    Returns:
        格式化的日报文本，当日无账户记录时返回空字符串。
    """
    display_date = date.fromisoformat(today_str).strftime("%m-%d") if today_str else "??"

    if not Path(db_path).exists():
        logger.info("LSTM 模拟盘 DB 还未创建，跳过日报")
        return ""

    with sqlite3.connect(db_path) as conn:
        account = _query_dict(
            conn,
            "SELECT * FROM sim_account_daily WHERE date=?",
            (today_str,),
        )
        positions = _query_all(conn, "SELECT * FROM sim_positions ORDER BY buy_date ASC")
        # 今日买入信号（pending 状态）
        today_signals = _query_all(
            conn,
            "SELECT * FROM sim_buy_signals WHERE buy_date=? AND status='pending'",
            (today_str,),
        )

    if account is None and not positions and not today_signals:
        return ""

    # ── 批量预加载所有需要的股票名称（一次 baostock 查询）──
    all_symbols: list[str] = []
    for sig in (today_signals or []):
        all_symbols.append(sig.get("symbol", ""))
    for pos in (positions or []):
        all_symbols.append(pos.get("symbol", ""))
    _get_stock_names_batch(all_symbols)

    lines = [
        f"LSTM-Transformer 模拟盘日报  {display_date}",
        "",
    ]

    # ── 账户概况 ──
    if account:
        total_value = account.get("total_value", 0)
        total_return = account.get("total_return", 0)
        cash = account.get("cash", 0)
        stock_value = account.get("stock_value", 0)
        pos_count = account.get("position_count", 0)

        lines.append("  ▶ 账户概况")
        lines.append(f"    总资产: {total_value:>10,.2f}")
        lines.append(f"    累计收益: {total_return:>+8.2%}")
        lines.append(f"    现金: {cash:>10,.2f}  |  持仓市值: {stock_value:>10,.2f}")
        lines.append(f"    持仓数量: {pos_count} 只")
        lines.append("")
    else:
        lines.append("  ▶ 账户概况")
        lines.append("    数据收集中（首日运行）")
        lines.append("")

    # ── 今日预测(信号) ──
    if today_signals:
        lines.append("  ▶ 今日 LSTM 选股")
        for sig in today_signals:
            sym = sig.get("symbol", "")
            name = _get_stock_name(sym)
            label = f"{name}({sym})" if name else sym
            from_str = sig.get("strategy_from", "LSTM选股")
            lines.append(f"    📡 {label} [{from_str}]")
        lines.append("")

    # ── 当前持仓 ──
    if positions:
        lines.append("  ▶ 当前持仓")
        sorted_pos = sorted(positions, key=lambda p: p.get("pnl_pct", 0), reverse=True)
        for i, pos in enumerate(sorted_pos[:10], 1):
            sym = pos.get("symbol", "")
            name = _get_stock_name(sym)
            days = pos.get("hold_days", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            shares = pos.get("shares", 0)
            cur_val = pos.get("current_value", 0)

            arrow = "📈" if pnl_pct >= 0 else "📉"
            lines.append(
                f"    {i:>2}. {arrow} {name}({sym}) "
                f"{shares}股 {cur_val:,.0f}元 "
                f"({pnl_pct:+.2%}) {days}日"
            )

        if len(positions) > 10:
            lines.append(f"    ... 还有 {len(positions) - 10} 只")

        lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  推送
# ════════════════════════════════════════════════════════════


def push_lstm_daily_report(today_str: str, db_path: str) -> None:
    """推送 LSTM 策略日报到微信。

    Args:
        today_str: 日期 "YYYY-MM-DD"。
        db_path: LSTM 模拟盘数据库路径。
    """
    from wxpusher import WxPusher

    from sequoia_x.core.config import get_settings

    try:
        settings = get_settings()
    except Exception as e:
        logger.warning(f"LSTM 日报推送: 获取配置失败 {e}")
        return

    text = build_lstm_daily_text(today_str, db_path)
    if not text:
        return

    try:
        result = WxPusher.send_message(
            content=text,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if result.get("code") == 1000:
            logger.info("LSTM 策略日报推送成功")
        else:
            logger.warning(f"LSTM 策略日报推送失败: {result}")
    except Exception as e:
        logger.warning(f"LSTM 策略日报推送异常: {e}")
