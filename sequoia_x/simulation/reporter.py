"""报告生成与推送模块。

包含：
  - build_trade_report_text()  — 单股平仓报告文本
  - build_daily_summary_text() — 组合日报文本
  - push_trade_report()        — 推送单股报告到微信
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  单股平仓报告
# ════════════════════════════════════════════════════════════


def build_trade_report_text(trade: dict) -> str:
    """生成单笔交易的平仓分析报告。

    Args:
        trade: 交易记录 dict（来自 engine._execute_sell）。

    Returns:
        格式化的报告文本。
    """
    symbol = trade.get("symbol", "?")
    name = trade.get("name", "") or ""
    name_display = f"{name} ({symbol})" if name else symbol

    strategy = trade.get("strategy_from", "LLM推荐") or "LLM推荐"
    buy_date = trade.get("buy_date", "?")
    sell_date = trade.get("sell_date", "?")
    hold_days = trade.get("hold_days", 0)
    buy_price = trade.get("buy_price", 0)
    sell_price = trade.get("sell_price", 0)
    pnl = trade.get("pnl", 0)
    pnl_pct = trade.get("pnl_pct", 0)
    sharpe = trade.get("sharpe_ratio")
    max_dd = trade.get("max_drawdown")
    exit_reason = trade.get("exit_reason", "?")
    commission = trade.get("commission", 0)
    stamp_tax = trade.get("stamp_tax", 0)

    # 收益判断
    if pnl > 0:
        pnl_tag = "✅ 盈利"
    elif pnl < 0:
        pnl_tag = "❌ 亏损"
    else:
        pnl_tag = "➖ 持平"

    lines = [
        f"Sequoia-X 交易报告  {name_display}",
        "",
        f"  【基本信息】",
        f"  策略来源: {strategy}",
        f"  持有周期: {buy_date} → {sell_date}（{hold_days}个交易日）",
        "",
        f"  【交易明细】",
        f"  买入价: {buy_price:.2f}",
        f"  卖出价: {sell_price:.2f}",
        f"  持仓股数: {trade.get('shares', 0)}",
        f"  总成本: {trade.get('total_cost', 0):.2f}",
        f"  净收入: {trade.get('total_revenue', 0):.2f}",
        f"  佣金: {commission:.2f} | 印花税: {stamp_tax:.2f}",
        "",
        f"  【收益分析】",
        f"  盈亏: {pnl:+.2f}",
        f"  收益率: {pnl_pct:+.2%}  {pnl_tag}",
    ]

    if sharpe is not None:
        lines.append(f"  持有期夏普率: {sharpe:.2f}")
    if max_dd is not None:
        lines.append(f"  最大回撤: {max_dd:.2%}")

    lines += [
        "",
        f"  【退出原因】",
        f"  {exit_reason}",
    ]

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  组合日报
# ════════════════════════════════════════════════════════════


def build_daily_summary_text(
    today_str: str,
    account: Optional[dict],
    positions: list[dict],
    bought: list[dict],
    sold: list[dict],
    cancelled: Optional[list[dict]] = None,
    pending_sells: Optional[list[dict]] = None,
) -> str:
    """生成每日模拟盘组合日报。

    Args:
        today_str: 日期字符串 "YYYY-MM-DD"。
        account: 账户总览记录（含 total_value, cash 等）。
        bought: 今日买入列表。
        sold: 今日卖出列表。
        cancelled: 今日被取消的买入信号及原因。
        pending_sells: 标记为待卖出的持仓（明日开盘执行）。

    Returns:
        格式化的日报文本，或空字符串（当日无交易时）。
    """
    display_date = date.fromisoformat(today_str).strftime("%m-%d") if today_str else "??"

    lines = [
        f"Sequoia-X 模拟盘日报  {display_date}",
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
        lines.append(f"    持仓数量: {pos_count}/{20}")
        lines.append("")
    else:
        lines.append("  ▶ 账户概况")
        lines.append("    数据收集中（首日运行）")
        lines.append("")

    # ── 今日操作 ──
    has_action = bool(bought or sold or cancelled or (pending_sells or []))

    if bought:
        lines.append("  ▶ 今日买入")
        for b in bought:
            name = _get_name(b.get("symbol", ""))
            lines.append(f"    ✅ {name}({b['symbol']}) {b['shares']}股 @ {b['price']:.2f}")
        lines.append("")

    if sold:
        lines.append("  ▶ 今日卖出")
        for s in sold:
            symbol = s.get("symbol", "")
            name = _get_name(symbol)
            pnl_str = f"{s.get('pnl', 0):+.2f} ({s.get('pnl_pct', 0):+.2%})"
            lines.append(f"    🔄 {name}({symbol}) → {pnl_str}")
            lines.append(f"       原因: {s.get('exit_reason', '?')[:40]}")
        lines.append("")

    cancelled = cancelled or []
    if cancelled:
        lines.append("  ▶ 未执行（取消原因）")
        for c in cancelled:
            lines.append(f"    ⏭️ {c.get('symbol', '?')} → {c.get('reason', '?')}")
        lines.append("")

    pending_sells = pending_sells or []
    if pending_sells:
        lines.append("  ▶ 明日待卖出")
        for ps in pending_sells:
            lines.append(f"    ⚠️ {ps.get('symbol', '?')} → {ps.get('reason', '?')[:40]}")
        lines.append("")

    if not has_action and not cancelled and not pending_sells:
        lines.append("  ▶ 今日无操作")
        lines.append("")

    # ── 当前持仓 ──
    if positions:
        lines.append("  ▶ 当前持仓")
        # 按收益率降序排列
        sorted_pos = sorted(positions, key=lambda p: p.get("pnl_pct", 0), reverse=True)
        for i, pos in enumerate(sorted_pos[:10], 1):
            symbol = pos.get("symbol", "")
            name = _get_name(symbol)
            days = pos.get("hold_days", 0)
            pnl_pct = pos.get("pnl_pct", 0)
            shares = pos.get("shares", 0)
            cur_val = pos.get("current_value", 0)

            if pnl_pct >= 0:
                arrow = "📈"
            else:
                arrow = "📉"

            lines.append(
                f"    {i:>2}. {arrow} {name}({symbol}) "
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


def push_trade_report(settings: Settings, trade: dict) -> None:
    """推送单股平仓报告到微信。"""
    try:
        text = build_trade_report_text(trade)
        from wxpusher import WxPusher
        result = WxPusher.send_message(
            content=text,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if result.get("code") == 1000:
            symbol = trade.get("symbol", "?")
            logger.info(f"sim: {symbol} 交易报告推送成功")
        else:
            logger.warning(f"sim: 交易报告推送失败: {result}")
    except Exception as e:
        logger.warning(f"sim: 交易报告推送异常: {e}")


def push_daily_summary(settings: Settings, text: str) -> None:
    """推送组合日报到微信。"""
    try:
        if not text:
            return
        from wxpusher import WxPusher
        result = WxPusher.send_message(
            content=text,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if result.get("code") == 1000:
            logger.info("sim: 组合日报推送成功")
    except Exception as e:
        logger.warning(f"sim: 组合日报推送异常: {e}")


# ════════════════════════════════════════════════════════════
#  月度汇总报告
# ════════════════════════════════════════════════════════════


def build_monthly_report_text(year: int, month: int, db_path: str) -> str:
    """生成月度模拟盘汇总报告。

    数据来源：
      - sim_closed_trades：本月已平仓交易（含盈亏/夏普/回撤）
      - sim_account_daily：本月账户日结（月初/月末资产）

    Args:
        year: 年份，如 2026。
        month: 月份，如 7。
        db_path: 数据库路径。

    Returns:
        格式化的月报文本。
    """
    import sqlite3
    import numpy as np

    start_str = f"{year}-{month:02d}-01"
    if month == 12:
        end_str = f"{year+1}-01-01"
    else:
        end_str = f"{year}-{month+1:02d}-01"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    trades = conn.execute(
        "SELECT * FROM sim_closed_trades "
        "WHERE sell_date >= ? AND sell_date < ? "
        "ORDER BY sell_date ASC",
        (start_str, end_str),
    ).fetchall()

    first_day = conn.execute(
        "SELECT * FROM sim_account_daily WHERE date >= ? ORDER BY date ASC LIMIT 1",
        (start_str,),
    ).fetchone()

    last_day = conn.execute(
        "SELECT * FROM sim_account_daily WHERE date < ? ORDER BY date DESC LIMIT 1",
        (end_str,),
    ).fetchone()

    conn.close()

    month_str = f"{year}年{month}月"
    lines = [f"Sequoia-X 模拟盘月报  {month_str}", ""]

    if first_day and last_day:
        begin_val = first_day["total_value"]
        end_val = last_day["total_value"]
        monthly_return = (end_val / begin_val - 1) if begin_val > 0 else 0.0
        lines.append("  ▶ 月度概况")
        lines.append(f"    月初总资产: {begin_val:>10,.2f}")
        lines.append(f"    月末总资产: {end_val:>10,.2f}")
        lines.append(f"    本月收益率: {monthly_return:>+8.2%}")
        lines.append(f"    月末持仓: {last_day['position_count']} 只")
        lines.append(f"    月末现金: {last_day['cash']:>10,.2f}")
        lines.append("")
    else:
        lines.append("  ▶ 月度概况")
        lines.append("    本月无模拟盘交易数据")
        lines.append("")

    total_trades = len(trades)
    lines.append(f"  ▶ 本月平仓交易（共 {total_trades} 笔）")
    lines.append("")

    if trades:
        # ── 按策略分组统计 ──
        from collections import defaultdict
        strategy_trades: dict[str, list] = defaultdict(list)
        for t in trades:
            sf = t["strategy_from"] or "未知策略"
            strategy_trades[sf].append(t)

        lines.append("  ── 各策略表现 ──")
        for sname in sorted(strategy_trades.keys()):
            st = strategy_trades[sname]
            pnl_arr_s = np.array([x["pnl"] for x in st])
            pnl_pct_s = np.array([x["pnl_pct"] for x in st])
            win_s = pnl_pct_s > 0
            wr_s = win_s.mean()
            lines.append(
                f"    {sname}: {len(st)}笔 "
                f"胜率{wr_s:.0%}({int(win_s.sum())}/{len(st)}) "
                f"合计{pnl_arr_s.sum():+,.0f}"
            )
        lines.append("")

        # ── 逐笔交易清单 ──
        pnl_pcts_list = []
        for t in trades:
            sym = t["symbol"]
            name = _get_name(sym)
            pnl_pct = t["pnl_pct"]
            sharpe = t["sharpe_ratio"]
            hold = t["hold_days"]
            tag = "✅" if pnl_pct >= 0 else "❌"
            sharpe_str = f" 夏普{sharpe:.2f}" if sharpe else ""
            sf = (t["strategy_from"] or "")[:8]
            lines.append(f"    {tag} {name}({sym}) {pnl_pct:+.2%}{sharpe_str} {hold}日 [{sf}]")
            pnl_pcts_list.append(pnl_pct)

        pnl_arr = np.array(pnl_pcts_list)
        win = pnl_arr > 0
        win_rate = win.mean()
        avg_win = pnl_arr[win].mean() if win.any() else 0.0
        avg_loss = pnl_arr[~win].mean() if (~win).any() else 0.0
        total_pnl = sum(t["pnl"] for t in trades)

        lines.append("")
        lines.append("  ▶ 绩效汇总")
        lines.append(f"    胜率: {win_rate:.0%}（{int(win.sum())}/{total_trades}）")
        lines.append(f"    平均盈利: {avg_win:+.2%}")
        lines.append(f"    平均亏损: {avg_loss:+.2%}")
        if abs(avg_loss) > 1e-10:
            lines.append(f"    盈亏比: {abs(avg_win/avg_loss):.2f}")
        lines.append(f"    平仓合计: {total_pnl:+,.2f}")

        # 月内最大回撤
        acct_conn = sqlite3.connect(db_path)
        rows = acct_conn.execute(
            "SELECT total_value FROM sim_account_daily "
            "WHERE date >= ? AND date < ? ORDER BY date",
            (start_str, end_str),
        ).fetchall()
        acct_conn.close()
        if len(rows) >= 5:
            vals = np.array([r[0] for r in rows])
            cum = vals / vals[0]
            run_max = np.maximum.accumulate(cum)
            dd = (cum - run_max) / run_max
            lines.append(f"    最大回撤: {dd.min():.2%}")
            daily_ret = np.diff(vals) / vals[:-1]
            if len(daily_ret) > 1 and daily_ret.std() > 1e-10:
                ms = ((daily_ret.mean() - 0.03/252) / daily_ret.std()) * np.sqrt(252)
                lines.append(f"    夏普率: {ms:.2f}")

    lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  辅助
# ════════════════════════════════════════════════════════════


def _get_name(symbol: str) -> str:
    """获取股票名称（本地 SQLite 优先，腾讯 API 回退）。"""
    from sequoia_x.core.config import get_settings
    from sequoia_x.data.engine import DataEngine
    return DataEngine(get_settings()).get_stock_name(symbol)
