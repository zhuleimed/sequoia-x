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
        "═" * 40,
        f"   Sequoia-X 交易报告",
        f"   {name_display}",
        "═" * 40,
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
        "",
        "═" * 40,
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
) -> str:
    """生成每日模拟盘组合日报。

    Args:
        today_str: 日期字符串 "YYYY-MM-DD"。
        account: 账户总览记录（含 total_value, cash 等）。
        positions: 当前所有持仓列表。
        bought: 今日买入列表。
        sold: 今日卖出列表。

    Returns:
        格式化的日报文本，或空字符串（当日无交易时）。
    """
    display_date = date.fromisoformat(today_str).strftime("%m-%d") if today_str else "??"

    lines = [
        "═" * 40,
        f"   Sequoia-X 模拟盘日报",
        f"   {display_date}",
        "═" * 40,
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
    has_action = bool(bought or sold)

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

    if not has_action:
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
    lines.append("═" * 40)

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
#  辅助
# ════════════════════════════════════════════════════════════


def _get_name(symbol: str) -> str:
    """获取股票名称（缓存 + baostock 查询）。"""
    try:
        import baostock as bs
        bs.login()
        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        rs = bs.query_stock_basic(code=f"{prefix}.{symbol}")
        while rs.next():
            name = rs.get_row_data()[1]
            bs.logout()
            return name
        bs.logout()
    except Exception:
        pass
    return ""
