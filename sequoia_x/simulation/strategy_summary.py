"""多策略汇总报告：LLM 策略 vs LSTM-Transformer 策略。

每日推送对比摘要到微信，格式：

📊 Sequoia-X 策略汇总 | 07-20
════════════════════════════════════════
❶ LLM选股  累计+X.X%  持仓X只
❷ LSTM-Transformer选股  累计+X.X%  持仓X只

数据来源：
  - data/sequoia_v2.db  — LLM 策略模拟盘
  - data/sim_lstm.db     — LSTM 策略模拟盘
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════

PROJECT_DIR: Path = Path(__file__).resolve().parent.parent.parent
LLM_DB: str = str(PROJECT_DIR / "data" / "sequoia_v2.db")
LSTM_DB: str = str(PROJECT_DIR / "data" / "sim_lstm.db")


# ════════════════════════════════════════════════════════════
#  查询
# ════════════════════════════════════════════════════════════


def _get_account_summary(db_path: str) -> Optional[dict]:
    """获取指定 DB 中最新的账户总览记录。"""
    if not Path(db_path).exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sim_account_daily ORDER BY date DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.debug(f"查询账户汇总失败 {db_path}: {e}")
        return None


def _get_position_count(db_path: str) -> int:
    """获取指定 DB 中的持仓数量。"""
    if not Path(db_path).exists():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM sim_positions").fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def _get_closed_trades_count(db_path: str) -> int:
    """获取指定 DB 中的已完成交易数量。"""
    if not Path(db_path).exists():
        return 0
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM sim_closed_trades").fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════
#  汇总构建
# ════════════════════════════════════════════════════════════


def _format_strategy_line(
    emoji: str,
    name: str,
    account: Optional[dict],
    pos_count: int,
    trade_count: int,
) -> str:
    """格式化单条策略行。"""
    if account is None:
        return f"{emoji} {name}  暂无数据"

    total_return = account.get("total_return", 0.0)
    total_value = account.get("total_value", 0.0)
    summary_parts = [
        f"累计{total_return:+.2%}",
        f"持仓{pos_count}只",
        f"资产{total_value:,.0f}",
    ]
    if trade_count > 0:
        summary_parts.append(f"交易{trade_count}笔")
    detail = " | ".join(summary_parts)
    return f"{emoji} {name}  {detail}"


def build_strategy_summary_text() -> str:
    """构建多策略汇总报告文本。

    Returns:
        格式化的报告文本。
    """
    today_str = date.today().strftime("%m-%d")

    # LLM 策略
    llm_account = _get_account_summary(LLM_DB)
    llm_positions = _get_position_count(LLM_DB)
    llm_trades = _get_closed_trades_count(LLM_DB)

    # LSTM 策略
    lstm_account = _get_account_summary(LSTM_DB)
    lstm_positions = _get_position_count(LSTM_DB)
    lstm_trades = _get_closed_trades_count(LSTM_DB)

    lines = [
        f"Sequoia-X 策略汇总 | {today_str}",
        "=" * 40,
        _format_strategy_line(
            "1. LLM", "LLM选股", llm_account, llm_positions, llm_trades
        ),
        _format_strategy_line(
            "2. LSTM", "LSTM-Transformer选股",
            lstm_account, lstm_positions, lstm_trades,
        ),
        "",
    ]

    # 如果都有数据，加两策略对比
    if llm_account is not None and lstm_account is not None:
        llm_ret = llm_account.get("total_return", 0.0)
        lstm_ret = lstm_account.get("total_return", 0.0)
        diff = llm_ret - lstm_ret
        if diff > 0.01:
            lines.append(f"LLM 领先 LSTM {diff:+.2%}")
        elif diff < -0.01:
            lines.append(f"LSTM 领先 LLM {abs(diff):+.2%}")
        else:
            lines.append("两策略累计收益接近")
        lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
#  推送
# ════════════════════════════════════════════════════════════


def push_strategy_summary() -> None:
    """推送多策略汇总报告到微信。"""
    from wxpusher import WxPusher

    from sequoia_x.core.config import get_settings

    try:
        settings = get_settings()
    except Exception as e:
        logger.warning(f"策略汇总推送: 获取配置失败 {e}")
        return

    text = build_strategy_summary_text()

    try:
        result = WxPusher.send_message(
            content=text,
            token=settings.wxpusher_token,
            topic_ids=settings.wxpusher_topic_ids,
            content_type=1,
        )
        if result.get("code") == 1000:
            logger.info("策略汇总推送成功")
        else:
            logger.warning(f"策略汇总推送失败: {result}")
    except Exception as e:
        logger.warning(f"策略汇总推送异常: {e}")


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════


def main() -> None:
    """CLI 入口：构建并打印策略汇总（可选推送）。"""
    import argparse

    parser = argparse.ArgumentParser(description="多策略汇总报告")
    parser.add_argument("--push", action="store_true", help="推送到微信")
    args = parser.parse_args()

    text = build_strategy_summary_text()
    print(text)

    if args.push:
        push_strategy_summary()


if __name__ == "__main__":
    main()
