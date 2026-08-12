"""V2 模拟盘日常操作（pipeline 步骤：LLM 模拟盘操作之后执行）

流程（与 LLM 模拟盘同一套 T+1 执行引擎，独立数据库 sim_v2.db）：
  1. V2 模拟盘日操作：SimEngine(sim_v2.db).run_daily()
     - 先执行待卖出（T 日开盘价）→ 释放仓位
     - 再执行待买入（T 日开盘价）→ 递补
     - 收盘价更新估值 + 13 条卖出规则评估 → 标记 pending_sell
  2. V2 组合日报推送（复用 reporter，wxpusher）
  3. 月末检测：今天是否月末最后交易日 → 日志提示重训 cron（每月 1 日 03:00）

用法：
  python scripts/v2_simulation_daily.py            # 日常（pipeline 调用）
  python scripts/v2_simulation_daily.py --no-push  # 不推送（调试）

依赖：
  - 数据同步/LLM 模拟盘操作已完成（pipeline 顺序保证）
  - sim_v2.db 独立数据库（与 LLM 完全隔离，100 万初始资金）
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from sequoia_x.core.config import get_settings
from sequoia_x.core.logger import get_logger
from sequoia_x.simulation.engine import SimEngine
from sequoia_x.simulation.models import (
    get_account_summary,
    get_all_positions,
    get_realized_unrealized_pnl,
)
from sequoia_x.simulation.reporter import build_daily_summary_text, push_daily_summary

logger = get_logger(__name__)

# V2 独立模拟盘数据库（与 LLM 模拟盘 data/sequoia_v2.db 完全隔离）
SIM_V2_DB = str(PROJECT_DIR / "data" / "sim_v2.db")


def is_last_trading_day_of_month(settings) -> bool:
    """判断今天是否为月末最后交易日（查主库 stock_daily，用于重训定时提示）。

    注意：数据只同步到今天，MAX(date) 恒等于 today，不能单独作为月末依据；
    必须叠加"今天已是当月下旬（≥25 日）"条件，否则月初（如 8-3）会误判。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if int(today[8:10]) < 25:
        return False
    conn = sqlite3.connect(settings.db_path)  # stock_daily 在主库
    try:
        row = conn.execute(
            "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?",
            (today[:7] + "%",),
        ).fetchone()
    finally:
        conn.close()
    return row is not None and row[0] == today


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 模拟盘日常操作")
    parser.add_argument("--no-push", action="store_true", help="不推送日报（调试）")
    args = parser.parse_args()

    settings = get_settings()
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info(f"V2 模拟盘日常操作 | {today} | 独立数据库: {SIM_V2_DB}")
    logger.info("=" * 60)

    # ── 1. V2 模拟盘日操作（T+1 模型，与 LLM 模拟盘同引擎）──
    # V2 规则：持仓上限 10 只 × 每只 10 万（与回测 M4+TOP_N=10 一致，100万满仓）
    sim = SimEngine(
        settings,
        db_path=SIM_V2_DB,
        max_positions=10,
        per_stock_budget=100_000,
    )
    result = sim.run_daily(push_report=False)  # 日报统一由本脚本推送
    logger.info(f"V2 模拟盘更新完成: {result}")

    # ── 2. V2 组合日报推送 ──
    if not args.no_push:
        try:
            account = get_account_summary(SIM_V2_DB, today)
            positions = get_all_positions(SIM_V2_DB)
            # 已实现/未实现盈亏拆分（2026-08-12 新增，日报展示）
            realized, unrealized = get_realized_unrealized_pnl(SIM_V2_DB)
            text = build_daily_summary_text(
                today,
                account,
                positions,
                result.get("bought", []),
                result.get("sold", []),
                cancelled=result.get("cancelled"),
                pending_sells=result.get("marked_sell"),
                max_positions=10,
                realized_pnl=realized,
                unrealized_pnl=unrealized,
            )
            header = f"【V2 模型模拟盘日报 {today}】\n"
            push_daily_summary(settings, header + text)
            logger.info("V2 组合日报已推送")
        except Exception as e:
            logger.warning(f"V2 日报推送失败: {e}")

    # ── 3. 月末检测：提示重训（cron 每月 1 日 03:00 自动触发）──
    if is_last_trading_day_of_month(settings):
        logger.info(
            "📌 今天是月末最后交易日——V2 月度重训将在下月 1 日 03:00 自动启动"
            "（cron: 0 3 1 * *（2026-08-10 由 00:00 调整） → v2_monthly_retrain.py）"
        )
    else:
        logger.info("非月末，无重训安排")


if __name__ == "__main__":
    main()
