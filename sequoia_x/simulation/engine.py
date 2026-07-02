"""模拟盘核心引擎 — SimEngine。

T+1 模型（同 019 ETF 模拟盘）：
  T 日 20:00（管线 Step 2: 策略选股+LLM）
    → 产生买入信号（save_llm_recommendations）
    → 保存到数据库，今日不交易

  T+1 日 20:00（管线 Step 2.5: --sim-update）
    → 执行 T 日待执行买入信号（用 T+1 日开盘价，检查涨停/停牌）
    → 更新所有持仓估值（用 T+1 日收盘价）
    → 运行卖出规则（多因子评分 ≥ 60 则触发）
    → 写入账户总览

每日流程（run_daily）：
  1. 交易日判断（跳过非交易日）
  2. 加载账户状态 + 检查数据就绪
  3. 查询待执行买入信号
  4. 执行买入（开盘价，检查涨停/停牌/仓位上限）
  5. 更新所有持仓估值（收盘价）
  6. 逐只持仓运行卖出规则
  7. 触发卖出者：以收盘价卖出 → 生成交易报告 → 推送
  8. 写入账户总览
  9. 推送组合日报
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

from sequoia_x.simulation.config import (
    INITIAL_CAPITAL,
    PER_STOCK_BUDGET,
    MAX_POSITIONS,
    COMMISSION_RATE,
    STAMP_TAX_RATE,
    SLIPPAGE,
    INDEX_SYMBOL,
)
from sequoia_x.simulation.models import (
    init_sim_tables,
    get_pending_signals,
    mark_signal_executed,
    mark_signal_cancelled,
    insert_position,
    get_all_positions,
    update_position_valuation,
    remove_position,
    get_positions_pending_sell,
    mark_position_for_sell,
    clear_pending_sell,
    get_today_recommended_symbols,
    increment_llm_override,
    reset_llm_override,
    insert_closed_trade,
    upsert_account_daily,
    get_account_summary,
    get_recent_account_days,
    get_cash_balance,
)
from sequoia_x.simulation.rules import evaluate_exit
from sequoia_x.simulation.reporter import (
    build_trade_report_text,
    build_daily_summary_text,
    push_trade_report,
)

logger = get_logger(__name__)

# ── 涨跌停限制（A 股） ──
LIMIT_10PCT = 0.10


class SimEngine:
    """模拟盘执行引擎（多仓位版本）。

    Attributes:
        settings: 系统配置。
        engine: 数据引擎（用于读取股票行情）。
        db_path: 数据库路径。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = DataEngine(settings)
        self.db_path = settings.db_path
        init_sim_tables(self.db_path)

    # ════════════════════════════════════════════════════════
    #  主入口
    # ════════════════════════════════════════════════════════

    def run_daily(self) -> dict:
        """执行一个交易日的完整模拟盘流程。

        T+1 模型（同 019 ETF 模拟盘）：
          Step 3: 执行待执行订单（用今日 OPEN）
            ├─ 先卖出（pending_sell → 释放仓位）
            └─ 再买入（pending 信号 → 递补）
          Step 4: 更新估值 + 运行卖出规则（用今日 CLOSE）
            └─ 触发则标记 pending_sell（明日 OPEN 执行）
          Step 5: 写入账户日结

        Returns:
            {"status": "ok"|"skipped"|"error", "actions": [...], ...}
        """
        today_str = date.today().isoformat()
        results: dict = {
            "date": today_str,
            "status": "ok",
            "actions": [],
            "bought": [],
            "sold": [],
            "marked_sell": 0,
            "positions_updated": 0,
        }

        # ── 1. 交易日检查 ──
        if not self._is_trade_day():
            logger.info(f"sim_run_daily: {today_str} 非交易日，跳过")
            results["status"] = "skipped"
            return results

        logger.info(f"═══ 模拟盘每日更新 [{today_str}] ═══")

        # ── 2. 检查数据就绪 ──
        latest_date = self.engine._get_last_date_range()
        if not latest_date or latest_date < today_str:
            logger.warning(f"sim_run_daily: 数据未就绪（latest={latest_date}），跳过")
            results["status"] = "skipped"
            results["detail"] = f"latest_data={latest_date}"
            return results

        # ── 3. 执行待执行订单（T+1 模型，均用今日 OPEN 价） ──
        #    先卖出（释放仓位）→ 再买入（递补）
        sold, bought = self._execute_pending_orders(today_str)
        results["sold"] = sold
        results["bought"] = bought
        if sold:
            results["actions"].append(f"卖出 {len(sold)} 只")
        if bought:
            results["actions"].append(f"买入 {len(bought)} 只")

        # ── 4. 更新估值 + 运行卖出规则（用今日 CLOSE 评估，标记待卖出） ──
        #    触发条件 → 写入 pending_sell_reason，明日 OPEN 再执行卖出
        marked = self._update_and_evaluate(today_str)
        results["marked_sell"] = marked
        if marked > 0:
            results["actions"].append(f"标记待卖出 {marked} 只")

        # ── 5. 写入账户日结 ──
        self._write_account_daily(today_str)
        results["positions_updated"] = len(get_all_positions(self.db_path))

        # ── 6. 推送组合日报 ──
        self._push_daily_summary(today_str, results)

        logger.info(f"═══ 模拟盘更新完成 [{today_str}] ═══")
        if bought:
            logger.info(f"  买入: {' '.join(b['symbol'] for b in bought)}")
        if sold:
            sold_str = " ".join(f'{c["symbol"]}({c["pnl_pct"]:+.1%})' for c in sold)
            logger.info(f"  卖出: {sold_str}")
        if marked:
            logger.info(f"  标记待卖出: {marked} 只")
        logger.info(f"  当前持仓: {results['positions_updated']} 只")

        return results

    # ════════════════════════════════════════════════════════
    #  待执行订单（T+1 模型：先卖后买，均用开盘价）
    # ════════════════════════════════════════════════════════

    def _execute_pending_orders(self, today_str: str) -> tuple[list[dict], list[dict]]:
        """执行所有待执行订单：先卖出（释放仓位），再买入（递补）。

        均用今日 OPEN 价格执行。卖出和买入的决策依据都是 T-1 日 CLOSE 数据。

        Returns:
            (sold_list, bought_list)
        """
        sold = self._execute_pending_sells(today_str)
        bought = self._execute_pending_buys(today_str)
        return sold, bought

    def _execute_pending_sells(self, today_str: str) -> list[dict]:
        """执行所有待卖出订单。

        读取 pending_sell_reason 标记的持仓，用今日 OPEN 价卖出。
        检查跌停/停牌 → 失败则取消标记（不顺延，今晚重新评估）。

        Returns:
            [{"symbol": "600519", "pnl": 1234.56, "pnl_pct": 0.0414, ...}, ...]
        """
        pending = get_positions_pending_sell(self.db_path)
        if not pending:
            return []

        sold: list[dict] = []

        for pos in pending:
            pos_id = pos["id"]
            sym = pos["symbol"]

            # 获取今日行情
            today_data = self._get_today_data(sym, today_str)
            if today_data is None:
                clear_pending_sell(self.db_path, pos_id)
                logger.warning(f"sim 卖 {sym}: 无今日行情，取消待卖出")
                continue

            open_price = today_data["open"]
            prev_close = today_data.get("prev_close", open_price)
            volume = today_data.get("volume", 0)

            # 跌停检查（跌停时卖不出）
            if self._is_limit_down(open_price, prev_close):
                clear_pending_sell(self.db_path, pos_id)
                logger.info(f"sim 卖 {sym}: 开盘跌停，取消待卖出（今晚重新评估）")
                continue

            # 停牌检查
            if volume == 0:
                clear_pending_sell(self.db_path, pos_id)
                logger.info(f"sim 卖 {sym}: 停牌，取消待卖出（今晚重新评估）")
                continue

            # 执行卖出
            exit_reason = pos.get("pending_sell_reason", "规则触发")
            trade = self._execute_sell(
                pos_id=pos_id,
                symbol=sym,
                shares=pos["shares"],
                buy_price=pos["buy_price"],
                buy_date=pos["buy_date"],
                sell_price=open_price,        # ← 关键：用开盘价卖出
                hold_days=pos["hold_days"],
                total_cost=pos["total_cost"],
                strategy_from=pos.get("strategy_from", ""),
                exit_reason=exit_reason,
            )
            if trade:
                sold.append(trade)
                push_trade_report(self.settings, trade)

        return sold

    def _execute_pending_buys(self, today_str: str) -> list[dict]:
        """执行待买入信号（T-1 日 LLM 推荐）。

        用今日 OPEN 价买入。检查涨停/停牌/仓位上限/资金。
        超出仓位的信号取消（不留存）。

        Returns:
            [{"symbol": "600519", "shares": 300, "price": 150.00, ...}, ...]
        """
        signals = get_pending_signals(self.db_path)
        if not signals:
            return []

        # 检查当前持仓数量（卖出已执行完，此时仓位已释放）
        current_positions = get_all_positions(self.db_path)
        if len(current_positions) >= MAX_POSITIONS:
            logger.info(f"sim 买: 持仓已达上限({MAX_POSITIONS}只)，取消所有买入信号")
            for s in signals:
                mark_signal_cancelled(self.db_path, s["id"], "持仓已达上限")
            return []

        slots_available = MAX_POSITIONS - len(current_positions)
        cash_balance = self._get_cash()

        bought: list[dict] = []

        for signal in signals[:slots_available]:
            sym = signal["symbol"]

            today_data = self._get_today_data(sym, today_str)
            if today_data is None:
                mark_signal_cancelled(self.db_path, signal["id"],
                                      f"{sym} 无今日行情数据")
                logger.warning(f"sim 买 {sym}: 无今日行情，信号取消")
                continue

            open_price = today_data["open"]
            prev_close = today_data.get("prev_close", open_price)
            volume = today_data.get("volume", 0)

            # 涨停检查
            if self._is_limit_up(open_price, prev_close):
                mark_signal_cancelled(self.db_path, signal["id"],
                                      f"{sym} 开盘涨停")
                logger.info(f"sim 买 {sym}: 涨停，信号取消")
                continue

            # 跌停检查
            if self._is_limit_down(open_price, prev_close):
                mark_signal_cancelled(self.db_path, signal["id"],
                                      f"{sym} 开盘跌停")
                logger.info(f"sim 买 {sym}: 跌停，信号取消")
                continue

            # 停牌检查
            if volume == 0:
                mark_signal_cancelled(self.db_path, signal["id"],
                                      f"{sym} 停牌")
                logger.info(f"sim 买 {sym}: 停牌，信号取消")
                continue

            # 执行买入
            budget = min(PER_STOCK_BUDGET, cash_balance)
            buy_price = open_price * (1 + SLIPPAGE)
            max_shares = int(budget // buy_price // 100) * 100

            if max_shares <= 0:
                mark_signal_cancelled(self.db_path, signal["id"],
                                      f"资金不足（预算{budget:.0f}，不够1手）")
                continue

            cost = max_shares * buy_price
            commission = max(cost * COMMISSION_RATE, 0.0)
            total_cost = cost + commission

            if total_cost > cash_balance:
                max_shares = int(cash_balance // buy_price // 100) * 100
                if max_shares <= 0:
                    mark_signal_cancelled(self.db_path, signal["id"], "现金余额不足")
                    continue
                cost = max_shares * buy_price
                commission = max(cost * COMMISSION_RATE, 0.0)
                total_cost = cost + commission

            insert_position(
                self.db_path,
                symbol=sym,
                strategy_from=signal.get("strategy_from", ""),
                buy_date=today_str,
                buy_price=round(buy_price, 4),
                shares=max_shares,
                total_cost=round(total_cost, 2),
                signal_id=signal["id"],
            )

            mark_signal_executed(self.db_path, signal["id"],
                                 round(buy_price, 4), max_shares)
            cash_balance -= total_cost

            bought.append({
                "symbol": sym,
                "shares": max_shares,
                "price": round(buy_price, 4),
                "cost": round(total_cost, 2),
            })
            logger.info(f"sim 买: {sym} {max_shares}股 @ {buy_price:.4f}")

        # 取消多余信号（仓位不够，不留存）
        remaining = signals[slots_available:]
        for s in remaining:
            mark_signal_cancelled(self.db_path, s["id"], "仓位不足，取消待下次推荐")
        if remaining:
            logger.info(f"sim 买: 取消 {len(remaining)} 条未执行信号")

        return bought

    # ════════════════════════════════════════════════════════
    #  估值更新 + 卖出判定（T+1 模型：只标记，不平仓）
    # ════════════════════════════════════════════════════════

    def _update_and_evaluate(self, today_str: str) -> int:
        """更新所有持仓估值，运行卖出规则，触发则标记待卖出（明日 OPEN 执行）。

        注意：这里只写入 pending_sell_reason，不实际平仓。
        实际卖出在次日 _execute_pending_sells() 中以开盘价执行。

        Returns:
            标记为待卖出的持仓数量。
        """
        positions = get_all_positions(self.db_path)
        if not positions:
            return 0

        index_df = self._get_index_df(today_str)
        # 获取今日 LLM 推荐的股票（用于卖出覆盖：推荐股不卖出）
        today_llm_picks = get_today_recommended_symbols(self.db_path, today_str)
        marked_count = 0

        for pos in positions:
            # 已标记待卖出的跳过（等待明日执行）
            if pos.get("pending_sell_reason"):
                continue

            sym = pos["symbol"]
            pos_id = pos["id"]

            df = self.engine.get_ohlcv(sym)
            if df.empty:
                continue

            today_mask = df["date"] == today_str
            if not today_mask.any():
                continue

            today_row = df[today_mask].iloc[-1]
            close_price = today_row["close"]

            # 更新估值
            update_position_valuation(
                self.db_path, pos_id, close_price,
                pos["shares"], pos["total_cost"],
            )

            # 重新读取持仓
            updated_pos = self._get_position(pos_id)
            if updated_pos is None:
                continue

            # 运行卖出规则
            result = evaluate_exit(
                entry_price=updated_pos["buy_price"],
                current_price=close_price,
                highest_price=updated_pos["highest_price"] or updated_pos["buy_price"],
                hold_days=updated_pos["hold_days"],
                symbol_df=df.tail(30),
                index_df=index_df.tail(30) if index_df is not None else None,
                today_opened=bool(updated_pos.get("today_opened", 0)),
            )

            if result.should_exit:
                # LLM 同日推荐覆盖：若该股也在今日 LLM 推荐中，则继续持有
                # 但连续覆盖 ≥ 3 次后强制卖出，不再覆盖
                if sym in today_llm_picks:
                    override_count = increment_llm_override(self.db_path, pos_id)
                    MAX_OVERRIDE = 3
                    if override_count >= MAX_OVERRIDE:
                        mark_position_for_sell(self.db_path, pos_id,
                                               f"LLM覆盖{MAX_OVERRIDE}次后强制卖出({result.reason})")
                        marked_count += 1
                        logger.info(
                            f"sim 评: {sym} LLM已连续覆盖{override_count}次，"
                            f"强制执行卖出"
                        )
                    else:
                        logger.info(
                            f"sim 评: {sym} 卖出触发但 LLM 同日推荐"
                            f"（第{override_count}次覆盖），继续持有"
                        )
                    continue

                # 未覆盖：标记待卖出（明日以开盘价执行）
                mark_position_for_sell(self.db_path, pos_id, result.reason)
                marked_count += 1
                logger.info(
                    f"sim 评: {sym} 触发卖出（{result.reason}），"
                    f"明日 OPEN 执行"
                )

        return marked_count

    # ════════════════════════════════════════════════════════
    #  卖出执行
    # ════════════════════════════════════════════════════════

    def _execute_sell(self, pos_id: int, symbol: str, shares: int,
                      buy_price: float, buy_date: str, sell_price: float,
                      hold_days: int, total_cost: float,
                      strategy_from: str, exit_reason: str) -> Optional[dict]:
        """以收盘价卖出持仓。

        计算含佣金+印花税+滑点的净收入，写入 closed_trades 表。

        Returns:
            交易记录 dict（用于生成报告），或 None（失败时）。
        """
        sell_px = sell_price * (1 - SLIPPAGE)
        revenue = shares * sell_px
        commission = max(revenue * COMMISSION_RATE, 0.0)
        stamp_tax = max(revenue * STAMP_TAX_RATE, 0.0)
        net_revenue = revenue - commission - stamp_tax
        pnl = net_revenue - total_cost
        pnl_pct = pnl / total_cost if total_cost > 0 else 0.0

        # 计算持有期最大回撤（从历史数据）
        max_dd = self._calc_max_drawdown(symbol, buy_date, date.today().isoformat())

        # 计算持有期夏普率
        sharpe = self._calc_hold_sharpe(symbol, buy_date, date.today().isoformat())

        trade = {
            "symbol": symbol,
            "name": self._get_stock_name(symbol),
            "strategy_from": strategy_from,
            "buy_date": buy_date,
            "sell_date": date.today().isoformat(),
            "hold_days": hold_days,
            "buy_price": buy_price,
            "sell_price": round(sell_px, 4),
            "shares": shares,
            "total_cost": total_cost,
            "total_revenue": round(net_revenue, 2),
            "commission": round(commission, 2),
            "stamp_tax": round(stamp_tax, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 6),
            "max_drawdown": round(max_dd, 6) if max_dd is not None else None,
            "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
            "exit_reason": exit_reason,
        }

        # 从持仓表删除
        removed = remove_position(self.db_path, pos_id)
        if removed is None:
            logger.error(f"sim: 卖出 {symbol} 时持仓记录不存在")
            return None

        # 写入已完成交易
        insert_closed_trade(self.db_path, trade)

        logger.info(
            f"sim: 卖出 {symbol} {shares}股 @ {sell_px:.4f} "
            f"PNL={pnl:+.2f}({pnl_pct:+.2%}) "
            f"原因: {exit_reason}"
        )

        return trade

    # ════════════════════════════════════════════════════════
    #  账户总览
    # ════════════════════════════════════════════════════════

    def _write_account_daily(self, today_str: str) -> None:
        """写入当日账户总览。"""
        positions = get_all_positions(self.db_path)
        cash = self._get_cash()
        stock_value = sum(p["current_value"] or 0 for p in positions)
        total_value = cash + stock_value
        position_count = len(positions)

        # 计算当日盈亏
        prev_day = get_account_summary(self.db_path,
                                       self._get_prev_trade_day(today_str))
        if prev_day:
            daily_pnl = total_value - prev_day["total_value"]
            daily_pnl_pct = daily_pnl / prev_day["total_value"] if prev_day["total_value"] > 0 else 0.0
        else:
            daily_pnl = total_value - INITIAL_CAPITAL
            daily_pnl_pct = daily_pnl / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 0.0

        total_pnl = total_value - INITIAL_CAPITAL
        total_return = total_pnl / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 0.0

        record = {
            "date": today_str,
            "cash": round(cash, 2),
            "stock_value": round(stock_value, 2),
            "total_value": round(total_value, 2),
            "daily_pnl": round(daily_pnl, 2),
            "daily_pnl_pct": round(daily_pnl_pct, 6),
            "total_pnl": round(total_pnl, 2),
            "total_return": round(total_return, 6),
            "position_count": position_count,
        }
        upsert_account_daily(self.db_path, record)
        logger.info(
            f"sim: 账户日结 {today_str} 总资产{total_value:.2f} "
            f"({total_return:+.2%}) 持仓{position_count}只"
        )

    # ════════════════════════════════════════════════════════
    #  日报推送
    # ════════════════════════════════════════════════════════

    def _push_daily_summary(self, today_str: str, results: dict) -> None:
        """推送组合日报到微信。"""
        try:
            positions = get_all_positions(self.db_path)
            account = get_account_summary(self.db_path, today_str)
            text = build_daily_summary_text(
                today_str=today_str,
                account=account,
                positions=positions,
                bought=results.get("bought", []),
                sold=results.get("sold", []),
            )
            if text:
                from wxpusher import WxPusher
                WxPusher.send_message(
                    content=text,
                    token=self.settings.wxpusher_token,
                    topic_ids=self.settings.wxpusher_topic_ids,
                    content_type=1,
                )
                logger.info("sim: 组合日报推送成功")
        except Exception as e:
            logger.warning(f"sim: 组合日报推送失败: {e}")

    # ════════════════════════════════════════════════════════
    #  辅助方法
    # ════════════════════════════════════════════════════════

    def _is_trade_day(self) -> bool:
        """判断当日是否为交易日（复用 DataSync 的判断逻辑）。"""
        from sequoia_x.data.sync import DataSync
        return DataSync(self.settings).is_trade_day()

    def _get_cash(self) -> float:
        """获取当前可用现金余额。"""
        return get_cash_balance(self.db_path)

    def _get_today_data(self, symbol: str, today_str: str) -> Optional[dict]:
        """获取某只股票今日的开盘价、前收盘价、成交量。

        从 stock_daily 表查询：
          - 今日行（date = today_str）：获取 open, volume
          - 前一日行（date < today_str 的最新行）：获取 close（作为前收盘价）
        """
        with sqlite3.connect(self.db_path) as conn:
            # 今日数据
            row = conn.execute(
                "SELECT open, close, volume FROM stock_daily WHERE symbol=? AND date=?",
                (symbol, today_str),
            ).fetchone()
            if row is None:
                return None

            # 前一日收盘
            prev = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol=? AND date<? "
                "ORDER BY date DESC LIMIT 1",
                (symbol, today_str),
            ).fetchone()

        return {
            "open": row[0] or 0.0,
            "close": row[1] or 0.0,
            "volume": row[2] or 0,
            "prev_close": prev[0] if prev else row[0],
        }

    def _get_position(self, pos_id: int) -> Optional[dict]:
        """按 ID 获取持仓记录。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM sim_positions WHERE id=?", (pos_id,)).fetchone()
        return dict(row) if row else None

    def _get_index_df(self, today_str: str) -> Optional[pd.DataFrame]:
        """获取上证指数日线数据（用于相对强弱比较）。"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql(
                    "SELECT date, close FROM index_daily WHERE symbol=? "
                    "ORDER BY date",
                    conn,
                    params=(INDEX_SYMBOL,),
                )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as e:
            logger.debug(f"sim: 指数数据获取失败: {e}")
        return None

    def _get_prev_trade_day(self, today_str: str) -> str:
        """获取前一个交易日。"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT date FROM sim_account_daily WHERE date<? "
                "ORDER BY date DESC LIMIT 1",
                (today_str,),
            ).fetchone()
        return row[0] if row else today_str

    def _get_stock_name(self, symbol: str) -> str:
        """通过 baostock 获取股票名称。"""
        try:
            import baostock as bs
            bs.login()
            prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
            rs = bs.query_stock_basic(code=f"{prefix}.{symbol}")
            while rs.next():
                return rs.get_row_data()[1]
            bs.logout()
        except Exception:
            pass
        return ""

    def _calc_max_drawdown(self, symbol: str, start_date: str, end_date: str) -> Optional[float]:
        """计算某只股票在指定区间内的最大回撤。"""
        try:
            df = self.engine.get_ohlcv(symbol)
            if df.empty:
                return None
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            segment = df[mask]
            if len(segment) < 2:
                return None
            closes = segment["close"].values
            peak = closes[0]
            max_dd = 0.0
            for c in closes:
                if c > peak:
                    peak = c
                dd = (peak - c) / peak
                if dd > max_dd:
                    max_dd = dd
            return -max_dd
        except Exception:
            return None

    def _calc_hold_sharpe(self, symbol: str, start_date: str, end_date: str) -> Optional[float]:
        """计算持有期间的年化夏普率。"""
        try:
            df = self.engine.get_ohlcv(symbol)
            if df.empty:
                return None
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            segment = df[mask]
            if len(segment) < 5:
                return None
            closes = segment["close"].values
            daily_rets = pd.Series(closes).pct_change().dropna().values
            if len(daily_rets) < 2:
                return None
            mean_r = float(np.mean(daily_rets))
            std_r = float(np.std(daily_rets, ddof=1))
            if std_r < 1e-10:
                return 0.0
            rf_daily = 0.03 / 252
            return (mean_r - rf_daily) / std_r * math.sqrt(252)
        except Exception:
            return None

    @staticmethod
    def _is_limit_up(open_price: float, prev_close: float) -> bool:
        """判断开盘是否涨停。"""
        if prev_close <= 0:
            return False
        return open_price >= prev_close * (1 + LIMIT_10PCT)

    @staticmethod
    def _is_limit_down(open_price: float, prev_close: float) -> bool:
        """判断开盘是否跌停。"""
        if prev_close <= 0:
            return False
        return open_price <= prev_close * (1 - LIMIT_10PCT)
