"""model_selection_v2 - 逐日回测引擎。"""
from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2 import backtest as bt_cfg

logger = get_logger(__name__)


class V2BacktestEngine:
    """V2 多任务树模型回测引擎。"""

    def __init__(self, engine: DataEngine,
                 model_t1, model_t2, model_t3,
                 cfg: V2Config | None = None):
        self.engine = engine
        self.model_t1 = model_t1
        self.model_t2 = model_t2
        self.model_t3 = model_t3
        self.cfg = cfg or get_config()
        self.cash = bt_cfg.INITIAL_CAPITAL
        self.positions: dict[str, dict] = {}
        self.closed_trades: list[dict] = []
        self.daily_records: list[dict] = []
        self.trade_records: list[dict] = []

    def run(self, start_date: str, end_date: str = "",
            predictions_cache: dict | None = None) -> dict:
        """运行回测。

        逐日循环：T-1日收盘数据构建特征 → 3模型预测 → T日开盘执行。
        """
        from sequoia_x.model_selection_v2.models.tree_cls import predict_cls
        from sequoia_x.model_selection_v2.models.tree_reg import predict_reg
        from sequoia_x.model_selection_v2.models.tree_vol import predict_vol

        # 获取交易日列表
        conn = sqlite3.connect(self.engine.db_path)
        date_cond = f"date >= '{start_date}'"
        if end_date:
            date_cond += f" AND date <= '{end_date}'"
        dates = pd.read_sql(
            f"SELECT DISTINCT date FROM stock_daily WHERE {date_cond} ORDER BY date",
            conn
        )["date"].tolist()
        conn.close()

        if len(dates) < 150:
            logger.error(f"回测: 数据不足 ({len(dates)} 天)")
            return {}

        base_pool = self.engine.get_base_stock_pool()
        logger.info(f"回测: {dates[0]} ~ {dates[-1]}, {len(dates)} 天, {len(base_pool)} 只")

        warmup = self.cfg.window
        cache_path = Path("output/backtest_v2/predictions_cache.json")

        for idx, today in enumerate(dates):
            if idx < warmup:
                continue
            prev_date = dates[idx - 1]

            # 获取预测
            if predictions_cache is not None and prev_date in predictions_cache:
                predictions = predictions_cache[prev_date]
            else:
                predictions = self._predict_batch(base_pool, prev_date, predict_cls,
                                                   predict_reg, predict_vol)

            if not predictions:
                continue

            # 生成信号（T1过滤→T2排序→T3调仓）
            # eval_date=prev_date: current_price 为 T-1 收盘，双轨止损用 T-1 开盘价
            signals = self._generate_signals(predictions, eval_date=prev_date)

            # 执行卖出
            self._execute_sells(signals.get("sell", []), today)

            # 执行买入
            self._execute_buys(signals.get("buy", []), today)

            # 日终估值
            self._mark_to_market(today)

            # 记录日结
            self._record_daily(today)

        return self._compute_metrics()

    def _predict_batch(self, pool: list[str], ref_date: str,
                       predict_cls_fn, predict_reg_fn, predict_vol_fn) -> list[dict]:
        """批量预测。"""
        from sequoia_x.model_selection_v2.features import build_prediction_features
        xs, symbols = [], []
        for symbol in pool:
            try:
                X = build_prediction_features(symbol, self.engine, self.cfg, ref_date=ref_date)
                if X is not None:
                    xs.append(X)
                    symbols.append(symbol)
            except Exception:
                continue
        if not xs:
            return []
        X_batch = np.vstack(xs)
        prob_up = predict_cls_fn(self.model_t1, X_batch)
        excess_ret = predict_reg_fn(self.model_t2, X_batch)
        volatility = predict_vol_fn(self.model_t3, X_batch)
        results = []
        for i, sym in enumerate(symbols):
            if np.isfinite(prob_up[i]):
                results.append({
                    "symbol": sym, "prob_up": float(prob_up[i]),
                    "excess_ret": float(excess_ret[i]),
                    "volatility": float(volatility[i]),
                })
        return results

    def _generate_signals(self, predictions: list[dict],
                          eval_date: str | None = None) -> dict:
        """生成买卖信号。

        Args:
            predictions: 预测结果列表。
            eval_date: 评估基准日（即 current_price 对应日 = T-1）。
                硬止损双轨触发（2026-08-12，与实盘 SimEngine 口径一致）需要
                T-1 开盘价参与止损判定；None=仅收盘确认（旧行为）。
        """
        signals: dict = {"buy": [], "sell": []}

        # 卖出：运行 rules.py 的 evaluate_exit（复用共享模块）
        from sequoia_x.simulation.rules import evaluate_exit
        for symbol, pos in list(self.positions.items()):
            current_price = pos.get("current_price", 0)
            if current_price <= 0:
                continue
            df = self.engine.get_ohlcv(symbol)
            idx_df = self._get_index_df()
            result = evaluate_exit(
                entry_price=pos["cost"] / pos["shares"] if pos["shares"] > 0 else pos["cost"],
                current_price=current_price,
                highest_price=pos.get("highest_price", current_price),
                hold_days=pos.get("hold_days", 0),
                symbol=symbol,
                symbol_df=df.tail(60) if df is not None and not df.empty else None,
                index_df=idx_df.tail(60) if idx_df is not None and not idx_df.empty else None,
                today_opened=False,
                # 双轨止损：T-1 开盘价也参与硬止损判定（跳空破线当天即触发）
                day_open=self._get_open_price(symbol, eval_date) if eval_date else None,
            )
            # V2 特有：叠加 T1/T2 预测因子
            pred_for_sym = next((p for p in predictions if p["symbol"] == symbol), None)
            if pred_for_sym:
                if pred_for_sym["prob_up"] < 0.3:
                    result.score += 20  # T1 强烈看空，加速卖出
                if pred_for_sym["excess_ret"] < -0.03:
                    result.score += 15  # T2 预期超额亏损
            if result.should_exit or result.score >= 60:
                signals["sell"].append(symbol)

        # 买入：T1 过滤 → T2 排序 → Top N
        candidates = [p for p in predictions
                      if p["symbol"] not in self.positions
                      and p["prob_up"] >= bt_cfg.MIN_BUY_PROB]
        candidates.sort(key=lambda x: x["excess_ret"], reverse=True)
        slots = bt_cfg.MAX_POSITIONS - len(self.positions)
        signals["buy"] = [c["symbol"] for c in candidates[:min(slots, bt_cfg.TOP_N_BUY_PER_DAY)]]
        return signals

    def _execute_sells(self, symbols: list[str], date_str: str) -> None:
        """以当日开盘价卖出。"""
        for symbol in symbols:
            if symbol not in self.positions:
                continue
            pos = self.positions[symbol]
            price = self._get_open_price(symbol, date_str)
            if price is None:
                continue
            sell_price = price * (1 - bt_cfg.SLIPPAGE)
            revenue = pos["shares"] * sell_price
            commission = revenue * bt_cfg.COMMISSION_RATE
            tax = revenue * bt_cfg.STAMP_TAX_RATE
            net = revenue - commission - tax
            pnl = net - pos["cost"]
            self.cash += net
            self.positions.pop(symbol)
            self.trade_records.append({
                "symbol": symbol, "type": "sell", "date": date_str,
                "price": round(sell_price, 4), "shares": pos["shares"],
                "pnl": round(pnl, 2),
            })

    def _execute_buys(self, symbols: list[str], date_str: str) -> None:
        """以当日开盘价买入。"""
        for symbol in symbols:
            price = self._get_open_price(symbol, date_str)
            if price is None:
                continue
            buy_price = price * (1 + bt_cfg.SLIPPAGE)
            budget = min(bt_cfg.PER_STOCK_BUDGET, self.cash * 0.9)
            shares = int(budget / buy_price / 100) * 100
            if shares < 100:
                continue
            total = shares * buy_price * (1 + bt_cfg.COMMISSION_RATE)
            if total > self.cash:
                continue
            self.cash -= total
            self.positions[symbol] = {
                "shares": shares, "cost": total,
                "buy_date": date_str, "highest_price": buy_price,
                "hold_days": 0, "current_price": buy_price,
                "current_value": total, "pnl": 0.0, "pnl_pct": 0.0,
            }
            self.trade_records.append({
                "symbol": symbol, "type": "buy", "date": date_str,
                "price": round(buy_price, 4), "shares": shares,
            })

    def _get_open_price(self, symbol: str, date_str: str) -> float | None:
        conn = sqlite3.connect(self.engine.db_path)
        row = conn.execute(
            "SELECT open FROM stock_daily WHERE symbol=? AND date=?", (symbol, date_str)
        ).fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else None

    def _get_index_df(self) -> pd.DataFrame:
        df = self.engine.get_ohlcv("sh.000001")
        if df.empty:
            conn = sqlite3.connect(self.engine.db_path)
            df = pd.read_sql(
                "SELECT * FROM index_daily WHERE symbol='sh.000001' ORDER BY date", conn
            )
            conn.close()
        return df if not df.empty else pd.DataFrame()

    def _mark_to_market(self, date_str: str) -> None:
        conn = sqlite3.connect(self.engine.db_path)
        for symbol, pos in self.positions.items():
            row = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol=? AND date=?", (symbol, date_str)
            ).fetchone()
            if row and row[0]:
                close = float(row[0])
                pos["current_price"] = close
                pos["current_value"] = pos["shares"] * close
                pos["pnl"] = pos["current_value"] - pos["cost"]
                pos["pnl_pct"] = pos["pnl"] / pos["cost"] if pos["cost"] > 0 else 0.0
                pos["hold_days"] = pos.get("hold_days", 0) + 1
                if close > pos["highest_price"]:
                    pos["highest_price"] = close
        conn.close()

    def _record_daily(self, date_str: str) -> None:
        stock_value = sum(p.get("current_value", p["cost"]) for p in self.positions.values())
        total = self.cash + stock_value
        self.daily_records.append({
            "date": date_str, "cash": round(self.cash, 2),
            "stock_value": round(stock_value, 2),
            "total_value": round(total, 2),
            "positions": len(self.positions),
        })

    def _compute_metrics(self) -> dict:
        if not self.daily_records:
            return {}
        n = len(self.daily_records)
        tv = np.array([r["total_value"] for r in self.daily_records])
        total_return = tv[-1] / bt_cfg.INITIAL_CAPITAL - 1
        annual_return = (1 + total_return) ** (252 / n) - 1 if n >= 20 else None
        daily_ret = np.diff(tv) / tv[:-1]
        mean_ret = np.mean(daily_ret) if len(daily_ret) > 0 else 0
        std_ret = np.std(daily_ret) if len(daily_ret) > 0 else 1e-10
        sharpe = (mean_ret - 0.03 / 252) / std_ret * np.sqrt(252) if std_ret > 1e-10 else 0
        cuml = tv / tv[0]
        running_max = np.maximum.accumulate(cuml)
        drawdown = (cuml - running_max) / running_max
        max_dd = float(drawdown.min())
        buys = [t for t in self.trade_records if t["type"] == "buy"]
        sells = [t for t in self.trade_records if t["type"] == "sell"]
        win_trades = [t for t in sells if t["pnl"] > 0]
        return {
            "total_return": total_return, "annual_return": annual_return,
            "sharpe": round(sharpe, 2), "max_drawdown": max_dd,
            "n_days": n, "n_buys": len(buys), "n_sells": len(sells),
            "win_rate": len(win_trades)/len(sells) if sells else 0,
            "total_value": float(tv[-1]), "final_cash": self.cash,
            "daily_records": self.daily_records,
            "trade_records": self.trade_records,
        }
