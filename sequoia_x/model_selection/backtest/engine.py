"""逐日回测引擎。

仿 ETF 项目 BacktestEngine，逐日循环：信号→执行→记录。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection.config import LSTMConfig, get_config as get_lstm_config
from sequoia_x.model_selection.backtest import config as bt_cfg
from sequoia_x.model_selection.features import build_prediction_features

logger = get_logger(__name__)


class LSTMBacktestEngine:
    """LSTM 策略逐日回测引擎。

    T+1 模型：
      - 信号使用 close[T-1] 数据构建特征
      - 执行使用 open[T] 价格
      - 收盘后估值检查止损止盈
    """

    def __init__(self, engine: DataEngine, model=None, model_train_fn=None):
        self.engine = engine
        self.cfg = get_lstm_config()
        self.model = model
        self.model_train_fn = model_train_fn  # 每月重训函数
        self.cash = bt_cfg.INITIAL_CAPITAL
        self.positions: dict[str, dict] = {}  # {symbol: {shares, cost, buy_date, highest_price}}
        self.closed_trades: list[dict] = []
        self.daily_records: list[dict] = []
        self.trade_records: list[dict] = []

    def run(self, start_date: str, end_date: str = "") -> dict:
        """运行回测。

        逐日循环：
          1. 每月末重训模型
          2. 用 T-1 数据生成信号
          3. 用 T 开盘价执行交易
          4. 日终估值

        Returns:
            dict with metrics.
        """
        from sequoia_x.model_selection.backtest.data import (
            get_trade_dates, get_monthly_boundaries,
        )

        dates = get_trade_dates(self.engine, start_date, end_date)
        if len(dates) < 100:
            logger.error(f"回测: 数据不足 ({len(dates)} 天)")
            return {}

        boundaries = get_monthly_boundaries(dates)
        logger.info(f"回测: {dates[0]} ~ {dates[-1]}, {len(dates)} 天, "
                     f"{len(boundaries)} 个月")

        # 初始训练（用起始日期之前的数据）
        # 跳过前 window 天用于特征构建（build_prediction_features 只需要 window 天历史数据）
        warmup = self.cfg.window
        prediction_cache: dict = {}

        # 缓存基础股票池：只在回测开始时调用一次 baostock，
        # 避免每交易日重复 login/logout（500+ 天回测可节省数千次连接）
        base_pool: list[str] = []
        try:
            base_pool = self.engine.get_base_stock_pool()
            logger.info(f"回测股票池: {len(base_pool)} 只（缓存复用，不再逐日查询）")
        except Exception as e:
            logger.error(f"获取基础股票池失败: {e}")
            return {}

        for idx, today in enumerate(dates):
            if idx < warmup:
                continue

            prev_date = dates[idx - 1]

            # 每月末重训
            if idx in boundaries and self.model_train_fn:
                logger.info(f"回测重训: {today}")
                self.model = self.model_train_fn(today)
                prediction_cache.clear()

            if self.model is None:
                # 尚无模型，跳过（使用动量策略会引入 bias）
                continue

            # Step 1: 使用缓存的股票池（已在 run() 开头加载）
            # Step 2: 预测（用 prev_date 数据避免 look-ahead）
            predictions = self._predict_batch(base_pool, prev_date)
            if not predictions:
                continue

            # Step 3: 生成信号 (v1.3: 使用完整 rules.py 评估)
            signals = self._generate_signals(predictions, today)

            # Step 4: 执行卖出（用今天开盘价）
            self._execute_sells(signals.get("sell", []), today)

            # Step 5: 执行买入
            self._execute_buys(signals.get("buy", []), today)

            # Step 6: 日终估值
            self._mark_to_market(today)

            # Step 7: 日结记录
            self._record_daily(today)

        return self._compute_metrics()

    # ──────────────────────────────────────────────────────────
    #  预测
    # ──────────────────────────────────────────────────────────

    def _predict_batch(self, pool: list[str], ref_date: str) -> list[tuple[str, float]]:
        """预测一批股票的收益率（批量推理，大幅快于逐个 predict）。

        收集所有有效股票的 (1,120,62) 特征，stack 为 (N,120,62) 后单次
        model.predict 调用，避免逐个 predict 的 TF session 开销。
        """
        xs: list[np.ndarray] = []
        symbols: list[str] = []
        for symbol in pool:  # 全量预测，批量推理足够快
            try:
                X = build_prediction_features(symbol, self.engine, self.cfg)
                if X is not None:
                    xs.append(X)
                    symbols.append(symbol)
            except Exception:
                continue

        if not xs:
            return []

        X_batch = np.vstack(xs)  # (N, window, n_features)
        preds = self.model.predict(X_batch, verbose=0).flatten()

        results: list[tuple[str, float]] = []
        for sym, pred in zip(symbols, preds):
            if np.isfinite(pred):
                results.append((sym, float(pred)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ──────────────────────────────────────────────────────────
    #  信号生成 (v1.3: 使用完整13条卖出规则 + LSTM因子)
    # ──────────────────────────────────────────────────────────

    def _generate_signals(
        self, predictions: list[tuple[str, float]], date_str: str,
    ) -> dict:
        """生成买卖信号。

        卖出: 使用 simulation/rules.py 的 evaluate_exit (13条规则 + LSTM因子 + min_hold)
        买入: 取预测收益率最高的 TOP_N 只
        """
        import json
        from pathlib import Path
        from sequoia_x.simulation.config import LSTM_PREDICT_CACHE

        signals: dict = {"buy": [], "sell": []}

        # ── 保存预测缓存 (供 evaluate_exit 中的 LSTM 因子读取) ──
        pred_dict = {sym: float(pred) for sym, pred in predictions}
        cache_path = Path(LSTM_PREDICT_CACHE)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w") as f:
                json.dump(pred_dict, f)
        except Exception:
            pass

        # ── 卖出: 使用完整 rules.py 评估 ──
        for symbol, pos in list(self.positions.items()):
            try:
                # 获取个股OHLCV数据(用于均线/夏普等规则)
                df = self.engine.get_ohlcv(symbol)
                if df.empty:
                    continue
                # 获取指数OHLCV(用于相对弱势规则)
                idx_df = self._get_index_df()
                # 当前收盘价
                current_price = pos.get("current_price", 0)
                if current_price <= 0:
                    continue

                from sequoia_x.simulation.rules import evaluate_exit
                result = evaluate_exit(
                    entry_price=pos["cost"] / pos["shares"] if pos["shares"] > 0 else pos["cost"],
                    current_price=current_price,
                    highest_price=pos.get("highest_price", current_price),
                    hold_days=pos.get("hold_days", 0),
                    symbol=symbol,
                    symbol_df=df.tail(60) if df is not None else None,
                    index_df=idx_df.tail(60) if idx_df is not None else None,
                    today_opened=False,
                )
                if result.should_exit:
                    signals["sell"].append(symbol)
            except Exception:
                # evaluate_exit 失败时回退到简单止损
                pnl_pct = pos.get("pnl_pct", 0)
                if pnl_pct < -0.08:
                    signals["sell"].append(symbol)

        # ── 买入：取预测最高的 N 只 ──
        bought_today = 0
        for symbol, pred in predictions:
            if symbol in self.positions:
                continue
            if pred < bt_cfg.MIN_PRED_RETURN:
                continue
            if len(self.positions) + bought_today >= bt_cfg.MAX_POSITIONS:
                break
            if bought_today >= bt_cfg.TOP_N_BUY_PER_DAY:
                break
            signals["buy"].append(symbol)
            bought_today += 1

        return signals

    # ──────────────────────────────────────────────────────────
    #  交易执行
    # ──────────────────────────────────────────────────────────

    def _execute_sells(self, symbols: list[str], date_str: str) -> None:
        """执行卖出（用当日开盘价，考虑滑点）。v1.3: evaluate_exit 决定的均为全额卖出。"""
        for symbol in symbols:
            if symbol not in self.positions:
                continue
            pos = self.positions[symbol]
            price = self._get_open_price(symbol, date_str)
            if price is None:
                continue

            sell_shares = pos["shares"]
            sell_price = price * (1 - bt_cfg.SLIPPAGE)
            revenue = sell_shares * sell_price
            commission = revenue * bt_cfg.COMMISSION_RATE
            tax = revenue * bt_cfg.STAMP_TAX_RATE
            net = revenue - commission - tax
            pnl = net - pos["cost"]
            self.cash += net
            self.positions.pop(symbol)

            self.trade_records.append({
                "symbol": symbol, "type": "sell", "date": date_str,
                "price": round(sell_price, 4), "shares": sell_shares,
                "pnl": round(pnl, 2),
            })

    def _execute_buys(self, symbols: list[str], date_str: str) -> None:
        """执行买入（用当日开盘价，考虑滑点和佣金）。"""
        for symbol in symbols:
            price = self._get_open_price(symbol, date_str)
            if price is None:
                continue
            buy_price = price * (1 + bt_cfg.SLIPPAGE)
            # 预算：单只上限 vs 账户余额 90%
            budget = min(bt_cfg.PER_STOCK_BUDGET, self.cash * 0.9)
            shares = int(budget / buy_price / 100) * 100
            if shares < 100:
                continue
            cost = shares * buy_price
            commission = cost * bt_cfg.COMMISSION_RATE
            total = cost + commission
            if total > self.cash:
                continue
            self.cash -= total
            self.positions[symbol] = {
                "shares": shares, "cost": total,
                "buy_date": date_str, "highest_price": buy_price,
                "hold_days": 0,  # v1.3: 追踪持有天数
            }
            self.trade_records.append({
                "symbol": symbol, "type": "buy", "date": date_str,
                "price": round(buy_price, 4), "shares": shares,
                "cost": round(total, 2),
            })

    # ──────────────────────────────────────────────────────────
    #  行情查询
    # ──────────────────────────────────────────────────────────

    def _get_index_df(self) -> "pd.DataFrame":
        """获取上证指数 OHLCV 数据（用于相对弱势规则）。"""
        import pandas as pd
        df = self.engine.get_ohlcv("sh.000001")
        if df.empty:
            # fallback: try local index_daily
            import sqlite3
            conn = sqlite3.connect(self.engine.db_path)
            df = pd.read_sql(
                "SELECT * FROM index_daily WHERE symbol='sh.000001' ORDER BY date",
                conn,
            )
            conn.close()
        return df if not df.empty else pd.DataFrame()

    def _get_open_price(self, symbol: str, date_str: str) -> float | None:
        """获取某日开盘价。"""
        import sqlite3
        conn = sqlite3.connect(self.engine.db_path)
        row = conn.execute(
            "SELECT open FROM stock_daily WHERE symbol=? AND date=?",
            (symbol, date_str)
        ).fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else None

    def _get_close_price(self, symbol: str, date_str: str) -> float | None:
        """获取某日收盘价。"""
        import sqlite3
        conn = sqlite3.connect(self.engine.db_path)
        row = conn.execute(
            "SELECT close FROM stock_daily WHERE symbol=? AND date=?",
            (symbol, date_str)
        ).fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else None

    # ──────────────────────────────────────────────────────────
    #  估值与记录
    # ──────────────────────────────────────────────────────────

    def _mark_to_market(self, date_str: str) -> None:
        """日终按收盘价估值。"""
        import sqlite3
        conn = sqlite3.connect(self.engine.db_path)
        for symbol, pos in self.positions.items():
            row = conn.execute(
                "SELECT close FROM stock_daily WHERE symbol=? AND date=?",
                (symbol, date_str)
            ).fetchone()
            if row and row[0]:
                close = float(row[0])
                pos["current_price"] = close
                pos["current_value"] = pos["shares"] * close
                pos["pnl"] = pos["current_value"] - pos["cost"]
                pos["pnl_pct"] = pos["pnl"] / pos["cost"]
                pos["hold_days"] = pos.get("hold_days", 0) + 1  # v1.3
                if close > pos["highest_price"]:
                    pos["highest_price"] = close
        conn.close()

    def _record_daily(self, date_str: str) -> None:
        """记录日结。"""
        stock_value = sum(
            p.get("current_value", p["cost"]) for p in self.positions.values()
        )
        total = self.cash + stock_value
        self.daily_records.append({
            "date": date_str,
            "cash": round(self.cash, 2),
            "stock_value": round(stock_value, 2),
            "total_value": round(total, 2),
            "positions": len(self.positions),
        })

    # ──────────────────────────────────────────────────────────
    #  绩效指标
    # ──────────────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        """计算绩效指标。

        包括：
          - 累计/年化收益率
          - 夏普比率（无风险利率 3%）
          - 最大回撤
          - 胜率
        """
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
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe": round(sharpe, 2),
            "max_drawdown": max_dd,
            "n_days": n,
            "n_buys": len(buys),
            "n_sells": len(sells),
            "win_rate": len(win_trades) / len(sells) if sells else 0,
            "total_value": float(tv[-1]),
            "final_cash": self.cash,
            "daily_records": self.daily_records,
            "trade_records": self.trade_records,
        }
