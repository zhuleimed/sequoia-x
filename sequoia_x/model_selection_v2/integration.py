"""T2+T4 集成信号生成与多周期回测框架。

核心: Rank 融合 — 两个模型排名的平均值作为最终排名。
选 Top N 只排名最靠前的股票，T1/T3 可选辅助。

使用方法:
    from sequoia_x.model_selection_v2.integration import IntegratedSignal, run_backtest
    signal = IntegratedSignal(t2_model, t4_model, cfg)
    results = run_backtest(signal, start="2025-01-01", end="2026-06-30")
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import V2Config, get_config

logger = get_logger(__name__)


# ════════════════════════════════════════════════════════
#  Rank 融合信号生成
# ════════════════════════════════════════════════════════

def rank_fusion(pred_t2: np.ndarray, pred_t4: np.ndarray) -> np.ndarray:
    """Rank 融合：两个模型预测排名的平均值。

    每个模型的预测值各自排名（值越大排名越靠前），
    取两个排名的平均作为最终排名。
    自动过滤分歧——T2看好但T4不看好 → 排名居中 → 不会被选中。

    Args:
        pred_t2: T2 (LightGBM) 预测值，shape (n_stocks,)
        pred_t4: T4 (LSTM) 预测值，shape (n_stocks,)

    Returns:
        融合排名，shape (n_stocks,)，值越小（排名越靠前）越好。
    """
    # 排名: 值越大排名越靠前 (1=最好)
    rank_t2 = rankdata(-pred_t2, method="average")
    rank_t4 = rankdata(-pred_t4, method="average")
    # 平均排名 → 越小越好
    avg_rank = (rank_t2 + rank_t4) / 2.0
    return avg_rank


def dynamic_weight_fusion(pred_t2: np.ndarray, pred_t4: np.ndarray,
                           ic_t2: float, ic_t4: float) -> np.ndarray:
    """动态 IC 加权融合（备选方案）。

    Args:
        pred_t2, pred_t4: 两个模型的原始预测值。
        ic_t2, ic_t4: 最近 N 个月的平均 Rank IC。

    Returns:
        加权信号，shape (n_stocks,)
    """
    total_ic = abs(ic_t2) + abs(ic_t4)
    if total_ic < 1e-6:
        w_t2 = w_t4 = 0.5
    else:
        w_t2 = abs(ic_t2) / total_ic
        w_t4 = abs(ic_t4) / total_ic
    return w_t2 * np.array(pred_t2) + w_t4 * np.array(pred_t4)


class IntegratedSignal:
    """T2+T4 集成信号生成器。

    参数:
        top_n: 每月选股数量。
        fusion_method: "rank" | "weighted"。
        use_t1_filter: 是否启用 T1 方向过滤（AUC>0.6时）。
        use_t3_sizing: 是否启用 T3 波动率仓位调节。
    """

    def __init__(
        self,
        cfg: V2Config | None = None,
        top_n: int = 10,
        fusion_method: str = "rank",
        use_t1_filter: bool = False,
        use_t3_sizing: bool = False,
    ):
        self.cfg = cfg or get_config()
        self.top_n = top_n
        self.fusion_method = fusion_method
        self.use_t1_filter = use_t1_filter
        self.use_t3_sizing = use_t3_sizing

    def generate_monthly_signals(
        self,
        symbols: list[str],
        pred_t2: np.ndarray,
        pred_t4: np.ndarray,
        pred_t1: np.ndarray | None = None,
        pred_t3: np.ndarray | None = None,
        ic_t2: float = 0.05,
        ic_t4: float = 0.05,
    ) -> list[dict]:
        """生成月度选股信号。

        Args:
            symbols: 股票代码列表。
            pred_t2: T2 预测值 (excess return)。
            pred_t4: T4 预测值 (excess return)。
            pred_t1: T1 预测概率 (optional, for direction filter)。
            pred_t3: T3 预测波动率 (optional, for position sizing)。
            ic_t2, ic_t4: 近期 IC 估计（仅 weighted 方法使用）。

        Returns:
            信号列表，每个元素: {symbol, score, rank, weight}
        """
        n = len(symbols)
        if n < self.top_n:
            logger.warning(f"股票池过小 ({n} < {self.top_n})")
            return []

        # 1. 计算融合分数
        if self.fusion_method == "rank":
            rank_scores = rank_fusion(
                np.array(pred_t2, dtype=float),
                np.array(pred_t4, dtype=float),
            )
        else:
            weighted = dynamic_weight_fusion(
                np.array(pred_t2, dtype=float),
                np.array(pred_t4, dtype=float),
                ic_t2, ic_t4,
            )
            rank_scores = rankdata(weighted, method="average")

        # 2. T1 方向过滤
        if self.use_t1_filter and pred_t1 is not None:
            t1_mask = np.array(pred_t1) > self.cfg.min_buy_prob
        else:
            t1_mask = np.ones(n, dtype=bool)

        # 3. 按融合排名排序，取 Top N
        order = np.argsort(rank_scores)
        selected = []
        for idx in order:
            if not t1_mask[idx]:
                continue
            sym = symbols[idx]
            weight = 1.0

            # T3 仓位调节
            if self.use_t3_sizing and pred_t3 is not None:
                vol = pred_t3[idx]
                median_vol = np.median(pred_t3[t1_mask]) if t1_mask.sum() > 0 else 0.01
                if median_vol > 0:
                    vol_ratio = vol / median_vol
                    weight = np.clip(1.0 / max(vol_ratio, 0.3), 0.5, 1.5)

            selected.append({
                "symbol": sym,
                "rank_score": float(rank_scores[idx]),
                "rank": len(selected) + 1,
                "weight": round(weight, 2),
                "t2_raw": float(pred_t2[idx]),
                "t4_raw": float(pred_t4[idx]),
            })

            if len(selected) >= self.top_n:
                break

        return selected


# ════════════════════════════════════════════════════════
#  多周期回测框架
# ════════════════════════════════════════════════════════

class MonthlyBacktest:
    """月度调仓回测引擎。

    每月最后一个交易日生成信号，次月第一个交易日开盘执行。
    """

    def __init__(
        self,
        signal: IntegratedSignal,
        engine: DataEngine | None = None,
        initial_capital: float = 500_000.0,
        commission: float = 0.00025,
        stamp_tax: float = 0.001,
        slippage: float = 0.0001,
    ):
        self.signal = signal
        self.engine = engine or DataEngine.__new__(DataEngine)
        self.initial_capital = initial_capital
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.slippage = slippage

    def _get_monthly_dates(self, start: str, end: str) -> list[tuple[str, str]]:
        """获取月度重训练日期对。

        Returns:
            [(train_end_date, test_start_date), ...]
            每月最后一个交易日训练 → 次月第一个交易日测试。
        """
        db_path = self.signal.cfg.db_path
        conn = sqlite3.connect(db_path)
        all_dates = pd.read_sql(
            f"SELECT DISTINCT date FROM stock_daily WHERE date >= '{start}' AND date <= '{end}' ORDER BY date",
            conn
        )["date"].tolist()
        conn.close()

        # 按月份分组
        month_dates = defaultdict(list)
        for d in all_dates:
            month_dates[d[:7]].append(d)

        pairs = []
        months = sorted(month_dates.keys())
        for i in range(1, len(months)):
            prev_last = month_dates[months[i - 1]][-1]
            curr_first = month_dates[months[i]][0]
            pairs.append((prev_last, curr_first))

        return pairs

    def run(self, start: str, end: str,
            predictions_by_month: dict[str, dict] | None = None) -> dict:
        """运行月度回测。

        Args:
            start, end: 回测起止日期。
            predictions_by_month: 预计算的月度预测。
                { "2026-01": {"symbols": [...], "t2": [...], "t4": [...]}, ... }

        Returns:
            回测指标: {total_return, annual_return, sharpe, max_drawdown, ...}
        """
        month_pairs = self._get_monthly_dates(start, end)
        if not month_pairs:
            logger.error("回测日期不足")
            return {}

        logger.info(f"月度回测: {start}~{end}, {len(month_pairs)} 个月")

        cash = self.initial_capital
        holdings: dict[str, dict] = {}  # symbol → {shares, cost}
        monthly_values: list[dict] = []

        for train_end, test_start in month_pairs:
            test_month = test_start[:7]

            # 获取本月预测
            if predictions_by_month and test_month in predictions_by_month:
                preds = predictions_by_month[test_month]
                monthly_signals = self.signal.generate_monthly_signals(
                    symbols=preds["symbols"],
                    pred_t2=np.array(preds["t2"]),
                    pred_t4=np.array(preds["t4"]),
                    pred_t1=np.array(preds.get("t1", [])),
                    pred_t3=np.array(preds.get("t3", [])),
                )
            else:
                monthly_signals = []

            # 卖出上月持仓
            for sym in list(holdings.keys()):
                pos = holdings[sym]
                exit_price = self._get_price(sym, test_start)
                if exit_price > 0:
                    proceeds = pos["shares"] * exit_price * (1 - self.commission - self.stamp_tax)
                    cash += proceeds
                del holdings[sym]

            # 买入本月信号
            if monthly_signals:
                budget_per_stock = cash / len(monthly_signals)
                for sig in monthly_signals:
                    sym = sig["symbol"]
                    entry_price = self._get_price(sym, test_start)
                    if entry_price <= 0:
                        continue
                    available = budget_per_stock * sig.get("weight", 1.0)
                    shares = int(available / entry_price / 100) * 100  # 整手
                    if shares < 100:
                        continue
                    cost = shares * entry_price * (1 + self.commission + self.slippage)
                    if cost <= cash:
                        cash -= cost
                        holdings[sym] = {"shares": shares, "cost": cost, "entry_price": entry_price}

            # 记录月末净值
            total_value = cash + sum(
                h["shares"] * self._get_price(s, test_start)
                for s, h in holdings.items()
            )
            monthly_values.append({
                "month": test_month,
                "cash": round(cash, 2),
                "holdings": len(holdings),
                "total_value": round(total_value, 2),
                "signals": len(monthly_signals),
            })

        # 计算指标
        total_value = cash + sum(
            h["shares"] * self._get_price(s, month_pairs[-1][1])
            for s, h in holdings.items()
        )
        return self._compute_metrics(monthly_values, total_value)

    def _get_price(self, symbol: str, date: str) -> float:
        """获取某日开盘价。"""
        try:
            conn = sqlite3.connect(self.signal.cfg.db_path)
            row = conn.execute(
                "SELECT open FROM stock_daily WHERE symbol=? AND date=?",
                (symbol, date)
            ).fetchone()
            conn.close()
            return float(row[0]) if row else 0.0
        except Exception:
            return 0.0

    def _compute_metrics(self, monthly_values: list[dict], final_value: float) -> dict:
        """计算回测指标。"""
        n_months = len(monthly_values)
        if n_months == 0:
            return {}

        total_return = (final_value - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (12 / n_months) - 1

        values = [self.initial_capital] + [mv["total_value"] for mv in monthly_values]
        monthly_returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]

        # Sharpe (annualized, assume 0 risk-free)
        if len(monthly_returns) > 1:
            monthly_std = np.std(monthly_returns)
            sharpe = (np.mean(monthly_returns) / monthly_std * np.sqrt(12)) if monthly_std > 0 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        peak = values[0]
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (v - peak) / peak
            max_dd = min(max_dd, dd)

        return {
            "total_return": round(total_return, 4),
            "annual_return": round(annual_return, 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
            "n_months": n_months,
            "final_value": round(final_value, 2),
            "monthly_values": monthly_values,
            "monthly_returns": [round(r, 4) for r in monthly_returns],
        }


def compare_top_n(
    predictions_by_month: dict,
    top_n_list: list[int] = [10, 15, 20, 25],
    start: str = "2025-08-01",
    end: str = "2026-06-30",
    initial_capital: float = 500_000.0,
) -> list[dict]:
    """对比不同 TOP_N 的回测效果。

    Args:
        predictions_by_month: 预计算的月度预测。
        top_n_list: 需要对比的 TOP_N 列表。

    Returns:
        每个 TOP_N 的指标: [{top_n, total_return, sharpe, max_dd, ...}, ...]
    """
    results = []
    for top_n in top_n_list:
        cfg = get_config()
        signal = IntegratedSignal(cfg, top_n=top_n, fusion_method="rank")
        engine = DataEngine(cfg)  # type: ignore[arg-type]
        bt = MonthlyBacktest(signal, engine, initial_capital=initial_capital)
        metrics = bt.run(start, end, predictions_by_month)
        metrics["top_n"] = top_n
        results.append(metrics)
        logger.info(f"TOP_N={top_n:>2d}: return={metrics.get('total_return', 0):+.2%} "
                    f"sharpe={metrics.get('sharpe', 0):.2f} max_dd={metrics.get('max_drawdown', 0):+.2%}")
    return results


def run_backtest(
    signal_cfg: dict | None = None,
    start: str = "2025-01-01",
    end: str = "2026-06-30",
    predictions_file: str | None = None,
) -> dict:
    """一键运行月度回测。

    Args:
        signal_cfg: IntegratedSignal 参数。
        start, end: 回测起止。
        predictions_file: JSON 文件，包含预计算的月度预测。

    Returns:
        回测指标。
    """
    cfg = get_config()
    signal = IntegratedSignal(**(signal_cfg or {}))
    engine = DataEngine(cfg)  # type: ignore[arg-type]

    predictions_by_month = None
    if predictions_file:
        with open(predictions_file) as f:
            predictions_by_month = json.load(f)

    bt = MonthlyBacktest(signal, engine)
    return bt.run(start, end, predictions_by_month)


# ════════════════════════════════════════════════════════
#  预测生成辅助（接入月度重训练管线）
# ════════════════════════════════════════════════════════

def dump_monthly_predictions(
    output_path: str | Path,
    test_month: str,
    symbols: list[str],
    pred_t2: list[float],
    pred_t4: list[float],
    pred_t1: list[float] | None = None,
    pred_t3: list[float] | None = None,
) -> None:
    """追加月度预测到 JSON 文件（供回测使用）。"""
    output_path = Path(output_path)
    data = {}
    if output_path.exists():
        data = json.loads(output_path.read_text())

    entry = {"symbols": symbols, "t2": pred_t2, "t4": pred_t4}
    if pred_t1 is not None:
        entry["t1"] = pred_t1
    if pred_t3 is not None:
        entry["t3"] = pred_t3

    data[test_month] = entry
    output_path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"月度预测已保存: {test_month} → {output_path}")
