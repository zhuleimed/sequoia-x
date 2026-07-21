"""LSTM 模型选股策略回测 -- CLI 入口。

用法:
  python -m sequoia_x.model_selection.backtest.run
  python -m sequoia_x.model_selection.backtest.run --period 2024
  python -m sequoia_x.model_selection.backtest.run --start 2024-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection.config import get_config as get_lstm_config
from sequoia_x.model_selection.backtest.engine import LSTMBacktestEngine
from sequoia_x.model_selection.backtest import config as bt_cfg
from sequoia_x.model_selection.backtest.reporter import save_results
from sequoia_x.model_selection.model import load_latest_model

logger = get_logger(__name__)

PERIODS: dict[str, tuple[str, str, str]] = {
    "2024": ("2024-01-01", "2024-12-31", "震荡市, HS300 +1.71%"),
    "2025": ("2025-01-01", "2025-12-31", "大牛市, HS300 +34.94%"),
    "2026": ("2026-01-01", "2026-07-20", "快牛, HS300 +24.25%"),
    "full": ("2024-01-01", "2026-07-20", "全周期"),
}


def run_period(
    engine: DataEngine, name: str, start: str, end: str, desc: str,
    model=None,
) -> dict:
    """运行单个期间回测。

    Args:
        model: 预加载的 Keras 模型（避免多期间回测时重复加载）。
    """
    logger.info(f"回测 {name}: {start} ~ {end} ({desc})")
    t0 = time.time()

    # model 由调用方传入（main() 中统一加载一次），不再重复加载
    if model is None:
        logger.warning("未找到 LSTM 模型，回测将跳过所有交易信号")

    bt = LSTMBacktestEngine(engine, model=model)
    metrics = bt.run(start, end)
    elapsed = time.time() - t0
    metrics["period"] = name
    metrics["description"] = desc
    metrics["duration_seconds"] = round(elapsed, 1)
    if metrics:
        logger.info(
            f"回测 {name} 完成 | 收益={metrics.get('total_return',0):+.1%} "
            f"夏普={metrics.get('sharpe',0)} "
            f"回撤={metrics.get('max_drawdown',0):.1%} "
            f"耗时={elapsed:.0f}s"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM-Transformer 选股策略回测")
    parser.add_argument(
        "--period", choices=list(PERIODS.keys()),
        help="回测期间 (默认全部)"
    )
    parser.add_argument(
        "--start", type=str, help="自定义开始日期 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, help="自定义结束日期 (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    settings = Settings()
    engine = DataEngine(settings)
    lstm_cfg = get_lstm_config()
    model_data = load_latest_model(lstm_cfg)
    model = model_data[0] if model_data else None

    if args.start:
        bt = LSTMBacktestEngine(engine, model=model)
        metrics = bt.run(args.start, args.end or "")
        _save_with_details(metrics, bt_cfg.OUTPUT_DIR)
    elif args.period:
        name, start, end, desc = PERIODS[args.period]
        bt = LSTMBacktestEngine(engine, model=model)
        metrics = bt.run(start, end)
        _save_with_details(metrics, bt_cfg.OUTPUT_DIR)
    else:
        # 运行全部4个期间（model 统一加载一次，复用给各期间）
        all_metrics: list[dict] = []
        all_daily: list[dict] = []
        all_trades: list[dict] = []
        for name, (start, end, desc) in PERIODS.items():
            metrics = run_period(engine, name, start, end, desc, model=model)
            # 分离绩效指标和明细数据
            all_daily.extend(metrics.pop("daily_records", []))
            all_trades.extend(metrics.pop("trade_records", []))
            all_metrics.append(metrics)
        save_results(all_metrics, bt_cfg.OUTPUT_DIR,
                     daily_records=all_daily, trade_records=all_trades)


def _save_with_details(metrics: dict, output_dir: str) -> None:
    """保存单个期间的回测结果（含日结和交易明细）。"""
    daily = metrics.pop("daily_records", None)
    trades = metrics.pop("trade_records", None)
    save_results([metrics], output_dir, daily_records=daily, trade_records=trades)


if __name__ == "__main__":
    main()
