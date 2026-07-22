#!/usr/bin/env python
"""v1.3 LSTM 回测 — 多配置对比（缓存预测，仅算一次）。

用法:
    python backtest_compare.py               # 首次: 算预测 + 4组回测
    python backtest_compare.py --replay-only  # 复用已有缓存, 仅回放

配置组:
    baseline  — v1.2 原参数 (10只/2买/5万)
    A         — 仅LSTM因子+min_hold (10只/2买)
    B         — 仅扩仓 (30只/5买/1.6万)
    A+B       — 扩仓+LSTM因子+min_hold (30只/5买/1.6万)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.core.logger import get_logger
from sequoia_x.model_selection.config import get_config as get_lstm_config
from sequoia_x.model_selection.model import load_latest_model
from sequoia_x.model_selection.backtest.engine import LSTMBacktestEngine
from sequoia_x.model_selection.backtest import config as bt_cfg
from sequoia_x.model_selection.backtest.reporter import save_results

logger = get_logger(__name__)

CACHE_PATH = Path("output/backtest_lstm/predictions_cache.json")
OUTPUT_DIR = Path("output/backtest_lstm")

PERIODS: dict[str, tuple[str, str, str]] = {
    "2024": ("2024-01-01", "2024-12-31", "震荡市, HS300 +1.71%"),
    "2025": ("2025-01-01", "2025-12-31", "大牛市, HS300 +34.94%"),
    "2026": ("2026-01-01", "2026-07-20", "快牛, HS300 +24.25%"),
    "full": ("2024-01-01", "2026-07-20", "全周期"),
}

# ════════════════════════════════════════════════════════════
# 对比配置组
# ════════════════════════════════════════════════════════════

CONFIGS = [
    {
        "name": "baseline",
        "desc": "v1.2 原参数",
        "max_positions": 10,
        "top_n_buy": 2,
        "per_stock_budget": 50_000.0,
        "min_pred_return": 0.01,
        "lstm_factor": False,
        "min_hold": False,
    },
    {
        "name": "A",
        "desc": "仅LSTM因子+min_hold",
        "max_positions": 10,
        "top_n_buy": 2,
        "per_stock_budget": 50_000.0,
        "min_pred_return": 0.01,
        "lstm_factor": True,
        "min_hold": True,
    },
    {
        "name": "B",
        "desc": "仅扩仓",
        "max_positions": 30,
        "top_n_buy": 5,
        "per_stock_budget": 16_000.0,
        "min_pred_return": 0.0,
        "lstm_factor": False,
        "min_hold": False,
    },
    {
        "name": "A+B",
        "desc": "扩仓+LSTM因子+min_hold",
        "max_positions": 30,
        "top_n_buy": 5,
        "per_stock_budget": 16_000.0,
        "min_pred_return": 0.0,
        "lstm_factor": True,
        "min_hold": True,
    },
]


def apply_config(cfg_dict: dict) -> None:
    """把回测配置写入 bt_cfg 模块 + rules.py 相关配置。"""
    bt_cfg.MAX_POSITIONS = cfg_dict["max_positions"]
    bt_cfg.TOP_N_BUY_PER_DAY = cfg_dict["top_n_buy"]
    bt_cfg.PER_STOCK_BUDGET = cfg_dict["per_stock_budget"]
    bt_cfg.MIN_PRED_RETURN = cfg_dict["min_pred_return"]

    # 控制 LSTM 因子开关
    import sequoia_x.simulation.config as sim_cfg
    if cfg_dict["lstm_factor"]:
        sim_cfg.LSTM_BUY_BONUS = -30
        sim_cfg.LSTM_SELL_PENALTY = 20
    else:
        sim_cfg.LSTM_BUY_BONUS = 0
        sim_cfg.LSTM_SELL_PENALTY = 0

    # 控制 min_hold 开关
    sim_cfg.MIN_HOLD_DAYS = 5 if cfg_dict["min_hold"] else 0


def load_cache() -> dict[str, list] | None:
    """加载预测缓存。"""
    if not CACHE_PATH.exists():
        return None
    with open(CACHE_PATH) as f:
        raw = json.load(f)
    return {k: [(s, p) for s, p in v] for k, v in raw.items()}


def run_phase1(model) -> None:
    """Phase 1: 计算并保存每日预测（全周期，仅一次）。"""
    logger.info("=" * 60)
    logger.info("Phase 1: 计算每日预测缓存（全周期 2024-2026）")
    logger.info("=" * 60)

    settings = Settings()
    engine = DataEngine(settings)
    bt = LSTMBacktestEngine(engine, model=model)

    t0 = time.time()
    # 只跑全周期，save_predictions=True 保存缓存
    bt.run("2024-01-01", "2026-07-20", save_predictions=True)
    elapsed = time.time() - t0
    logger.info(f"Phase 1 完成: {elapsed:.0f}s ({elapsed/3600:.1f}h)")


def run_phase2(model, cache: dict) -> list[dict]:
    """Phase 2: 用缓存预测回放 4 组配置。"""
    logger.info("=" * 60)
    logger.info("Phase 2: 多配置回放（复用预测缓存）")
    logger.info("=" * 60)

    settings = Settings()
    engine = DataEngine(settings)
    all_results: list[dict] = []

    for cfg_dict in CONFIGS:
        name = cfg_dict["name"]
        desc = cfg_dict["desc"]
        logger.info(f"\n{'='*40}\n  配置 [{name}]: {desc}\n{'='*40}")
        apply_config(cfg_dict)

        config_metrics: list[dict] = []
        all_daily: list[dict] = []
        all_trades: list[dict] = []

        for period_name, (start, end, period_desc) in PERIODS.items():
            t0 = time.time()
            bt = LSTMBacktestEngine(engine, model=model)
            metrics = bt.run(start, end, predictions_cache=cache)
            if not metrics:
                continue
            elapsed = time.time() - t0

            all_daily.extend(metrics.pop("daily_records", []))
            all_trades.extend(metrics.pop("trade_records", []))

            metrics["period"] = period_name
            metrics["description"] = period_desc
            metrics["config"] = name
            metrics["config_desc"] = desc

            logger.info(
                f"  [{name}] {period_name}: 收益={metrics.get('total_return',0):+.2%} "
                f"夏普={metrics.get('sharpe',0)} 回撤={metrics.get('max_drawdown',0):.2%} "
                f"耗时={elapsed:.0f}s"
            )
            config_metrics.append(metrics)

        # 保存单配置结果
        tag = name.replace("+", "p")
        save_results(
            config_metrics, str(OUTPUT_DIR / tag),
            daily_records=all_daily, trade_records=all_trades,
        )
        all_results.extend(config_metrics)

    return all_results


def print_comparison(all_results: list[dict]) -> None:
    """打印多配置对比表。"""
    print("\n" + "=" * 110)
    print("  v1.3 LSTM 回测 — 多配置对比")
    print("=" * 110)
    print(f"{'配置':>10s} {'期间':>6s} {'策略收益':>8s} {'HS300':>8s} {'超额':>8s} "
          f"{'夏普':>6s} {'回撤':>7s} {'胜率':>6s} {'交易':>5s}")
    print("-" * 110)

    # 按配置分组
    from collections import defaultdict
    by_config = defaultdict(dict)
    for m in all_results:
        by_config[m.get("config", "?")][m.get("period", "?")] = m

    hs300_map = {
        "2024": "+1.71%", "2025": "+34.94%", "2026": "+24.25%", "full": "+27.4%"
    }

    for cfg_name in [c["name"] for c in CONFIGS]:
        for period in ["2024", "2025", "2026", "full"]:
            m = by_config.get(cfg_name, {}).get(period)
            if m:
                print(
                    f"{cfg_name:>10s} {period:>6s} "
                    f"{m.get('total_return',0):>+7.1%} "
                    f"{hs300_map.get(period,'?'):>8s} "
                    f"{m.get('total_return',0)-(float(hs300_map[period].rstrip('%'))/100):>+7.1%} "
                    f"{m.get('sharpe',0):>6.2f} "
                    f"{m.get('max_drawdown',0):>7.1%} "
                    f"{m.get('win_rate',0):>5.1%} "
                    f"{m.get('n_buys',0)+m.get('n_sells',0):>5d}"
                )
        if cfg_name != CONFIGS[-1]["name"]:
            print("-" * 110)

    print("=" * 110)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="LSTM 多配置回测对比")
    parser.add_argument("--replay-only", action="store_true", help="仅回放,跳过预测")
    args = parser.parse_args()

    lstm_cfg = get_lstm_config()
    model_data = load_latest_model(lstm_cfg)
    model = model_data[0] if model_data else None
    if model is None:
        logger.error("无可用模型")
        return

    if not args.replay_only:
        run_phase1(model)

    cache = load_cache()
    if cache is None:
        logger.error("预测缓存不存在,请先跑 Phase 1 (不加 --replay-only)")
        return

    logger.info(f"预测缓存已加载: {len(cache)} 天")

    all_results = run_phase2(model, cache)
    print_comparison(all_results)

    # 保存汇总
    summary_path = OUTPUT_DIR / "comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"对比汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
