"""model_selection_v2 - 回测报告输出。"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)


def save_results(
    metrics_list: list[dict], output_dir: str,
    daily_records: list[dict] | None = None,
    trade_records: list[dict] | None = None,
) -> None:
    """保存回测结果。

    Args:
        metrics_list: 多个期间的绩效指标列表。
        output_dir: 输出目录路径。
        daily_records: 逐日净值记录。
        trade_records: 逐笔交易记录。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 指标 JSON
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics_list, f, indent=2, default=str)
    logger.info(f"绩效指标已保存: {out / 'metrics.json'}")

    # 逐日净值 CSV
    if daily_records:
        _save_csv(out / "daily_records.csv", daily_records)
        logger.info(f"逐日净值已保存: {out / 'daily_records.csv'} ({len(daily_records)} 行)")

    # 交易明细 CSV
    if trade_records:
        _save_csv(out / "trade_records.csv", trade_records)
        logger.info(f"交易明细已保存: {out / 'trade_records.csv'} ({len(trade_records)} 笔)")


def _save_csv(path: Path, records: list[dict]) -> None:
    """保存 CSV 文件。"""
    if not records:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def print_comparison_table(all_metrics: list[dict]) -> None:
    """打印多期间对比表。"""
    periods = {"2024": "+1.71%", "2025": "+34.94%", "2026": "+24.25%", "full": "+27.4%"}
    print("\n" + "=" * 80)
    print("  V2 多任务树模型 — 回测报告")
    print("=" * 80)
    print(f"{'期间':>6s} {'策略收益':>8s} {'HS300':>8s} {'超额':>8s} "
          f"{'夏普':>6s} {'回撤':>7s} {'胜率':>6s} {'交易':>5s}")
    print("-" * 80)
    for m in all_metrics:
        period = m.get("period", "?")
        hs300 = periods.get(period, "?")
        hs300_val = float(hs300.rstrip("%")) / 100 if hs300 != "?" else 0
        print(
            f"{period:>6s} "
            f"{m.get('total_return', 0):>+7.1%} "
            f"{hs300:>8s} "
            f"{m.get('total_return', 0) - hs300_val:>+7.1%} "
            f"{m.get('sharpe', 0):>6.2f} "
            f"{m.get('max_drawdown', 0):>7.1%} "
            f"{m.get('win_rate', 0):>5.1%} "
            f"{m.get('n_buys', 0) + m.get('n_sells', 0):>5d}"
        )
    print("=" * 80)
