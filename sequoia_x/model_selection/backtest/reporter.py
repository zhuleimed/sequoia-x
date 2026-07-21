"""回测结果报告生成。

格式化输出回测指标 + 多期对比报告。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# 沪深300各期间基准收益率（用于对比）
HS300_BENCHMARKS: dict[str, float] = {
    "2024": 0.0171,
    "2025": 0.3494,
    "2026": 0.2425,
    "full": 0.2735,
}


def save_results(
    all_metrics: list[dict],
    output_dir: str,
    daily_records: list[dict] | None = None,
    trade_records: list[dict] | None = None,
) -> None:
    """保存回测结果并打印对比报告。

    Args:
        all_metrics: 各期间回测指标列表。
        output_dir: 输出目录。
        daily_records: 逐日净值记录（可选，导出 CSV 便于绘图分析）。
        trade_records: 逐笔交易记录（可选，导出 CSV 便于审计）。
    """
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 转储 JSON（绩效指标）
    with open(out / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))

    # 导出日结记录 CSV（净值曲线绘图用）
    if daily_records:
        df_daily = pd.DataFrame(daily_records)
        df_daily.to_csv(out / "daily_records.csv", index=False)
        logger.info(f"逐日净值已保存: {out / 'daily_records.csv'} ({len(df_daily)} 行)")

    # 导出交易记录 CSV（审计每笔买卖）
    if trade_records:
        df_trades = pd.DataFrame(trade_records)
        df_trades.to_csv(out / "trade_records.csv", index=False)
        logger.info(f"交易明细已保存: {out / 'trade_records.csv'} ({len(df_trades)} 笔)")

    # 打印对比报告
    print_benchmark_report(all_metrics)

    logger.info(f"回测结果已保存: {out.resolve()}")


def print_benchmark_report(all_metrics: list[dict]) -> None:
    """打印多期对比报告。"""
    print(f"\n{'='*80}")
    print(f"  LSTM-Transformer 选股策略 --- 回测报告")
    print(f"{'='*80}")
    print(f"{'期间':<12} {'策略收益':>10} {'HS300':>10} {'超额':>10} "
          f"{'夏普':>8} {'回撤':>8} {'胜率':>8}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for m in all_metrics:
        name = m.get("period", "?")
        ret = m.get("total_return", 0)
        hs = HS300_BENCHMARKS.get(name, 0)
        excess = ret - hs
        sh = m.get("sharpe", 0)
        dd = m.get("max_drawdown", 0)
        wr = m.get("win_rate", 0)
        print(f"{name:<12} {ret:>+9.1%} {hs:>+9.1%} {excess:>+9.1%} "
              f"{sh:>7.2f} {dd:>7.1%} {wr:>7.1%}")

    print(f"{'='*80}")
