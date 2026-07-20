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


def save_results(all_metrics: list[dict], output_dir: str) -> None:
    """保存回测结果并打印对比报告。

    Args:
        all_metrics: 各期间回测指标列表。
        output_dir: 输出目录。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 转储 JSON
    with open(out / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))

    # 打印对比报告
    print_benchmark_report(all_metrics)

    # 保存交易记录摘要
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
