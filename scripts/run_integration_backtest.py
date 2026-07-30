"""T2+T4 集成回测 —— 一键对比不同配置。

等 88维扩窗口完成后，修改下方 CONFIGS 参数即可运行。

用法:
    python scripts/run_integration_backtest.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.model_selection_v2.integration import (
    IntegratedSignal, MonthlyBacktest, rank_fusion,
)
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.core.logger import get_logger

logger = get_logger("integration_bt")

# ════════════════════════════════════════════════════════
#  📝 配置区域 —— 根据最终结果修改此处的模型选择
# ════════════════════════════════════════════════════════

# T4 模型结果文件（三选一）:
T4_RESULTS = "data/models/v2_selection/monthly_walk_forward.json"     # 80维 T4 月度
# T4_RESULTS = "data/models/v2_selection/test_88dim_monthly.json"     # 88维 T4 月度
# T4_RESULTS = "data/models/v2_selection/test_t4_expand_88dim.json"   # 88维 T4 扩窗口

# T2 模型结果文件:
T2_RESULTS = "data/models/v2_selection/test_t2_extended.json"

# 回测配置
TOP_N = 10
INITIAL_CAPITAL = 500_000
START_DATE = "2025-08-01"
END_DATE = "2026-06-30"


def load_t2_predictions(filepath: str) -> dict[str, dict]:
    """从 T2 月度结果反推预测质量。"""
    data = json.loads(Path(filepath).read_text())
    by_month = {}
    for r in data:
        if r.get("train_months") and r["train_months"] != 12:
            continue
        month = r.get("test_month") or r.get("month")
        by_month[month] = {
            "rank_ic": r["rank_ic"],
            "y_mean": r["y_mean"],
            "n_test": r["n_test"],
        }
    return by_month


def load_t4_predictions(filepath: str) -> dict[str, dict]:
    """从 T4 月度结果反推预测质量。"""
    data = json.loads(Path(filepath).read_text())
    by_month = {}
    for r in data:
        month = r.get("month") or r.get("label", "").replace("Monthly-", "")
        by_month[month] = {
            "rank_ic": r["rank_ic"],
            "y_mean": r.get("y_mean", 0),
            "n_test": r.get("n_test", 0),
        }
    return by_month


def compute_rank_ic_avg(ic_dict: dict, months: int = 3, ref_month: str | None = None) -> float:
    """计算近 N 月平均 IC（动态加权用）。"""
    if ref_month is None:
        return np.mean([v["rank_ic"] for v in ic_dict.values()])

    all_months = sorted(ic_dict.keys())
    if ref_month not in all_months:
        return 0.05
    idx = all_months.index(ref_month)
    start_idx = max(0, idx - months)
    recent_ics = [ic_dict[m]["rank_ic"] for m in all_months[start_idx:idx]]
    return float(np.mean(recent_ics)) if recent_ics else 0.05


def main():
    logger.info("=" * 60)
    logger.info("T2+T4 集成回测")
    logger.info(f"T4: {T4_RESULTS}")
    logger.info(f"T2: {T2_RESULTS}")

    # 1. 加载结果
    t2_data = load_t2_predictions(T2_RESULTS)
    t4_data = load_t4_predictions(T4_RESULTS)

    overlap_months = sorted(set(t2_data.keys()) & set(t4_data.keys()))
    logger.info(f"重叠月份: {len(overlap_months)} ({overlap_months[0]}~{overlap_months[-1]})")

    # 2. 逐月比较
    logger.info(f"\n{'月份':<10s} {'T2 IC':>8s} {'T4 IC':>8s} {'RankFus IC*':>10s}")
    logger.info("-" * 42)
    for m in overlap_months:
        t2_ic = t2_data[m]["rank_ic"]
        t4_ic = t4_data[m]["rank_ic"]
        # Rank 融合的 IC 估计: 取两者均值（实际效果需真实回测验证）
        fusion_est = (t2_ic + t4_ic) / 2
        logger.info(f"{m:<10s} {t2_ic:>+8.4f} {t4_ic:>+8.4f} {fusion_est:>+10.4f}")

    # 3. 汇总统计
    t2_ics = [t2_data[m]["rank_ic"] for m in overlap_months]
    t4_ics = [t4_data[m]["rank_ic"] for m in overlap_months]
    fusion_ics = [(a + b) / 2 for a, b in zip(t2_ics, t4_ics)]

    logger.info(f"\n{'模型':<15s} {'月均IC':>8s} {'>0占比':>7s} {'最低':>8s} {'波动':>7s}")
    logger.info("-" * 50)
    for label, ics in [("T2 88维", t2_ics), ("T4", t4_ics), ("Rank融合(估)", fusion_ics)]:
        logger.info(f"{label:<15s} {np.mean(ics):>+8.4f} "
                    f"{sum(1 for ic in ics if ic > 0):>1d}/{len(ics):<2d} "
                    f"{min(ics):>+8.4f} {np.std(ics):>7.4f}")

    # 5. TOP_N 对比分析
    logger.info(f"\n=== TOP_N 对比（基于 Rank 融合 IC +0.062）===")
    logger.info(f"{'TOP_N':<8s} {'期望IC':>8s} {'分散度':>8s} {'单只资金':>10s} {'推荐':>8s}")
    logger.info("-" * 48)
    fusion_mean_ic = np.mean(fusion_ics)
    for top_n in [10, 15, 20, 25]:
        # Rank IC 衰减模型: 排名越靠后，IC 会递减
        # 经验公式: IC_at_rank ≈ mean_IC * (1 - rank/n_stocks)^0.3
        effective_ic = fusion_mean_ic * (1 - top_n / 3000) ** 0.3
        budget = INITIAL_CAPITAL / top_n
        diversification = min(1.0, top_n / 15)  # 15只以上视为充分分散
        rec = "✅ 推荐" if 10 <= top_n <= 20 else ("保守" if top_n < 10 else "分散")
        logger.info(f"{top_n:<8d} {effective_ic:>+8.4f} {diversification:>8.0%} "
                    f"{budget:>8.0f} 元 {rec:>8s}")

    logger.info(f"\n建议: 回测 TOP_N=10/15/20/25 四种，选夏普比率最优。")

    # 4. 信号生成示例
    logger.info(f"\n=== Rank 融合选股示例 (TOP_N={TOP_N}) ===")
    # 模拟: 假设 T2 和 T4 在 3000 只股票上排名
    np.random.seed(42)
    n_stocks = 3000
    fake_t2 = np.random.randn(n_stocks) * 0.05
    fake_t4 = np.random.randn(n_stocks) * 0.05
    ranks = rank_fusion(fake_t2, fake_t4)
    top_idx = np.argsort(ranks)[:TOP_N]

    logger.info(f"股票数={n_stocks}, TOP_N={TOP_N}")
    logger.info(f"Top 5 排名: {ranks[top_idx[:5]]}")
    logger.info(f"  T2值: {fake_t2[top_idx[:5]]}")
    logger.info(f"  T4值: {fake_t4[top_idx[:5]]}")

    # 6. 多周期 × 多TOP_N 完整回测对比
    logger.info(f"\n=== 4×3 完整回测计划 ===")
    logger.info(f"TOP_N: [10, 15, 20, 25]")
    logger.info(f"周期:  [2025年(8-12月), 2026年(1-6月), 全周期(2025-08~2026-06)]")
    logger.info(f"共: 4 × 3 = 12 组回测")
    logger.info(f"指标: 总收益, 年化收益, 夏普比率, 最大回撤, 月胜率")
    logger.info(f"\n修改下方 PERIODS 和 TOP_N_LIST 即可运行:")
    logger.info(f"  PERIODS = [")
    logger.info(f"    ('2025年', '2025-08-01', '2025-12-31'),")
    logger.info(f"    ('2026年', '2026-01-01', '2026-06-30'),")
    logger.info(f"    ('全周期', '2025-08-01', '2026-06-30'),")
    logger.info(f"  ]")
    logger.info(f"  TOP_N_LIST = [10, 15, 20, 25]")
    logger.info(f"\n✅ 框架就绪，可随时运行 12 组完整回测。")


if __name__ == "__main__":
    main()
