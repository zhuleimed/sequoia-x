"""Phase 2: 72 组共享预测缓存回测 —— 一键运行。

流程:
  1. 加载/构建预测缓存 (T2+T1+T3 月度预测值)
  2. 循环 72 组 (4 TOP_N × 3 时段 × 6 风控)
  3. 每组仅运行模拟循环（秒级/组），共享同一份预测值
  4. 输出 CSV 汇总 + JSON 明细 + 最优配置

用法:
  # 完整流程（先构建缓存，再跑 72 组）
  python scripts/run_shared_backtest.py --all

  # 使用已有缓存
  python scripts/run_shared_backtest.py --all --cache output/backtest_v2/prediction_cache.json

  # 快速测试（3个月×500股票×3组）
  python scripts/run_shared_backtest.py --months 3 --max-stocks 500 --top-n 10 --mode M0

  # 仅构建缓存（不跑回测）
  python scripts/build_prediction_cache.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine

logger = get_logger("shared_backtest")

# ── 常量 ──
OUTPUT_DIR = Path("output/backtest_v2")
CACHE_PATH = OUTPUT_DIR / "prediction_cache.json"

TOP_N_LIST = [10, 15, 20, 25]
PERIODS = [
    ("2025年", "2025-08", "2025-12"),
    ("2026年", "2026-01", "2026-06"),
    ("全周期", "2025-08", "2026-06"),
]
# 2026-08-20: V4 129维 70 个月完整回测（2020-09~2026-06，与旧 121 维 70 月同口径对比）。
# 用 `--period full` 选择此完整全周期段。
FULL_PERIODS = [
    ("全周期70月", "2020-09", "2026-06"),
]
RISK_MODES = [
    ("M0", "裸融合"),
    ("M1", "+T1过滤"),
    ("M2", "+T3仓位"),
    ("M3", "+大盘风控"),
    ("M4", "+IC加权"),
    ("M5", "全风控"),
]
INITIAL_CAPITAL = 500_000.0


def load_or_build_cache(
    cfg: V2Config,
    engine: DataEngine,
    cache_path: Path,
    test_months: list[str],
    max_stocks: int = 0,
    force_rebuild: bool = False,
    skip_t4: bool = False,
) -> dict:
    """加载预测缓存，如不存在则自动构建。

    Returns:
        {month: {symbols, t2, t1, t3}, ...}
    """
    if not force_rebuild and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        months_ok = sum(1 for m in test_months if m in cache)
        if months_ok >= len(test_months):
            logger.info(f"预测缓存已就绪: {len(cache)} 个月 (来自 {cache_path})")
            return cache
        logger.info(f"缓存不完整 ({months_ok}/{len(test_months)} 个月)，重新构建...")
    else:
        logger.info("构建预测缓存...")

    from scripts.build_prediction_cache import build_cache
    return build_cache(cfg, engine, test_months, max_stocks, output_path=cache_path, skip_t4=skip_t4)


def run_single_group(
    cfg: V2Config,
    engine: DataEngine,
    prediction_cache: dict,
    top_n: int,
    risk_mode: str,
    start_month: str,
    end_month: str,
) -> dict:
    """运行单组回测（使用共享预测缓存）。

    Returns:
        {total_return, annual_return, sharpe, max_drawdown, win_rate, ...}
    """
    bt = MonthlyBacktestEngine(
        cfg=cfg,
        engine=engine,
        top_n=top_n,
        risk_mode=risk_mode,
        initial_capital=INITIAL_CAPITAL,
        use_real_t4=False,
        prediction_cache=prediction_cache,
    )
    metrics = bt.run(start_month, end_month)
    return metrics


def save_summary_csv(all_results: list[dict]) -> None:
    """生成汇总 CSV。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary_all.csv"
    fieldnames = [
        "风控模式", "TOP_N", "时段", "月数",
        "总收益率", "年化收益率", "夏普比率", "最大回撤",
        "月胜率", "交易笔数", "月均换手率", "胜率(笔)", "终值",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    logger.info(f"汇总表: {summary_path} ({len(all_results)} 行)")


def save_monthly_matrix(all_results: list[dict]) -> None:
    """生成月度收益矩阵 CSV。"""
    monthly_data: dict[str, dict[str, float]] = {}
    for r in all_results:
        key = f"{r['风控模式']}_T{r['TOP_N']}_{r['时段']}"
        labels = r.get("_monthly_labels", [])
        returns = r.get("_monthly_returns", [])
        if labels and returns and len(labels) == len(returns):
            monthly_data[key] = dict(zip(labels, returns))

    if not monthly_data:
        return

    all_months = sorted(set(m for d in monthly_data.values() for m in d))
    matrix_path = OUTPUT_DIR / "monthly_returns.csv"
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["配置"] + all_months)
        for key in sorted(monthly_data.keys()):
            row = [key] + [
                f"{monthly_data[key].get(m, 0):+.4f}" if m in monthly_data[key] else ""
                for m in all_months
            ]
            writer.writerow(row)
    logger.info(f"收益矩阵: {matrix_path}")


def compute_and_save_optimal(all_results: list[dict]) -> dict:
    """综合评分选出最优配置。"""
    valid = [r for r in all_results
             if r.get("夏普比率") is not None and r.get("最大回撤") is not None]
    if not valid:
        return {}

    n = len(valid)
    idx = list(range(n))
    by_sharpe = sorted(idx, key=lambda i: valid[i]["夏普比率"])
    by_ret = sorted(idx, key=lambda i: valid[i]["总收益率"])
    by_win = sorted(idx, key=lambda i: valid[i]["月胜率"])
    by_dd = sorted(idx, key=lambda i: -valid[i]["最大回撤"])

    s = np.zeros(n)
    for rank, i in enumerate(by_sharpe):
        s[i] += 0.4 * (rank + 1) / n
    for rank, i in enumerate(by_ret):
        s[i] += 0.2 * (rank + 1) / n
    for rank, i in enumerate(by_win):
        s[i] += 0.15 * (rank + 1) / n
    for rank, i in enumerate(by_dd):
        s[i] += 0.25 * (rank + 1) / n

    best_idx = int(np.argmax(s))
    optimal = valid[best_idx].copy()
    optimal["综合评分"] = round(float(s[best_idx]), 4)

    opt_path = OUTPUT_DIR / "optimal_config.json"
    opt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(opt_path, "w") as f:
        json.dump(optimal, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"最优配置: {opt_path}")

    return optimal


def get_test_months(start_month: str, end_month: str) -> list[str]:
    """生成月份列表。"""
    start_ym = (int(start_month[:4]), int(start_month[5:7]))
    end_ym = (int(end_month[:4]), int(end_month[5:7]))
    months = []
    y, m = start_ym
    while (y, m) <= end_ym:
        months.append(f"{y}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def main():
    parser = argparse.ArgumentParser(description="Phase 2: 72 组共享预测缓存回测")
    parser.add_argument("--all", action="store_true", help="运行全部 72 组")
    parser.add_argument("--top-n", type=int, default=0, help="限制 TOP_N")
    parser.add_argument("--mode", type=str, default="", help="限制风控模式")
    parser.add_argument("--period", type=str, default="",
                        help="限制时段: 2025/2026/all")
    parser.add_argument("--cache", type=str, default=str(CACHE_PATH),
                        help="预测缓存路径")
    parser.add_argument("--months", type=int, default=0,
                        help="限制月份数（0=全部）")
    parser.add_argument("--max-stocks", type=int, default=0,
                        help="限制股票池大小（0=全量）")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="强制重建预测缓存")
    parser.add_argument("--skip-t4", action="store_true",
                        help="跳过 T4 LSTM 训练")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="输出目录")
    args = parser.parse_args()

    # 确定运行范围
    if args.all:
        top_n_list = TOP_N_LIST
        # 2026-08-20: --period full 用完整 70 月段（V4 129维 与旧 121维 70月 同口径对比）
        periods = FULL_PERIODS if args.period == "full" else PERIODS
        modes = RISK_MODES
    elif args.top_n and args.mode:
        top_n_list = [args.top_n]
        periods = PERIODS if not args.period else [
            (p[0], p[1], p[2]) for p in PERIODS if args.period in p[0]
        ] or PERIODS
        modes = [(args.mode, args.mode)]
    else:
        parser.print_help()
        return

    total_groups = len(top_n_list) * len(periods) * len(modes)
    logger.info(f"\n{'='*60}")
    logger.info(f"共享预测缓存回测")
    logger.info(f"TOP_N: {top_n_list}")
    logger.info(f"时段: {[p[0] for p in periods]}")
    logger.info(f"风控: {[m[0] for m in modes]}")
    logger.info(f"总计: {total_groups} 组")
    logger.info(f"{'='*60}")

    # 初始化
    cfg = get_config()
    engine = DataEngine(Settings())

    # 收集所有需要的月份
    all_months_set = set()
    for _, p_start, p_end in periods:
        all_months_set.update(get_test_months(p_start, p_end))
    test_months = sorted(all_months_set)

    if args.months > 0:
        test_months = test_months[:args.months]

    # Phase 1: 加载/构建预测缓存
    cache = load_or_build_cache(
        cfg, engine, Path(args.cache), test_months,
        max_stocks=args.max_stocks,
        force_rebuild=args.rebuild_cache,
        skip_t4=args.skip_t4,
    )

    actual_months = sorted(set(test_months) & set(cache.keys()))
    if not actual_months:
        logger.error("预测缓存为空！请先运行 build_prediction_cache.py")
        return

    logger.info(f"缓存就绪: {len(actual_months)} 个月 ({actual_months[0]}~{actual_months[-1]})")

    # Phase 2: 运行所有组合
    all_results = []
    t_total = time.time()
    group_idx = 0

    for top_n in top_n_list:
        for period_name, start_month, end_month in periods:
            for mode_name, mode_desc in modes:
                group_idx += 1
                t_group = time.time()

                # 过滤该时段内的月份
                period_months = [m for m in get_test_months(start_month, end_month)
                                 if m in cache]

                logger.info(f"\n{'─'*50}")
                logger.info(f"[{group_idx}/{total_groups}] TOP_N={top_n} "
                            f"{period_name}({len(period_months)}月) {mode_name}({mode_desc})")

                try:
                    metrics = run_single_group(
                        cfg, engine, cache, top_n, mode_name,
                        start_month, end_month,
                    )
                except Exception as e:
                    logger.error(f"  ❌ 失败: {e}", exc_info=True)
                    metrics = {"total_return": None, "sharpe": None,
                               "max_drawdown": None, "error": str(e)}

                elapsed = time.time() - t_group
                row = {
                    "风控模式": mode_name,
                    "TOP_N": top_n,
                    "时段": period_name,
                    "月数": metrics.get("n_months", 0),
                    "总收益率": metrics.get("total_return"),
                    "年化收益率": metrics.get("annual_return"),
                    "夏普比率": metrics.get("sharpe"),
                    "最大回撤": metrics.get("max_drawdown"),
                    "月胜率": metrics.get("win_rate"),
                    "交易笔数": metrics.get("n_trades", 0),
                    "月均换手率": metrics.get("avg_turnover", 0),
                    "胜率(笔)": metrics.get("win_trade_pct", 0),
                    "终值": metrics.get("final_value", 0),
                }
                all_results.append(row)

                ret_str = f"{metrics.get('total_return', 0):+.2%}" \
                    if metrics.get('total_return') is not None else "N/A"
                logger.info(f"  ✅ 收益={ret_str} "
                            f"夏普={metrics.get('sharpe', 'N/A')} "
                            f"耗时={elapsed:.0f}s ({'缓存' if cache else '实时'})")

    total_elapsed = time.time() - t_total

    # Phase 3: 保存结果
    save_summary_csv(all_results)
    save_monthly_matrix(all_results)
    optimal = compute_and_save_optimal(all_results)

    logger.info(f"\n{'='*60}")
    logger.info(f"回测完成: {total_groups} 组, 总耗时={total_elapsed:.0f}s "
                f"({total_elapsed/60:.1f}min)")
    logger.info(f"{'='*60}")

    if optimal:
        print(f"\n🏆 最优配置:")
        print(f"  风控: {optimal.get('风控模式')}  TOP_N: {optimal.get('TOP_N')}  "
              f"时段: {optimal.get('时段')}")
        print(f"  年化: {optimal.get('年化收益率', 0):+.2%}  "
              f"夏普: {optimal.get('夏普比率', 0):.2f}  "
              f"回撤: {optimal.get('最大回撤', 0):+.2%}  "
              f"评分: {optimal.get('综合评分', 0):.4f}")


if __name__ == "__main__":
    main()
