"""T2+T4 Rank融合 + 多风控模式 月度综合回测系统。

完整流程:
  每月: 训练T2+T4+T1+T3 → 全股票池预测 → Rank融合 → 选股
       → 风控管线 → 执行买入 → 日级别持仓管理(13条卖出规则)
       → 月末清仓 → 循环

72组对比:
  4 TOP_N (10/15/20/25) × 3 时段 (2025/2026/全周期) × 6 风控 (M0-M5)

输出:
  output/backtest_v2/summary_all.csv      — 汇总对比表
  output/backtest_v2/summary_top{N}.csv   — 各 TOP_N 详细对比
  output/backtest_v2/details_{mode}.csv   — 月度明细
  output/backtest_v2/monthly_returns.csv  — 收益矩阵
  output/backtest_v2/optimal_config.json  — 最优配置

用法:
  # 快速测试（单组，跳过T4）
  python scripts/run_comprehensive_backtest.py --period 2025 --top-n 10 --mode M0 --fast

  # 单组完整回测（含T4真实训练）
  python scripts/run_comprehensive_backtest.py --period 2025 --top-n 10 --mode M0

  # 全部72组
  python scripts/run_comprehensive_backtest.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# 确保项目根在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import V2Config, get_config
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine

logger = get_logger("comprehensive_bt")

# ── 输出目录 ──
OUTPUT_DIR = Path("output/backtest_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 回测参数网格 ──
TOP_N_LIST = [10, 15, 20, 25]

PERIODS = [
    ("2025年", "2025-08", "2025-12"),
    ("2026年", "2026-01", "2026-06"),
    ("全周期", "2025-08", "2026-06"),
]

# 风控模式: (标识, 描述)
RISK_MODES = [
    ("M0", "裸融合"),
    ("M1", "+T1过滤"),
    ("M2", "+T3仓位"),
    ("M3", "+大盘风控"),
    ("M4", "+IC加权"),
    ("M5", "全风控"),
]

INITIAL_CAPITAL = 500_000.0


def run_single_backtest(
    cfg: V2Config,
    engine: DataEngine,
    top_n: int,
    risk_mode: str,
    start_month: str,
    end_month: str,
    use_real_t4: bool = True,
    prediction_cache: dict | None = None,
    fusion_method: str = "pred_std",
) -> dict:
    """运行单组回测。

    Returns:
        {total_return, annual_return, sharpe, max_drawdown, win_rate, ...}
    """
    bt = MonthlyBacktestEngine(
        cfg=cfg,
        engine=engine,
        top_n=top_n,
        risk_mode=risk_mode,
        initial_capital=INITIAL_CAPITAL,
        use_real_t4=use_real_t4,
        prediction_cache=prediction_cache,
        fusion_method=fusion_method,
    )
    metrics = bt.run(start_month, end_month)

    # 保存明细
    detail_path = OUTPUT_DIR / f"details_{risk_mode}_top{top_n}_{start_month}_{end_month}.json"
    with open(detail_path, "w") as f:
        json.dump({
            "config": {
                "top_n": top_n, "risk_mode": risk_mode,
                "start_month": start_month, "end_month": end_month,
                "use_real_t4": use_real_t4,
            },
            "metrics": {k: v for k, v in metrics.items()
                        if k not in ("daily_records", "trades")},
            "daily_records": metrics.get("daily_records", []),
            "trades": metrics.get("trades", []),
        }, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"  明细已保存: {detail_path}")

    return metrics


def build_summary_csv(all_results: list[dict]) -> None:
    """生成汇总 CSV 文件。"""
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

    logger.info(f"汇总表已保存: {summary_path} ({len(all_results)} 行)")

    # 按 TOP_N 分组
    for top_n in TOP_N_LIST:
        subset = [r for r in all_results if r["TOP_N"] == top_n]
        if not subset:
            continue
        topn_path = OUTPUT_DIR / f"summary_top{top_n}.csv"
        with open(topn_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in subset:
                writer.writerow(r)
        logger.info(f"  TOP_N={top_n}: {len(subset)} 组 → {topn_path}")


def build_monthly_returns_csv(all_results: list[dict]) -> None:
    """生成月度收益矩阵 CSV。"""
    # 收集所有月度收益
    monthly_data: dict[str, dict[str, float]] = {}  # {config_key: {month: return}}

    for r in all_results:
        config_key = f"{r['风控模式']}_T{r['TOP_N']}_{r['时段']}"
        monthly_returns = r.get("_monthly_returns", [])
        monthly_labels = r.get("_monthly_labels", [])
        if monthly_returns and monthly_labels:
            monthly_data[config_key] = dict(zip(monthly_labels, monthly_returns))

    if not monthly_data:
        return

    # 收集所有月份
    all_months = sorted(set(
        m for data in monthly_data.values() for m in data.keys()
    ))

    matrix_path = OUTPUT_DIR / "monthly_returns.csv"
    with open(matrix_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["配置"] + all_months)
        for config_key in sorted(monthly_data.keys()):
            row = [config_key]
            for month in all_months:
                ret = monthly_data[config_key].get(month, "")
                row.append(f"{ret:+.4f}" if isinstance(ret, (int, float)) else "")
            writer.writerow(row)

    logger.info(f"月度收益矩阵已保存: {matrix_path}")


def compute_optimal(all_results: list[dict]) -> dict:
    """计算最优配置（综合评分 = 0.4×夏普排序 + 0.25×回撤排序 + 0.2×收益排序 + 0.15×胜率排序）。"""
    if not all_results:
        return {}

    # 提取有效结果（有夏普和回撤的）
    valid = [r for r in all_results
             if r.get("夏普比率") is not None and r.get("最大回撤") is not None]

    if not valid:
        return {}

    n = len(valid)
    # 排序分（值越大越好 → 分越高）
    sharpe_rank = np.zeros(n)
    ret_rank = np.zeros(n)
    win_rank = np.zeros(n)
    # 回撤：值越小（越接近0）越好 → 排序反转
    dd_rank = np.zeros(n)

    sorted_by_sharpe = sorted(range(n), key=lambda i: valid[i]["夏普比率"])
    sorted_by_ret = sorted(range(n), key=lambda i: valid[i]["总收益率"])
    sorted_by_win = sorted(range(n), key=lambda i: valid[i]["月胜率"])
    sorted_by_dd = sorted(range(n), key=lambda i: -valid[i]["最大回撤"])  # 负值：-0.1 > -0.5

    for rank, idx in enumerate(sorted_by_sharpe):
        sharpe_rank[idx] = (rank + 1) / n
    for rank, idx in enumerate(sorted_by_ret):
        ret_rank[idx] = (rank + 1) / n
    for rank, idx in enumerate(sorted_by_win):
        win_rank[idx] = (rank + 1) / n
    for rank, idx in enumerate(sorted_by_dd):
        dd_rank[idx] = (rank + 1) / n

    scores = (0.4 * sharpe_rank + 0.25 * dd_rank +
              0.2 * ret_rank + 0.15 * win_rank)
    best_idx = int(np.argmax(scores))

    optimal = valid[best_idx].copy()
    optimal["综合评分"] = round(float(scores[best_idx]), 4)

    # 保存
    opt_path = OUTPUT_DIR / "optimal_config.json"
    with open(opt_path, "w") as f:
        json.dump(optimal, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"最优配置已保存: {opt_path}")
    return optimal


def main():
    parser = argparse.ArgumentParser(description="T2+T4月度综合回测")
    parser.add_argument("--period", type=str, default="",
                        help="时段: 2025, 2026, all")
    parser.add_argument("--top-n", type=int, default=0,
                        help="选股数: 10/15/20/25")
    parser.add_argument("--mode", type=str, default="",
                        help="风控模式: M0/M1/M2/M3/M4/M5")
    parser.add_argument("--all", action="store_true",
                        help="运行全部72组")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式（跳过T4训练）")
    parser.add_argument("--fusion-method", type=str, default="pred_std",
                        choices=["pred_std", "ic_weighted"],
                        help="融合方法: pred_std=原启发式(默认) | ic_weighted=滚动IC加权(§25方案1)")
    parser.add_argument("--start-month", type=str, default="",
                        help="自定义起始月 (YYYY-MM)")
    parser.add_argument("--end-month", type=str, default="",
                        help="自定义结束月 (YYYY-MM)")
    args = parser.parse_args()

    # 确定运行范围
    if args.all:
        top_n_list = TOP_N_LIST
        periods = PERIODS
        modes = RISK_MODES
    elif args.top_n and args.period and args.mode:
        top_n_list = [args.top_n]
        periods = [(args.period, args.start_month or "", args.end_month or "")]
        modes = [(args.mode, args.mode)]
    elif args.top_n or args.period or args.mode:
        # 部分指定：补齐默认值
        top_n_list = [args.top_n] if args.top_n else TOP_N_LIST
        periods = [(args.period, args.start_month or "", args.end_month or "")] \
            if args.period else PERIODS
        modes = [(args.mode, args.mode)] if args.mode else RISK_MODES
    else:
        parser.print_help()
        return

    # 如果 --period 指定但没给起止日期，从 PERIODS 中匹配
    resolved_periods: list[tuple[str, str, str]] = []
    for p_name, p_start, p_end in periods:
        if not p_start and args.period:
            # 尝试从默认 PERIODS 匹配
            for def_name, def_start, def_end in PERIODS:
                if def_name.startswith(args.period):
                    resolved_periods.append((def_name, def_start, def_end))
                    break
        else:
            resolved_periods.append((p_name, p_start, p_end))

    if not resolved_periods:
        logger.error("无法解析时段参数")
        return

    total_groups = len(top_n_list) * len(resolved_periods) * len(modes)
    logger.info(f"\n{'='*60}")
    logger.info(f"综合回测启动")
    logger.info(f"TOP_N: {top_n_list}, 时段: {[p[0] for p in resolved_periods]}, "
                f"风控: {[m[0] for m in modes]}")
    logger.info(f"总计: {len(top_n_list)}×{len(resolved_periods)}×{len(modes)} = {total_groups} 组")
    logger.info(f"T4: {'真实训练' if not args.fast else '跳过(快速模式)'}")
    logger.info(f"{'='*60}")

    # 初始化
    cfg = get_config()
    engine = DataEngine(Settings())
    use_real_t4 = not args.fast

    # 加载预测缓存（T2+T4+T1+T3 已预计算，无需实时训练）
    prediction_cache = None
    cache_path = OUTPUT_DIR / "prediction_cache.json"
    if cache_path.exists() and not args.fast:
        prediction_cache = json.loads(cache_path.read_text())
        logger.info(f"预测缓存加载: {cache_path} ({len(prediction_cache)} 个月)")
    elif args.fast:
        logger.info("快速模式: 跳过预测缓存，使用实时训练")
    else:
        logger.warning(f"预测缓存不存在: {cache_path}，将使用实时训练（耗时较长）")

    all_results = []
    t_total_start = time.time()
    group_idx = 0

    for top_n in top_n_list:
        for period_name, start_month, end_month in resolved_periods:
            for mode_name, mode_desc in modes:
                group_idx += 1
                t_group_start = time.time()

                logger.info(f"\n{'─'*50}")
                logger.info(f"[{group_idx}/{total_groups}] TOP_N={top_n} "
                            f"{period_name} {mode_name}({mode_desc})")

                try:
                    metrics = run_single_backtest(
                        cfg, engine, top_n, mode_name,
                        start_month, end_month,
                        use_real_t4=use_real_t4,
                        prediction_cache=prediction_cache,
                        fusion_method=args.fusion_method,
                    )
                except Exception as e:
                    logger.error(f"  ❌ 回测失败: {e}", exc_info=True)
                    metrics = {
                        "total_return": None, "annual_return": None,
                        "sharpe": None, "max_drawdown": None,
                        "win_rate": None, "n_months": 0, "n_trades": 0,
                        "error": str(e),
                    }

                elapsed = time.time() - t_group_start

                # 构造汇总行
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
                    "_monthly_returns": metrics.get("monthly_returns", []),
                    "_monthly_labels": metrics.get("_monthly_labels", []),
                }
                all_results.append(row)

                # 打印当前结果
                ret_str = f"{metrics.get('total_return', 0):+.2%}" \
                    if metrics.get('total_return') is not None else "N/A"
                sharpe_str = f"{metrics.get('sharpe', 0):.2f}" \
                    if metrics.get('sharpe') is not None else "N/A"
                dd_str = f"{metrics.get('max_drawdown', 0):+.2%}" \
                    if metrics.get('max_drawdown') is not None else "N/A"
                logger.info(f"  ✅ 收益={ret_str} 夏普={sharpe_str} "
                            f"回撤={dd_str} 耗时={elapsed:.0f}s")

    total_elapsed = time.time() - t_total_start
    logger.info(f"\n{'='*60}")
    logger.info(f"全部回测完成: {total_groups} 组, "
                f"总耗时={total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"{'='*60}")

    # 输出
    build_summary_csv(all_results)
    build_monthly_returns_csv(all_results)
    optimal = compute_optimal(all_results)

    # 打印最优结果
    if optimal:
        print(f"\n{'='*60}")
        print("🏆 最优配置 (综合评分):")
        print(f"  风控: {optimal.get('风控模式')}  TOP_N: {optimal.get('TOP_N')}  "
              f"时段: {optimal.get('时段')}")
        print(f"  年化收益: {optimal.get('年化收益率', 0):+.2%}  "
              f"夏普: {optimal.get('夏普比率', 0):.2f}  "
              f"最大回撤: {optimal.get('最大回撤', 0):+.2%}")
        print(f"  月胜率: {optimal.get('月胜率', 0):.0%}  "
              f"综合评分: {optimal.get('综合评分', 0):.3f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
