"""回测结果分析脚本。

读取 output/backtest_v2/summary_all.csv，产出：
  1. 全景对比表（终端输出+Markdown）
  2. 综合评分热力图
  3. Top 10 最优组合
  4. 最优 vs 最差 对比
  5. 2026年5月（最极端月）各风控表现对比

用法:
  python scripts/analyze_backtest_results.py
  python scripts/analyze_backtest_results.py --csv output/backtest_v2/summary_all.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("output/backtest_v2")
REPORT_PATH = OUTPUT_DIR / "analysis_report.md"


def load_results(csv_path: Path) -> list[dict]:
    """加载汇总 CSV。"""
    results = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数字字段
            for key in ["TOP_N", "月数", "交易笔数"]:
                if key in row and row[key]:
                    row[key] = int(float(row[key]))
            for key in ["总收益率", "年化收益率", "夏普比率", "最大回撤",
                        "月胜率", "月均换手率", "胜率(笔)", "终值"]:
                if key in row and row[key]:
                    row[key] = float(row[key])
            results.append(row)
    return results


def compute_composite_score(results: list[dict]) -> list[dict]:
    """计算综合评分。

    0.4×夏普排序分 + 0.25×回撤排序分 + 0.2×收益排序分 + 0.15×胜率排序分
    """
    valid = [r for r in results
             if r.get("夏普比率") is not None and r.get("最大回撤") is not None]
    if not valid:
        return results

    n = len(valid)
    indices = list(range(n))

    # 各项排序（从最差到最好）
    by_sharpe = sorted(indices, key=lambda i: valid[i]["夏普比率"] or -999)
    by_ret = sorted(indices, key=lambda i: valid[i]["总收益率"] or -999)
    by_win = sorted(indices, key=lambda i: valid[i]["月胜率"] or -999)
    by_dd = sorted(indices, key=lambda i: -(valid[i]["最大回撤"] or -999))  # 回撤越小越好

    for rank, idx in enumerate(by_sharpe):
        valid[idx]["score_sharpe"] = (rank + 1) / n
    for rank, idx in enumerate(by_ret):
        valid[idx]["score_return"] = (rank + 1) / n
    for rank, idx in enumerate(by_win):
        valid[idx]["score_win"] = (rank + 1) / n
    for rank, idx in enumerate(by_dd):
        valid[idx]["score_dd"] = (rank + 1) / n

    for r in valid:
        r["综合评分"] = round(
            0.4 * r["score_sharpe"] +
            0.25 * r["score_dd"] +
            0.2 * r["score_return"] +
            0.15 * r["score_win"], 4
        )

    return sorted(valid, key=lambda r: r.get("综合评分", 0), reverse=True)


def print_top_results(scored: list[dict], top_n: int = 10) -> str:
    """打印 Top N 最优组合。"""
    lines = []
    lines.append(f"\n## Top {top_n} 最优组合\n")
    lines.append("| # | 风控 | TOP_N | 时段 | 年化收益 | 夏普 | 最大回撤 | 月胜率 | 综合评分 |")
    lines.append("|---|------|-------|------|---------|------|---------|--------|---------|")

    for i, r in enumerate(scored[:top_n]):
        lines.append(
            f"| {i+1} | {r['风控模式']} | {r['TOP_N']} | {r['时段']} | "
            f"{r.get('年化收益率', 0):+.2%} | {r.get('夏普比率', 0):.2f} | "
            f"{r.get('最大回撤', 0):+.2%} | {r.get('月胜率', 0):.0%} | "
            f"{r.get('综合评分', 0):.4f} |"
        )

    return "\n".join(lines)


def print_heatmap(scored: list[dict]) -> str:
    """生成综合评分热力图（ASCII 版）。

    行=TOP_N×时段, 列=风控模式。
    """
    rows = []
    for top_n in [10, 15, 20, 25]:
        for period in ["2025年", "2026年", "全周期"]:
            rows.append((top_n, period))

    cols = ["M0", "M1", "M2", "M3", "M4", "M5"]

    # 构建评分矩阵
    matrix = {}
    for r in scored:
        key = (r["TOP_N"], r["时段"], r["风控模式"])
        matrix[key] = r.get("综合评分", 0)

    lines = []
    lines.append("\n## 综合评分热力图\n")
    lines.append("| TOP_N | 时段 | " + " | ".join(cols) + " |")
    lines.append("|-------|------|" + "|".join(["-" * 6] * len(cols)) + "|")

    for top_n, period in rows:
        vals = []
        best_val = max(matrix.get((top_n, period, c), 0) for c in cols)
        for c in cols:
            v = matrix.get((top_n, period, c), 0)
            marker = " ★" if v == best_val and best_val > 0 else ""
            vals.append(f"{v:.4f}{marker}")
        lines.append(f"| {top_n} | {period} | " + " | ".join(vals) + " |")

    return "\n".join(lines)


def print_worst_vs_best(scored: list[dict]) -> str:
    """最优 vs 最差对比。"""
    if len(scored) < 2:
        return ""

    best = scored[0]
    worst = scored[-1]

    lines = []
    lines.append("\n## 最优 vs 最差 对比\n")
    lines.append("| 指标 | 最优 | 最差 |")
    lines.append("|------|------|------|")
    lines.append(f"| 配置 | {best['风控模式']} T{best['TOP_N']} {best['时段']} | "
                 f"{worst['风控模式']} T{worst['TOP_N']} {worst['时段']} |")
    lines.append(f"| 年化收益 | {best.get('年化收益率', 0):+.2%} | {worst.get('年化收益率', 0):+.2%} |")
    lines.append(f"| 夏普比率 | {best.get('夏普比率', 0):.2f} | {worst.get('夏普比率', 0):.2f} |")
    lines.append(f"| 最大回撤 | {best.get('最大回撤', 0):+.2%} | {worst.get('最大回撤', 0):+.2%} |")
    lines.append(f"| 月胜率 | {best.get('月胜率', 0):.0%} | {worst.get('月胜率', 0):.0%} |")
    lines.append(f"| 综合评分 | {best.get('综合评分', 0):.4f} | {worst.get('综合评分', 0):.4f} |")

    return "\n".join(lines)


def print_mode_comparison(results: list[dict]) -> str:
    """各风控模式平均表现对比。"""
    lines = []
    lines.append("\n## 风控模式效果对比\n")
    lines.append("| 模式 | 平均年化 | 平均夏普 | 平均回撤 | 平均胜率 |")
    lines.append("|------|---------|---------|---------|---------|")

    for mode in ["M0", "M1", "M2", "M3", "M4", "M5"]:
        subset = [r for r in results if r["风控模式"] == mode
                  and r.get("年化收益率") is not None]
        if not subset:
            continue
        avg_ret = np.mean([r["年化收益率"] for r in subset])
        avg_sharpe = np.mean([r["夏普比率"] for r in subset])
        avg_dd = np.mean([r["最大回撤"] for r in subset])
        avg_win = np.mean([r["月胜率"] for r in subset])
        lines.append(f"| {mode} | {avg_ret:+.2%} | {avg_sharpe:.2f} | {avg_dd:+.2%} | {avg_win:.0%} |")

    return "\n".join(lines)


def print_period_comparison(results: list[dict]) -> str:
    """各时段平均表现对比。"""
    lines = []
    lines.append("\n## 时段适应性\n")
    lines.append("| 时段 | 平均年化 | 平均夏普 | 平均回撤 | 平均胜率 |")
    lines.append("|------|---------|---------|---------|---------|")

    for period in ["2025年", "2026年", "全周期"]:
        subset = [r for r in results if r["时段"] == period
                  and r.get("年化收益率") is not None]
        if not subset:
            continue
        avg_ret = np.mean([r["年化收益率"] for r in subset])
        avg_sharpe = np.mean([r["夏普比率"] for r in subset])
        avg_dd = np.mean([r["最大回撤"] for r in subset])
        avg_win = np.mean([r["月胜率"] for r in subset])
        lines.append(f"| {period} | {avg_ret:+.2%} | {avg_sharpe:.2f} | {avg_dd:+.2%} | {avg_win:.0%} |")

    return "\n".join(lines)


def generate_report(results: list[dict]) -> str:
    """生成完整分析报告。"""
    scored = compute_composite_score(results)

    parts = [
        "# Sequoia-X V2 综合回测分析报告\n",
        f"> 自动生成 | 共 {len(results)} 组对比\n",
        print_top_results(scored, 15),
        "\n---\n",
        print_heatmap(scored),
        "\n---\n",
        print_worst_vs_best(scored),
        "\n---\n",
        print_mode_comparison(results),
        "\n---\n",
        print_period_comparison(results),
        "\n---\n",
        "## 结论与建议\n",
        _generate_recommendation(scored),
    ]

    report = "\n".join(parts)
    return report


def _generate_recommendation(scored: list[dict]) -> str:
    """生成小白友好的建议。"""
    if not scored:
        return "_无数据，无法生成建议。_"

    best = scored[0]

    # 找 TOP_N 的最佳值
    top10_best = [r for r in scored if r["TOP_N"] == 10]
    top15_best = [r for r in scored if r["TOP_N"] == 15]
    top20_best = [r for r in scored if r["TOP_N"] == 20]
    top25_best = [r for r in scored if r["TOP_N"] == 25]

    lines = []
    lines.append(f"### 推荐配置\n")

    rr = best.get('年化收益率', 0)
    lines.append(f"**建议使用 {best['风控模式']} + TOP_N={best['TOP_N']} + {best['时段']}**")
    lines.append(f"- 预期年化收益率: **{rr:+.1%}**")
    lines.append(f"- 预期最大回撤: **{best.get('最大回撤', 0):+.1%}**")
    lines.append(f"- 每月选股 **{best['TOP_N']} 只**")
    lines.append(f"- 综合评分: {best.get('综合评分', 0):.4f}")

    lines.append(f"\n### 各 TOP_N 最优配置\n")
    for top_n, subset in [(10, top10_best), (15, top15_best),
                           (20, top20_best), (25, top25_best)]:
        if subset:
            s = subset[0]
            lines.append(f"- **TOP_N={top_n}**: {s['风控模式']} | "
                         f"年化={s.get('年化收益率', 0):+.1%} | "
                         f"夏普={s.get('夏普比率', 0):.2f} | "
                         f"回撤={s.get('最大回撤', 0):+.1%}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="回测结果分析")
    parser.add_argument("--csv", type=str,
                        default=str(OUTPUT_DIR / "summary_all.csv"),
                        help="汇总 CSV 路径")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        print("请先运行 run_comprehensive_backtest.py 生成结果。")
        sys.exit(1)

    results = load_results(csv_path)
    if not results:
        print("❌ CSV 为空")
        sys.exit(1)

    print(f"加载 {len(results)} 组回测结果")

    # 生成报告
    report = generate_report(results)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"报告已保存: {REPORT_PATH}")

    # 终端打印摘要
    scored = compute_composite_score(results)
    print(print_top_results(scored, 5))
    print(print_mode_comparison(results))


if __name__ == "__main__":
    main()
