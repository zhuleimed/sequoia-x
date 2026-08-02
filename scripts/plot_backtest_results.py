"""
Sequoia-X V2 回测结果可视化。

图表:
  1. 净值曲线（策略 vs 沪深300 + 回撤区间）
  2. 月度收益柱状图
  3. 综合评分热力图（72组）
  4. TOP_N 夏普对比
  5. 最优 vs 最差 雷达对比
  6. 风控模式效果对比

用法:
  python scripts/plot_backtest_results.py              # 基于已有summary生成图表
  python scripts/plot_backtest_results.py --with-curve # 含净值曲线（需运行单组回测）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_DIR = Path("output/backtest_v2")
CHART_DIR = OUTPUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ── CJK 字体全局配置（根治中文乱码）──
# 策略：注册 Noto Sans CJK 字体 → 设为 matplotlib 默认 sans-serif 字体
# 这样 title、tick labels、legend、colorbar 全部自动使用中文字体
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_fm = matplotlib.font_manager
# 1. 清除 matplotlib 字体缓存（否则可能加载不到 CJK 字体）
_font_cache_dir = matplotlib.get_cachedir()
for _cache_file in Path(_font_cache_dir).glob("fontlist-v*.json"):
    _cache_file.unlink(missing_ok=True)
# 2. 注册 CJK 字体
_fm.fontManager.addfont(_FONT_PATH)
# 3. 获取字体家族名并设为全局默认
_FONT_FAMILY = _fm.FontProperties(fname=_FONT_PATH).get_name()  # "Noto Sans CJK JP"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [_FONT_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# 4. 重建字体管理器
_fm._load_fontmanager(try_read_cache=False)

# 便捷 FontProperties 对象（用于需要特殊大小的场景，大部分已由全局字体覆盖）
FONT_SM = FontProperties(fname=_FONT_PATH, size=9)
FONT_MD = FontProperties(fname=_FONT_PATH, size=11)
FONT_TITLE = FontProperties(fname=_FONT_PATH, size=16, weight="bold")
FONT_BOLD = FontProperties(fname=_FONT_PATH, weight="bold")


def load_results(csv_path: Path) -> list[dict]:
    results = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            for k in ["总收益率", "年化收益率", "夏普比率", "最大回撤",
                       "月胜率", "月均换手率", "胜率(笔)", "终值"]:
                if k in row and row[k]:
                    row[k] = float(row[k])
            for k in ["TOP_N", "月数", "交易笔数"]:
                if k in row and row[k]:
                    row[k] = int(float(row[k]))
            results.append(row)
    return results


def compute_scores(results: list[dict]) -> list[dict]:
    valid = [r for r in results if r.get("夏普比率") is not None
             and r.get("最大回撤") is not None]
    if not valid:
        return results
    n = len(valid)
    idx = list(range(n))
    s = np.zeros(n)
    for rank, i in enumerate(sorted(idx, key=lambda i: valid[i]["夏普比率"])):
        s[i] += 0.4 * (rank + 1) / n
    for rank, i in enumerate(sorted(idx, key=lambda i: valid[i]["总收益率"])):
        s[i] += 0.2 * (rank + 1) / n
    for rank, i in enumerate(sorted(idx, key=lambda i: valid[i]["月胜率"])):
        s[i] += 0.15 * (rank + 1) / n
    for rank, i in enumerate(sorted(idx, key=lambda i: -valid[i]["最大回撤"])):
        s[i] += 0.25 * (rank + 1) / n
    for i, r in enumerate(valid):
        r["综合评分"] = round(float(s[i]), 4)
    # 按综合评分降序排列（最优在前）
    valid.sort(key=lambda r: r["综合评分"], reverse=True)
    return valid


# ═══════════════════════════════════════════════
#  净值曲线（策略 vs 沪深300 + 回撤区间）
# ═══════════════════════════════════════════════

def generate_net_value_curve() -> None:
    """运行最佳配置回测，获取日级别净值，与沪深300对比。"""
    from sequoia_x.core.config import Settings
    from sequoia_x.data.engine import DataEngine
    from sequoia_x.model_selection_v2.config import get_config
    from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine

    print("  运行全周期回测获取日级别净值...")
    cfg = get_config()
    engine = DataEngine(Settings())

    # 加载预测缓存
    cache_path = OUTPUT_DIR / "prediction_cache.json"
    if not cache_path.exists():
        print("  ⚠ prediction_cache.json 不存在，使用实时训练模式...")
        cache = None
    else:
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  使用缓存: {len(cache)} 个月")

    bt = MonthlyBacktestEngine(
        cfg=cfg, engine=engine, top_n=10, risk_mode="M4",  # 全周期冠军配置
        initial_capital=500_000, use_real_t4=False,
        prediction_cache=cache,
    )
    metrics = bt.run("2020-09", "2026-07")

    daily = metrics.get("daily_records", [])
    if not daily:
        print("  ❌ 无日级别记录")
        return

    # 策略净值
    dates = [r["date"] for r in daily]
    values = np.array([r["total_value"] for r in daily])
    nav = values / values[0]

    # 沪深300 同期净值
    idx_nav, idx_dates = _get_index_nav(dates[0], dates[-1])
    if idx_nav is not None and len(idx_nav) > 0:
        # 对齐日期：方式简单——取指数同期累计收益
        idx_nav_aligned = idx_nav / idx_nav[0]
    else:
        idx_nav_aligned = None

    # 计算回撤
    running_max = np.maximum.accumulate(nav)
    drawdown = (nav - running_max) / running_max

    # ── 绘制 ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    # 上轴：净值曲线
    ax1.plot(dates, nav, color="#2196F3", linewidth=2, label="策略净值 (M5 TOP15)")
    if idx_nav_aligned is not None:
        # 取同期的指数日期
        ax1.plot(idx_dates[:len(idx_nav_aligned)], idx_nav_aligned,
                 color="#FF5722", linewidth=1.5, linestyle="--", alpha=0.7,
                 label="沪深300")
    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax1.fill_between(dates, 1.0, nav, where=(nav >= 1.0),
                     color="#4CAF50", alpha=0.15, label="盈利区间")
    ax1.fill_between(dates, nav, 1.0, where=(nav < 1.0),
                     color="#FF5722", alpha=0.15, label="亏损区间")

    # 标注关键事件
    _add_market_annotations(ax1, dates, nav)

    ax1.set_title("Sequoia-X V2 净值曲线 vs 沪深300", fontproperties=FONT_TITLE, pad=15)
    ax1.set_ylabel("净值", fontproperties=FONT_MD)
    ax1.legend(loc="upper left", prop=FONT_SM, framealpha=0.9)
    ax1.grid(alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # 标注终值
    final_val = nav[-1]
    ax1.annotate(f"{final_val:.2f}",
                 xy=(len(dates) - 1, final_val),
                 xytext=(5, 10), textcoords="offset points",
                 fontproperties=FONT_BOLD, fontsize=13, color="#2196F3")

    # 下轴：回撤区间
    ax2.fill_between(dates, drawdown, 0, color="#FF5722", alpha=0.3)
    ax2.plot(dates, drawdown, color="#FF5722", linewidth=1)
    ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.3)
    ax2.set_ylabel("回撤", fontproperties=FONT_MD)
    ax2.set_xlabel("日期", fontproperties=FONT_MD)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.grid(alpha=0.3)

    # 标注最大回撤
    max_dd_idx = np.argmin(drawdown)
    max_dd_val = drawdown[max_dd_idx]
    ax2.annotate(f"最大回撤 {max_dd_val:.1%}",
                 xy=(max_dd_idx, max_dd_val),
                 xytext=(10, -20), textcoords="offset points",
                 fontproperties=FONT_SM, color="#D32F2F",
                 arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.2))

    # 格式化 x 轴
    _format_date_axis(ax1, dates)
    _format_date_axis(ax2, dates)

    plt.tight_layout()
    fig.savefig(CHART_DIR / "net_value_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 月度收益柱状图 ──
    monthly_rets = metrics.get("monthly_returns", [])
    monthly_labels = metrics.get("_monthly_labels", [])
    if monthly_rets and monthly_labels:
        _plot_monthly_bars(monthly_labels, monthly_rets, metrics)

    print("  ✅ net_value_curve.png + monthly_bars.png")


def _get_index_nav(start_date: str, end_date: str) -> tuple:
    """获取沪深300同期净值。"""
    try:
        conn = sqlite3.connect("data/sequoia_v2.db")
        rows = conn.execute(
            "SELECT date, close FROM index_daily WHERE symbol='sh.000300' "
            "AND date >= ? AND date <= ? ORDER BY date",
            (start_date, end_date)
        ).fetchall()
        conn.close()
        if rows:
            dates = [r[0] for r in rows]
            closes = np.array([r[1] for r in rows], dtype=float)
            return closes, dates
    except Exception as e:
        print(f"  ⚠ 获取指数数据失败: {e}")
    return np.array([]), []


def _add_market_annotations(ax, dates: list, nav: np.ndarray) -> None:
    """在净值曲线上标注关键市场事件。"""
    events = {
        "2025-09-24": "9/24新政",
        "2026-03-01": "3月反弹",
        "2026-05-01": "5月崩盘(y2=-42%)",
    }
    for date_str, label in events.items():
        # 找最近的交易日
        for i, d in enumerate(dates):
            if d >= date_str:
                if i < len(nav):
                    ax.annotate(label, xy=(i, nav[i]),
                                xytext=(0, 15), textcoords="offset points",
                                fontproperties=FONT_SM, fontsize=7,
                                color="#333", ha="center",
                                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
                break


def _format_date_axis(ax, dates: list) -> None:
    """格式化日期轴，只显示关键刻度。"""
    n = len(dates)
    if n <= 60:
        step = max(1, n // 10)
    else:
        step = max(1, n // 12)
    ticks = list(range(0, n, step))
    labels = [dates[i] for i in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, fontsize=7, ha="right")


def _plot_monthly_bars(labels: list[str], returns: list[float], metrics: dict) -> None:
    """月度收益柱状图。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#4CAF50" if r >= 0 else "#FF5722" for r in returns]
    bars = ax.bar(labels, returns, color=colors, edgecolor="white", linewidth=0.5)

    for bar, r in zip(bars, returns):
        y_pos = bar.get_height() + (0.01 if r >= 0 else -0.03)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{r:+.1%}", ha="center", fontproperties=FONT_SM, fontsize=8)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_title("月度收益率", fontproperties=FONT_TITLE, pad=10)
    ax.set_ylabel("收益率", fontproperties=FONT_MD)
    ax.set_xlabel("月份", fontproperties=FONT_MD)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)

    # 添加汇总标注
    total_ret = metrics.get("total_return", sum(returns))
    sharpe = metrics.get("sharpe", 0)
    ax.text(0.98, 0.95,
            f"总收益: {total_ret:+.1%}\n夏普: {sharpe:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontproperties=FONT_MD, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(CHART_DIR / "monthly_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════
#  综合评分热力图
# ═══════════════════════════════════════════════

def plot_heatmap(results: list[dict]) -> None:
    scored = compute_scores(results)
    periods = ["2025年", "2026年", "全周期"]
    top_ns = [10, 15, 20, 25]
    modes = ["M0", "M1", "M2", "M3", "M4", "M5"]

    data = np.zeros((len(top_ns) * len(periods), len(modes)))
    row_labels = []
    for ri, (tn, pd_) in enumerate([(tn, pd_) for tn in top_ns for pd_ in periods]):
        row_labels.append(f"T{tn} {pd_}")
        for ci, mode in enumerate(modes):
            matches = [r for r in scored if r["风控模式"] == mode
                       and r["TOP_N"] == tn and r["时段"] == pd_]
            data[ri, ci] = matches[0]["综合评分"] if matches else 0

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(len(row_labels)):
        for j in range(len(modes)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            best_in_row = np.max(data[i, :])
            marker = " ★" if val == best_in_row and best_in_row > 0 else ""
            ax.text(j, i, f"{val:.3f}{marker}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, fontsize=11)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title("72组综合评分热力图 ★ = 该行最优", fontproperties=FONT_TITLE, pad=15)
    plt.colorbar(im, ax=ax, label="综合评分")
    plt.tight_layout()
    fig.savefig(CHART_DIR / "heatmap_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ heatmap_scores.png")


# ═══════════════════════════════════════════════
#  TOP_N 对比
# ═══════════════════════════════════════════════

def plot_sharpe_by_topn(results: list[dict]) -> None:
    periods = ["2025年", "2026年", "全周期"]
    top_ns = [10, 15, 20, 25]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(top_ns))
    width = 0.25

    for pi, period in enumerate(periods):
        sharpes = []
        for tn in top_ns:
            subset = [r for r in results if r["TOP_N"] == tn and r["时段"] == period
                      and r.get("夏普比率") is not None]
            sharpe = max(r["夏普比率"] for r in subset) if subset else 0
            sharpes.append(sharpe)
        ax.bar(x + pi * width, sharpes, width, label=period, color=colors[pi],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"TOP_{tn}" for tn in top_ns], fontsize=11)
    ax.set_ylabel("夏普比率", fontproperties=FONT_MD)
    ax.set_title("各时段最优夏普比率", fontproperties=FONT_TITLE, pad=10)
    ax.legend(loc="upper left", prop=FONT_SM)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHART_DIR / "sharpe_by_topn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ sharpe_by_topn.png")


# ═══════════════════════════════════════════════
#  时段收益对比
# ═══════════════════════════════════════════════

def plot_period_returns(results: list[dict]) -> None:
    periods = ["2025年", "2026年", "全周期"]
    top_ns = [10, 15, 20, 25]
    colors = {10: "#2196F3", 15: "#4CAF50", 20: "#FF9800", 25: "#9C27B0"}

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(periods))
    width = 0.2

    for ti, tn in enumerate(top_ns):
        returns = []
        for period in periods:
            subset = [r for r in results if r["TOP_N"] == tn and r["时段"] == period
                      and r.get("总收益率") is not None]
            ret = max(r["总收益率"] for r in subset) if subset else 0
            returns.append(ret)
        ax.bar(x + ti * width, returns, width, label=f"TOP_{tn}",
               color=colors[tn], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(periods, fontsize=12)
    ax.set_ylabel("总收益率", fontproperties=FONT_MD)
    ax.set_title("各时段最优总收益率", fontproperties=FONT_TITLE, pad=10)
    ax.legend(loc="upper left", prop=FONT_SM)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHART_DIR / "period_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ period_returns.png")


# ═══════════════════════════════════════════════
#  风控模式对比
# ═══════════════════════════════════════════════

def plot_risk_mode_comparison(results: list[dict]) -> None:
    modes = ["M0", "M1", "M2", "M3", "M4", "M5"]
    metrics_map = {
        "年化收益率": ("#4CAF50", ":%"),
        "夏普比率": ("#2196F3", ":.2f"),
        "最大回撤": ("#FF5722", ":.2f"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ai, (metric, ax) in enumerate(zip(metrics_map.keys(), axes)):
        color, fmt = metrics_map[metric]
        values = []
        for mode in modes:
            subset = [r for r in results if r["风控模式"] == mode
                      and r.get(metric) is not None]
            values.append(np.mean([r[metric] for r in subset]) if subset else 0)

        bars = ax.bar(modes, values, color=color, edgecolor="white")
        ax.set_title(metric, fontproperties=FONT_MD, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            label = f"{val:+.2f}" if "比率" in metric or "回撤" in metric else f"{val:+.1%}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    label, ha="center", va="bottom", fontsize=8)

    fig.suptitle("风控模式效果对比（全周期均值）", fontproperties=FONT_TITLE)
    plt.tight_layout()
    fig.savefig(CHART_DIR / "risk_mode_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ risk_mode_comparison.png")


# ═══════════════════════════════════════════════
#  最优 vs 最差
# ═══════════════════════════════════════════════

def plot_best_vs_worst(results: list[dict]) -> None:
    scored = compute_scores(results)
    if len(scored) < 2:
        return
    best, worst = scored[0], scored[-1]
    # 四个指标分两组：前三"越高越好"，回撤"越低越好"（原始负值，越接近0越好）
    metrics_labels = ["年化收益率", "夏普比率", "月胜率", "最大回撤"]
    display_labels = ["年化收益率\n(越高越好)", "夏普比率\n(越高越好)",
                      "月胜率\n(越高越好)", "最大回撤\n(越低越好→)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_labels))
    width = 0.35

    best_vals = [best.get(m, 0) for m in metrics_labels]
    worst_vals = [worst.get(m, 0) for m in metrics_labels]

    bars_green = ax.bar(x - width / 2, best_vals, width,
           label=f"最优: {best['风控模式']} T{best['TOP_N']} {best['时段']} (评分{best.get('综合评分', 0):.3f})",
           color="#4CAF50", edgecolor="white")
    bars_red = ax.bar(x + width / 2, worst_vals, width,
           label=f"最差: {worst['风控模式']} T{worst['TOP_N']} {worst['时段']} (评分{worst.get('综合评分', 0):.3f})",
           color="#FF5722", edgecolor="white")

    # 为每根柱子标注数值
    for bar, val in zip(bars_green, best_vals):
        y_pos = bar.get_height() + (0.02 if val >= 0 else -0.04)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.1%}" if abs(val) < 10 else f"{val:+.1f}" if abs(val) < 100 else f"{val:+.0%}",
                ha="center", fontsize=9, fontweight="bold", color="#2E7D32")
    for bar, val in zip(bars_red, worst_vals):
        y_pos = bar.get_height() + (0.02 if val >= 0 else -0.04)
        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                f"{val:+.1%}" if abs(val) < 10 else f"{val:+.1f}" if abs(val) < 100 else f"{val:+.0%}",
                ha="center", fontsize=9, fontweight="bold", color="#C62828")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.set_title("最优 vs 最差配置对比", fontproperties=FONT_TITLE, pad=10)
    ax.legend(loc="upper left", prop=FONT_SM, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(CHART_DIR / "best_vs_worst.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✅ best_vs_worst.png")


def main():
    parser = argparse.ArgumentParser(description="Sequoia-X V2 回测图表生成")
    parser.add_argument("--with-curve", action="store_true",
                        help="生成净值曲线（需运行完整单组回测）")
    parser.add_argument("--csv", type=str,
                        default=str(OUTPUT_DIR / "summary_all.csv"),
                        help="汇总CSV路径")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"❌ {csv_path} 不存在，请先运行回测")
        sys.exit(1)

    results = load_results(csv_path)
    print(f"加载 {len(results)} 组回测结果\n")

    # 净值曲线（独立于 summary CSV，需要实际运行回测）
    if args.with_curve:
        try:
            generate_net_value_curve()
        except Exception as e:
            print(f"  ⚠ 净值曲线生成失败: {e}")

    # 基于 summary CSV 的图表
    plot_heatmap(results)
    plot_sharpe_by_topn(results)
    plot_period_returns(results)
    plot_risk_mode_comparison(results)
    plot_best_vs_worst(results)

    print(f"\n图表已保存到: {CHART_DIR}/")


if __name__ == "__main__":
    main()
