"""全周期累积收益率曲线 vs 沪深300（含最大回撤区间标识）（2026-08-02）

- 重跑全周期回测（M4+TOP_N=10，缓存模式）获取日级别净值
- 计算策略累积收益率、沪深300 同期累积收益率
- 精确计算最大回撤区间（峰值日→谷底日），图上阴影标识
- 输出: output/backtest_v2/charts/full_period_return_curve.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sequoia_x.core.config import Settings
from sequoia_x.data.engine import DataEngine
from sequoia_x.model_selection_v2.config import get_config
from sequoia_x.model_selection_v2.backtest.monthly_engine import MonthlyBacktestEngine

OUTPUT_DIR = Path("output/backtest_v2")
CHART_DIR = OUTPUT_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(__file__).resolve().parent.parent / "data/sequoia_v2.db"

# CJK 字体
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_fm = matplotlib.font_manager
_font_cache_dir = matplotlib.get_cachedir()
for _cache_file in Path(_font_cache_dir).glob("fontlist-v*.json"):
    _cache_file.unlink(missing_ok=True)
_fm.fontManager.addfont(_FONT_PATH)
_FONT_FAMILY = _fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [_FONT_FAMILY, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    print("重跑全周期回测获取日级别净值（M4+TOP_N=10, 2020-09~2026-06）...")
    cfg = get_config()
    engine = DataEngine(Settings())
    with open(OUTPUT_DIR / "prediction_cache.json") as f:
        cache = json.load(f)

    bt = MonthlyBacktestEngine(
        cfg=cfg, engine=engine, top_n=10, risk_mode="M4",
        initial_capital=500_000, use_real_t4=False,
        prediction_cache=cache,
    )
    metrics = bt.run("2020-09", "2026-07")
    daily = metrics.get("daily_records", [])
    if not daily:
        print("❌ 无日级别记录")
        return

    dates = [r["date"] for r in daily]
    values = np.array([r["total_value"] for r in daily])
    nav = values / values[0]
    cum_ret = nav - 1  # 策略累积收益率

    # ── 最大回撤计算 ──
    running_max = np.maximum.accumulate(nav)
    drawdown = nav / running_max - 1
    trough_idx = int(np.argmin(drawdown))
    # 峰值 = trough 之前的最大值
    peak_idx = int(np.argmax(nav[:trough_idx + 1]))
    max_dd = drawdown[trough_idx]
    print(f"最大回撤: {max_dd:.2%} | 峰值日={dates[peak_idx]} → 谷底日={dates[trough_idx]}")

    # ── 沪深300 同期累积收益率 ──
    conn = sqlite3.connect(DB_PATH)
    idx_df = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol='sh.000300' "
        "AND date >= ? AND date <= ? ORDER BY date",
        conn, params=(dates[0], dates[-1]),
    )
    conn.close()
    idx_nav = idx_df["close"].values / idx_df["close"].iloc[0]
    idx_cum = idx_nav - 1
    idx_dates = idx_df["date"].tolist()
    # 对齐到策略日期（取交集附近）
    idx_map = dict(zip(idx_dates, idx_cum))
    idx_aligned = np.array([idx_map.get(d, np.nan) for d in dates])

    # ── 绘制（最初样式：主图累积收益率 + 回撤副图，只标注最大回撤）──
    x = np.arange(len(dates))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]},
                             sharex=True)
    ax1, ax2 = axes

    # 主图：累积收益率（%）
    ax1.plot(x, cum_ret * 100, label="策略（M4+TOP_N=10）", color="#C0392B", lw=1.8)
    ax1.plot(x, idx_aligned * 100, label="沪深300（同期）", color="#2980B9", lw=1.2, alpha=0.9)
    ax1.axhline(0, color="gray", lw=0.8, ls="--")

    # 最大回撤区间（峰值→谷底阴影 + 两点标记 + 箭头标注）
    ax1.axvspan(peak_idx, trough_idx, color="#E74C3C", alpha=0.18)
    ax1.scatter([peak_idx, trough_idx],
                [cum_ret[peak_idx] * 100, cum_ret[trough_idx] * 100],
                color=["#27AE60", "#C0392B"], s=45, zorder=5)
    ax1.annotate(
        f"峰值 {dates[peak_idx]}\n{cum_ret[peak_idx]*100:+.0f}%",
        xy=(peak_idx, cum_ret[peak_idx] * 100),
        xytext=(peak_idx - len(dates) * 0.14, cum_ret[peak_idx] * 100 + 25),
        arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2),
        fontsize=10.5, color="#27AE60", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#27AE60", alpha=0.9),
    )
    ax1.annotate(
        f"最大回撤 {max_dd:.1%}\n谷底 {dates[trough_idx]}",
        xy=(trough_idx, cum_ret[trough_idx] * 100),
        xytext=(trough_idx - len(dates) * 0.22, max(cum_ret) * 100 * 0.42),
        arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
        fontsize=11, color="#C0392B", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#C0392B", alpha=0.92),
    )
    ax1.plot([peak_idx, trough_idx],
             [cum_ret[peak_idx] * 100, cum_ret[trough_idx] * 100],
             color="#E74C3C", ls="--", lw=1.0, alpha=0.7)

    ax1.set_ylabel("累积收益率 (%)")
    ax1.set_title(f"Sequoia-X V2 全周期累积收益率 vs 沪深300（{dates[0]} ~ {dates[-1]}，共 {len(dates)} 交易日）",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(alpha=0.3)
    # 终值标注
    ax1.annotate(f"策略终值 {cum_ret[-1]:+.0%}",
                 xy=(len(dates) - 1, cum_ret[-1] * 100),
                 xytext=(len(dates) * 0.62, cum_ret[-1] * 100 * 0.92),
                 arrowprops=dict(arrowstyle="->", color="#C0392B"),
                 fontsize=11, color="#C0392B", fontweight="bold")
    ax1.annotate(f"沪深300 {idx_aligned[-1]:+.0%}",
                 xy=(len(dates) - 1, idx_aligned[-1] * 100),
                 xytext=(len(dates) * 0.62, idx_aligned[-1] * 100 * 2.2),
                 arrowprops=dict(arrowstyle="->", color="#2980B9"),
                 fontsize=11, color="#2980B9", fontweight="bold")

    # 副图：回撤曲线
    ax2.fill_between(x, drawdown * 100, 0, color="#E74C3C", alpha=0.35)
    ax2.plot(x, drawdown * 100, color="#C0392B", lw=0.9)
    ax2.axvspan(peak_idx, trough_idx, color="#E74C3C", alpha=0.2)
    ax2.set_ylabel("回撤 (%)")
    ax2.set_ylim(min(drawdown) * 100 * 1.3, 2)
    ax2.grid(alpha=0.3)
    ax2.annotate(f"{max_dd:.1%}", xy=(trough_idx, drawdown[trough_idx] * 100),
                 fontsize=11, color="#C0392B", fontweight="bold")

    # X 轴：年份刻度
    years = []
    prev = None
    for i, d in enumerate(dates):
        y = d[:4]
        if y != prev:
            years.append((i, y))
            prev = y
    ax2.set_xticks([i for i, _ in years])
    ax2.set_xticklabels([y for _, y in years])

    plt.tight_layout()
    out = CHART_DIR / "full_period_return_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 已保存: {out}")


if __name__ == "__main__":
    main()
