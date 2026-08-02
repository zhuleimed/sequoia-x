"""逐月 T2/T4 Rank IC 分析（2026-08-02）

用途：验证"温和市 T2 强、极端市 T4 强"假设（BACKTEST_PLAN §25.4 第 1 步），
为 2025 年收益优化方案（滚动 IC 加权）提供数据依据。

方法：
  对 prediction_cache.json 的每个月份（70 个月）：
  1. 取当月测试日（该月最后一个交易日）的 T2/T4 预测
  2. 从 stock_daily 计算每只股票未来 20 个交易日的收益（y2 标签同款定义）
  3. 从 index_daily 计算沪深 300 同期收益 → 超额收益 y2
  4. Rank IC = Spearman(预测, y2_actual)（当月全部股票）
  5. 市场状态标签：按沪深300 当月收益分档（温和/极端）

输出：
  output/backtest_v2/monthly_ic_analysis.csv + 控制台汇总表
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "output/backtest_v2/prediction_cache.json"
DB_PATH = PROJECT_ROOT / "data/sequoia_v2.db"
OUTPUT_CSV = PROJECT_ROOT / "output/backtest_v2/monthly_ic_analysis.csv"

LOOKBACK_DAYS = 20  # y2 窗口：未来 20 个交易日


def load_prices(conn, symbols, start_date):
    """批量加载股票价格（start_date 起，含未来 20 日）。"""
    ph = ",".join("?" * len(symbols))
    df = pd.read_sql(
        f"SELECT symbol, date, close FROM stock_daily "
        f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
        conn, params=symbols + [start_date],
    )
    return df


def next_trading_dates(dates_arr, idx, n):
    """从 dates_arr[idx] 起取第 n 个交易日的日期。"""
    if idx + n < len(dates_arr):
        return dates_arr[idx + n]
    return None


def analyze_month(month, cache_entry, conn, all_dates_idx):
    """分析单个月份：返回 {month, n, t2_ic, t4_ic, y2_mean, idx_ret, market_state}。"""
    symbols = cache_entry["symbols"]
    t2_pred = np.array(cache_entry["t2"])
    t4_pred = np.array(cache_entry["t4"])

    # 当月最后一个交易日（测试日 T）
    ym = month
    last_date = conn.execute(
        "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?", (ym + "%",)
    ).fetchone()[0]
    if last_date is None:
        return None

    # T 日全市场日期序列（用于找 T+20）
    dates_arr = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= ? ORDER BY date",
        (last_date,),
    ).fetchall()]
    # 找到 T 在日期序列中的位置（需往前找 T 自身及之前的对齐）
    # 简化：从 stock_daily 查 T 日的所有股票收盘 + 未来第 20 个交易日收盘
    ph = ",".join("?" * len(symbols))
    prices = pd.read_sql(
        f"SELECT symbol, date, close FROM stock_daily "
        f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
        conn, params=symbols + [last_date],
    )
    # 沪深 300 指数
    idx_df = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol='sh.000300' AND date >= ? ORDER BY date",
        conn, params=(last_date,),
    )

    # 构建 T 日收盘价和 T+20 收盘价
    t_close = {}
    t20_close = {}
    dates_sorted = sorted(set(prices["date"]))
    for sym, g in prices.groupby("symbol"):
        g = g.sort_values("date")
        row_t = g[g["date"] == last_date]
        if row_t.empty:
            continue
        t_close[sym] = float(row_t["close"].iloc[0])
        # 未来第 20 个交易日（不含 T 日，取 T 之后第 20 个）
        future = g[g["date"] > last_date]
        if len(future) >= LOOKBACK_DAYS:
            t20_close[sym] = float(future["close"].iloc[LOOKBACK_DAYS - 1])

    # 指数同期收益
    idx_t = idx_df[idx_df["date"] == last_date]
    if idx_t.empty:
        return None
    idx_t_close = float(idx_t["close"].iloc[0])
    idx_future = idx_df[idx_df["date"] > last_date]
    if len(idx_future) < LOOKBACK_DAYS:
        return None
    idx_t20_close = float(idx_future["close"].iloc[LOOKBACK_DAYS - 1])
    idx_ret = idx_t20_close / idx_t_close - 1

    # 计算 y2（超额收益）
    syms, y2s, t2v, t4v = [], [], [], []
    for sym in symbols:
        if sym in t_close and sym in t20_close:
            stock_ret = t20_close[sym] / t_close[sym] - 1
            y2 = np.clip(stock_ret - idx_ret, -0.5, 0.5)
            syms.append(sym)
            y2s.append(y2)
            t2v.append(t2_pred[symbols.index(sym)])
            t4v.append(t4_pred[symbols.index(sym)])

    if len(syms) < 50:
        return None
    y2_arr = np.array(y2s)
    t2_ic, _ = spearmanr(t2v, y2_arr)
    t4_ic, _ = spearmanr(t4v, y2_arr)

    # 市场状态：按沪深300 当月收益分档
    if idx_ret > 0.03:
        state = "牛市"
    elif idx_ret > 0:
        state = "温和涨"
    elif idx_ret > -0.03:
        state = "温和跌"
    elif idx_ret > -0.10:
        state = "熊市"
    else:
        state = "极端熊"

    return {
        "month": month,
        "n": len(syms),
        "t2_ic": round(t2_ic, 4),
        "t4_ic": round(t4_ic, 4),
        "y2_mean": round(float(y2_arr.mean()), 4),
        "idx_ret": round(float(idx_ret), 4),
        "market_state": state,
    }


def main():
    print("=" * 70)
    print("  逐月 T2/T4 Rank IC 分析")
    print("  数据: prediction_cache.json | y2: 未来20日超额收益(沪深300基准)")
    print("=" * 70)

    cache = json.loads(CACHE_PATH.read_text())
    months = sorted(cache.keys())
    print(f"缓存月份: {len(months)}")

    conn = sqlite3.connect(DB_PATH)
    results = []
    for i, m in enumerate(months):
        r = analyze_month(m, cache[m], conn, None)
        if r:
            results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(months)}")
    conn.close()

    if not results:
        print("无有效结果！")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n结果已保存: {OUTPUT_CSV}\n")

    # ── 汇总表 ──
    pd.set_option("display.width", 100)
    pd.set_option("display.max_rows", 200)
    print(df[["month", "n", "t2_ic", "t4_ic", "y2_mean", "idx_ret", "market_state"]].to_string(index=False))

    # ── 按年度汇总 ──
    df["year"] = df["month"].str[:4]
    print("\n" + "=" * 70)
    print("  按年度汇总")
    print("=" * 70)
    for y, g in df.groupby("year"):
        print(f"  {y} ({len(g)}月): T2 IC 均值={g['t2_ic'].mean():+.4f} | "
              f"T4 IC 均值={g['t4_ic'].mean():+.4f} | "
              f"T2胜出月份={sum(g['t2_ic'] > g['t4_ic'])}/{len(g)} | "
              f"y2均值={g['y2_mean'].mean():+.4f}")

    # ── 按市场状态汇总（验证假设）──
    print("\n" + "=" * 70)
    print("  按市场状态汇总（验证'温和市T2强、极端市T4强'假设）")
    print("=" * 70)
    for s, g in df.groupby("market_state"):
        print(f"  {s} ({len(g)}月): T2 IC={g['t2_ic'].mean():+.4f} | "
              f"T4 IC={g['t4_ic'].mean():+.4f} | "
              f"T2胜出={sum(g['t2_ic'] > g['t4_ic'])}/{len(g)}")

    # ── 2025年（优化目标）特写 ──
    print("\n" + "=" * 70)
    print("  2025-08~2026-06（原 11 个月回测区间）特写")
    print("=" * 70)
    focus = df[df["month"].between("2025-08", "2026-06")]
    print(focus[["month", "t2_ic", "t4_ic", "y2_mean", "market_state"]].to_string(index=False))
    if len(focus):
        print(f"  T2 胜出: {sum(focus['t2_ic'] > focus['t4_ic'])}/{len(focus)} | "
              f"T2 IC={focus['t2_ic'].mean():+.4f} | T4 IC={focus['t4_ic'].mean():+.4f}")


if __name__ == "__main__":
    main()
