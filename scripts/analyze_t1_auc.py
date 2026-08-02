"""T1 模型真实 AUC 分析（2026-08-02）

目的：核查 T1 方向过滤器为什么从未生效（M0=M1）。
方法：对 prediction_cache 的每个月份，用缓存 t1_prob（5日看涨概率）vs
实际 y1（T 日后第 5 个交易日是否上涨）计算 roc_auc_score。

输出：70 个月 AUC 分布 + 汇总（多少月份 > 0.58 阈值）。
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_ROOT / "output/backtest_v2/prediction_cache.json"
DB_PATH = PROJECT_ROOT / "data/sequoia_v2.db"
OUTPUT_CSV = PROJECT_ROOT / "output/backtest_v2/t1_auc_analysis.csv"


def main():
    cache = json.loads(CACHE_PATH.read_text())
    months = sorted(cache.keys())
    print(f"缓存月份: {len(months)}")

    conn = sqlite3.connect(DB_PATH)
    results = []
    for i, m in enumerate(months):
        e = cache[m]
        symbols = e["symbols"]
        t1_prob = np.array(e["t1"])

        last_date = conn.execute(
            "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?", (m + "%",)
        ).fetchone()[0]
        if last_date is None:
            continue

        # T 日收盘与未来第 5 个交易日收盘
        ph = ",".join("?" * len(symbols))
        prices = pd.read_sql(
            f"SELECT symbol, date, close FROM stock_daily "
            f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
            conn, params=symbols + [last_date],
        )
        y1s, probs = [], []
        for sym, g in prices.groupby("symbol"):
            g = g.sort_values("date")
            row_t = g[g["date"] == last_date]
            future = g[g["date"] > last_date]
            if row_t.empty or len(future) < 5:
                continue
            y1 = 1 if float(future["close"].iloc[4]) > float(row_t["close"].iloc[0]) else 0
            y1s.append(y1)
            probs.append(t1_prob[symbols.index(sym)])

        if len(y1s) < 50 or len(set(y1s)) < 2:
            continue
        try:
            auc = roc_auc_score(y1s, probs)
        except Exception:
            continue
        results.append({"month": m, "auc": round(auc, 4), "n": len(y1s),
                        "up_ratio": round(np.mean(y1s), 3)})
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(months)}")

    conn.close()

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n结果已保存: {OUTPUT_CSV}\n")

    # 汇总
    aucs = df["auc"]
    print("=" * 60)
    print(f"  70 个月 T1 AUC 统计")
    print("=" * 60)
    print(f"  均值: {aucs.mean():.4f} | 中位数: {aucs.median():.4f} | "
          f"标准差: {aucs.std():.4f}")
    print(f"  >0.58 (阈值): {sum(aucs > 0.58)}/{len(aucs)} 个月 ({sum(aucs > 0.58)/len(aucs):.0%})")
    print(f"  >0.55: {sum(aucs > 0.55)} 个月 | >0.52: {sum(aucs > 0.52)} 个月")
    print(f"  <0.50 (反向): {sum(aucs < 0.50)} 个月")
    print()
    print("  按年度:")
    df["year"] = df["month"].str[:4]
    for y, g in df.groupby("year"):
        print(f"    {y}: AUC 均值={g['auc'].mean():.4f} | "
              f">0.58 月份={sum(g['auc'] > 0.58)}/{len(g)}")
    print()
    print("  近 12 个月明细 (2025-07~2026-06):")
    recent = df[df["month"] >= "2025-07"]
    for _, r in recent.iterrows():
        flag = "✅" if r["auc"] > 0.58 else ("⚠️" if r["auc"] > 0.52 else "❌")
        print(f"    {r['month']}: AUC={r['auc']:.4f} 上涨占比={r['up_ratio']:.0%} {flag}")


if __name__ == "__main__":
    main()
