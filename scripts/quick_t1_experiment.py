"""T1 方向可学性快速实证（2026-08-02）

背景：T1（5日方向分类）70 个月 AUC=0.499 无预测能力。改造前先验证
"方向分类任务本身是否可学"——用**现有模型预测**（无需重训）评估：

  A. AUC(sign(t2_pred), sign(y2))   —— 20 日超额方向可学性上限
     （T2 回归是当前最优超额收益模型；若其方向化 AUC≈0.5，则方向分类任务本质不可学）
  B. AUC(t1_prob, y1_10d)           —— T1 换 10 日方向标签的泛化
  C. AUC(t1_prob, y1_20d)           —— T1 换 20 日方向标签的泛化

输出：70 个月逐月 AUC + 汇总。
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


def load_prices_future(conn, symbols, last_date, horizon):
    """T 日收盘 + 未来第 horizon 交易日收盘。返回 {sym: (t_close, fut_close)}。"""
    ph = ",".join("?" * len(symbols))
    prices = pd.read_sql(
        f"SELECT symbol, date, close FROM stock_daily "
        f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
        conn, params=symbols + [last_date],
    )
    out = {}
    for sym, g in prices.groupby("symbol"):
        g = g.sort_values("date")
        row_t = g[g["date"] == last_date]
        future = g[g["date"] > last_date]
        if row_t.empty or len(future) < horizon:
            continue
        out[sym] = (float(row_t["close"].iloc[0]), float(future["close"].iloc[horizon - 1]))
    return out


def main():
    cache = json.loads(CACHE_PATH.read_text())
    months = sorted(cache.keys())
    conn = sqlite3.connect(DB_PATH)
    rows = []

    for i, m in enumerate(months):
        e = cache[m]
        symbols = e["symbols"]
        t2 = np.array(e["t2"])
        t1 = np.array(e["t1"])

        last_date = conn.execute(
            "SELECT MAX(date) FROM stock_daily WHERE date LIKE ?", (m + "%",)
        ).fetchone()[0]
        if last_date is None:
            continue

        # 指数（y2 基准）
        idx_df = pd.read_sql(
            "SELECT date, close FROM index_daily WHERE symbol='sh.000300' AND date >= ? ORDER BY date",
            conn, params=(last_date,),
        )
        idx_t = idx_df[idx_df["date"] == last_date]
        if idx_t.empty:
            continue

        p5 = load_prices_future(conn, symbols, last_date, 5)
        p10 = load_prices_future(conn, symbols, last_date, 10)
        p20 = load_prices_future(conn, symbols, last_date, 20)

        # 指数未来 20 日收益
        idx_future = idx_df[idx_df["date"] > last_date]
        idx_ret = None
        if len(idx_future) >= 20:
            idx_ret = float(idx_future["close"].iloc[19]) / float(idx_t["close"].iloc[0]) - 1

        # 组装各标签
        syms = [s for s in symbols if s in p5 and s in p10 and s in p20]
        if len(syms) < 100:
            continue
        idx_map = {s: i for i, s in enumerate(symbols)}

        y1_5d = np.array([1 if p5[s][1] > p5[s][0] else 0 for s in syms])
        y1_10d = np.array([1 if p10[s][1] > p10[s][0] else 0 for s in syms])
        y1_20d = np.array([1 if p20[s][1] > p20[s][0] else 0 for s in syms])
        if idx_ret is not None:
            sign_y2 = np.array([1 if p20[s][1] / p20[s][0] - 1 > idx_ret else 0 for s in syms])
        else:
            sign_y2 = None

        t2_s = np.array([t2[idx_map[s]] for s in syms])
        t1_s = np.array([t1[idx_map[s]] for s in syms])

        def safe_auc(y_true, proba):
            if len(set(y_true)) < 2:
                return None
            try:
                return roc_auc_score(y_true, proba)
            except Exception:
                return None

        auc_a = safe_auc(sign_y2, t2_s) if sign_y2 is not None else None
        auc_b = safe_auc(y1_10d, t1_s)
        auc_c = safe_auc(y1_20d, t1_s)
        rows.append({
            "month": m,
            "A_超额方向T2": round(auc_a, 4) if auc_a else None,
            "B_10日T1": round(auc_b, 4) if auc_b else None,
            "C_20日T1": round(auc_c, 4) if auc_c else None,
            "n": len(syms),
        })
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(months)}")

    conn.close()

    df = pd.DataFrame(rows)
    df.to_csv(PROJECT_ROOT / "output/backtest_v2/t1_direction_experiment.csv", index=False)

    print("\n" + "=" * 60)
    print("  方向可学性实证（70 个月均值）")
    print("=" * 60)
    for col, desc in [("A_超额方向T2", "20日超额方向（T2回归方向化）"),
                      ("B_10日T1", "10日方向（T1泛化）"),
                      ("C_20日T1", "20日方向（T1泛化）")]:
        vals = df[col].dropna()
        print(f"  {desc:28}: 均值={vals.mean():.4f} | >0.55: {sum(vals>0.55)}/{len(vals)} | <0.50: {sum(vals<0.50)}/{len(vals)}")
    print("\n  近 12 个月明细:")
    recent = df[df["month"] >= "2025-07"]
    print(recent[["month", "A_超额方向T2", "B_10日T1", "C_20日T1"]].to_string(index=False))


if __name__ == "__main__":
    main()
