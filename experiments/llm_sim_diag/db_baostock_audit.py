"""stock_daily 与 baostock 的不复权价系统对拍（数据质量测绘）。

背景
----
归因诊断中发现 stock_daily 存在价格异常。最初以为是"2026-06-09 单日错乱"，
但对拍 000813 发现真相相反：
    DB   2026-05-25~06-08：9.85 / 9.71 / 9.48 ...
    baostock(不复权) 同日：3.36 / 3.28 / 3.32 ...
    DB   2026-06-09 起   ：3.07 / 3.13 / 3.05 ...   ← 与 baostock 一致
即：**6/9 之前的数据被整体放大（本例 ×2.889），6/9 起才是对的**。
"跳变"只是两段数据口径不同的交界，不是单日事故。

这说明该问题不能靠"抓单日跳变"定位——必须**直接与独立源对拍**。

本脚本做什么
------------
随机抽样 N 只股票（沪深各半），对每个抽样日拿 baostock 不复权收盘价，
与库内值算比值 DB/baostock，统计比值偏离 1 的比例与分布，按日期/交易所分组。

判读
----
比值 ≈ 1.0000 → 该日该股票库内值正确
比值明显 ≠ 1（如 2.89、0.35）→ 口径不一致或数据错误
"""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "sequoia_v2.db"

# 抽样日期：跨年份、跨月度，覆盖 2020~2026 全区间，用于定位"口径分界"
SAMPLE_DATES = [
    "2022-12-09", "2023-06-09", "2023-12-11", "2024-06-07",
    "2024-12-10", "2025-03-10", "2025-06-10", "2025-09-10",
    "2026-01-09", "2026-03-10", "2026-05-08", "2026-06-05",
    "2026-06-09", "2026-06-16", "2026-07-10",
]
N_STOCKS = 40          # 每类抽多少只（深市 / 沪市各自）
TOL = 0.005            # 比值容差

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main() -> int:
    con = sqlite3.connect(DB)
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM stock_daily WHERE date >= '2025-06-01'").fetchall()]
    con.close()
    deep = sorted([s for s in syms if s.startswith(("0", "3"))])
    sh = sorted([s for s in syms if s.startswith(("6",))])
    rng = random.Random(20260925)
    sample = rng.sample(deep, min(N_STOCKS, len(deep))) + rng.sample(sh, min(N_STOCKS, len(sh)))
    print(f"抽样 {len(sample)} 只（深市 {min(N_STOCKS,len(deep))} + 沪市 {min(N_STOCKS,len(sh))}）")

    import baostock as bs
    bs.login()
    rows = []
    try:
        for i, sym in enumerate(sample, 1):
            code = f"sz.{sym}" if sym.startswith(("0", "3")) else f"sh.{sym}"
            try:
                rs = bs.query_history_k_data_plus(
                    code, "date,close", start_date=SAMPLE_DATES[0], end_date=SAMPLE_DATES[-1],
                    frequency="d", adjustflag="3")
                if rs.error_code != "0":
                    continue
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
                if not data:
                    continue
                bdf = pd.DataFrame(data, columns=rs.fields)
                bdf["close"] = pd.to_numeric(bdf["close"], errors="coerce")
                rows.append(bdf.assign(symbol=sym))
            except Exception as e:
                print(f"  {sym}: {e}")
            if i % 20 == 0:
                print(f"  ... {i}/{len(sample)}")
    finally:
        bs.logout()

    bs_all = pd.concat(rows, ignore_index=True)
    con = sqlite3.connect(DB)
    local = pd.read_sql(
        "SELECT symbol, date, close FROM stock_daily WHERE date >= ? ORDER BY symbol, date",
        con, params=(SAMPLE_DATES[0],))
    con.close()
    local["close"] = pd.to_numeric(local["close"], errors="coerce")
    local = local[local["date"].isin(SAMPLE_DATES)]

    m = bs_all.merge(local, on=["symbol", "date"], suffixes=("_bs", "_db"))
    m = m[(m["close_bs"] > 0) & (m["close_db"] > 0)]
    m["ratio"] = m["close_db"] / m["close_bs"]
    m["exch"] = np.where(m["symbol"].str.startswith(("0", "3")), "深市", "沪市")

    print("\n" + "=" * 100)
    print("按抽样日：库内值 / baostock 不复权值 的比值分布")
    print("=" * 100)
    out = []
    for d, g in m.groupby("date"):
        off = (g["ratio"] - 1).abs() > TOL
        out.append({
            "日期": d, "样本": len(g),
            "比值≈1占比": (~off).mean(),
            "比值中位": g["ratio"].median(),
            "异常(比值≠1)": int(off.sum()),
            "深市异常": int((off & (g["exch"] == "深市")).sum()),
            "沪市异常": int((off & (g["exch"] == "沪市")).sum()),
        })
    print(pd.DataFrame(out).to_string(index=False))

    print("\n按交易所汇总：")
    for e, g in m.groupby("exch"):
        bad = ((g["ratio"] - 1).abs() > TOL)
        print(f"  {e}: 样本 {len(g)}, 异常 {bad.sum()} ({bad.mean()*100:.1f}%)")

    bad = m[(m["ratio"] - 1).abs() > TOL]
    if len(bad):
        print(f"\n异常样本的比值分布（{len(bad)} 条）：")
        print(bad["ratio"].describe(percentiles=[.05, .25, .5, .75, .95]).to_string())
        print("\n抽样 12 条异常明细：")
        print(bad.head(12)[["symbol", "date", "close_bs", "close_db", "ratio"]].to_string(index=False))

    out_path = PROJECT / "experiments" / "llm_sim_diag" / "db_baostock_audit.csv"
    m.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
