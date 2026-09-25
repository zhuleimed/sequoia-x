"""最终版诊断：剔除 2026-07-06 / 07-07 的坏数据后重做全部对照。

★ 数据污染事实（本次分析发现）
  stock_daily 在 2026-07-06、07-07 两天存在**价格尺度错误**：
  例：002371 于 07-03 收 816.00，07-06 该行却是 open 5.61/close 5.76（缩小约 141 倍），
  07-07 又回到 794.23/806.39。两天合计 1,316 行 |日涨幅|>20%，涉及 673 只股票。
  后果：这两天"全市场等权日均涨幅"被抬到 +20%，任何用它做基准的分析都会被带偏。

  本脚本的处理：把这两天的坏行标记为不可用，凡前瞻窗口跨越坏行的样本一律置 NaN。
"""

from __future__ import annotations

import glob
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "sequoia_v2.db"
RESULTS_GLOB = str(PROJECT / "data" / "results" / "results_2026*.json")
HORIZONS = (1, 3, 5, 10)
BAD_DAYS = ("2026-07-06", "2026-07-07")
JUMP = 0.25  # 单日 |涨幅| 超过 25% 视为坏行（A股涨跌停 20%，留余量）

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def load_prices(con: sqlite3.Connection):
    b = pd.read_sql(
        "SELECT symbol, date, open, close FROM stock_daily WHERE date >= '2026-05-01'", con
    )
    for c in ("open", "close"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna().sort_values(["symbol", "date"])
    b["prev"] = b.groupby("symbol")["close"].shift(1)
    # 坏行：已知坏日 且 跳变超过阈值
    bad = (b["date"].isin(BAD_DAYS)) & (
        (b["close"] / b["prev"] - 1).abs() > JUMP
    ) | ((b["date"].isin(BAD_DAYS)) & ((b["open"] / b["prev"] - 1).abs() > JUMP))
    b["corrupt"] = bad.fillna(False)
    px = {}
    for s, g in b.groupby("symbol"):
        px[s] = dict(zip(g["date"], zip(g["open"], g["close"], g["corrupt"])))
    n_bad = int(b["corrupt"].sum())
    return px, n_bad


def main() -> int:
    con = sqlite3.connect(DB)
    cal = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2026-05-01' ORDER BY date", con
    )["date"].tolist()
    cal_idx = {d: i for i, d in enumerate(cal)}
    px, n_bad = load_prices(con)
    sig = pd.read_sql(
        "SELECT DISTINCT symbol, buy_date FROM sim_buy_signals WHERE status='executed'", con
    )
    con.close()
    llm_by_day: dict[str, set[str]] = {}
    for r in sig.itertuples(index=False):
        llm_by_day.setdefault(r.buy_date, set()).add(r.symbol)

    def fwd(sym: str, d0: str, n: int) -> float:
        """T+1 开盘买入 → 第 n 日收盘。窗口内任一坏行/缺数据 → NaN。"""
        i = cal_idx.get(d0)
        if i is None or i + 1 >= len(cal):
            return np.nan
        bd = cal[i + 1]
        d = px.get(sym, {})
        j = cal_idx[bd] + n
        if j >= len(cal):
            return np.nan
        window = cal[cal_idx[bd]: j + 1]
        o = d.get(bd, (np.nan, np.nan, True))[0]
        c = d.get(cal[j], (np.nan, np.nan, True))[1]
        if not (o > 0) or not (c > 0):
            return np.nan
        if any(d.get(w, (0, 0, True))[2] for w in window):  # 窗口含坏行
            return np.nan
        return c / o - 1.0

    rows = []
    for fp in sorted(glob.glob(RESULTS_GLOB)):
        dd = json.load(open(fp, encoding="utf-8"))
        sd = dd["date"]
        if sd not in cal_idx:
            continue
        freq: Counter = Counter()
        pool: list[str] = []
        for syms in (v for v in dd["strategies"].values() if v):
            pool.extend(syms)
            freq.update(syms)
        pool = sorted(set(pool))
        if not pool:
            continue
        top10 = {s for s, _ in freq.most_common(10)}
        picked = llm_by_day.get(sd, set())
        groups = {
            "A_LLM选中": sorted(picked & set(pool)),
            "B_池内淘汰_top10": sorted((top10 & set(pool)) - picked),
            "C_市场其余": [s for i, s in enumerate(px) if s not in set(pool) and i % 5 == 0],
        }
        for gname, syms in groups.items():
            for s in syms:
                rec = {"date": sd, "group": gname, "symbol": s}
                for n in HORIZONS:
                    rec[f"r{n}"] = fwd(s, sd, n)
                rows.append(rec)
    df = pd.DataFrame(rows)

    print("=" * 104)
    print(f"坏行标记 {n_bad} 条（{BAD_DAYS[0]} / {BAD_DAYS[1]}，跳变>{JUMP:.0%}）")
    print(f"样本 {len(df)} 条 | 信号日 {df['date'].nunique()} 天 | "
          f"{df['date'].min()} ~ {df['date'].max()}")
    print("=" * 104)

    print("\n【一】各组平均 / 中位数收益（已剔除坏数据）")
    g = df.groupby("group")
    print(pd.DataFrame({
        "样本": g.size(),
        **{f"+{n}日均值": g[f"r{n}"].mean() for n in HORIZONS},
        **{f"+{n}日中位": g[f"r{n}"].median() for n in HORIZONS},
    }).to_string())

    print("\n【二】逐日配对检验（消除日期效应；差值 = A当日均值 − 基准当日均值）")
    for base, lbl in (("B_池内淘汰_top10", "LLM 看过的其他候选(top10 内)"),
                      ("C_市场其余", "全市场其余股票")):
        piv = df[df["group"].isin(["A_LLM选中", base])].pivot_table(
            index="date", columns="group", values=[f"r{n}" for n in HORIZONS], aggfunc="mean")
        out = []
        for n in HORIZONS:
            diff = (piv[("r%d" % n, "A_LLM选中")] - piv[("r%d" % n, base)]).dropna()
            if len(diff) < 5:
                continue
            t, p = stats.ttest_1samp(diff, 0.0)
            out.append({"持有期": f"+{n}日", "天数": len(diff), "差值均值": diff.mean(),
                        "差值中位": diff.median(), "t值": t, "p值": p,
                        "显著": "★是" if p < 0.05 else "否"})
        print(f"\n  ▸ A_LLM选中  vs  {lbl}")
        print(pd.DataFrame(out).to_string(index=False))

    print("\n【三】候选池 vs 全市场（逐日配对）")
    piv = df[df["group"].isin(["B_池内淘汰_top10", "C_市场其余"])].pivot_table(
        index="date", columns="group", values=[f"r{n}" for n in HORIZONS], aggfunc="mean")
    out = []
    for n in HORIZONS:
        diff = (piv[("r%d" % n, "B_池内淘汰_top10")] - piv[("r%d" % n, "C_市场其余")]).dropna()
        t, p = stats.ttest_1samp(diff, 0.0)
        out.append({"持有期": f"+{n}日", "天数": len(diff), "差值均值": diff.mean(),
                    "t值": t, "p值": p, "显著": "★是" if p < 0.05 else "否"})
    print(pd.DataFrame(out).to_string(index=False))

    print("\n【四】全市场等权基准（逐日截面中位数，抗异常值）")
    m = df[df["group"] == "C_市场其余"].groupby("date")[[f"r{n}" for n in HORIZONS]].median()
    for n in HORIZONS:
        print(f"  +{n}日 日中位数的均值: {m[f'r{n}'].mean()*100:+.3f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
