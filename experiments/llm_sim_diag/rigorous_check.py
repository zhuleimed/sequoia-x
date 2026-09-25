"""对 pool_vs_llm.py 结论的稳健性复核（配对检验 + 数据完整性检查）。

pool_vs_llm.py 的结论"LLM 选中的票比它淘汰的差"必须先排除两个伪影，否则不可信：

  伪影 1 · 日期效应：小样本组（A 组 100 只）可能恰好集中在少数暴跌日。
      → 解法：**逐日配对**。每个信号日算 (组内均值 - 同日基准均值)，
        得到一条日度超额序列，再对这条序列做 t 检验。日效应被完全抵消。

  伪影 2 · 数据可得性偏差：远期收益（+5/+10）要求股票在 N 日后仍有行情，
        停牌/退市的票会缺失，可能让各组样本不一致。
      → 解法：统计各组的 NaN 率，并对只取"所有组都有数据"的公共样本重算。

另外把"全市场等权"基准与指数对齐，确认基准数字本身没算错。
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

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def build() -> tuple[pd.DataFrame, dict[str, dict[str, tuple[float, float]]], list[str], dict[str, int]]:
    con = sqlite3.connect(DB)
    cal = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2026-05-01' ORDER BY date", con
    )["date"].tolist()
    cal_idx = {d: i for i, d in enumerate(cal)}
    bars = pd.read_sql(
        "SELECT symbol, date, open, close FROM stock_daily WHERE date >= '2026-05-01'", con
    )
    bars["open"] = pd.to_numeric(bars["open"], errors="coerce")
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["open", "close"])
    bars = bars[bars["open"] > 0]
    px = {s: dict(zip(g["date"], zip(g["open"], g["close"]))) for s, g in bars.groupby("symbol")}
    sig = pd.read_sql(
        "SELECT DISTINCT symbol, buy_date FROM sim_buy_signals WHERE status='executed'", con
    )
    con.close()
    llm_by_day: dict[str, set[str]] = {}
    for r in sig.itertuples(index=False):
        llm_by_day.setdefault(r.buy_date, set()).add(r.symbol)

    def fwd(sym: str, d0: str, n: int) -> float:
        i = cal_idx.get(d0)
        if i is None or i + 1 >= len(cal):
            return np.nan
        bd = cal[i + 1]
        d = px.get(sym, {})
        o = d.get(bd, (np.nan, np.nan))[0]
        j = cal_idx[bd] + n
        if not (o > 0) or j >= len(cal):
            return np.nan
        c = d.get(cal[j], (np.nan, np.nan))[1]
        return c / o - 1.0 if c > 0 else np.nan

    rows = []
    for fp in sorted(glob.glob(RESULTS_GLOB)):
        d = json.load(open(fp, encoding="utf-8"))
        sd = d["date"]
        if sd not in cal_idx:
            continue
        freq: Counter = Counter()
        pool: list[str] = []
        for syms in (v for v in d["strategies"].values() if v):
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
            "B2_池内淘汰_全池": sorted(set(pool) - picked - top10),
            "C_市场其余": [s for i, s in enumerate(px) if s not in set(pool) and i % 5 == 0],
        }
        for gname, syms in groups.items():
            for s in syms:
                if s not in px:
                    continue
                rec = {"date": sd, "group": gname, "symbol": s}
                for n in HORIZONS:
                    rec[f"r{n}"] = fwd(s, sd, n)
                rows.append(rec)
    return pd.DataFrame(rows), px, cal, cal_idx


def main() -> int:
    df, px, cal, _ = build()
    print("=" * 104)
    print(f"样本 {len(df)} 条 | 信号日 {df['date'].nunique()} 天 | "
          f"{df['date'].min()} ~ {df['date'].max()}")
    print("=" * 104)

    print("\n【一】数据可得性（NaN 率）——排除远期数据缺失造成的样本漂移")
    nan_tbl = df.groupby("group")[[f"r{n}" for n in HORIZONS]].apply(
        lambda g: g.isna().mean()
    )
    print(nan_tbl.to_string())

    print("\n【二】逐日配对检验 —— 每个信号日内先做差，再做 t 检验（消除日期效应）")
    print("     差值 = 组A当日均值 − 组B当日均值（单位：收益率）")
    for base, lbl in (("B_池内淘汰_top10", "LLM 看过的其他候选(top10 内)"),
                      ("B2_池内淘汰_全池", "候选池中未进 top10 的"),
                      ("C_市场其余", "全市场其余股票")):
        piv = df[df["group"].isin(["A_LLM选中", base])].pivot_table(
            index="date", columns="group", values=[f"r{n}" for n in HORIZONS], aggfunc="mean"
        )
        print(f"\n  ▸ A_LLM选中  vs  {lbl}")
        lines = []
        for n in HORIZONS:
            a = piv[("r%d" % n, "A_LLM选中")]
            b = piv[("r%d" % n, base)]
            diff = (a - b).dropna()
            if len(diff) < 5:
                continue
            t, p = stats.ttest_1samp(diff, 0.0)
            lines.append({
                "持有期": f"+{n}日", "可比天数": len(diff),
                "日均差值": diff.mean(), "中位差值": diff.median(),
                "t值": t, "p值": p, "显著": "是" if p < 0.05 else "否",
            })
        print(pd.DataFrame(lines).to_string(index=False))

    print("\n【三】候选池 vs 全市场（同样是逐日配对）——8 个策略本身有没有选股能力")
    piv = df[df["group"].isin(["B_池内淘汰_top10", "C_市场其余"])].pivot_table(
        index="date", columns="group", values=[f"r{n}" for n in HORIZONS], aggfunc="mean"
    )
    lines = []
    for n in HORIZONS:
        diff = (piv[("r%d" % n, "B_池内淘汰_top10")] - piv[("r%d" % n, "C_市场其余")]).dropna()
        t, p = stats.ttest_1samp(diff, 0.0)
        lines.append({"持有期": f"+{n}日", "可比天数": len(diff), "日均差值": diff.mean(),
                      "t值": t, "p值": p, "显著": "是" if p < 0.05 else "否"})
    print(pd.DataFrame(lines).to_string(index=False))

    print("\n【四】全市场等权基准（每日截面均值再跨日平均）——核对基准量级")
    mkt = df[df["group"] == "C_市场其余"].groupby("date")[[f"r{n}" for n in HORIZONS]].mean()
    idx = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol='sh.000300' AND date>='2026-05-01' ORDER BY date",
        sqlite3.connect(DB),
    )
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    print("  全市场等权（日均）:", {f"+{n}日": f"{mkt[f'r{n}'].mean()*100:+.3f}%" for n in HORIZONS})
    print(f"  沪深300 区间: {idx['close'].iloc[0]:.0f} → {idx['close'].iloc[-1]:.0f} "
          f"({(idx['close'].iloc[-1]/idx['close'].iloc[0]-1)*100:+.2f}%)")

    print("\n【五】公共样本复核 —— 只保留同一信号日内 A/B 两组都有数据的配对（消除样本不同）")
    sub = df[df["group"].isin(["A_LLM选中", "B_池内淘汰_top10"])].copy()
    cnt = sub.groupby(["date", "group"]).size().unstack(fill_value=0)
    ok_days = cnt[(cnt.get("A_LLM选中", 0) > 0) & (cnt.get("B_池内淘汰_top10", 0) > 0)].index
    sub = sub[sub["date"].isin(ok_days)]
    p2 = sub.pivot_table(index="date", columns="group", values=[f"r{n}" for n in HORIZONS], aggfunc="mean")
    print(pd.DataFrame([{
        "持有期": f"+{n}日",
        "A均值": p2[("r%d" % n, "A_LLM选中")].mean(),
        "B均值": p2[("r%d" % n, "B_池内淘汰_top10")].mean(),
        "配对天数": p2[("r%d" % n, "A_LLM选中")].notna().sum(),
    } for n in HORIZONS]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
