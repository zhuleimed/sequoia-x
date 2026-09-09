#!/usr/bin/env python
"""窄条件候选(D+1 科创 / 机构净买+低开) 的时序单次 OOS 存活检验。

背景: 已在整1年窗里"看到"科创板块 D+1 有正超额、机构坐席净买相对更佳。此处用看得见的纪律测试它能否**存活到样本外**:
  1. 时序切分(事前定, 非事后挑最优点): IS=D<2026-05-01(观察/选门), OOS=D≥2026-05-01(只放这一次)
  2. 候选规则写死(不对OOS重调):  b=科创  OR  (机构净买 org_net>0 且 E1 开盘溢价 gap≤+3%)
       —— 即"上一档候选"的字面重述(科创面板 / 机构买低gap), 规则非从OOS里挖
  3. OOS(仅此一次)去验。若候选优势在 OOS 消失/反号 ⇒ 判该窄条件 alpha 属窗内现象(regime/幸存偏差), 不作为可选稳定因子。

对照 = 同窗口"全部可交易上榜"。口径 ex 相对沪深300, D+1 可买(剔一字/停牌/持有窗不全)。输出 holdout_narrow_cand.csv + 表。
caveat: OOS 仅~4个月单段、样本小; 结果=指示级(换血前观察), 不作定论。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
HOLD = [10, 20]


def stat_series(s):
    s = s.dropna()
    if not len(s):
        return (np.nan, np.nan, np.nan, 0)
    return s.mean() * 100, (s > 0).mean() * 100, s.median() * 100, len(s)


def F(x):
    return "  ." if (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"


def P(x):
    return "  ." if (isinstance(x, float) and np.isnan(x)) else f"{x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(OUT / "events_20260909.csv"))
    ap.add_argument("--cut", default="2026-05-01")
    a = ap.parse_args()
    ev = pd.read_csv(a.csv, dtype={"symbol": str})
    ev["symbol"] = ev["symbol"].str.zfill(6)
    sub = ev[ev["drop"].isna() & ev["E1"].notna()].copy()
    sub["D"] = pd.to_datetime(sub["D"])
    org = pd.to_numeric(sub["org_net"], errors="coerce")
    gap = pd.to_numeric(sub["gap"], errors="coerce")
    is_ke = sub["symbol"].str.startswith("688")
    is_inst_low = org.notna() & (org > 0) & gap.notna() & (gap <= 0.03)
    sub["cand"] = is_ke | is_inst_low

    ism = sub["D"] < pd.Timestamp(a.cut)
    print(f"规则(事前定死): b=科创 OR (机构净买 org>0 且 E1开盘gap≤+3%)\n切分 IS D<{a.cut} / OOS D≥{a.cut}")
    out = []
    for wlab, m in (("IS", ism), ("OOS", ~ism)):
        s = sub[m]
        for name, own in (("全部", np.ones(len(s), dtype=bool)), ("候选", s["cand"].values)):
            g = s[own]
            row = {"window": wlab, "grp": name, "n": int(len(g))}
            for H in HOLD:
                mv, wv, md, nn = stat_series(g[f"ex{H}"])
                row[f"ex{H}"] = mv; row[f"win{H}"] = wv; row[f"med{H}"] = md; row[f"n{H}"] = nn
            out.append(row)
            print(f"[{wlab}] {name:4s} n={len(g):6d}  "
                  f"h10 ex{F(row['ex10'])} win{P(row['win10'])} med{F(row['med10'])}   "
                  f"h20 ex{F(row['ex20'])} win{P(row['win20'])} med{F(row['med20'])} n20={row['n20']}")
    pd.DataFrame(out).to_csv(OUT / "holdout_narrow_cand.csv", index=False, encoding="utf-8-sig")
    print("\ncaveat: OOS 仅~4个月单段、样本小; 若候选在 OOS 优势消失→判窗内现象; 仍属'初步,换血前观察'，非定论。")


if __name__ == "__main__":
    main()
