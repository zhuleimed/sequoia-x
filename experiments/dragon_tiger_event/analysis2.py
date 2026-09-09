#!/usr/bin/env python
"""第二迭代: 基于富字段(dt_org_net机构/dt_hm_net游资/dt_chg) 龙虎榜事件的机构/游资"封形"
   + 换中证1000(sh.000852)剥离大小盘 的前向超额。

读 events_20260909.csv(已带 r/ex/org_net/hm_net/sell_date*)：
  A. 机构/游资封形：按上榜当天 机构坐席(org_net)/游资(hm_net) 有/无与正负 分 6 组比 ex(对沪深300)
  B. 换基准归因：用 sh.000852(中证1000, ~至2026-08-07) 在 [E1 开盘, sellH 收盘] 同窗算 x1000 = r - (1000同窗)，
     与 对沪深300 ex 对照 → 看"总亏"里有几成是 大小盘风格(被沪深300当基准的偏差)。

用 py312, 只读不改 data。
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "data" / "sequoia_v2.db"
HOLD = [2, 5, 10, 20]
OUT = Path(__file__).resolve().parent


def board(sym):
    if sym.startswith(("300", "301")): return "创业"
    if sym.startswith("688"): return "科创"
    if sym.startswith(("60", "00")): return "主板"
    return "其他"


def pct(x):
    return "    ." if np.isnan(x) else f"{x:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(OUT / "events_20260909.csv"))
    a = ap.parse_args()
    ev = pd.read_csv(a.csv, dtype={"symbol": str})
    ev["symbol"] = ev["symbol"].str.zfill(6)
    for c in ["symbol", "org_net", "hm_net", "dt_net_buy"]:
        if c in ev.columns:
            ev[c] = ev[c] if c == "symbol" else pd.to_numeric(ev[c], errors="coerce")
    sub = ev[ev["drop"].isna() & ev["E1"].notna()].copy()
    sub["b"] = sub["symbol"].map(board)
    org, hm = sub["org_net"], sub["hm_net"]
    nb = pd.to_numeric(sub["dt_net_buy"], errors="coerce")
    print(f"可交易事件 {len(sub)}；板块 " + " ".join(f"{b}:{len(g)}" for b, g in sub.groupby("b", sort=False)))

    # ---------- A. 机构/游资封形 ----------
    groups = [
        ("机构坐席净买 org>0",       org.notna() & (org > 0)),
        ("机构坐席净卖 org<0",       org.notna() & (org < 0)),
        ("纯游资净买 hm>0 & (机构或≈0)", hm.notna() & (hm > 0) & ((org.isna()) | (org.abs() < 1e6))),
        ("游资净买(含带机构) hm>0",   hm.notna() & (hm > 0)),
        ("游资净卖 hm<0",           hm.notna() & (hm < 0)),
        ("net_buy净卖 全标",         nb < 0),
        ("整体(全可交易)",           pd.Series(True, index=sub.index)),
    ]
    print("\nA. 机构/游资封形: 每格 = ex20均值 / ex5 / ex（对hs300）+N")
    rows = []
    for lab, m in groups:
        g = sub[m]
        r0 = {"组": lab, "Nbuy": len(g)}
        for h in HOLD:
            r0[f"N{h}"] = g[f"ex{h}"].dropna().shape[0]
            r0[f"ex{h}"] = g[f"ex{h}"].dropna().mean() * 100
        rows.append(r0)
    tab = pd.DataFrame(rows).set_index("组")
    print(tab[[f"ex{h}" for h in [2, 5, 10, 20]]].to_string(float_format=pct))
    tab.to_csv(OUT / "inst_hot_form_summary.csv", encoding="utf-8-sig")

    # ---------- B. 换中证1000 剥离大小盘 ----------
    print("\nB. 中证1000(sh.000852) 为基准 re-bench（大小盘风格剥离）")
    con = sqlite3.connect(DB)
    i852 = pd.read_sql("SELECT date,open,close FROM index_daily WHERE symbol='sh.000852'", con)
    con.close()
    i852["date"] = pd.to_datetime(i852["date"])
    i852 = i852.set_index("date")
    c1k_hi = i852["close"].index.max()
    c_close = i852["close"].to_dict(); c_open = i852["open"].to_dict()
    # 只对 E1≤000852末日的可 rebase 行
    sub2 = sub[pd.to_datetime(sub["E1"]) <= c1k_hi].copy()
    rr = []
    for _, row in sub2.iterrows():
        E1 = pd.Timestamp(row["E1"])
        if E1 not in c_open:
            continue
        o = {"symbol": row["symbol"], "b": row["b"]}
        # 机构/游资标签(粗)
        cls = "机构买" if (row["org_net"] is not None and row["org_net"] > 0) else \
              ("游资买" if (row["hm_net"] is not None and row["hm_net"] > 0) else "净卖/其他")
        o["cls"] = cls
        for h in HOLD:
            sd = row.get(f"sell_date{h}")
            if pd.isna(sd) or (pd.to_datetime(sd) not in c_close):
                o[f"x{h}"] = np.nan
                continue
            x = float(row[f"r{h}"]) - (c_close[pd.to_datetime(sd)] / c_open[E1] - 1.0)
            o[f"x{h}"] = x
        rr.append(o)
    xdf = pd.DataFrame(rr)
    if xdf.shape[0] == 0:
        print("无可 rebase 子样本(000852 覆盖不足)"); return
    print(f"可 rebase 子样本 {len(xdf)} (E1≤{c1k_hi.date()})；中证1000 净额均值% 按板:")
    tot = {}
    for lab, g in xdf.groupby("b", sort=False):
        for h in (10, 20):
            s = g[f"x{h}"].dropna()
            tot.setdefault((lab, h), (s.agg(["mean", "count"])))
    # 直接打印
    print(f"{'板':<6}{'n':>6}{'x10均值':>9}{'x20均值':>9}{'x20n':>6}   (中证1000净额; 可与对hs300对比)")
    for b_, g in xdf.groupby("b", sort=False):
        a5 = g["x5"].dropna().mean() * 100
        a20 = g["x20"].dropna().mean() * 100
        n20 = int(g["x20"].dropna().shape[0])
        print(f"{b_:<6}{len(g):>6}{pct(a5):>9}{pct(a20):>9}{n20:>6}")
    a5all = xdf["x5"].dropna().mean() * 100; a20all = xdf["x20"].dropna().mean() * 100
    n20all = int(xdf["x20"].dropna().shape[0])
    print(f"{'全部':<6}{len(xdf):>6}{pct(a5all):>9}{pct(a20all):>9}{n20all:>6}")
    xdf.to_csv(OUT / "rebench_x1000.csv", index=False)
    # 分 机构/游资
    print("\n  按 机构vs游资 标签的中证1000净额: ")
    for lab, g in xdf.groupby("cls", sort=False):
        a20 = g["x20"].dropna().mean() * 100; n20 = int(g["x20"].dropna().shape[0])
        print(f"   {lab:<8} n={len(g):5d} x20={pct(a20):>8} (n20={n20})")
    print("\ncaveat: B 只覆盖 000852 有价的可 E1 rebase 行(≤2026-08-07),样本与A不完全同 set; 单一市场期~1年, 未剔成本, 均初步。")

if __name__ == "__main__":
    main()
