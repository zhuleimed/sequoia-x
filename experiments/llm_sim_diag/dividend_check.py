"""核查：不复权价中的"除权除息跳空"对前述结论的影响有多大。

问题
----
LLM 模拟盘窗口（2026-07 ~ 2026-09）正好覆盖 A 股年度分红季。
全库是不复权实际成交价，**除权日有真实跳空**——但这部分跌不是亏损，
是现金/股票已经打到股东账上。

若这笔"假跌"集中在某些组，前述"买入即跌"的量级就会被高估。

做法
----
复用项目既有复权模块 `sequoia_x.model_selection_v2.adjust.build_adjust_factors`
（口径与特征/标签层一致），对每笔交易算：
    不复权收益 R_raw = close_end / open_buy − 1
    后复权收益 R_adj = (close_end × F_end) / (open_buy × F_buy) − 1
    分红修正量  Δ = R_adj − R_raw
Δ 就是"不复权口径下被误算成亏损的那部分"。
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

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from sequoia_x.model_selection_v2.adjust import build_adjust_factors, _load_events  # noqa: E402

DB = PROJECT / "data" / "sequoia_v2.db"
RESULTS_GLOB = str(PROJECT / "data" / "results" / "results_2026*.json")
BAD_DAYS = ("2026-07-06", "2026-07-07")
JUMP = 0.25
HOLD = 5

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main() -> int:
    con = sqlite3.connect(DB)

    # ── 1. 全市场除权事件普查（2026-06 ~ 2026-09）──
    print("=" * 100)
    print("一、窗口内全市场除权事件普查（数据源 data/extra_features/xdxr/）")
    print("=" * 100)
    ev_rows = []
    for fp in sorted(glob.glob(str(PROJECT / "data" / "extra_features" / "xdxr" / "*.parquet"))):
        code = Path(fp).stem
        ev = _load_events(code)
        if ev is None:
            continue
        m = (ev["avail"] >= "2026-06-01") & (ev["avail"] <= "2026-09-30")
        for _, r in ev[m].iterrows():
            ev_rows.append({
                "symbol": code, "date": r["avail"].strftime("%Y-%m-%d"),
                "现金分红(元/股)": r["div_ps"], "送转(股/股)": r["szg_ps"],
                "配股(股/股)": r["peigu_ps"],
            })
    evd = pd.DataFrame(ev_rows)
    if evd.empty:
        print("窗口内无除权事件")
        return 1
    print(f"事件总数: {len(evd)}  涉及股票: {evd.symbol.nunique()}")
    print(f"  纯现金分红: {((evd['送转(股/股)']==0)&(evd['配股(股/股)']==0)).sum()} 笔")
    print(f"  含送转    : {(evd['送转(股/股)']>0).sum()} 笔")
    big = evd[evd["送转(股/股)"] > 0].nlargest(8, "送转(股/股)")
    if len(big):
        print("\n  送转比例最大的 8 笔（这类会造成 10%~50% 的假跌）：")
        print(big.to_string(index=False))
    print(f"\n  按月分布:\n{evd.groupby(evd['date'].str[:7]).size().to_string()}")

    # ── 2. LLM 买入的除权影响 ──
    con2 = sqlite3.connect(DB)
    sig = pd.read_sql(
        "SELECT DISTINCT symbol, buy_date FROM sim_buy_signals WHERE status='executed'", con2)
    cal = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2026-05-01' ORDER BY date",
        con2)["date"].tolist()
    cal_idx = {d: i for i, d in enumerate(cal)}
    b = pd.read_sql(
        "SELECT symbol, date, open, close FROM stock_daily WHERE date >= '2026-05-01'", con2)
    for c in ("open", "close"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna().sort_values(["symbol", "date"])
    b["prev"] = b.groupby("symbol")["close"].shift(1)
    b["corrupt"] = (b["date"].isin(BAD_DAYS)) & (
        ((b["close"] / b["prev"] - 1).abs() > JUMP) | ((b["open"] / b["prev"] - 1).abs() > JUMP))
    b["corrupt"] = b["corrupt"].fillna(False)
    px = {s: dict(zip(g["date"], zip(g["open"], g["close"], g["corrupt"])))
          for s, g in b.groupby("symbol")}
    con2.close()

    # 全量因子缓存
    fcache: dict[str, pd.Series] = {}

    def factors(sym: str) -> pd.Series | None:
        if sym in fcache:
            return fcache[sym]
        d = px.get(sym)
        if not d:
            return None
        ser = pd.Series([v[1] for v in d.values()], index=pd.to_datetime(list(d.keys())))
        f = build_adjust_factors(sym, ser)
        fcache[sym] = f
        return f

    rows = []
    for r in sig.itertuples(index=False):
        i = cal_idx.get(r.buy_date)
        if i is None or i + 1 + HOLD >= len(cal):
            continue
        bd, ex = cal[i + 1], cal[i + 1 + HOLD]
        d = px.get(r.symbol, {})
        o, cl = d.get(bd), d.get(ex)
        if not o or not cl or not (o[0] > 0) or not (cl[1] > 0):
            continue
        if any(d.get(w, (0, 0, True))[2] for w in cal[i + 1: i + 2 + HOLD]):
            continue
        F = factors(r.symbol)
        raw = cl[1] / o[0] - 1.0
        adj = raw
        if F is not None:
            try:
                fb, fe = F.loc[bd], F.loc[ex]
                adj = (cl[1] * fe) / (o[0] * fb) - 1.0
            except KeyError:
                pass
        rows.append({"symbol": r.symbol, "signal_date": r.buy_date, "buy_date": bd,
                     "R_raw": raw, "R_adj": adj, "Δ分红修正": adj - raw})
    t = pd.DataFrame(rows)

    print("\n" + "=" * 100)
    print(f"二、LLM 模拟盘买入样本（T+1 开盘买入，持有 {HOLD} 日，n={len(t)}）")
    print("=" * 100)
    hit = t[t["Δ分红修正"].abs() > 1e-9]
    print(f"窗口内遇除权的: {len(hit)} 笔 / {len(t)} 笔 ({len(hit)/len(t)*100:.1f}%)")
    if len(hit):
        print("\n  受影响明细（按修正量降序）：")
        print(hit.nlargest(15, "Δ分红修正").to_string(index=False))
    print(f"\n  平均：不复权 {t['R_raw'].mean()*100:+.3f}%  后复权 {t['R_adj'].mean()*100:+.3f}%  "
          f"（差 {t['Δ分红修正'].mean()*100:+.3f}pp）")
    print(f"  中位：不复权 {t['R_raw'].median()*100:+.3f}%  后复权 {t['R_adj'].median()*100:+.3f}%")
    print(f"  Δ分红修正 分布：均值 {t['Δ分红修正'].mean()*100:+.3f}pp  "
          f"最大 {t['Δ分红修正'].max()*100:+.3f}pp  最小 {t['Δ分红修正'].min()*100:+.3f}pp")
    print(f"  修正量 >0.5pp 的笔数: {(t['Δ分红修正']>0.005).sum()}")
    print(f"  修正量 >2pp  的笔数: {(t['Δ分红修正']>0.02).sum()}")

    # ── 3. 市场基准的除权影响（抽样）──
    print("\n" + "=" * 100)
    print("三、同期全市场（等权抽样）：除权是否也同等影响基准？")
    print("=" * 100)
    days = [d for d in cal if d in set(sig["buy_date"])]
    days = [d for d in days if cal_idx[d] + 1 + HOLD < len(cal)]
    smp = list(px)[::7]
    pairs = {"raw": [], "adj": []}
    for sd in days:
        i = cal_idx[sd]
        bd, ex = cal[i + 1], cal[i + 1 + HOLD]
        raw_l, adj_l = [], []
        for s in smp:
            d = px.get(s, {})
            o, cl = d.get(bd), d.get(ex)
            if not o or not cl or not (o[0] > 0) or not (cl[1] > 0):
                continue
            if any(d.get(w, (0, 0, True))[2] for w in cal[i + 1: i + 2 + HOLD]):
                continue
            rr = cl[1] / o[0] - 1.0
            aa = rr
            F = factors(s)
            if F is not None:
                try:
                    aa = (cl[1] * F.loc[ex]) / (o[0] * F.loc[bd]) - 1.0
                except KeyError:
                    pass
            raw_l.append(rr)
            adj_l.append(aa)
        if raw_l:
            pairs["raw"].append(np.mean(raw_l))
            pairs["adj"].append(np.mean(adj_l))
    print(f"  全市场等权 持有{HOLD}日 日均: 不复权 {np.mean(pairs['raw'])*100:+.4f}%  "
          f"后复权 {np.mean(pairs['adj'])*100:+.4f}%  "
          f"（差 {(np.mean(pairs['adj'])-np.mean(pairs['raw']))*100:+.4f}pp）")
    print(f"  对照 LLM 组:                 不复权 {t['R_raw'].mean()*100:+.4f}%  "
          f"后复权 {t['R_adj'].mean()*100:+.4f}%  "
          f"（差 {t['Δ分红修正'].mean()*100:+.4f}pp）")
    print(f"\n  → 超额（LLM − 市场）: 不复权 {(t['R_raw'].mean()-np.mean(pairs['raw']))*100:+.3f}pp  "
          f"后复权 {(t['R_adj'].mean()-np.mean(pairs['adj']))*100:+.3f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
