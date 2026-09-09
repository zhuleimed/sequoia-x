#!/usr/bin/env python
"""龙虎榜(dragon-tiger)(+涨停面)上榜"可交易"事件研究 —— 手册方向⑦ 第一档(a)。

把"某股在某交易日 D 上龙虎榜"当一个只在 D 收盘后才披露的事件；量从 **事件后第一可买开盘**
进场、持 H 个市场交易日后收盘出场的**相对沪深300 超额**(双方同开同卖, 公平可交易口经)。

防前视(铁律, 同生产 features_extra next_day=True)：
  - 龙虎榜/涨停只在 D **收盘后**公告 → 最早可买 = D 之后第一个市场交易日 E1 的开盘
  - E1 该股若停牌(无bar)或一字封板(open==high==low 无成交区间, 买不进) ⇒ 判"不可进场", 单列 drop 计数
  - 事件/E1/E_h 缺价或持仓窗内停牌 ⇒ 该(事件,horizon)在(持有窗不全)剔除并计数, 透明不进超额池
一个事件若连最乐观的可买都做不到，就不该把"账面绩效"记成你可交易的超额。

输出:
  events_<tag>.csv     每股×每上榜日明细(场进/drop原因, r{H}/ex{H}, gap, net_buy…)
用法(铁律六=py312):
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/dragon_tiger_event/event_study.py [--out-tag X]
再跑 analysis.py 读该 CSV 出四组分桶摘要(见 analyze_*.py)。
"""
import argparse
import glob
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from sequoia_x.features_extra.build_extra_features import _load  # 复用每股 parquet 读取(带 lru_cache)

DB_PATH = PROJECT_ROOT / "data" / "sequoia_v2.db"
EXTRA = PROJECT_ROOT / "data" / "extra_features"
OUT = PROJECT_ROOT / "experiments" / "dragon_tiger_event"
HOLD = [2, 5, 10, 20]                  # 持有市场交易日(买=次日开盘计第1, 卖=第H交易日收盘)


def load_all():
    """载 market 日历(sh.000300)+ 全股 bar。返回 idxdict, trade_days, per-symbol group。"""
    conn = sqlite3.connect(DB_PATH)
    sq = pd.read_sql("SELECT symbol,date,open,high,low,close FROM stock_daily "
                     "WHERE date>='2025-07-01'", conn)
    idx = pd.read_sql("SELECT date,open,close FROM index_daily WHERE symbol='sh.000300'", conn)
    conn.close()
    sq["date"] = pd.to_datetime(sq["date"])
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date").set_index("date")
    idx_close = idx["close"].to_dict(); idx_open = idx["open"].to_dict()
    td = pd.DatetimeIndex(idx["close"].index)
    pos = {d: i for i, d in enumerate(td)}
    px = {s: g.sort_index().set_index("date") for s, g in
          sq.groupby("symbol", sort=False)}
    return idx_close, idx_open, td, pos, px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="20260909")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/3] 载 data ...", flush=True)
    idx_close, idx_open, td, pos, px = load_all()
    td_arr = td

    print("[2/3] 列龙虎榜事件, 合并涨停面 ...", flush=True)
    files = sorted(glob.glob(str(EXTRA / "dragon_tiger" / "*.parquet")))
    rows = []
    n_ev_d = 0; n_e1_nobar = 0; n_e1_oneword = 0
    for fp in files:
        sym = Path(fp).stem
        if sym not in px:
            continue
        try:
            ev = pd.read_parquet(fp)
        except Exception:
            continue
        if ev is None or len(ev) == 0:
            continue
        ev = ev.dropna(subset=["dt_net_buy", "dt_net_rate"], how="all")
        if ev.empty:
            continue
        ev["date"] = pd.to_datetime(ev["date"])
        g = px[sym]
        ev = ev[ev["date"].isin(g.index)]
        if ev.empty:
            continue
        # 涨停面(该股自身涨停timeline, "妖/连板/首板"动量, 视窗=D前6个市场日含D)
        lu_pos = np.array([], dtype=int); lu_val = np.array([])
        lp = EXTRA / "limit_up" / f"{sym}.parquet"
        if lp.exists():
            ld = _load("limit_up", sym)
            if ld is not None and len(ld):
                ld["date"] = pd.to_datetime(ld["date"])
                ok = pd.Series(ld["date"]).map(pos)
                m = ok.notna().values
                lu_pos = ok.values[m].astype(int)
                lu_val = pd.to_numeric(pd.Series(ld["lu_lianban"].values)[m], errors="coerce").values
                sortable = np.argsort(lu_pos)
                lu_pos = lu_pos[sortable]; lu_val = lu_val[sortable]
        def _recent_flag(pi):
            """D 位次 pi; 返回 (D当日是否涨停入池, D前6市场日内最大连板数)。窗口含D。
            该股涨停位次数组升序; 二分出 [pi-6, pi] 的段取 max连板。"""
            if not lu_pos.size:
                return False, 0.0
            lo = int(np.searchsorted(lu_pos, pi - 6, side="left"))
            hi = int(np.searchsorted(lu_pos, pi + 1, side="left"))
            if lo >= hi:
                return False, 0.0
            return bool(pi in lu_pos[lo:hi]), float(np.nanmax(lu_val[lo:hi]))
        # 富字段(from enrich-schema): 机构净买/游资净买/上榜当日涨幅(可能 None)
        orgs = pd.to_numeric(ev.get("dt_org_net"), errors="coerce") if "dt_org_net" in ev.columns else pd.Series(np.nan, index=ev.index)
        hms  = pd.to_numeric(ev.get("dt_hm_net"),  errors="coerce") if "dt_hm_net"  in ev.columns else pd.Series(np.nan, index=ev.index)
        chgd = pd.to_numeric(ev.get("dt_chg"),     errors="coerce") if "dt_chg"     in ev.columns else pd.Series(np.nan, index=ev.index)
        for i in range(len(ev)):
            d = ev["date"].iloc[i]; nb = ev["dt_net_buy"].iloc[i]; nr = ev["dt_net_rate"].iloc[i]
            org = orgs.iloc[i] if i < len(orgs) else np.nan
            hm = hms.iloc[i] if i < len(hms) else np.nan
            echg = chgd.iloc[i] if i < len(chgd) else np.nan
            n_ev_d += 1
            dkey = d.date()
            pi = pos.get(d)
            if pi is None or pi + 1 >= len(td_arr):
                _push(rows, sym, dkey, nb, nr, _recent_flag(pi)[1], org=org, hm=hm, chg=echg, drop="市场日历无次日")
                continue
            is_limD, max_lb = _recent_flag(pi)
            e1 = td_arr[pi + 1]
            if e1 not in g.index:
                n_e1_nobar += 1
                _push(rows, sym, dkey, nb, nr, max_lb, org=org, hm=hm, chg=echg, drop="E1停牌无bar")
                continue
            ro1 = g.loc[e1]
            o1 = float(ro1["open"])
            # 一字/无成交打不开 → 买不进(按 OHLC 精确判: open==high==low 无成交区间打不开)
            if ro1["open"] == ro1["high"] == ro1["low"]:
                n_e1_oneword += 1
                _push(rows, sym, dkey, nb, nr, max_lb, org=org, hm=hm, chg=echg,
                      gap=o1 / float(g.loc[d, "close"]) - 1.0 if d in g.index else None,
                      drop="E1一字无成交区间")
                continue
            d_close = float(g.loc[d, "close"]) if d in g.index else np.nan
            # D 日本股涨幅(相对股自身前一 bar 收盘)——"消息当天冲没冲"/动量进D
            di = g.index.get_loc(d)
            d_pct = (d_close / float(g.iloc[di - 1]["close"]) - 1.0) if (di > 0 and d_close == d_close) else np.nan
            gap = o1 / d_close - 1.0 if d_close == d_close and d_close else None
            rec = {"symbol": sym, "D": dkey, "dt_net_buy": nb, "dt_net_rate": nr,
                   "lianban6d_max": max_lb if max_lb else 0, "is_limit_D": int(is_limD),
                   "org_net": org, "hm_net": hm, "chg_onD": echg,   # 富字段(元/小数)
                   "E1": e1.date(), "E1open": round(o1, 3),
                   "gap": round(gap, 4) if gap is not None else None,
                   "Dclose": round(d_close, 3) if d_close == d_close else None,
                   "D_pct": round(d_pct, 4) if d_pct == d_pct else None}
            for _H in HOLD:
                rec[f"sell_date{_H}"] = None          # 预置, 若某 H 无窗则留 None
            for H in HOLD:
                eh_pos = pi + H
                if eh_pos >= len(td_arr):
                    rec[f"drop{H}"] = "无未来行情"; continue
                eh = td_arr[eh_pos]
                if eh not in g.index or eh not in idx_close:
                    rec[f"drop{H}"] = "持有窗停牌/无价"; continue
                cH = float(g.loc[eh, "close"])
                if pd.isna(cH) or pd.isna(o1):
                    rec[f"drop{H}"] = "无价"; continue
                rH = cH / o1 - 1.0
                idxr = idx_close[eh] / idx_open[e1] - 1.0
                rec[f"r{H}"] = round(rH, 6); rec[f"idx_r{H}"] = round(idxr, 6)
                rec[f"ex{H}"] = round(rH - idxr, 6)
                rec[f"sell_date{H}"] = eh.date()   # 保存卖点市场日, 便于换基准(rebase)对齐
            rows.append(rec)
    print(f"      dragon 文件 {len(files)}; 有效事件 {n_ev_d}; "
          f"E1停牌 {n_e1_nobar}, E1一字 {n_e1_oneword}", flush=True)
    evdf = pd.DataFrame(rows)
    evdf.to_csv(OUT / f"events_{a.out_tag}.csv", index=False)
    print(f"[3/3] 明细已存 events_{a.out_tag}.csv ({len(evdf)} 行)", flush=True)
    return evdf


def _push(rows, sym, dkey, nb, nr, max_lb, org=None, hm=None, chg=None, drop="", gap=None):
    rec = {"symbol": sym, "D": dkey, "dt_net_buy": nb, "dt_net_rate": nr,
           "lianban6d_max": float(max_lb) if max_lb else 0.0, "is_limit_D": 0,
           "org_net": org, "hm_net": hm, "chg_onD": chg,
           "E1": None, "E1open": None,
           "gap": round(gap, 4) if gap is not None else None,
           "Dclose": None, "drop": drop}
    for H in HOLD:
        rec[f"ex{H}"] = np.nan; rec[f"r{H}"] = np.nan
    rows.append(rec)


if __name__ == "__main__":
    df = main()
    live = df[df["drop"].isna()]
    print(f"\n可交易(未drop)事件池: {len(df)} 行中 {len(live)} 行")
    print(df[df["drop"].isna()].head(5).to_string())
