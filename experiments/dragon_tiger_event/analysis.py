#!/usr/bin/env python
"""读取 event_study.py 产出的 events_<tag>.csv, 出龙虎榜"可交易"分桶摘要 + 表格。

口径(与 event_study 一致, 摘要只统计可交易且该 horizon 有完整窗 buyable 的子集):
  ex{h}      相对沪深300 净超额(D+1开盘买/持h市场日后收盘卖, 基准同时段)
  r{h}       绝对收益
  涨占比      ex>0 的占比
  N          buyable 事件里该 horizon 完整窗(可交易+持有完整)样本
报告每组一行均值风格, 与仓库 analyze_kline summary_table 风格接近。

用法(py312):
  /home/zhulei/anaconda3/envs/zhulei_py312/bin/python experiments/dragon_tiger_event/analysis.py [--csv ev.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HOLD = [2, 5, 10, 20]
OUT = Path(__file__).resolve().parent


def _rowstats(sub: pd.DataFrame) -> dict:
    """给一个 buyable 子集, 返回各 H 的均值ex/r/涨占比/N(每列各取 non-nan)。"""
    out = {}
    for h in HOLD:
        s = sub[f"ex{h}"].dropna()
        r_ = sub[f"r{h}"].dropna()
        out[f"ex{h}_n"] = len(s)
        out[f"ex{h}_{'mean'}"] = s.mean() * 100 if len(s) else np.nan
        out[f"ex{h}_win"] = (s > 0).mean() * 100 if len(s) else np.nan
        out[f"ex{h}_med"] = s.median() * 100 if len(s) else np.nan
        out[f"r{h}_{'mean'}"] = r_.mean() * 100 if len(r_) else np.nan
    return out


def show(df: pd.DataFrame, label: str):
    row = {"组": label}
    row.update(_rowstats(df))
    return row


def print_block(rows: list[dict]):
    """并按H把 ex_n/mean/win/med 画成行列更易读的表。"""
    t = pd.DataFrame(rows).set_index("组")
    hdr_ord = []
    cols = []
    for h in HOLD:
        for k in ["mean", "win", "med", "n"]:
            cols += [f"ex{h}_{k}"]
    print(t[cols].to_string(float_format=lambda x: f"{x:6.2f}" if not (isinstance(x, float) and np.isnan(x)) else "   ."))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(OUT / "events_20260819.csv"))
    ap.add_argument("--skip-in-hold-drop", action="store_true",
                    help="不把(买进但个别H窗停牌)计入: buyable 即计入每列自行取非空")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    # buyable = 未 drop(事件级无 'drop' 原因)
    live = df[df["drop"].isna() & df["E1"].notna()].copy()
    live["D"] = pd.to_datetime(live["D"])
    print(f"事件{len(df)}行; 可交易{len(live)} ({len(live)/len(df):.1%}); "
          f"可交易中 E1 一字被排除请见 CSV 的 drop", flush=True)
    live["gap"] = pd.to_numeric(live["gap"], errors="coerce")
    live["D_pct"] = pd.to_numeric(live["D_pct"], errors="coerce")

    rows = []
    # 1) 整体
    rows.append(show(live, "整体(全部可交易上榜)"))
    # 2) 净买入方向/强度
    m = live["dt_net_buy"]; q = m.quantile([.25, .5, .75])
    rows.append(show(live[m < 0], "净卖(d<0)"))
    rows.append(show(live[m >= 0], "净买(d>=0)"))
    pos = live[m >= 0]
    if len(pos):
        pq = pos["dt_net_buy"].quantile([.5, .8])
        rows.append(show(pos[pos["dt_net_buy"] <= pq[.5]], "净买-弱(≤中位正)"))
        rows.append(show(pos[(pos["dt_net_buy"] > pq[.5]) & (pos["dt_net_buy"] <= pq[.8])], "净买-中"))
        rows.append(show(pos[pos["dt_net_buy"] > pq[.8]], "净买-强(top20%)"))
    # 3) 当日自身/高开兑现(D_pct 方向 + gap 开盘溢价)
    rows.append(show(live[live["D_pct"] > 0.095], "上榜日已大涨(+9.5%+)"))
    rows.append(show(live[live["D_pct"] < -0.09], "上榜日-9%大跌(近跌停)"))
    rows.append(show(live[(live["D_pct"] >= -0.09) & (live["D_pct"] <= 0.095)], "上榜日平稳"))
    #   gap 溢价段
    rows.append(show(live[live["gap"] <= 0], "E1开盘≤平(g≈≤0)"))
    rows.append(show(live[(live["gap"] > 0) & (live["gap"] <= 0.03)], "E1低开~+3%以内"))
    rows.append(show(live[(live["gap"] > 0.03) & (live["gap"] <= 0.07)], "E1高开+3~7%"))
    rows.append(show(live[live["gap"] > 0.07], "E1显著高开>+7%"))
    # 4) 近期涨停动量(妖/连板)
    lb = live["lianban6d_max"].fillna(0).astype(float)
    rows.append(show(live[lb <= 0], "无近6日涨停"))
    rows.append(show(live[lb == 1], "近6日 1 连板(首板)"))
    rows.append(show(live[lb.between(2, 3)], "近6日 2-3 连板"))
    rows.append(show(live[lb >= 4], "近6日 ≥4 连板(妖)"))
    # 时间前半/后半(诚实: 单调覆盖的稳定性代理)
    halves = pd.qcut(pd.to_datetime(live["D"]).rank(method="first"), 2, labels=["前半年(s1)", "后半年(s2)"])
    rows.append(show(live[halves == "前半年(s1)"], "时间前半 D≤中位"))
    rows.append(show(live[halves == "后半年(s2)"], "时间后半 D>中位"))

    print("=== 分桶摘要: 每格 = ex{H} 的 [均值%/ 涨占比%/ 中位%/ N] ===")
    t = print_block(rows)
    t.to_csv(OUT / "buckets_summary_table.csv", encoding="utf-8-sig")
    print("\ncaveat: 数据仅 ~1 年(2025-08~2026-08)单一市场期; 均值为该条件样本而非随机; "
          "只反映上榜公布后才可做的 D+1 开盘口径; 未剔交易成本(一二级冲击); 需后续OOS补窗复验。")


if __name__ == "__main__":
    main()
