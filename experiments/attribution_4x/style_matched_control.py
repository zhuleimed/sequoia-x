#!/usr/bin/env python3
"""风格匹配对照（审计建议实验③，2026-09-27）—— 判定"选股超额"是能力还是风格暴露。

问题
----
#17 测出"模型选股在持有期比同池随机 300 只多约 +1.6~1.9%/月"（边界显著）。
但审计指出：**β≈0.95 只能排除"高 β"，排除不了规模/波动/流动性/行业暴露**；
而原始 IC 里确实混着风格成分（中性化后 IC 下降）。
⇒ 必须用**风格匹配的对照**：从与选股**同一（行业 × 成交额分位 × 波动分位）桶**里抽对照股。

设计
----
每月（买入日）：
  1. 池内每只股票算：行业 + 过去 20 日平均成交额(log) + 过去 20 日收益波动（**只用过去数据**）
  2. 分桶：(行业, 成交额五分位, 波动五分位)
  3. 模型选股（引擎实际成交买单）各配一只**同桶**随机对照（排除自身）⇒ 组成"风格匹配组合"
     · 桶内无其他成员时**逐级回退**：0=(行业,成交额,波动) → 1=(行业,成交额) → 2=(行业) → 3=全池
     · 报告各级别的使用频率（回退越多，对照越不"风格匹配"）
  4. 持有期收益口径与 #17 一致：`close(月末)/close(买入日) − 1`
  5. **多随机种子**（默认 200）重复抽样 ⇒ 报配对差的分布（避免单次抽样的种子偶然性）

判读
----
  · 模型 − 风格匹配 的配对差**塌到 ≈0** ⇒ #17 的 +1.6%/月 **是风格暴露**，"有选股能力"须降级
  · 若仍显著为正 ⇒ **才能说"在风格之外还有真选股能力"**
  · 居中 ⇒ 报出"其中多少来自风格"，作为限定

用法（断点续跑：结果已存在即跳过；`--force` 强制重算）
    env -u KMP_AFFINITY python experiments/attribution_4x/style_matched_control.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.pop("KMP_AFFINITY", None)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "out"

CACHE = ROOT / "output/backtest_v2/.purged_70m.json"
N_SEEDS = 200
FORCE = "--force" in sys.argv

from experiments.attribution_4x.selection_alpha import engine_buys  # noqa: E402


def main() -> int:
    res_path = OUT / "style_matched_control.json"
    if res_path.exists() and not FORCE:
        print(f"已有结果 {res_path}（--force 可强制重算）")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)

    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    print("加载行情/行业…", flush=True)
    px = pd.read_sql("SELECT symbol,date,close,amount FROM stock_daily "
                     "WHERE date>='2020-09-01' AND close>0 ORDER BY symbol,date", cx)
    cal = np.array([r[0] for r in cx.execute(
        "SELECT DISTINCT date FROM index_daily WHERE symbol='sh.000300' "
        "AND date>='2020-09-01' ORDER BY date")])
    ind = pd.read_sql("SELECT symbol, industry_l1 FROM tdx_stock_industry", cx)
    pool = set(json.load(open(ROOT / "output/backtest_v2/.stock_pool.json")))
    cx.close()
    ind_map = dict(zip(ind["symbol"], ind["industry_l1"]))

    print("取引擎实际买单（与 #17 同口径：policy=all, hard_stop=-0.08）…", flush=True)
    buys = engine_buys(CACHE)
    buy_dates = sorted(buys["date"].unique())
    month_end = {dt: [x for x in cal if x[:7] == dt[:7]][-1] for dt in buy_dates}
    print(f"  {len(buys)} 笔 / {len(buy_dates)} 个买入日", flush=True)

    feats = {sym: (g["date"].to_numpy(), g["close"].to_numpy(float), g["amount"].to_numpy(float))
             for sym, g in px.groupby("symbol", sort=False)}

    months = []          # 每月：模型收益 + 每只选股的候选池（逐级回退）
    for dt in buy_dates:
        de = month_end[dt]
        recs = []
        for sym, (dts, cls, amts) in feats.items():
            if sym not in pool:
                continue
            j0 = np.searchsorted(dts, dt)
            j1 = np.searchsorted(dts, de)
            if j0 >= len(dts) or j1 >= len(dts) or dts[j0] != dt or dts[j1] != de or j0 < 20:
                continue
            amt = float(np.nanmean(amts[j0 - 20:j0 + 1]))
            r = np.diff(cls[j0 - 20:j0 + 1]) / cls[j0 - 20:j0]
            recs.append({"symbol": sym, "ret": cls[j1] / cls[j0] - 1,
                         "amt": np.log(max(amt, 1.0)),
                         "vol": float(np.std(r)) if len(r) > 1 else 0.0,
                         "ind": ind_map.get(sym, "未知")})
        if len(recs) < 200:
            continue
        df = pd.DataFrame(recs).reset_index(drop=True)
        df["amt_q"] = pd.qcut(df["amt"], 5, labels=False, duplicates="drop")
        df["vol_q"] = pd.qcut(df["vol"], 5, labels=False, duplicates="drop")
        # 四个回退级别的成员索引
        keys = [list(zip(df["ind"], df["amt_q"], df["vol_q"])),
                list(zip(df["ind"], df["amt_q"])),
                list(df["ind"]),
                [None] * len(df)]
        level_members = []
        for lv, k in enumerate(keys):
            m = {}
            for i, kk in enumerate(k):
                m.setdefault(kk, []).append(i)
            level_members.append(m)
        pick_syms = set(buys[buys["date"] == dt]["symbol"])
        picks = []
        for i, sym in enumerate(df["symbol"]):
            if sym not in pick_syms:
                continue
            cand = {}
            for lv in range(4):
                kk = keys[lv][i]
                members = [j for j in level_members[lv].get(kk, []) if j != i]
                cand[lv] = members
            picks.append({"i": i, "ret": float(df.at[i, "ret"]), "cand": cand})
        if len(picks) < 3:
            continue
        months.append({"date": dt, "model": float(np.mean([p["ret"] for p in picks])),
                       "pool": float(df["ret"].mean()), "picks": picks,
                       "rets": df["ret"].to_numpy()})
    print(f"参与月份 {len(months)}", flush=True)
    if len(months) < 10:
        print("⚠️ 月份太少，中止")
        return 1

    model = np.array([m["model"] for m in months])
    pool_r = np.array([m["pool"] for m in months])
    t0, p0 = stats.ttest_rel(model, pool_r)
    print(f"\n【基准】模型 {model.mean()*100:+.3f}%/月 ｜ 全池 {pool_r.mean()*100:+.3f}%"
          f" ｜ 配对 {t0:+.2f} p={p0:.4f}")

    lv_use = np.zeros(4)
    diffs, pvals, matched_means = [], [], []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        mm = []
        for m in months:
            ctrl = []
            for p in m["picks"]:
                for lv in range(4):
                    if p["cand"][lv]:
                        ctrl.append(m["rets"][p["cand"][lv][rng.randint(len(p["cand"][lv]))]])
                        if seed == 0:
                            lv_use[lv] += 1
                        break
            mm.append(np.mean(ctrl))
        mm = np.array(mm)
        matched_means.append(mm.mean())
        t, p = stats.ttest_rel(model, mm)
        diffs.append((model - mm).mean())
        pvals.append(p)
    diffs = np.array(diffs)
    pvals = np.array(pvals)
    matched_means = np.array(matched_means)

    print(f"\n【风格匹配对照】{N_SEEDS} 个随机种子")
    print(f"  对照组月均收益: {matched_means.mean()*100:+.3f}%"
          f"（种子间 σ {matched_means.std()*100:.3f}pp，min {matched_means.min()*100:+.3f}%）")
    print(f"  模型 − 风格匹配 的月均差: {diffs.mean()*100:+.3f}pp"
          f"  （种子间 σ {diffs.std()*100:.3f}pp，范围 [{diffs.min()*100:+.3f}, {diffs.max()*100:+.3f}]）")
    print(f"  配对 p 的中位数: {np.median(pvals):.4f} ｜ p<0.05 的种子占比: {(pvals < 0.05).mean()*100:.0f}%")
    tot = lv_use.sum() or 1
    print(f"  候选回退级别使用占比: L0(行业+成交额+波动) {lv_use[0]/tot*100:.1f}% ｜ "
          f"L1 {lv_use[1]/tot*100:.1f}% ｜ L2(仅行业) {lv_use[2]/tot*100:.1f}% ｜ L3(全池) {lv_use[3]/tot*100:.1f}%")
    print(f"\n【判读】模型超额 {model.mean()*100:+.3f}%/月；扣掉风格后剩 {diffs.mean()*100:+.3f}pp/月"
          f"（保留 {diffs.mean()/(model.mean()-pool_r.mean())*100:.0f}% 的原始超额）")
    json.dump({"n_months": len(months), "model_mean": float(model.mean()),
               "pool_mean": float(pool_r.mean()), "pool_paired_t": float(t0), "pool_paired_p": float(p0),
               "matched_mean": float(matched_means.mean()), "diff_mean": float(diffs.mean()),
               "diff_std": float(diffs.std()), "p_median": float(np.median(pvals)),
               "p_lt05_share": float((pvals < 0.05).mean()),
               "level_use": (lv_use / tot).tolist(), "n_seeds": N_SEEDS},
              open(res_path, "w"), ensure_ascii=False, indent=2)
    print(f"\n产物: {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
