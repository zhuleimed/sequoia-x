#!/usr/bin/env python3
"""T4 融合权重扫描（待办第 9 项，2026-09-27）

为什么做
--------
实测 **T4 在每个周期都强于 T2**（69 个月，20 日：T4 +0.0423 vs T2 +0.0292；
10 日中性化：T4 +0.0342 vs T2 +0.0202），但生产的融合权重仍由 2026-07 写的一条启发式
决定：`w_t4 = 0.25 + 0.25 × min(std_t4 / 0.02, 1)` ⇒ **w_t4 只能落在 0.25~0.50**。
本脚本在**现有缓存**上扫一遍权重，看融合还有没有提升空间（纯计算，不改任何生产代码）。

⚠️ 防过拟合（本脚本最重要的设计）
--------------------------------
"在同一份样本上挑权重" = **样本内优化**，挑出来的"最优权重"很可能是噪声。
所以同时报两段，并只认**两段一致**的结论：
    · 拟合段 2020-10 ~ 2023-12
    · 检验段 2024-01 ~ 2026-06
若两段的最优权重不同 ⇒ 说明没有稳定的最优权重，应维持现状或走动态加权。

口径（与 ic_by_horizon.py 一致，保证可比）
----------------------------------------
  · 融合分数 = w·rank_t4 + (1−w)·rank_t2（秩降序）
  · y = clip(H 日个股收益 − 沪深300 同期, ±0.5)，ref_date = 上月最后交易日
  · IC = 逐月 Spearman 再取均值（**非池化**）；60 日做 Newey-West 修正
  · 中性化 = y 对 申万一级行业 + 成交额分位 + 波动分位 回归取残差

用法：
    python experiments/attribution_4x/t4_weight_sweep.py \
        --cache=output/backtest_v2/.t4_70m_random.json --tag=70m_random
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "out"
CACHE = ROOT / "output/backtest_v2/.t4_70m_random.json"
TAG = "70m_random"
for _a in sys.argv[1:]:
    if _a.startswith("--cache="):
        CACHE = Path(_a[len("--cache="):])
    elif _a.startswith("--tag="):
        TAG = _a[len("--tag="):]

WEIGHTS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]   # 0=纯T2，1=纯T4
HS = [20, 60]
CLIP = 0.5
SEGS = [("拟合段 2020-10~2023-12", "2020-10", "2023-12"),
        ("检验段 2024-01~2026-06", "2024-01", "2026-06")]


def newey_west_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    n = len(x)
    mu = float(np.mean(x))
    xc = x - mu
    s = float(np.dot(xc, xc) / n)
    for k in range(1, min(lag, n - 1) + 1):
        s += 2.0 * (1.0 - k / (lag + 1)) * float(np.dot(xc[:-k], xc[k:]) / n)
    se = np.sqrt(max(s, 1e-18) / n)
    t = float(mu / se) if se > 0 else 0.0
    return t, float(2 * (1 - stats.t.cdf(abs(t), df=max(n - 1, 1))))


def main() -> int:
    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    print("加载行情/行业…", flush=True)
    px = pd.read_sql("SELECT symbol,date,close,amount FROM stock_daily "
                     "WHERE date>='2019-12-01' AND close>0 ORDER BY symbol,date", cx)
    idx = pd.read_sql("SELECT date,close FROM index_daily WHERE symbol='sh.000300' "
                      "ORDER BY date", cx)
    ind = pd.read_sql("SELECT symbol, industry_l1 FROM tdx_stock_industry", cx)
    cx.close()
    ind_map = dict(zip(ind["symbol"], ind["industry_l1"]))
    cal = np.array(sorted(set(idx["date"])))
    icl = dict(zip(idx["date"], idx["close"].astype(float)))
    cm = {s: (g["date"].to_numpy(), g["close"].to_numpy(float), g["amount"].to_numpy(float))
          for s, g in px.groupby("symbol", sort=False)}
    cache = json.load(open(CACHE))
    print(f"  {len(cm):,} 只 / {len(cal):,} 日 / 缓存 {len(cache)} 个月", flush=True)

    # ── 先把每个月算一次：各权重共用的（y_raw, y_neu, rank_t2, rank_t4）──
    per = {}          # month → dict(h → (y_raw, y_neu, r2, r4))
    for m in sorted(cache):
        pv = cal[cal < f"{m}-01"]
        if len(pv) < 70:
            continue
        pos = int(np.searchsorted(cal, pv[-1]))
        rec = {}
        syms = cache[m]["symbols"]
        p2 = np.array(cache[m]["t2"], dtype=float)
        p4 = np.array(cache[m]["t4"], dtype=float)
        for H in HS:
            if pos + H >= len(cal):
                continue
            d0, d1 = cal[pos + 1], cal[pos + H]
            ir = icl[d1] / icl[d0] - 1.0
            ys, i2, i4, amt, vol, inds = [], [], [], [], [], []
            for k, s in enumerate(syms):
                a = cm.get(s)
                if a is None:
                    continue
                dt, cl, am = a
                j0, j1 = np.searchsorted(dt, d0), np.searchsorted(dt, d1)
                if j0 >= len(dt) or j1 >= len(dt) or dt[j0] != d0 or dt[j1] != d1 or j0 - 20 < 0:
                    continue
                ys.append(np.clip(cl[j1] / cl[j0] - 1.0 - ir, -CLIP, CLIP))
                i2.append(k); i4.append(k)
                amt.append(np.log(max(np.nanmean(am[max(j0 - 20, 0):j0 + 1]), 1.0)))
                r = np.diff(cl[max(j0 - 20, 0):j0 + 1]) / cl[max(j0 - 20, 0):j0]
                vol.append(np.std(r) if len(r) > 1 else 0.0)
                inds.append(ind_map.get(s, "未知"))
            if len(ys) < 100:
                continue
            # 中性化：y ~ 行业 + 成交额分位 + 波动分位 → 残差
            d = pd.DataFrame({"y": ys, "amt": amt, "vol": vol, "ind": inds})
            d["amt_q"] = pd.qcut(d["amt"], 5, labels=False, duplicates="drop")
            d["vol_q"] = pd.qcut(d["vol"], 5, labels=False, duplicates="drop")
            D = pd.get_dummies(d[["ind", "amt_q", "vol_q"]].astype(str), drop_first=True)
            X = np.column_stack([np.ones(len(d)), D.to_numpy(dtype=float)])
            beta, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
            rec[H] = (np.array(ys), d["y"].to_numpy() - X @ beta,
                      p2[i2], p4[i4])
        if rec:
            per[m] = rec

    # ── 扫权重 ──
    rows = []
    for w in WEIGHTS:
        row = {"w_t4": w}
        for H in HS:
            for seg_name, lo, hi in [("全段", "2020-01", "2026-12")] + SEGS:
                ics_raw, ics_neu, tops = [], [], []
                for m, rec in per.items():
                    if H not in rec or not (lo <= m <= hi):
                        continue
                    y, y_neu, p2, p4 = rec[H]
                    # 秩降序（rankdata(-x) 的等价写法，用 pandas 简化）
                    r2 = pd.Series(p2).rank().to_numpy()
                    r4 = pd.Series(p4).rank().to_numpy()
                    score = w * r4 + (1 - w) * r2
                    ics_raw.append(spearmanr(score, y).statistic)
                    ics_neu.append(spearmanr(score, y_neu).statistic)
                    if H == 20:
                        k = min(10, len(y))
                        tops.append(float(np.mean(y[np.argsort(-score)[:k]])))
                a = np.array(ics_raw, float); a = a[~np.isnan(a)]
                b = np.array(ics_neu, float); b = b[~np.isnan(b)]
                if len(a) < 10:
                    continue
                t_raw = newey_west_t(a, max(0, H // 20 - 1))[0]
                row[f"{seg_name}_IC{H}"] = a.mean()
                row[f"{seg_name}_t{H}"] = t_raw
                if H == 20:
                    row[f"{seg_name}_ICIR20"] = a.mean() / a.std(ddof=1)
                    row[f"{seg_name}_neuIC20"] = b.mean() if len(b) else np.nan
                    row[f"{seg_name}_TOP10"] = float(np.mean(tops)) if tops else np.nan
                    row[f"{seg_name}_n"] = len(a)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"t4_weight_sweep_{TAG}.csv", index=False, encoding="utf-8-sig")

    def show(seg: str):
        print(f"\n{'='*104}\n【{seg}】\n{'='*104}")
        print(f"{'w_t4':>5s} {'月数':>5s} {'IC20':>9s} {'ICIR20':>7s} {'t20':>6s} "
              f"{'中性化IC20':>10s} {'TOP10超额':>9s} {'IC60':>9s}")
        print("-" * 104)
        for _, r in df.iterrows():
            if f"{seg}_IC20" not in r or pd.isna(r.get(f"{seg}_IC20")):
                continue
            print(f"{r['w_t4']:>5.1f} {int(r[f'{seg}_n']):>5d} {r[f'{seg}_IC20']:>+9.4f} "
                  f"{r[f'{seg}_ICIR20']:>+7.3f} {r[f'{seg}_t20']:>+6.2f} "
                  f"{r[f'{seg}_neuIC20']:>+10.4f} {r[f'{seg}_TOP10']*100:>+8.2f}% "
                  f"{r[f'{seg}_IC60']:>+9.4f}")

    for seg in ["全段"] + [s[0] for s in SEGS]:
        show(seg)

    # ── 候选"权重策略"对照（2026-09-27 追加：共享函数该用哪条公式？）──
    #   背景：回测引擎用自适应 0.40+0.30q（q=min(std_t4/0.02,1)），生产用固定 0.5。
    #   抽成共享函数必须二选一 ⇒ 用同一批数据把两个候选比出来，别凭喜好定。
    print(f"\n{'='*104}")
    print("【权重策略对照】固定 0.5（=生产现状） vs 自适应 0.40+0.30q（=引擎现状）")
    print("=" * 104)
    policies = {
        "固定 0.50（生产现状）": lambda std, n: np.full(n, 0.50),
        "自适应 0.40+0.30q（引擎现状）": lambda std, n: np.full(n, 0.40 + 0.30 * min(std / 0.02, 1.0)),
        "固定 0.60": lambda std, n: np.full(n, 0.60),
    }
    for seg_name, lo, hi in SEGS:
        print(f"\n  【{seg_name}】")
        print(f"  {'策略':26s} {'IC20':>9s} {'ICIR20':>7s} {'中性化IC20':>10s} {'TOP10超额':>9s} {'IC60':>9s}")
        for pname, pol in policies.items():
            ic20, icn20, tops, ic60 = [], [], [], []
            for m, rec in per.items():
                if not (lo <= m <= hi):
                    continue
                for H, acc in ((20, ic20), (60, ic60)):
                    if H not in rec:
                        continue
                    y, y_neu, p2, p4 = rec[H]
                    r2 = pd.Series(p2).rank().to_numpy()
                    r4 = pd.Series(p4).rank().to_numpy()
                    w4 = float(pol(float(np.std(p4)), 1)[0])
                    score = w4 * r4 + (1 - w4) * r2
                    acc.append(spearmanr(score, y).statistic)
                    if H == 20:
                        icn20.append(spearmanr(score, y_neu).statistic)
                        k = min(10, len(y))
                        tops.append(float(np.mean(y[np.argsort(-score)[:k]])))
            a = np.array(ic20, float); a = a[~np.isnan(a)]
            if len(a) < 10:
                continue
            print(f"  {pname:26s} {a.mean():>+9.4f} {a.mean()/a.std(ddof=1):>+7.3f} "
                  f"{np.nanmean(icn20):>+10.4f} {np.mean(tops)*100:>+8.2f}% "
                  f"{np.nanmean(ic60):>+9.4f}")

    print(f"\n{'='*104}")
    print("【怎么读】只认两段**一致**的结论：若拟合段与检验段的最优 w_t4 不同，")
    print("  说明'最优权重'是噪声（样本内优化），应维持现状或改走动态加权。")
    print(f"明细: {OUT}/t4_weight_sweep_{TAG}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
