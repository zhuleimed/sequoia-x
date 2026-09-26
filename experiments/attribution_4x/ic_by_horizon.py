#!/usr/bin/env python3
"""多周期 + 中性化 IC 评估（2026-09-27）

为什么做这个
------------
目前的结论卡在"测不准"上：月度 IC ≈ 0.015、月频 σ≈0.11，69 个月观测 ⇒ t≈1.09（不显著）。
用户希望"日频 IC 提高功效"。**先纠正一个我自己的说法**：光把标签换成日频**不会**提高功效 ——
预测本身是**每月一次**的横截面（模型月频重训），独立观测数就是月数（69），
日内只是把同一个月的结果反复重采样。真正能做的有三件：

  ① **IC 衰减曲线**（1/3/5/10/20/40/60 交易日）→ 回答"信号活在哪个周期"，
     直接决定该不该把月度换仓改成季度换仓/信号平滑；
  ② **中性化 IC**（行业 + 市值/流动性 + 波动）→ 降 σ(IC)、提高 t，
     同时回答"这是 alpha 还是行业/风格暴露"；
  ③ **功效分析** → 把"测不出来"量化成"还需要 N 个月"（`stock_daily` 最早 2020-01-02，
     无法向后扩样本，所以这是硬约束）。

口径（与 ic_compare.py / cheap_checks.py / t4_ic.py 一致，保证可比）
------------------------------------------------------------------
  · ref_date = 该月**上一个月的最后一个交易日**
  · d0 = ref 之后**第 1 个**交易日，dH = ref 之后**第 H 个**交易日
  · y_H = clip(个股 dH/d0 − 1 − 沪深300 同期, ±0.5)      ← 与 labels.py 一致（不含第 1 日）
  · IC = 逐月 Spearman，再取月度均值（**不是池化**）
  · 重叠修正：H 日的窗口跨 ⌈H/20⌉ 个月 → 月度 IC 序列有自相关，
    用 **Newey-West** 标准误修正 t（否则 60 日的 t 会被系统性放大）

中性化（y_neutral 的做法）
--------------------------
  用最小二乘把 y 对以下因子回归，取**残差**：
    行业哑变量（`tdx_stock_industry.industry_l1`，申万一级）
    + 市值/流动性代理分位（过去 20 日平均成交额的对数，5 档）
    + 波动分位（过去 20 日收益率标准差，5 档）
  注：行业分类是**当前快照**（2026-07-24），历史月份用它属近似 —— 行业调整不频繁，可接受。

用法
----
    python experiments/attribution_4x/ic_by_horizon.py                       # 默认 purged_70m（T2）
    python experiments/attribution_4x/ic_by_horizon.py --cache=<路径> --tag=xxx
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
CACHE = ROOT / "output/backtest_v2/.purged_70m.json"
PRED_FIELD = "t2"          # 评哪个预测头（t2 / t4 / 融合另算）
TAG = "t2_70m"
for _a in sys.argv[1:]:
    if _a.startswith("--cache="):
        CACHE = Path(_a[len("--cache="):])
    elif _a.startswith("--tag="):
        TAG = _a[len("--tag="):]
    elif _a.startswith("--field="):
        PRED_FIELD = _a[len("--field="):]

# 第 H 个交易日的口径（H=1 退化：只有 1 天，d0=d1 → 收益恒为 0，故从 2 开始）
HORIZONS = [2, 3, 5, 10, 20, 40, 60]
CLIP = 0.5


def newey_west_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    """对月度 IC 序列做 Newey-West 修正的 t 检验（重叠窗口必需）。

    朴素 t 假设各月独立；但 H 日窗口跨多个月 → 相邻月的 y 重叠 → IC 自相关 →
    t 被放大。NW 用自协方差修标准误：
        SE² = γ0 + 2·Σ_{k=1..lag} (1 − k/(lag+1))·γ_k
    """
    n = len(x)
    mu = float(np.mean(x))            # ← 先存均值，再离差化（否则 t 恒为 0）
    xc = x - mu
    g0 = float(np.dot(xc, xc) / n)
    s = g0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(np.dot(xc[:-k], xc[k:]) / n)
        s += 2.0 * (1.0 - k / (lag + 1)) * gk
    se = np.sqrt(max(s, 1e-18) / n)
    t = float(mu / se) if se > 0 else 0.0
    p = float(2 * (1 - stats.t.cdf(abs(t), df=max(n - 1, 1))))
    return t, p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    print("加载行情与行业…", flush=True)
    px = pd.read_sql("SELECT symbol,date,close,amount FROM stock_daily "
                     "WHERE date>='2019-12-01' AND close>0 ORDER BY symbol,date", cx)
    idx = pd.read_sql("SELECT date,close FROM index_daily WHERE symbol='sh.000300' "
                      "ORDER BY date", cx)
    ind = pd.read_sql("SELECT symbol, industry_l1 FROM tdx_stock_industry", cx)
    cx.close()
    ind_map = dict(zip(ind["symbol"], ind["industry_l1"]))

    cal = np.array(sorted(set(idx["date"])))
    icl = dict(zip(idx["date"], idx["close"].astype(float)))
    # 每只股票：日期 / 收盘 / 成交额（做市值-流动性代理）
    cm = {s: (g["date"].to_numpy(), g["close"].to_numpy(float),
              g["amount"].to_numpy(float)) for s, g in px.groupby("symbol", sort=False)}
    print(f"  个股 {len(cm):,} 只 / 交易日 {len(cal):,} 天 / 行业 {len(ind_map):,} 只", flush=True)

    cache = json.load(open(CACHE))
    months = sorted(cache)
    print(f"  缓存 {len(months)} 个月：{months[0]} ~ {months[-1]}，预测头={PRED_FIELD}\n", flush=True)

    rows = []
    for m in months:
        pv = cal[cal < f"{m}-01"]
        if len(pv) < 70:
            continue
        pos = int(np.searchsorted(cal, pv[-1]))
        rec = {"month": m, "ref_date": pv[-1]}
        syms = cache[m]["symbols"]
        pred = np.array(cache[m][PRED_FIELD], dtype=float)
        for H in HORIZONS:
            if pos + H >= len(cal):
                continue
            d0, d1 = cal[pos + 1], cal[pos + H]
            ir = icl[d1] / icl[d0] - 1.0
            ys, ps, amt, vol, inds = [], [], [], [], []
            for s, p in zip(syms, pred):
                a = cm.get(s)
                if a is None:
                    continue
                dt, cl, am = a
                j0, j1 = np.searchsorted(dt, d0), np.searchsorted(dt, d1)
                if j0 >= len(dt) or j1 >= len(dt) or dt[j0] != d0 or dt[j1] != d1:
                    continue
                # 停牌/缺日剔除：d0、d1 都要有价
                if j0 - 20 < 0:
                    continue
                ys.append(np.clip(cl[j1] / cl[j0] - 1.0 - ir, -CLIP, CLIP))
                ps.append(p)
                amt.append(np.log(max(np.nanmean(am[max(j0 - 20, 0):j0 + 1]), 1.0)))
                r = np.diff(cl[max(j0 - 20, 0):j0 + 1]) / cl[max(j0 - 20, 0):j0]
                vol.append(np.std(r) if len(r) > 1 else 0.0)
                inds.append(ind_map.get(s, "未知"))
            if len(ys) < 100:
                continue
            ys = np.array(ys)
            ps = np.array(ps, dtype=float)
            rec[f"ic_raw_{H}"] = float(spearmanr(ps, ys).statistic)
            # ── 中性化：y ~ 行业 + 成交额分位 + 波动分位，取残差 ──
            d = pd.DataFrame({"y": ys, "amt": amt, "vol": vol, "ind": inds})
            d["amt_q"] = pd.qcut(d["amt"], 5, labels=False, duplicates="drop")
            d["vol_q"] = pd.qcut(d["vol"], 5, labels=False, duplicates="drop")
            D = pd.get_dummies(d[["ind", "amt_q", "vol_q"]].astype(str), drop_first=True)
            X = np.column_stack([np.ones(len(d)), D.to_numpy(dtype=float)])
            beta, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
            resid = d["y"].to_numpy() - X @ beta
            rec[f"ic_neu_{H}"] = float(spearmanr(ps, resid).statistic)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"ic_by_horizon_{TAG}.csv", index=False, encoding="utf-8-sig")

    print("=" * 100)
    print(f"【IC 衰减曲线 + 中性化对照】{CACHE.name}，{len(df)} 个月，预测头={PRED_FIELD}")
    print("=" * 100)
    print(f"{'H(交易日)':>9s} {'口径':>6s} {'均值IC':>9s} {'σ':>7s} {'ICIR':>7s} {'IC>0':>5s} "
          f"{'朴素t':>7s} {'NW_t':>7s} {'NW_p':>7s} {'显著':>4s}")
    print("-" * 100)
    for H in HORIZONS:
        for kind in ("raw", "neu"):
            col = f"ic_{kind}_{H}"
            if col not in df:
                continue
            c = df[col].dropna()
            if len(c) < 10:
                continue
            t0, _ = stats.ttest_1samp(c, 0)
            lag = max(1, int(round(H / 20))) - 1          # 重叠窗口的月数（H=20→0）
            tn, pn = newey_west_t(c.to_numpy(), lag) if lag > 0 else (t0, float(stats.ttest_1samp(c, 0).pvalue))
            lbl = "原始" if kind == "raw" else "中性化"
            print(f"{H:>9d} {lbl:>6s} {c.mean():>+9.4f} {c.std(ddof=1):>7.4f} "
                  f"{c.mean()/c.std(ddof=1):>+7.3f} {(c>0).mean()*100:>4.0f}% "
                  f"{t0:>+7.2f} {tn:>+7.2f} {pn:>7.3f} {'✅' if pn<0.05 else '  ':>4s}")

    # ── 功效分析：给定观测到的 μ 与 σ，要多少个月才能到 80% 功效 ──
    print("\n" + "=" * 100)
    print("【功效分析】要在 α=0.05、功效 80% 下检出观测到的 IC，需要多少个月？（n = ((1.96+0.84)·σ/μ)²）")
    print("=" * 100)
    for H in HORIZONS:
        col = f"ic_neu_{H}"
        if col not in df:
            continue
        c = df[col].dropna()
        if len(c) < 10 or c.mean() <= 0:
            continue
        need = ((1.96 + 0.84) * c.std(ddof=1) / c.mean()) ** 2
        print(f"  H={H:>2d} 日：均值 IC={c.mean():+.4f}  σ={c.std(ddof=1):.4f} "
              f"→ 需要 **{need:.0f} 个月（{need/12:.1f} 年）**；现有 {len(c)} 个月")
    print("\n  注：这是**中性化后**的功效。stock_daily 最早 2020-01-02，历史无法再向前延伸，"
          "\n     所以「需要 N 个月」意味着只能靠**时间累积**或**提高预测频率**，不能靠加历史。")
    print(f"\n明细: {OUT}/ic_by_horizon_{TAG}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
