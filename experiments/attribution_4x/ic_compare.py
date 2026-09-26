#!/usr/bin/env python3
"""4.9× 归因 · 第②层：V4 与 V5 的预测，谁排得更准？

为什么先做这一层
----------------
已确认**回测引擎在两版之间没有实质变化**（`git diff 0cd1dcf HEAD -- monthly_engine.py`
只新增了默认关闭的 `intra_exit_policy`，`today_opened=(di==0)`→`False` 因外层已
`continue` 而等价）。所以 4.9× 的差异**必然全部来自预测本身** —— 即"选了哪些股票"。

那就可以直接度量：拿 V4 与 V5 两份预测缓存，在**同样的 70 个月**上算
**已实现的 Rank IC**（预测值 vs 事后真实 20 日超额收益）。
这不需要跑回测，是纯计算。

口径
----
严格复刻 `labels.py::_label_t2`：
  · ref_date = 该月**上一个月的最后一个交易日**（回测日志口径："[N/70] 2026-06 训练截止=2026-05-29"）
  · stock_ret = close(第20个交易日) / close(第1个交易日) - 1   （均为 ref_date **之后**）
  · idx_ret   同法，用 index_daily 的 sh.000300
  · y = clip(stock_ret - idx_ret, -0.5, 0.5)
  · IC = Spearman(pred_t2, y)

用法
----
    python experiments/attribution_4x/ic_compare.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "sequoia_v2.db"
CACHES = {
    "V4": ROOT / "output/backtest_v2/prediction_cache_v4_60m.json",
    "V5": ROOT / "output/backtest_v2/prediction_cache_v5_60m.json",
}
OUT = Path(__file__).resolve().parent / "out"
H = 20          # T2 预测期（交易日）
CLIP = 0.5


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)

    print("加载行情（2020 起）...", flush=True)
    px = pd.read_sql(
        "SELECT symbol, date, close FROM stock_daily WHERE date >= '2019-12-01' "
        "AND close > 0 ORDER BY symbol, date", conn)
    idx = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol='sh.000300' AND date >= '2019-12-01' "
        "ORDER BY date", conn)
    conn.close()
    print(f"  个股 {len(px):,} 行 / 指数 {len(idx):,} 行", flush=True)

    # 每只股票的 (日期数组, 收盘数组)，用 searchsorted 定位
    close_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, g in px.groupby("symbol", sort=False):
        close_map[sym] = (g["date"].to_numpy(), g["close"].to_numpy(dtype=float))
    cal = np.array(sorted(set(idx["date"])))
    idx_close = dict(zip(idx["date"], idx["close"].astype(float)))
    print(f"  个股 {len(close_map):,} 只 / 交易日 {len(cal):,} 天", flush=True)

    caches = {k: json.load(open(v)) for k, v in CACHES.items()}
    months = sorted(caches["V5"])
    assert months == sorted(caches["V4"]), "两份缓存的月份不一致"

    rows = []
    for i, m in enumerate(months, 1):
        # ref_date = 上个月最后一个交易日
        prev_cal = cal[cal < f"{m}-01"]
        if len(prev_cal) < H + 2:
            continue
        ref = prev_cal[-1]
        pos = int(np.searchsorted(cal, ref))
        if pos + H >= len(cal):
            continue
        d0, d1 = cal[pos + 1], cal[pos + H]          # 第1个 / 第20个交易日（ref 之后）
        i0, i1 = idx_close.get(d0), idx_close.get(d1)
        if i0 is None or i1 is None:
            continue
        idx_ret = i1 / i0 - 1.0

        rec = {"month": m, "ref_date": ref, "d0": d0, "d1": d1}
        for tag, cache in caches.items():
            syms = cache[m]["symbols"]
            pred = np.array(cache[m]["t2"], dtype=float)
            ys, ps = [], []
            for s, p in zip(syms, pred):
                a = close_map.get(s)
                if a is None:
                    continue
                dts, cls = a
                j0 = np.searchsorted(dts, d0)
                j1 = np.searchsorted(dts, d1)
                if j0 >= len(dts) or j1 >= len(dts) or dts[j0] != d0 or dts[j1] != d1:
                    continue                              # 停牌/缺日 → 剔除
                y = cls[j1] / cls[j0] - 1.0 - idx_ret
                ys.append(np.clip(y, -CLIP, CLIP))
                ps.append(p)
            if len(ys) < 100:
                rec[f"{tag}_ic"] = None
                continue
            ys, ps = np.array(ys), np.array(ps)
            rec[f"{tag}_ic"] = float(spearmanr(ps, ys).statistic)
            rec[f"{tag}_n"] = len(ys)
            # 经济口径：各自 TOP10 的事后平均超额收益
            k = min(10, len(ys))
            rec[f"{tag}_top10_ret"] = float(np.mean(ys[np.argsort(-ps)[:k]]))
            rec[f"{tag}_all_ret"] = float(np.mean(ys))
        rows.append(rec)
        if i % 10 == 0 or i == len(months):
            print(f"  [{i}/{len(months)}] {m} 完成", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "monthly_ic.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 88)
    print("【预测质量对比：已实现 Rank IC（70 个月，同口径同月份）】")
    print("=" * 88)
    for tag in ("V4", "V5"):
        c = df[f"{tag}_ic"].dropna()
        tops = df[f"{tag}_top10_ret"].dropna()
        print(f"  {tag}: 有效月数 {len(c)}  |  均值 IC={c.mean():+.4f}  std={c.std(ddof=1):.4f}  "
              f"ICIR={c.mean()/c.std(ddof=1):+.3f}  IC>0 占比 {(c>0).mean()*100:.0f}%"
              f"  |  TOP10 事后月均超额={tops.mean()*100:+.2f}%")
    c4, c5 = df["V4_ic"].dropna(), df["V5_ic"].dropna()
    com = df.dropna(subset=["V4_ic", "V5_ic"])
    d = com["V5_ic"] - com["V4_ic"]
    from scipy import stats
    t, p = stats.ttest_rel(com["V5_ic"], com["V4_ic"])
    print(f"\n  配对（n={len(com)} 个月）：V5 − V4 均值差={d.mean():+.4f}  "
          f"t={t:+.2f}  p={p:.4f}  {'✅ 显著' if p<0.05 else '❌ 不显著'}")
    print(f"  V5 更好的月份: {(d>0).sum()}/{len(d)}")
    print(f"\n明细: {OUT/'monthly_ic.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
