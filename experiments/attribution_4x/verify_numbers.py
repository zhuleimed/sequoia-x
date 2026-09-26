#!/usr/bin/env python3
"""数字溯源核验（2026-09-27）：把研究文档里引用过的 IC/收益数字，逐个回到**产物文件**上复算。

背景：用户质疑引用的数字。已发现一处真错（深市 120.2 万行被写成 189 万）。
本脚本用**同一份行情、同一个 y 口径**（与 ic_compare.py / cheap_checks.py / t4_ic.py 一致）
批量复算各缓存的关键指标，输出一行一个缓存，便于与文档逐条对照。

口径：
  ref_date = 该月上一个月的最后一个交易日
  y        = clip(个股 20 日收益 − 沪深300 同期收益, −0.5, 0.5)
  IC       = 逐月 Spearman，再取均值（**不是池化**——池化会把月间涨跌算进来，见 2026-09-27 结论）
  TOP10    = 按预测排序取前 10 只的事后平均超额收益（经济口径）

用法：
    python experiments/attribution_4x/verify_numbers.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3
from scipy import stats
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "output/backtest_v2"
CLIP, H = 0.5, 20

# (标签, 缓存路径) —— 覆盖文档里出现过的所有 IC 数字的口径
TARGETS = [
    ("V4_70m_tail      ", C / "prediction_cache_v4_60m.json"),
    ("V5_70m_tail      ", C / "prediction_cache_v5_60m.json"),
    ("purged_70m       ", C / ".purged_70m.json"),
    ("18m_purge_off    ", C / ".purge_off.json"),      # 泄漏版（tail）
    ("18m_purge_on     ", C / ".purge_on.json"),       # 去泄漏版
    ("18m_sample_random", C / ".sample_random.json"),
    ("18m_random_purged", C / ".random_purged.json"),
]


def main() -> int:
    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    px = pd.read_sql("SELECT symbol,date,close FROM stock_daily WHERE date>='2019-12-01' "
                     "AND close>0 ORDER BY symbol,date", cx)
    idx = pd.read_sql("SELECT date,close FROM index_daily WHERE symbol='sh.000300' "
                      "ORDER BY date", cx)
    cx.close()
    cm = {s: (g["date"].to_numpy(), g["close"].to_numpy(float))
          for s, g in px.groupby("symbol", sort=False)}
    cal = np.array(sorted(set(idx["date"])))
    icl = dict(zip(idx["date"], idx["close"].astype(float)))
    print(f"行情：{len(cm):,} 只 / {len(cal):,} 交易日\n")

    print(f"{'缓存':18s} {'月数':>4s} {'覆盖区间':22s} {'T2均值IC':>9s} {'ICIR':>7s} "
          f"{'IC>0':>5s} {'t':>6s} {'p':>6s} {'TOP10':>8s} {'全池':>7s}")
    print("-" * 108)
    for tag, path in TARGETS:
        if not path.exists():
            print(f"{tag} 缺失: {path}")
            continue
        cache = __import__("json").load(open(path))
        ics, tops, alls = [], [], []
        for m in sorted(cache):
            pv = cal[cal < f"{m}-01"]
            if len(pv) < H + 2:
                continue
            pos = int(np.searchsorted(cal, pv[-1]))
            if pos + H >= len(cal):
                continue
            d0, d1 = cal[pos + 1], cal[pos + H]
            ir = icl[d1] / icl[d0] - 1.0
            ys, ps = [], []
            for s, p in zip(cache[m]["symbols"], cache[m]["t2"]):
                a = cm.get(s)
                if a is None:
                    continue
                dt, cl = a
                j0, j1 = np.searchsorted(dt, d0), np.searchsorted(dt, d1)
                if j0 >= len(dt) or j1 >= len(dt) or dt[j0] != d0 or dt[j1] != d1:
                    continue
                ys.append(np.clip(cl[j1] / cl[j0] - 1.0 - ir, -CLIP, CLIP))
                ps.append(p)
            if len(ys) < 100:
                continue
            ys, ps = np.array(ys), np.array(ps, dtype=float)
            ics.append(spearmanr(ps, ys).statistic)
            k = min(10, len(ys))
            tops.append(float(np.mean(ys[np.argsort(-ps)[:k]])))
            alls.append(float(np.mean(ys)))
        a = np.array(ics, dtype=float)
        a = a[~np.isnan(a)]
        t, p = stats.ttest_1samp(a, 0)
        ms = sorted(cache)
        print(f"{tag:18s} {len(a):4d} {ms[0]}~{ms[-1]} {a.mean():+9.4f} "
              f"{a.mean()/a.std(ddof=1):+7.3f} {(a>0).mean()*100:4.0f}% "
              f"{t:+6.2f} {p:6.3f} {np.mean(tops)*100:+7.2f}% {np.mean(alls)*100:+6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
