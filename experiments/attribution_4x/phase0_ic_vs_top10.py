#!/usr/bin/env python
"""Phase 0：验证「平均 vs 顶端」机制是否在**其它脱节场景**中也成立

研究计划 #19 的 Phase 0（`docs/2026-09-29_研究计划19_IC与组合层脱节.md`）。

问题
----
2026-09-29 在**抽样 A/B** 上锁定了机制：
  全横截面（~2900 只）Rank IC 高的一边，**实际买入的 TOP10 反而更差**。
但那只验证了**一个**场景。本脚本把同一判别用在**另两次脱节**上：

| 场景 | IC 层面 | 组合层 |
|---|---|---|
| ② 组合层中性化 | 中性化**更好**（p=0.002） | 组合**不显著**（p=0.291） |
| 18 月口径 tail vs random | random 更好 | （当年未做钱口径） |

判据
----
若"IC 上升但 TOP10 不升（甚至下降）"在②也成立 ⇒ 机制**可推广**，
主指标应正式改为「TOP10 事后超额」。

算法与 `t4_ic.py` **逐字一致**（同 y 口径、同融合=秩平均、同 TOP10 定义），
锚点用缓存语义（`ref_date` = 上月最后交易日），避免重蹈 `analyze_monthly_ic.py` 的锚点错位。

用法: python experiments/attribution_4x/phase0_ic_vs_top10.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent / "out"
H, TOPK = 20, 10

CELLS = [
    ("②中性化·原始  ", ".purged_70m.json", "t2"),
    ("②中性化·中性化", ".t2_neutralized.json", "t2"),
    ("18月·tail     ", ".t4_test_18m.json", "fuse"),
    ("18月·random   ", ".t4_test_18m_random.json", "fuse"),
    ("70月·tail     ", ".t4_70m_tail.json", "fuse"),
    ("70月·random   ", ".t4_70m_random.json", "fuse"),
]


def fuse(p2: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """与 t4_ic.py:127-130 一致：两模型秩的平均。"""
    return (pd.Series(p2).rank().to_numpy() + pd.Series(p4).rank().to_numpy()) / 2.0


def main() -> int:
    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    print("加载行情（2019-12 起）...", flush=True)
    px = pd.read_sql("SELECT symbol,date,close FROM stock_daily WHERE date>='2019-12-01' "
                     "AND close>0 ORDER BY symbol,date", cx)
    idx = pd.read_sql("SELECT date,close FROM index_daily WHERE symbol='sh.000300' "
                      "ORDER BY date", cx)
    cx.close()
    cm = {s: (g["date"].to_numpy(), g["close"].to_numpy(float))
          for s, g in px.groupby("symbol", sort=False)}
    cal = np.array(sorted(set(idx["date"])))
    icl = dict(zip(idx["date"], idx["close"].astype(float)))
    print(f"  个股 {len(cm):,} 只 / 交易日 {len(cal):,} 天", flush=True)

    rows = []
    for label, fname, field in CELLS:
        p = ROOT / "output/backtest_v2" / fname
        if not p.exists():
            print(f"  ⚠️ {label} 缓存缺失: {fname}，跳过")
            continue
        cache = json.load(open(p))
        ics, tops = [], []
        for m in sorted(cache):
            # ref_date = 上月最后交易日（= 缓存语义；锚点错位是本项目的重大历史教训）
            prev = cal[cal < f"{m}-01"]
            if len(prev) == 0:
                continue
            ref = prev[-1]
            i0 = int(np.searchsorted(cal, ref))
            if i0 + H >= len(cal):
                continue
            d_end = cal[i0 + H]
            ys, ps = [], []
            for s in cache[m]["symbols"]:
                if s not in cm:
                    continue
                ds, cs = cm[s]
                j = int(np.searchsorted(ds, ref))
                k = int(np.searchsorted(ds, d_end))
                if j >= len(ds) or k >= len(ds) or ds[j] != ref or ds[k] != d_end or cs[j] <= 0:
                    continue
                ys.append(cs[k] / cs[j] - 1.0)
                ps.append(s)
            if len(ys) < 100:
                continue
            ys = np.asarray(ys)
            ir = icl[d_end] / icl[ref] - 1.0
            y = np.clip(ys - ir, -0.5, 0.5)
            pos = {s: i for i, s in enumerate(cache[m]["symbols"])}
            t2 = np.array([cache[m]["t2"][pos[s]] for s in ps], float)
            if field == "fuse":
                t4 = np.array([cache[m]["t4"][pos[s]] for s in ps], float)
                if np.std(t4) < 1e-9:      # T4 未跑成（全 0）→ 退化为纯 T2
                    pred = t2
                else:
                    pred = fuse(t2, t4)
            else:
                pred = t2
            ics.append(spearmanr(pred, y).statistic)
            tops.append(float(np.mean(y[np.argsort(-pred)[:TOPK]])))
        ics, tops = np.array(ics), np.array(tops)
        t_ic, _ = ttest_1samp(ics, 0)
        t_tp, p_tp = ttest_1samp(tops, 0)
        rows.append({"场景": label, "月数": len(ics), "全池IC20": ics.mean(),
                     "ICIR": ics.mean() / ics.std(), "IC>0": (ics > 0).mean(),
                     "TOP10超额%": tops.mean() * 100, "TOP10_t": t_tp, "TOP10_p": p_tp})

    df = pd.DataFrame(rows)
    print("\n" + "=" * 104)
    print("★ Phase 0：同一判别用在多个场景 —— IC 与「实际买入的 TOP10」是否同向？")
    print("=" * 104)
    show = df.copy()
    for c, f in [("全池IC20", "{:+.4f}"), ("ICIR", "{:.3f}"), ("IC>0", "{:.0%}"),
                 ("TOP10超额%", "{:+.2f}"), ("TOP10_t", "{:.2f}"), ("TOP10_p", "{:.3f}")]:
        show[c] = show[c].map(f.format)
    print(show.to_string(index=False))
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "phase0_ic_vs_top10.csv", index=False, encoding="utf-8-sig")
    print(f"\n明细: {OUT / 'phase0_ic_vs_top10.csv'}")

    print("\n── 读法 ──")
    for i in range(0, len(df), 2):
        if i + 1 >= len(df):
            break
        a, b = df.iloc[i], df.iloc[i + 1]
        dic = b["全池IC20"] - a["全池IC20"]
        dtp = b["TOP10超额%"] - a["TOP10超额%"]
        verdict = "同向" if np.sign(dic) == np.sign(dtp) else "★反向（机制成立）"
        print(f"  {a['场景'].strip()} → {b['场景'].strip()}: "
              f"IC {dic:+.4f}（{'升' if dic > 0 else '降'}） ｜ "
              f"TOP10 {dtp:+.2f}pp（{'升' if dtp > 0 else '降'}） ⇒ {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
