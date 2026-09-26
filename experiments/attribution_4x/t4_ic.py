#!/usr/bin/env python3
"""T4（LSTM）补测的分析层：拿昨晚建好的 purge 缓存，算 T4 的真实 IC。

背景
----
本轮"70 月回测前视泄漏"的所有结论只覆盖 **T2（LightGBM）**：
我构建的缓存与 70 月回测都带 `--skip-t4`，缓存里的 t4 字段全是 0。
而 T4 在生产 Step2 参与 T2+T4 Rank 融合选股，**从未被本轮检验**。

昨晚（2026-09-26 20:13~22:26）用 `scripts/run_t4_check.sh` 补建了一份
**含 T4** 的缓存：2025-01 ~ 2026-06 共 18 个月、purge 25 交易日（去泄漏）、
lastday_agg 视图、workers=2。本脚本对它做纯计算分析。

要回答的问题
------------
1. 去泄漏后，T4 的已实现 Rank IC 是多少？（对比项目自身历史测量 Fold3 +0.0712 /
   Fold4 +0.1007 —— 那些走的是**路径 B**，泄漏 <1%，理论上不该被本轮否定）
2. 与 T2 同口径对比，谁更强？
3. T4 的 IC 在不同预测期（5/20/60 日）上表现如何？
4. T2 与 T4 的预测**相关性**多高？（决定融合是否还能带来增量）

口径（与 ic_compare.py / cheap_checks.py 完全一致，保证可比）
------------------------------------------------------------
  · ref_date = 该月**上一个月的最后一个交易日**
  · y = clip(个股 20 日收益 − 沪深300 同期收益, −0.5, 0.5)
  · IC = Spearman(预测值, y)，逐月算，再做月度统计（均值 / ICIR / t 检验）

用法
----
    python experiments/attribution_4x/t4_ic.py                       # 默认：tail 抽样版
    python experiments/attribution_4x/t4_ic.py --cache=<路径> --tag=random
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
CACHE = ROOT / "output/backtest_v2/.t4_test_18m.json"   # 昨晚补建的含 T4 缓存
TAG = "tail"                                            # 输出文件名后缀
for _a in sys.argv[1:]:
    if _a.startswith("--cache="):
        CACHE = Path(_a[len("--cache="):])
    elif _a.startswith("--tag="):
        TAG = _a[len("--tag="):]
HS = [5, 20, 60]        # 预测期（交易日）
CLIP = 0.5              # y 的截断（与 labels.py 一致，防极端值主导）


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cx = sqlite3.connect(ROOT / "data/sequoia_v2.db")
    print("加载行情（2019-12 起，覆盖 70 个月口径）...", flush=True)
    px = pd.read_sql(
        "SELECT symbol,date,close FROM stock_daily WHERE date>='2019-12-01' AND close>0 "
        "ORDER BY symbol,date", cx)
    idx = pd.read_sql(
        "SELECT date,close FROM index_daily WHERE symbol='sh.000300' ORDER BY date", cx)
    cx.close()
    # 每只股票一个 (日期数组, 收盘数组)，用 searchsorted 定位日期
    cm = {s: (g["date"].to_numpy(), g["close"].to_numpy(float))
          for s, g in px.groupby("symbol", sort=False)}
    cal = np.array(sorted(set(idx["date"])))
    icl = {d: c for d, c in zip(idx["date"], idx["close"].astype(float))}
    print(f"  个股 {len(cm):,} 只 / 交易日 {len(cal):,} 天", flush=True)

    cache = json.load(open(CACHE))
    months = sorted(cache)
    print(f"  缓存 {len(months)} 个月：{months[0]} ~ {months[-1]}", flush=True)

    # ── 先做一次"T4 到底跑没跑成"的自检 ──
    # 训练失败时该月 t4 会是全 0（改动前是静默的），std≈0 即可识别。
    bad = [m for m in months if float(np.std(cache[m]["t4"])) < 1e-7]
    if bad:
        print(f"  ⚠️ 以下月份 T4 无方差（训练失败？）：{bad}", flush=True)
    else:
        print("  ✅ T4 自检通过：18 个月预测值均有方差（LSTM 确实跑成了）", flush=True)

    rows = []
    pool = {}          # 供"合并口径"（Fold 口径）复算：month → (pred_t4, pred_t2, y)
    for m in months:
        pv = cal[cal < f"{m}-01"]
        if len(pv) < max(HS) + 2:
            continue
        ref = pv[-1]
        pos = int(np.searchsorted(cal, ref))
        rec = {"month": m, "ref_date": ref}
        for H in HS:
            if pos + H >= len(cal):
                rec[f"t4_ic{H}"] = None
                rec[f"t2_ic{H}"] = None
                continue
            d0, d1 = cal[pos + 1], cal[pos + H]
            ir = icl[d1] / icl[d0] - 1.0                       # 指数同期收益
            ys, p4, p2 = [], [], []
            for s, a4, a2 in zip(cache[m]["symbols"], cache[m]["t4"], cache[m]["t2"]):
                a = cm.get(s)
                if a is None:
                    continue
                dt, cl = a
                j0 = np.searchsorted(dt, d0)
                j1 = np.searchsorted(dt, d1)
                # 停牌 / 缺日 → 剔除（d0、d1 两天都必须有价格）
                if j0 >= len(dt) or j1 >= len(dt) or dt[j0] != d0 or dt[j1] != d1:
                    continue
                ys.append(np.clip(cl[j1] / cl[j0] - 1.0 - ir, -CLIP, CLIP))
                p4.append(a4)
                p2.append(a2)
            if len(ys) < 100:
                rec[f"t4_ic{H}"] = None
                rec[f"t2_ic{H}"] = None
                continue
            ys = np.array(ys)
            p4 = np.array(p4, dtype=float)
            p2 = np.array(p2, dtype=float)
            rec[f"t4_ic{H}"] = float(spearmanr(p4, ys).statistic)
            rec[f"t2_ic{H}"] = float(spearmanr(p2, ys).statistic)
            # 生产口径：T2+T4 Rank 融合（秩平均）——生产选股用的是这个信号，不是单模型
            r2 = pd.Series(p2).rank().to_numpy()
            r4 = pd.Series(p4).rank().to_numpy()
            pf = r2 + r4
            rec[f"fuse_ic{H}"] = float(spearmanr(pf, ys).statistic)
            rec[f"n{H}"] = len(ys)
            if H == 20:
                pool[m] = (p4, p2, r2 + r4, ys)
                k = min(10, len(ys))
                rec["fuse_top10"] = float(np.mean(ys[np.argsort(-pf)[:k]]))
                # 经济口径：各自 TOP10 的事后平均超额收益
                k = min(10, len(ys))
                rec["t4_top10"] = float(np.mean(ys[np.argsort(-p4)[:k]]))
                rec["t2_top10"] = float(np.mean(ys[np.argsort(-p2)[:k]]))
                rec["all_ret"] = float(np.mean(ys))
                # T2 与 T4 预测的秩相关：融合能否带来增量的先验
                rec["t2t4_rankcorr"] = float(spearmanr(p2, p4).statistic)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"t4_ic_{TAG}.csv", index=False, encoding="utf-8-sig")

    def report(tag: str, col: str) -> None:
        c = df[col].dropna()
        if len(c) < 3:
            return
        t, p = stats.ttest_1samp(c, 0)
        print(f"  {tag}: 均值 IC={c.mean():+.4f}  std={c.std(ddof=1):.4f}  "
              f"ICIR={c.mean()/c.std(ddof=1):+.3f}  IC>0 {(c>0).mean()*100:.0f}%  "
              f"t={t:+.2f} p={p:.3f} {'✅显著' if p < 0.05 else '❌不显著'}")

    print("\n" + "=" * 88)
    print("【T4 vs T2 已实现 Rank IC —— 2025-01~2026-06，purge 25 去泄漏，同口径】")
    print("=" * 88)
    for H in HS:
        print(f"\n  预测期 {H} 交易日：")
        report(f"T4(LSTM)     ", f"t4_ic{H}")
        report(f"T2(LightGBM) ", f"t2_ic{H}")
        report(f"融合(T2+T4)  ", f"fuse_ic{H}")
    print("\n  ⚠️ 60 日口径的 t 值被**重叠窗口**放大了：相邻月份的 60 日收益区间互相重叠"
          "（每个月的观测不独立），\n     有效的独立样本数远小于 18。这里的 p 值只作排序参考，"
          "不能当作独立样本的显著性检验。")

    print("\n  逐月 20 日 IC：")
    dd = df.dropna(subset=["t2_ic20"])
    for _, r in dd.iterrows():
        t4 = f"{r['t4_ic20']:+.4f}" if pd.notna(r["t4_ic20"]) else "  (T4缺)"
        print(f"    {r['month']}  T2 {r['t2_ic20']:+.4f} ｜ T4 {t4} ｜ 融合 {r['fuse_ic20']:+.4f}")
    print(f"\n  T4 有效月数 {df['t4_ic20'].notna().sum()}/{len(df)}"
          f"（2025-08 因陈旧 checkpoint 训练失败，T4 全 0 已剔除）")

    for tag, col in [("T4", "t4"), ("T2", "t2")]:
        v = df[f"{col}_top10"].dropna()
        if len(v):
            print(f"\n  {tag} TOP10 事后月均超额={v.mean()*100:+.2f}%  "
                  f"（全池 {df['all_ret'].dropna().mean()*100:+.2f}%）")

    rc = df["t2t4_rankcorr"].dropna()
    if len(rc):
        print(f"\n  T2 与 T4 预测的月内秩相关：均值 {rc.mean():+.3f}（"
              f"越低越互补；接近 1 说明融合无增量）")

    # ── 与项目自身的历史测量对照（Fold3/Fold4 走路径 B，泄漏 <1%）──
    print("\n" + "=" * 88)
    print("【与项目历史 T4 测量对照】")
    print("=" * 88)
    d = df.dropna(subset=["t4_ic20"]).copy()
    d["year"] = d["month"].str[:4]
    for y, g in d.groupby("year"):
        print(f"  {y} 年 20 日 IC 均值：{g['t4_ic20'].mean():+.4f}（n={len(g)} 个月）")

    # ── Fold 口径复算：项目的 Fold IC 是**把整个测试期池化**算一个 Spearman
    #    （evaluate.py::_compute_rank_ic(pred_t4, y2_test)），不是"逐月算再平均"。
    #    两种口径不可直接比 —— 这里按 Fold 口径重算，才是能对照的数字。
    def pooled(tag: str, ms: list[str]) -> None:
        ms = [m for m in ms if m in pool]
        if len(ms) < 2:
            return
        p4 = np.concatenate([pool[m][0] for m in ms])
        p2 = np.concatenate([pool[m][1] for m in ms])
        pf = np.concatenate([pool[m][2] for m in ms])
        ys = np.concatenate([pool[m][3] for m in ms])
        print(f"  {tag}（池化，{len(ms)} 个月 / {len(ys):,} 样本）: "
              f"T2 IC={spearmanr(p2, ys).statistic:+.4f} ｜ "
              f"T4 IC={spearmanr(p4, ys).statistic:+.4f} ｜ "
              f"融合 IC={spearmanr(pf, ys).statistic:+.4f}")

    print()
    print("【按 Fold 的池化口径复算（可与 +0.0712 / +0.1007 直接对照）】")
    pooled("2025 全年 = Fold3", [m for m in pool if m.startswith("2025")])
    pooled("2025 Q2-Q4 = Fold4", [m for m in pool if m.startswith("2025") and m[5:] in
                                  ("04", "05", "06", "07", "08", "09", "10", "11", "12")])
    pooled("2026 H1 = Fold5", [m for m in pool if m.startswith("2026") and m[5:] in
                               ("01", "02", "03", "04", "05", "06")])
    print("  历史参照：Fold3(2025 全年) +0.0712 ｜ Fold4(2025 Q2-Q4) +0.1007 ｜ "
          "Fold5(2026 H1) −0.2584 ｜ Fold6(2026 Q2) −0.0909")
    print(f"\n明细: {OUT}/t4_ic_{TAG}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
