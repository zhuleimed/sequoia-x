#!/usr/bin/env python3
"""3a 70 个月汇总分析（2026-08-08, monitor_70m 完成后自动触发）

输出:
  - output/70m_summary.csv : 逐月 Rank IC / corr(Kronos,T2) / corr(Kronos,T4) / 沪深300基准
  - 年度聚合表 + §9.6 门槛判定 + 微信推送（由调用方推送, 本脚本 print 汇总）

口径: 与 step3（Rank IC, y2=未来20日超额 vs 沪深300）和 step4（Spearman corr）一致。
用法: py312 python step5_summary.py
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments/kronos"))
import step3_analyze as S3  # noqa: E402  复用 load_predictions/future_ret（同口径保证）

DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
POOLS_DIR = PROJECT_ROOT / "output/backtest_v2/kronos_pools"
CACHE_F = PROJECT_ROOT / "output/backtest_v2/prediction_cache.json"
SUMMARY_CSV = OUT_DIR / "70m_summary.csv"
HORIZON = 20


def months() -> list[str]:
    return sorted(p.stem for p in POOLS_DIR.glob("*.json"))


def monthly_ic(month: str, conn: sqlite3.Connection) -> dict | None:
    """单月 Rank IC（同 step3 口径）。"""
    fp = OUT_DIR / f"month_{month}.jsonl"
    if not fp.exists():
        return None
    rows = []
    for line in fp.read_text(encoding="utf-8").strip().splitlines():
        if line:
            rows.append(json.loads(line))
    if not rows:
        return None
    # 沪深300 基准（20 交易日）
    idx_rows = conn.execute(
        "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date>? "
        "ORDER BY date LIMIT ?", (month + "-30", HORIZON)).fetchall()
    ref_row = conn.execute(
        "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date<=? "
        "ORDER BY date DESC LIMIT 1", (month + "-30",)).fetchone()
    if len(idx_rows) < HORIZON or not ref_row:
        return None
    idx_ret = idx_rows[-1][0] / ref_row[0] - 1.0

    exps, y2s = [], []
    for r in rows:
        fr = S3.future_ret(r["code"], r["date"], conn)
        if fr is None:
            continue
        exps.append(r["exp_ret"])
        y2s.append(fr - idx_ret)
    if len(exps) < 50:
        return None
    ic, p = spearmanr(exps, y2s)
    n = len(exps)
    k = max(n // 10, 1)
    top = np.argsort(-np.array(exps))[:k]
    bot = np.argsort(np.array(exps))[:k]
    return {"month": month, "ic": ic, "p": p, "samples": n,
            "top10": np.array(y2s)[top].mean(), "bot10": np.array(y2s)[bot].mean(),
            "idx_ret": idx_ret}


def monthly_corr(month: str, cache: dict) -> dict | None:
    """单月 corr(Kronos, T2/T4)（同 step4 口径）。"""
    fp = OUT_DIR / f"month_{month}.jsonl"
    m = cache.get(month)
    if not fp.exists() or m is None:
        return None
    kronos = {json.loads(l)["code"]: json.loads(l)["exp_ret"]
              for l in fp.read_text(encoding="utf-8").strip().splitlines() if l}
    sym_map = {c: i for i, c in enumerate(m["symbols"])}
    kr, t2, t4 = [], [], []
    for code, v in kronos.items():
        i = sym_map.get(code)
        if i is not None:
            kr.append(v)
            t2.append(m["t2"][i])
            t4.append(m["t4"][i])
    if len(kr) < 50:
        return None
    return {"c_t2": spearmanr(kr, t2)[0], "c_t4": spearmanr(kr, t4)[0]}


def main() -> None:
    cache = json.loads(CACHE_F.read_text())
    conn = sqlite3.connect(DB)
    ms = months()
    rows = []
    for i, month in enumerate(ms):
        ic = monthly_ic(month, conn)
        cor = monthly_corr(month, cache)
        if ic is None:
            print(f"[{i+1}/{len(ms)}] {month} 跳过（无预测/无 y2）", flush=True)
            continue
        rows.append({**ic, "c_t2": cor["c_t2"] if cor else np.nan,
                     "c_t4": cor["c_t4"] if cor else np.nan})
        print(f"[{i+1}/{len(ms)}] {month}: IC={ic['ic']:+.4f} (n={ic['samples']}) "
              f"corrT2={rows[-1]['c_t2']:+.3f} corrT4={rows[-1]['c_t4']:+.3f}",
              flush=True)
    conn.close()

    if not rows:
        print("❌ 无任何月份可汇总")
        return
    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已写 {SUMMARY_CSV}（{len(df)} 个月）")

    # ── 全期汇总 ──
    print("\n=== 70 个月全期汇总（门槛 §9.6）===")
    ic_mean = df["ic"].mean()
    pos_frac = (df["ic"] > 0).mean()
    c2_mean = df["c_t2"].abs().mean()
    c4_mean = df["c_t4"].abs().mean()
    print(f"Rank IC 均值    = {ic_mean:+.4f}（门槛 ≥ +0.01）{'✅' if ic_mean >= 0.01 else '❌'}")
    print(f"正 IC 月占比    = {pos_frac:.0%}（参考 T2 57% / T4 61%）")
    print(f"|corr(T2)| 均值 = {c2_mean:.3f}（门槛 < 0.3）{'✅' if c2_mean < 0.3 else '❌'}")
    print(f"|corr(T4)| 均值 = {c4_mean:.3f}（门槛 < 0.3）{'✅' if c4_mean < 0.3 else '❌'}")
    print(f"TOP10-BOT10 价差均值 = {(df['top10']-df['bot10']).mean():+.2%}")
    print(f"沪深300 月基准均值 = {df['idx_ret'].mean():+.2%}")

    # ── 年度聚合 ──
    df["year"] = df["month"].str[:4]
    print("\n=== 按年度 ===")
    print(df.groupby("year")["ic"].agg(["mean", "count", lambda x: (x > 0).mean()])
          .round(4).to_string())
    # 参考对比（T2/T4 年度 IC, 来自 §11.3）
    ref = {"2020": (+0.041, +0.070), "2021": (+0.002, +0.015), "2022": (+0.020, +0.033),
           "2023": (+0.026, +0.018), "2024": (-0.001, +0.019), "2025": (-0.005, -0.006),
           "2026": (+0.012, -0.016)}
    print("\n年度对照（Kronos vs T2 / T4, 参考 §11.3）:")
    for y, (t2, t4) in ref.items():
        k = df.loc[df["year"] == y, "ic"].mean()
        if len(df.loc[df["year"] == y]) > 0:
            print(f"  {y}: Kronos {k:+.4f} | T2 {t2:+.4f} | T4 {t4:+.4f}")

    # ── 判定 ──
    print("\n=== 判定 ===")
    if ic_mean >= 0.01 and c2_mean < 0.3 and c4_mean < 0.3:
        print("✅ 70 个月全量达标 → 进入融合矩阵实验（门槛 IC > +0.0165 与 T2+T4 对照）")
    elif ic_mean >= 0.01:
        print("⚠️ IC 达标但相关性超限 → 融合价值有限, 讨论替代/加权方案")
    else:
        print("❌ 70 个月 IC 均值未达 +0.01 → 仅记录基线, 不融合")


if __name__ == "__main__":
    main()
