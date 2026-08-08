#!/usr/bin/env python3
"""Kronos vs 简单基线因子对照（2026-08-08, 评估口径修正后）

背景: 2026-06 单月 Kronos IC +0.4865, 但均值回归因子(m120/c-1) IC 也有 +0.4581,
corr(Kronos, m120) = 0.74 → Kronos 零样本输出 ≈ 均值回归/反转启发式,
"高 IC"是市场效应而非模型时序预测力（ADR V3-17: 评估改用增量口径）。

本脚本对任意月份输出:
  - Kronos IC / 反转因子 IC / 均值回归因子 IC（同口径 Spearman）
  - corr(Kronos, 各基线) —— 判断 Kronos 输出与基线的重合度
  - 增量 IC = Kronos IC - max(两基线 IC) —— 融合价值的判断标准
  - 与对照月（2026-06 下跌月）并排打印

用法: py312 python analyze_baseline.py [月份...]
  例: python analyze_baseline.py 2026-03 2026-06
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
OUT_DIR = PROJECT_ROOT / "experiments/kronos/output"
H = 20


def analyze(month: str, conn: sqlite3.Connection, idx_ret: float,
            suffix: str = "") -> dict | None:
    fp = OUT_DIR / f"month_{month}{suffix}.jsonl"
    if not fp.exists():
        return None
    rows = [json.loads(l) for l in
            fp.read_text(encoding="utf-8").strip().splitlines() if l]
    exps, y2, past20, m120 = [], [], [], []
    for r in rows:
        c, ref = r["code"], r["date"]
        hist = pd.read_sql(
            "SELECT close FROM stock_daily WHERE symbol=? AND date<=? "
            "ORDER BY date DESC LIMIT 140", conn, params=[c, ref])
        hist = hist.iloc[::-1]["close"].reset_index(drop=True)
        if len(hist) < 121:
            continue
        fut = conn.execute(
            "SELECT close FROM stock_daily WHERE symbol=? AND date>? "
            "ORDER BY date LIMIT ?", (c, ref, H)).fetchall()
        if len(fut) < H:
            continue
        close = r["close"]
        exps.append(r["exp_ret"])
        y2.append(fut[-1][0] / close - 1 - idx_ret)
        past20.append(close / hist.iloc[-21] - 1)          # 过去20日收益
        m120.append(hist.iloc[-120:].mean() / close - 1)   # 相对120日均值
    if len(exps) < 50:
        return None
    exps, y2 = np.array(exps), np.array(y2)
    past20, m120 = np.array(past20), np.array(m120)
    ic_k = spearmanr(exps, y2)[0]
    ic_rev = spearmanr(-past20, y2)[0]
    ic_mr = spearmanr(m120, y2)[0]
    return {"month": month, "n": len(exps),
            "ic_kronos": ic_k, "ic_reversal": ic_rev, "ic_meanrev": ic_mr,
            "corr_k_rev": spearmanr(exps, -past20)[0],
            "corr_k_mr": spearmanr(exps, m120)[0],
            "increment": ic_k - max(ic_rev, ic_mr)}


def main() -> None:
    args = sys.argv[1:]
    suffix = ""
    if "--suffix" in args:
        i = args.index("--suffix")
        suffix = args[i + 1]
        del args[i:i + 2]
    months = args or ["2026-03", "2026-06"]
    conn = sqlite3.connect(DB)
    results = []
    for month in months:
        # 沪深300 基准（未来 20 日）
        last = conn.execute(
            "SELECT MAX(date) FROM stock_daily WHERE date>=? AND date<?",
            (month + "-01", f"{int(month[:4]) + (1 if month[5:] == '12' else 0)}-"
             f"{1 if month[5:] == '12' else int(month[5:]) + 1:02d}-01")).fetchone()[0]
        fut = conn.execute(
            "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date>? "
            "ORDER BY date LIMIT ?", (last, H)).fetchall()
        refc = conn.execute(
            "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date<=? "
            "ORDER BY date DESC LIMIT 1", (last,)).fetchone()
        if len(fut) >= H and refc:
            idx_ret = fut[-1][0] / refc[0] - 1
            r = analyze(month, conn, idx_ret, suffix)
            if r:
                r["idx_ret"] = idx_ret
                results.append(r)
    conn.close()

    if not results:
        print("❌ 无结果")
        return
    print("\n=== Kronos vs 基线因子（同口径 Spearman, 增量口径 ADR V3-17）===")
    print(f"{'月份':<9}{'沪深300未来20日':>14}{'Kronos IC':>11}{'反转IC':>9}"
          f"{'均值回归IC':>11}{'corr(K,mR)':>11}{'增量IC':>9}")
    for r in results:
        print(f"{r['month']:<9}{r['idx_ret']:>+13.1%}{r['ic_kronos']:>+11.4f}"
              f"{r['ic_reversal']:>+9.4f}{r['ic_meanrev']:>+11.4f}"
              f"{r['corr_k_mr']:>+11.3f}{r['increment']:>+9.4f}")
    print("\n判定（增量口径）:")
    for r in results:
        if r["increment"] >= 0.01:
            print(f"  {r['month']}: 增量 {r['increment']:+.4f} ✅ 有真实增量（> +0.01）")
        else:
            print(f"  {r['month']}: 增量 {r['increment']:+.4f} ❌ 无增量（≈简单因子）")


if __name__ == "__main__":
    main()
