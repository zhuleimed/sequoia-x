#!/usr/bin/env python3
"""零成本复核（2026-10-14）：深市后复权 bug 是否污染了 Kronos 3a/3b 的评估结论。

背景
----
- 2026-09-25 发现并修复：stock_daily 深市（000/001/002/003/300/301）在
  2024-01-02 ~ 2026-06-08 存的是**后复权价**（水平被放大，含复权因子 ~2.3x）。
  修复脚本 scripts/fix_deep_market_basis_20260925.py。
- Kronos 3a/3b 评估月 = 2026-03（ref 03-31）与 2026-06（ref 06-30），
  **当时库还没修** ⇒ 深市股票的 120 天输入窗口内跨 2026-06-09 边界：
  边界前是后复权（高）、边界后是不复权（低）⇒ 窗口里有一个假"崩盘"。
- 沪市股票**从未受此 bug 影响** ⇒ 天然的干净对照组。

本脚本不重新推理（kronos 预测沿用当时 jsonl），只做两件事：
  1. 用**已修复的库**重算基线因子（m120 均值回归 / past20 反转）与 y2
     ⇒ "当时记录的基线 IC" vs "干净基线 IC" 的差 = 污染对基线的影响；
  2. 按市场（沪市=对照 / 深市=污染）拆分，看 kronos 的 IC 在两组间是否系统性不同。

判读：
  · 若深市 kronos IC ≈ 沪市 kronos IC ⇒ 污染对 kronos 预测质量影响小 ⇒ 原结论稳；
  · 若干净基线 IC 明显低于当时记录 ⇒ 污染抬高了基线 ⇒ 当时的"增量 IC"被低估
    ⇒ kronos 可能被冤枉（值得重跑）。

用法: py312 python recheck_after_datafix.py [月份...]
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
DB = str(ROOT / "data/sequoia_v2.db")
OUT = ROOT / "experiments/kronos/output"
H = 20  # 与 V2 T2 口径一致：未来 20 交易日

# 当时（2026-08-08/09）用污染库算出的记录，用于对照。★ 两个模型的预测不同，别混看：
#   "_ft"  = 微调模型（来源 output/ft_eval_result.json，五数齐全）
#   ""     = 零样本（来源项目记忆 kronos-3a-zero-shot / analyze_baseline 输出；基线与微调同池同值）
RECORDED = {
    "_ft": {
        "2026-06": {"ic_kronos": 0.4562, "ic_reversal": 0.4252, "ic_meanrev": 0.4581,
                    "corr_k_mr": 0.578},
        "2026-03": {"ic_kronos": -0.0003, "ic_reversal": 0.0993, "ic_meanrev": -0.0501,
                    "corr_k_mr": 0.250},
    },
    "": {
        "2026-06": {"ic_kronos": 0.4865, "ic_reversal": 0.4252, "ic_meanrev": 0.4581,
                    "corr_k_mr": None},
        "2026-03": {"ic_kronos": 0.0130, "ic_reversal": 0.0993, "ic_meanrev": -0.0501,
                    "corr_k_mr": None},
    },
}


def market_of(code: str) -> str:
    p = code[:3]
    if p in ("000", "001", "002", "003", "300", "301"):
        return "SZ"          # 深市 = 当时被污染组
    if p in ("600", "601", "603", "605", "688"):
        return "SH"          # 沪市 = 天然干净对照组
    return "OTHER"


def load_rows(month: str, suffix: str = "") -> list[dict]:
    fp = OUT / f"month_{month}{suffix}.jsonl"
    if not fp.exists():
        return []
    return [json.loads(l) for l in fp.read_text(encoding="utf-8").strip().splitlines() if l]


def idx_ret_20d(conn: sqlite3.Connection, ref: str) -> float:
    fut = conn.execute(
        "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date>? "
        "ORDER BY date LIMIT ?", (ref, H)).fetchall()
    refc = conn.execute(
        "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date<=? "
        "ORDER BY date DESC LIMIT 1", (ref,)).fetchone()
    return fut[-1][0] / refc[0] - 1.0


def compute(rows: list[dict], conn: sqlite3.Connection, idx: float) -> dict:
    """用当前（已修复）库重算 y2 与两个基线因子。"""
    e, y, rev, mr = [], [], [], []
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
        e.append(r["exp_ret"])
        y.append(fut[-1][0] / close - 1.0 - idx)
        rev.append(-(close / hist.iloc[-21] - 1.0))            # 反转因子
        mr.append(hist.iloc[-120:].mean() / close - 1.0)       # 均值回归因子
    e, y, rev, mr = map(np.asarray, (e, y, rev, mr))
    if len(e) < 50:
        return {}
    ic_k = spearmanr(e, y)[0]
    ic_rev = spearmanr(rev, y)[0]
    ic_mr = spearmanr(mr, y)[0]
    return {
        "n": len(e),
        "ic_kronos": ic_k,
        "ic_reversal": ic_rev,
        "ic_meanrev": ic_mr,
        "corr_k_mr": spearmanr(e, mr)[0],
        "increment": ic_k - max(ic_rev, ic_mr),
    }


def fmt(name: str, d: dict) -> str:
    if not d:
        return f"{name:<14} (样本不足)"
    return (f"{name:<14} n={d['n']:>4}  Kronos {d['ic_kronos']:+.4f}  "
            f"反转 {d['ic_reversal']:+.4f}  均值回归 {d['ic_meanrev']:+.4f}  "
            f"corr(K,mR) {d['corr_k_mr']:+.3f}  增量 {d['increment']:+.4f}")


def main() -> None:
    import sys
    argv = sys.argv[1:]
    suffix = ""
    if "--suffix" in argv:
        i = argv.index("--suffix")
        suffix = argv[i + 1]
        del argv[i:i + 2]
    ms = argv or ["2026-03", "2026-06"]
    conn = sqlite3.connect(DB)

    for month in ms:
        rows = load_rows(month, suffix)
        if not rows:
            print(f"❌ 无预测文件 month_{month}.jsonl")
            continue
        idx = idx_ret_20d(conn, rows[0]["date"])
        rec = RECORDED.get(suffix, {}).get(month, {})

        model = "微调" if suffix == "_ft" else "零样本"
        print(f"\n{'='*100}\n月份 {month}  (ref={rows[0]['date']}, 沪深300未来20日 {idx:+.2%}, 模型={model})")
        print(f"{'':14} {'IC':>8} {'反转IC':>10} {'均值回归IC':>12} {'corr(K,mR)':>12} {'增量IC':>9}")
        if rec:
            ck = rec["corr_k_mr"]
            corr_s = f"{ck:>+12.3f}" if ck is not None else f"{'—':>12}"
            print(f"{'当时记录(污染库)' + model:>18} {rec['ic_kronos']:>+8.4f} {rec['ic_reversal']:>+10.4f} "
                  f"{rec['ic_meanrev']:>+12.4f} {corr_s} "
                  f"{rec['ic_kronos']-max(rec['ic_reversal'],rec['ic_meanrev']):>+9.4f}")

        # 全池 + 分市场（全部用当前的干净库重算 y 与基线；kronos 预测仍是当时那份）
        for label, sub in [
            ("全池", rows),
            ("沪市(对照/干净)", [r for r in rows if market_of(r["code"]) == "SH"]),
            ("深市(污染组)", [r for r in rows if market_of(r["code"]) == "SZ"]),
        ]:
            print(fmt(f"{label}", compute(sub, conn, idx)))

        print("怎么读：'当时记录'那行是 2026-08 用污染库算出来的；下面三行是**用今天已修复的库**"
              "重算 y 与基线（kronos 预测仍是当时那份、未重跑）。")
        print("       若'均值回归IC'明显下降 ⇒ 污染当年抬高了基线 ⇒ 当时的增量被低估；"
              "深市 vs 沪市两行可直接对比污染组与对照组的 kronos IC。")

    conn.close()


if __name__ == "__main__":
    main()
