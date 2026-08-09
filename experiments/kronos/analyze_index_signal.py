#!/usr/bin/env python3
"""指数择时信号规则离线模拟（2026-08-09, 中金路线第③环"滚动调整推理参数"）

用已存推理结果（index_timing_check_2y.json, 145 点 pred_ret5/actual_ret5）离线模拟
不同信号规则, 无需重跑模型推理。目标: 对冲 Kronos 零样本的看跌偏置（踏空上涨段）。

规则:
  R0 基线:  信号 = pred_ret5 > 0                        （当前 +59.2% 超额）
  A  固定阈值: 信号 = pred_ret5 > -k%, k∈{0.5,1,1.5,2}  （放宽看多阈值）
  B  动量自适应: 20 日动量 m>0 → 阈值 -1%; m≤0 → 阈值 0 （上涨期敢看多）
  C  混合过滤: 动量 < -3% 强制空仓; Kronos 看多 或 动量>+3% → 持仓

指标: 方向准确率 / 择时累计 / 买入持有 / 超额 / 上涨段-下跌段拆分。
额外: 正信号占比（暴露度）。
用法: py312 python experiments/kronos/analyze_index_signal.py
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
RESULT = PROJECT_ROOT / "experiments/kronos/output/index_timing_check_2y.json"
INDEX_CODE = "sh.000852"
MOM_DAYS = 20


def load_momentum() -> dict[str, float]:
    """每 ref 日期前 20 交易日动量（close[t-MOM]/close[t] - 1）。"""
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol=? ORDER BY date",
        conn, params=[INDEX_CODE])
    conn.close()
    closes = df.set_index("date")["close"]
    mom = {}
    for i in range(MOM_DAYS, len(df)):
        d = df["date"].iloc[i]
        mom[d] = closes.iloc[i] / closes.iloc[i - MOM_DAYS] - 1.0
    return mom


def simulate(pred: np.ndarray, act: np.ndarray, rule: str, k: float = 0.0,
             mom: np.ndarray | None = None) -> dict:
    """按规则生成信号（持仓=1/空仓=0）并计算指标。"""
    if rule == "R0":
        sig = pred > 0
    elif rule == "A":
        sig = pred > -k / 100.0
    elif rule == "B":
        sig = np.where(mom > 0, pred > -1.0 / 100.0, pred > 0)
    elif rule == "C":
        sig = (pred > 0) | (mom > 0.03)
        sig &= mom > -0.03
    else:
        raise ValueError(rule)
    ret = np.where(sig, act, 0.0)
    acc = np.mean(np.sign(pred) == np.sign(act))       # 方向准确率（预测符号 vs 实际）
    nav = np.prod(1 + ret)
    bh = np.prod(1 + act)
    return {"rule": rule, "k": k, "exposure": float(sig.mean()),
            "accuracy": float(acc), "nav": float(nav),
            "excess": float(nav / bh - 1), "bh": float(bh)}


def main() -> None:
    res = json.loads(RESULT.read_text())
    rows = res["rows"]
    pred = np.array([r["pred_ret5"] for r in rows])
    act = np.array([r["actual_ret5"] for r in rows])
    dates = [r["date"] for r in rows]
    mom_map = load_momentum()
    mom = np.array([mom_map.get(d, 0.0) for d in dates])
    # 上涨/下跌段: 用 30 日滚动实际收益均值（平滑分段的代理）
    def seg_split(win=6):
        s = pd.Series(act).rolling(win, min_periods=1).mean() > 0
        return s.values
    up_mask = seg_split()

    print(f"📊 离线模拟（{len(rows)} 点, {dates[0]} ~ {dates[-1]}）")
    print(f"买入持有累计: {res['buyhold_nav']:+.1%}\n")
    print(f"{'规则':<28}{'暴露':>6}{'方向准':>7}{'择时累计':>10}{'超额':>8}"
          f"{'涨段超额':>9}{'跌段超额':>9}")
    results = []
    rules = [("R0 基线(pred>0)", "R0", 0.0),
             ("A1 阈值>-0.5%", "A", 0.5),
             ("A2 阈值>-1%", "A", 1.0),
             ("A3 阈值>-1.5%", "A", 1.5),
             ("A4 阈值>-2%", "A", 2.0),
             ("B 动量自适应(±1%)", "B", 0.0),
             ("C 动量混合过滤(±3%)", "C", 0.0)]
    for label, rule, k in rules:
        s = simulate(pred, act, rule, k, mom)
        sig = (pred > 0) if rule == "R0" else (
            (pred > -k / 100) if rule == "A" else (
                np.where(mom > 0, pred > -0.01, pred > 0) if rule == "B"
                else ((pred > 0) | (mom > 0.03)) & (mom > -0.03)))
        ret = np.where(sig, act, 0.0)
        up_ex = np.prod(1 + ret[up_mask]) / np.prod(1 + act[up_mask]) - 1
        dn_ex = np.prod(1 + ret[~up_mask]) / np.prod(1 + act[~up_mask]) - 1
        results.append((label, s, up_ex, dn_ex))
        print(f"{label:<28}{s['exposure']*100:>5.0f}%{s['accuracy']*100:>6.1f}%"
              f"{s['nav']:>+9.1%}{s['excess']:>+7.1%}{up_ex:>+8.1%}{dn_ex:>+8.1%}")

    best = max(results, key=lambda r: r[1]["excess"])
    print(f"\n🏆 最优: {best[0]}  超额 {best[1]['excess']:+.1%}  "
          f"(nav {best[1]['nav']:+.1%} vs bh {best[1]['bh']:+.1%})")
    # 最优规则的保守性检查: 是否保留下跌段保护
    if best[3] < 0:
        print(f"⚠️ 注意: 最优规则跌段超额 {best[3]:+.1%} 为负（保护减弱）, 需权衡")
    out = PROJECT_ROOT / "experiments/kronos/output/index_signal_sim.json"
    out.write_text(json.dumps(
        {"dates": [dates[0], dates[-1]], "n": len(rows),
         "rules": [{"label": l, **s} for l, s, _, _ in results],
         "best": {"label": best[0], **best[1], "up_excess": best[2], "down_excess": best[3]}},
        ensure_ascii=False, indent=2))
    print(f"结果已写 {out}")


if __name__ == "__main__":
    main()
