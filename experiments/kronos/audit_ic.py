#!/usr/bin/env python3
"""Kronos 3a 单月 IC 审计（2026-08-08, 用户质疑 +0.4503 是否仍含错误）

对"假 IC 修复"做四重独立验证（不依赖 step3/step4 代码路径）:
  A. 时间戳正确性: jsonl.date == DB 当月最后交易日; x 历史 120 天; y 为 ref 后 20 交易日
  B. 个股抽检: 8 只股票 exp_ret vs DB 实际未来 20 日收益（人眼判断量级/方向）
  C. 全量重算: Spearman(exp_ret, 未来20日超额收益) —— 应 ≈ +0.45
  D. 反证测试: Spearman(exp_ret, 过去20日收益) —— 若仍高相关 → 预测与历史价格相关 → 可疑
     另加 10 日/40 日未来窗口 IC 作为 horizon 一致性参考

用法: py312 python audit_ic.py [月份]
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MONTH = sys.argv[1] if len(sys.argv) > 1 else "2026-06"
DB = str(PROJECT_ROOT / "data/sequoia_v2.db")
FP = PROJECT_ROOT / "experiments/kronos/output" / f"month_{MONTH}.jsonl"
H = 20

conn = sqlite3.connect(DB)

# ── A. 时间戳正确性 ──
print("=" * 60)
print(f"A. 时间戳检查（{MONTH}）")
last_td = conn.execute(
    "SELECT MAX(date) FROM stock_daily WHERE date>=? AND date<?",
    (MONTH + "-01", f"{int(MONTH[:4]) + (1 if MONTH[5:] == '12' else 0)}-"
     f"{1 if MONTH[5:] == '12' else int(MONTH[5:]) + 1:02d}-01")).fetchone()[0]
rows = [json.loads(l) for l in FP.read_text(encoding="utf-8").strip().splitlines() if l]
dates = {r["date"] for r in rows}
print(f"DB 当月最后交易日: {last_td} | jsonl 中 date 集合: {sorted(dates)}")
print(f"  → {'✅ 一致' if dates == {last_td} else '❌ 不一致（严重）'}")

# 600519 抽样验证 x/y 区间
code = "600519"
ref = rows[0]["date"]
df = pd.read_sql("SELECT date FROM stock_daily WHERE symbol=? AND date<=? "
                 "ORDER BY date DESC LIMIT ?", conn,
                 params=[code, ref, 120])
x_last = df["date"].iloc[0]
y_first = conn.execute("SELECT DISTINCT date FROM stock_daily WHERE date>? "
                       "ORDER BY date LIMIT 1", (ref,)).fetchone()[0]
print(f"{code} 抽查: x 最后一天={x_last} (<=ref) | y 第一天={y_first} (>ref)"
      f" | 断言 y 全部在未来: "
      f"{'✅' if str(y_first) > str(ref) else '❌'}")

# ── B. 个股抽检 ──
print("\n" + "=" * 60)
print("B. 个股抽检（8 只: 茅台 + 随机 7）")
rng = np.random.RandomState(42)
sample_codes = [code] + [r["code"] for r in rng.choice(
    rows, 7, replace=False)] if len(rows) > 8 else [code]
print(f"{'代码':<8}{'close':>9}{'pred_close':>11}{'exp_ret':>9}"
      f"{'实际未来20日':>12}{'过去20日':>10}  判断")
for r in rows:
    if r["code"] not in sample_codes:
        continue
    c = r["code"]
    close = r["close"]
    fut = conn.execute("SELECT close FROM stock_daily WHERE symbol=? AND date>? "
                       "ORDER BY date LIMIT ?", (c, r["date"], H)).fetchall()
    past = conn.execute("SELECT close FROM stock_daily WHERE symbol=? AND date<=? "
                        "ORDER BY date DESC LIMIT ?", (c, r["date"], H + 1)).fetchall()
    if len(fut) < H or len(past) < H + 1:
        print(f"{c:<8}{close:>9.2f}{r['pred_close']:>11.2f}{r['exp_ret']:>+9.1%}"
              f"  (数据不足)")
        continue
    fut_ret = fut[-1][0] / close - 1
    past_ret = close / past[-1][0] - 1
    ok = "✅" if (r["exp_ret"] > 0) == (fut_ret > 0) else "?"
    print(f"{c:<8}{close:>9.2f}{r['pred_close']:>11.2f}{r['exp_ret']:>+9.1%}"
          f"{fut_ret:>+12.1%}{past_ret:>+10.1%}  {ok}")

# ── C. 全量重算未来 20 日 IC ──
print("\n" + "=" * 60)
print("C. 全量重算（独立于 step3 的路径）")
exps, fut20, past20 = [], [], []
idx_rows = conn.execute(
    "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date>? "
    "ORDER BY date LIMIT ?", (MONTH + "-30", H)).fetchall()
ref_close = conn.execute(
    "SELECT close FROM index_daily WHERE symbol='sh.000300' AND date<=? "
    "ORDER BY date DESC LIMIT 1", (MONTH + "-30",)).fetchone()
idx_ret = idx_rows[-1][0] / ref_close[0] - 1 if len(idx_rows) >= H else None
for r in rows:
    fut = conn.execute("SELECT close FROM stock_daily WHERE symbol=? AND date>? "
                       "ORDER BY date LIMIT ?", (r["code"], r["date"], H)).fetchall()
    past = conn.execute("SELECT close FROM stock_daily WHERE symbol=? AND date<=? "
                        "ORDER BY date DESC LIMIT ?", (r["code"], r["date"], H + 1)).fetchall()
    if len(fut) >= H and len(past) >= H + 1 and idx_ret is not None:
        exps.append(r["exp_ret"])
        fut20.append(fut[-1][0] / r["close"] - 1 - idx_ret)
        past20.append(r["close"] / past[-1][0] - 1)
conn.close()
exps = np.array(exps)
ic, p = spearmanr(exps, fut20)
ic_past, p_past = spearmanr(exps, past20)
print(f"有效样本: {len(exps)}")
print(f"Rank IC (exp_ret vs 未来20日超额收益) = {ic:+.4f} (p={p:.4f})  {'✅ ≈ +0.45' if abs(ic - 0.45) < 0.05 else '⚠️ 与 +0.4503 不符'}")

# ── D. 反证测试 ──
print("\n" + "=" * 60)
print("D. 反证测试（预测是否与历史收益相关 → 错位 bug 的指纹）")
print(f"corr(exp_ret, 过去20日收益)      = {ic_past:+.4f} (p={p_past:.4f})"
      f"  {'✅ 不相关（无历史错位指纹）' if abs(ic_past) < 0.1 else '❌ 高相关（可疑!）'}")
# 10 日 / 40 日 horizon 一致性
conn = sqlite3.connect(DB)
for hh in (10, 40):
    exps2, y2s = [], []
    for r in rows[:800]:  # 抽样 800 只加速
        fut = conn.execute("SELECT close FROM stock_daily WHERE symbol=? AND date>? "
                           "ORDER BY date LIMIT ?", (r["code"], r["date"], hh)).fetchall()
        if len(fut) >= hh:
            exps2.append(r["exp_ret"])
            y2s.append(fut[-1][0] / r["close"] - 1 - idx_ret)
    ic_h, _ = spearmanr(exps2, y2s)
    print(f"corr(exp_ret, 未来{hh}日超额收益) = {ic_h:+.4f}（与 20 日口径比对）")
conn.close()
