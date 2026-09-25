"""P1-1：8 个指标策略的选股能力检验。

问题
----
前述对照实验发现：8 个策略选出的**整个候选池**，相对全市场在 +3/+5/+10 日
显著为负（+10 日 -3.92%，t=-5.27）。也就是"这 8 个策略合起来没有选股能力"。

但那是一个整体结论。本脚本**逐个策略**检验，回答：
  哪些策略在 2026-06~09 这段行情里有效（正超额且显著）？
  哪些是中性的？哪些是显著负贡献、应当下线？

方法（与前述分析同一口径，保证可比）
--------------------------------
- 数据：`data/results/results_YYYYMMDD.json` 存档的每日各策略选股结果
- 买点：信号日**次日（T+1）开盘价**（与模拟盘 SimEngine 执行口径一致）
- 卖点：T+1 之后第 N 个交易日收盘（N = 1/3/5/10）
- 基准：同日**全市场其余股票**等权均值（去掉当日全部候选池）
  —— 逐日配对做差，再对日度差值序列做 t 检验，日期效应被完全消除
- 剔除 2026-07-06 / 07-07 / 06-09 的坏行（凡前瞻窗口跨越坏行的样本置 NaN）

读法
----
"日均超额"是**每个信号日**该策略选股相对当日市场平均的超额收益。
t 值 > 2 约当 p<0.05；本表样本天数有限（~60），单日噪声大，看符号与量级为主。
"""

from __future__ import annotations

import glob
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "sequoia_v2.db"
RESULTS_GLOB = str(PROJECT / "data" / "results" / "results_2026*.json")
HORIZONS = (1, 3, 5, 10)
BAD_DAYS = ("2026-06-09", "2026-07-06", "2026-07-07")
JUMP = 0.25
MIN_DAYS = 20   # 少于这么多天有选股的策略不单独下结论

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main() -> int:
    con = sqlite3.connect(DB)
    cal = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2026-05-01' ORDER BY date", con
    )["date"].tolist()
    cal_idx = {d: i for i, d in enumerate(cal)}
    b = pd.read_sql(
        "SELECT symbol, date, open, close FROM stock_daily WHERE date >= '2026-05-01'", con)
    for c in ("open", "close"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna().sort_values(["symbol", "date"])
    b["prev"] = b.groupby("symbol")["close"].shift(1)
    b["corrupt"] = (b["date"].isin(BAD_DAYS)) & (
        ((b["close"] / b["prev"] - 1).abs() > JUMP) | ((b["open"] / b["prev"] - 1).abs() > JUMP))
    b["corrupt"] = b["corrupt"].fillna(False)
    px = {s: dict(zip(g["date"], zip(g["open"], g["close"], g["corrupt"])))
          for s, g in b.groupby("symbol")}
    con.close()

    def fwd(sym: str, d0: str, n: int) -> float:
        i = cal_idx.get(d0)
        if i is None or i + 1 + n >= len(cal):
            return np.nan
        bd, ex = cal[i + 1], cal[i + 1 + n]
        d = px.get(sym, {})
        o, c = d.get(bd), d.get(ex)
        if not o or not c or not (o[0] > 0) or not (c[1] > 0):
            return np.nan
        if any(d.get(w, (0, 0, True))[2] for w in cal[i + 1: i + 2 + n]):
            return np.nan
        return c[1] / o[0] - 1.0

    # 逐日：每个策略的选股 + 当日市场基准
    per_day: dict[str, list[float]] = {}      # 市场基准日度序列
    strat_days: dict[str, dict[str, list[float]]] = {}   # strategy -> date -> [rets]
    days_used = 0
    for fp in sorted(glob.glob(RESULTS_GLOB)):
        d = json.load(open(fp, encoding="utf-8"))
        sd = d["date"]
        if sd not in cal_idx:
            continue
        strategies = {k: v for k, v in d["strategies"].items() if v}
        if not strategies:
            continue
        days_used += 1
        pool = set()
        for syms in strategies.values():
            pool.update(syms)
        # 市场基准：当日全市场其余股票（每 7 只抽 1，降低计算量）
        mkt = {n: [] for n in HORIZONS}
        for i, s in enumerate(px):
            if s in pool or i % 7:
                continue
            for n in HORIZONS:
                r = fwd(s, sd, n)
                if not np.isnan(r):
                    mkt[n].append(r)
        for n in HORIZONS:
            if mkt[n]:
                per_day.setdefault(n, []).append((sd, float(np.mean(mkt[n]))))
        for name, syms in strategies.items():
            for n in HORIZONS:
                rs = [fwd(s, sd, n) for s in syms]
                rs = [x for x in rs if not np.isnan(x)]
                if rs:
                    strat_days.setdefault(name, {}).setdefault(n, []).append((sd, float(np.mean(rs))))

    mkt_map = {n: dict(v) for n, v in per_day.items()}

    print("=" * 110)
    print(f"覆盖信号日 {days_used} 天 | 各策略选股数（当日 top5 上限）")
    print("=" * 110)

    rows = []
    for name, byn in sorted(strat_days.items()):
        row = {"策略": name}
        row["选股日数"] = len(byn.get(10, byn.get(5, byn.get(1, []))))
        total_picks = 0
        for n in HORIZONS:
            ser = byn.get(n, [])
            if len(ser) < MIN_DAYS:
                row[f"+{n}日超额"] = np.nan
                row[f"+{n}日t"] = np.nan
                continue
            arr = [(d, r) for d, r in ser if d in mkt_map[n]]
            if len(arr) < MIN_DAYS:
                row[f"+{n}日超额"] = np.nan
                row[f"+{n}日t"] = np.nan
                continue
            diff = np.array([r - mkt_map[n][d] for d, r in arr])
            t, _ = stats.ttest_1samp(diff, 0.0)
            row[f"+{n}日超额"] = diff.mean()
            row[f"+{n}日t"] = t
        rows.append(row)

    tbl = pd.DataFrame(rows).sort_values("+10日超额")
    print("\n【各策略相对全市场的日均超额（逐日配对）】")
    print(tbl.to_string(index=False))

    print("\n【汇总统计】")
    for n in HORIZONS:
        col, tcol = f"+{n}日超额", f"+{n}日t"
        sub = tbl.dropna(subset=[col])
        pos = (sub[col] > 0).sum()
        sig_pos = ((sub[col] > 0) & (sub[tcol] > 2)).sum()
        sig_neg = ((sub[col] < 0) & (sub[tcol] < -2)).sum()
        print(f"  +{n}日: {len(sub)} 个策略有足够样本 | 正超额 {pos} 个 "
              f"| 显著为正(有选股能力) {sig_pos} 个 | 显著为负 {sig_neg} 个")

    print("\n【市场基准（同期全市场等权，T+1 开盘买入）】")
    for n in HORIZONS:
        v = [r for _, r in per_day.get(n, [])]
        if v:
            print(f"  +{n}日: 均值 {np.mean(v)*100:+.3f}%")

    out = PROJECT / "experiments" / "llm_sim_diag" / "strategy_ic.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
