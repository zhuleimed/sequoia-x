"""反事实组合模拟：同样每天买 2 只、同样 5 日轮换，只换"选股来源"。

把"选股能力"单独拎出来量化——三个组合除了选股来源，其余（买点 T+1 开盘、持有 5 日、
等权、不含费用）**完全相同**，所以净值差异只能归因于选股：

  P1 = LLM 实际推荐的那 2 只                       （现实）
  P2 = 同日候选池 top10 中 LLM 没选的（等权取2只）  （反事实：换掉 LLM 的挑选）
  P3 = 全市场随机（每日抽样，等权 2 只）            （反事实：完全放弃选股）

另附 P4 = 8 策略全候选池等权（不选 top2），衡量"8 个策略"这一层本身。
"""

from __future__ import annotations

import glob
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "sequoia_v2.db"
RESULTS_GLOB = str(PROJECT / "data" / "results" / "results_2026*.json")
BAD_DAYS = ("2026-07-06", "2026-07-07")
JUMP = 0.25
HOLD = 5
N_PICK = 2

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main() -> int:
    con = sqlite3.connect(DB)
    cal = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2026-05-01' ORDER BY date", con
    )["date"].tolist()
    cal_idx = {d: i for i, d in enumerate(cal)}
    b = pd.read_sql(
        "SELECT symbol, date, open, close FROM stock_daily WHERE date >= '2026-05-01'", con
    )
    for c in ("open", "close"):
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b.dropna().sort_values(["symbol", "date"])
    b["prev"] = b.groupby("symbol")["close"].shift(1)
    b["corrupt"] = (b["date"].isin(BAD_DAYS)) & (
        ((b["close"] / b["prev"] - 1).abs() > JUMP) | ((b["open"] / b["prev"] - 1).abs() > JUMP)
    )
    b["corrupt"] = b["corrupt"].fillna(False)
    px = {s: dict(zip(g["date"], zip(g["open"], g["close"], g["corrupt"])))
          for s, g in b.groupby("symbol")}
    sig = pd.read_sql(
        "SELECT DISTINCT symbol, buy_date FROM sim_buy_signals WHERE status='executed'", con)
    con.close()
    by_day: dict[str, list[str]] = {}
    for r in sig.itertuples(index=False):
        by_day.setdefault(r.buy_date, []).append(r.symbol)

    # 每日持仓列表：group -> {date: [symbol,...]}
    holds: dict[str, dict[str, list[str]]] = {k: {} for k in ("P1_LLM选中", "P2_top10淘汰", "P3_全市场", "P4_全池")}
    for fp in sorted(glob.glob(RESULTS_GLOB)):
        d = json.load(open(fp, encoding="utf-8"))
        sd = d["date"]
        if sd not in cal_idx:
            continue
        freq: Counter = Counter()
        pool: list[str] = []
        for v in d["strategies"].values():
            if v:
                pool.extend(v)
                freq.update(v)
        pool = sorted(set(pool))
        if not pool:
            continue
        top10 = [s for s, _ in freq.most_common(10)]
        picked = by_day.get(sd, [])
        rej = [s for s in top10 if s not in picked]
        rng = np.random.default_rng(abs(hash(sd)) % (2**32))
        market = [s for s in px if s not in set(pool)]
        holds["P1_LLM选中"][sd] = picked[:N_PICK]
        holds["P2_top10淘汰"][sd] = list(rng.choice(rej, size=min(N_PICK, len(rej)), replace=False)) if rej else []
        holds["P3_全市场"][sd] = list(rng.choice(market, size=N_PICK, replace=False))
        holds["P4_全池"][sd] = pool

    # 组合模拟：T+1 开盘等权买入，持有 HOLD 个交易日后收盘卖出
    def simulate(plan: dict[str, list[str]]) -> pd.Series:
        daily: dict[str, list[float]] = {}
        for sd, syms in plan.items():
            i = cal_idx.get(sd)
            if i is None or i + 1 + HOLD >= len(cal):
                continue
            bd, sd_exit = cal[i + 1], cal[i + 1 + HOLD]
            window = cal[i + 1: i + 2 + HOLD]
            rets = []
            for s in syms:
                dd = px.get(s, {})
                o = dd.get(bd, (np.nan, np.nan, True))[0]
                c = dd.get(sd_exit, (np.nan, np.nan, True))[1]
                if not (o > 0) or not (c > 0):
                    continue
                if any(dd.get(w, (0, 0, True))[2] for w in window):
                    continue
                rets.append(c / o - 1.0)
            if not rets:
                continue
            # 把这段持有期收益按日摊到窗口（几何分摊，保证复利可加）
            r = float(np.mean(rets))
            per_day = (1 + r) ** (1 / HOLD) - 1
            for w in window[:-1]:
                daily.setdefault(w, []).append(per_day)
        ser = pd.Series({k: float(np.mean(v)) for k, v in sorted(daily.items())})
        return ser

    print("=" * 96)
    print(f"反事实组合模拟：每日等权买 {N_PICK} 只（P4 为全池），持有 {HOLD} 个交易日，T+1 开盘买入")
    print("=" * 96)
    res = {}
    for name, plan in holds.items():
        ser = simulate(plan)
        nav = (1 + ser).cumprod()
        res[name] = {"日数": len(ser), "日均": ser.mean(), "累计净值": nav.iloc[-1],
                     "累计收益": nav.iloc[-1] - 1,
                     "日波动": ser.std(), "年化夏普(近似)": ser.mean() / ser.std() * np.sqrt(244) if ser.std() else np.nan}
    print(pd.DataFrame(res).T.to_string())

    print("\n真实 LLM 模拟盘（含 20 只上限/5万均摊/卖出规则/手续费）实际结果：")
    con = sqlite3.connect(DB)
    a = pd.read_sql("SELECT date, total_value FROM sim_account_daily ORDER BY date", con)
    con.close()
    print(f"  {a['date'].iloc[0]} 100.00万 → {a['date'].iloc[-1]} "
          f"{a['total_value'].iloc[-1]/10000:.2f}万  "
          f"（{(a['total_value'].iloc[-1]/1e6-1)*100:+.2f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
