"""对照实验：LLM 选中的股票 vs 同一天被它淘汰的候选池。

回答用户问题"难道经过 8 个指标策略和 LLM 筛选出的股票表现就这么差？"

方法
----
`data/results/results_YYYYMMDD.json` 存档了每个信号日 8 个策略的完整选股结果。
据此重建每天喂给 LLM 的候选池，然后对比三组的**同期前瞻收益**（T+1 开盘买入口径）：

  A. LLM 最终推荐的 2 只（sim_buy_signals 里 buy_date=T 的信号）
  B. 候选池里的其余股票（LLM 看过但没挑）
  C. 全市场（当日除候选池外的所有股票，作为基准组）

三组用**完全相同的买入口径**（T+1 开盘价买入），所以差异只能来自"选股"本身。

若 A 显著差于 B/C → LLM 筛选在毁灭价值（负 alpha）；
若 A ≈ C → LLM 只是没有增加价值（随机挑选）；
若 A > B > C → 筛选链条有效，亏损来自大盘 beta。
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
BENCH = "sh.000300"
HORIZONS = (1, 3, 5, 10)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main() -> int:
    con = sqlite3.connect(DB)

    # ── 交易日历（用沪深300的日期，全市场一致）──
    cal = pd.read_sql(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2026-05-01' ORDER BY date",
        con,
    )["date"].tolist()
    cal_idx = {d: i for i, d in enumerate(cal)}

    # ── 全市场日线（开盘价用于模拟 T+1 买入）──
    bars = pd.read_sql(
        "SELECT symbol, date, open, close FROM stock_daily WHERE date >= '2026-05-01'",
        con,
    )
    bars["open"] = pd.to_numeric(bars["open"], errors="coerce")
    bars["close"] = pd.to_numeric(bars["close"], errors="coerce")
    bars = bars.dropna(subset=["open", "close"])
    bars = bars[bars["open"] > 0]
    # {symbol: {date: (open, close)}}
    px: dict[str, dict[str, tuple[float, float]]] = {}
    for sym, g in bars.groupby("symbol"):
        px[sym] = dict(zip(g["date"], zip(g["open"], g["close"])))

    # ── LLM 实际推荐的股票（信号表 buy_date = 信号日 T；只取 executed）──
    sig = pd.read_sql(
        "SELECT DISTINCT symbol, buy_date FROM sim_buy_signals "
        "WHERE status='executed' AND buy_date >= '2026-06-01'",
        con,
    )
    llm_by_day: dict[str, set[str]] = {}
    for r in sig.itertuples(index=False):
        llm_by_day.setdefault(r.buy_date, set()).add(r.symbol)
    con.close()

    # ── 基准 ──
    bench_ret: dict[tuple[str, int], float] = {}

    def fwd_ret(sym: str, sig_date: str, n: int) -> float | None:
        """信号日 sig_date → 第 n 个交易日后，以 T+1 开盘买入的收益。"""
        i = cal_idx.get(sig_date)
        if i is None or i + 1 >= len(cal):
            return None
        buy_date = cal[i + 1]
        d = px.get(sym, {})
        o = d.get(buy_date, (None, None))[0]
        if not o:
            return None
        j = cal_idx.get(buy_date) + n
        if j >= len(cal):
            return None
        c = d.get(cal[j], (None, None))[1]
        if not c:
            return None
        return c / o - 1.0

    rows: list[dict] = []
    files = sorted(glob.glob(RESULTS_GLOB))
    n_days = 0
    for fp in files:
        d = json.load(open(fp, encoding="utf-8"))
        sig_date = d["date"]
        if sig_date not in cal_idx:
            continue
        strat = {k: v for k, v in d["strategies"].items() if v}
        pool: list[str] = []
        freq: Counter = Counter()
        for syms in strat.values():
            pool.extend(syms)
            freq.update(syms)
        pool = sorted(set(pool))
        if not pool:
            continue
        # LLM 实际看到的是"多策略重叠频率前 10"（analyst.analyze 限流）
        top10 = {s for s, _ in freq.most_common(10)}
        picked = llm_by_day.get(sig_date, set())
        n_days += 1

        for group, syms in (
            ("A_LLM选中", sorted(picked & set(pool))),
            ("B_池内淘汰", sorted(set(pool) - picked)),
            ("C_市场其余", []),  # 占位，下面单独算
        ):
            for s in syms:
                if s not in px:
                    continue
                row = {"date": sig_date, "group": group, "symbol": s,
                       "in_top10": s in top10,
                       "signal_ret": None}
                for n in HORIZONS:
                    row[f"r{n}"] = fwd_ret(s, sig_date, n)
                rows.append(row)

        # C 组：当日全市场（去掉候选池），抽样降低计算量（每 5 只取 1）
        others = [s for i, s in enumerate(px) if s not in set(pool) and i % 5 == 0]
        for s in others:
            row = {"date": sig_date, "group": "C_市场其余", "symbol": s,
                   "in_top10": False, "signal_ret": None}
            for n in HORIZONS:
                row[f"r{n}"] = fwd_ret(s, sig_date, n)
            rows.append(row)

    df = pd.DataFrame(rows)
    print("=" * 100)
    print(f"覆盖信号日 {n_days} 天 | 样本 {len(df)} 条")
    print("=" * 100)

    print("\n【一】各组 T+1 开盘买入后的平均收益 —— 同口径对比")
    g = df.groupby("group")
    tbl = pd.DataFrame({
        "样本": g.size(),
        **{f"+{n}日": g[f"r{n}"].mean() for n in HORIZONS},
        **{f"+{n}日中位数": g[f"r{n}"].median() for n in HORIZONS},
    })
    print(tbl.to_string())

    print("\n【二】LLM 选中 vs 池内淘汰（只看 LLM 真实看过的 top10，严格对照）")
    sub = df[(df["group"] == "A_LLM选中") | ((df["group"] == "B_池内淘汰") & df["in_top10"])]
    g2 = sub.groupby("group")
    print(pd.DataFrame({
        "样本": g2.size(),
        **{f"+{n}日": g2[f"r{n}"].mean() for n in HORIZONS},
    }).to_string())

    print("\n【三】候选池 vs 全市场（检验'8个策略'本身有没有选股能力）")
    sub3 = df[df["group"].isin(["B_池内淘汰", "C_市场其余"])].copy()
    sub3["简化组"] = np.where(sub3["group"] == "C_市场其余", "全市场(基准)", "8策略候选池")
    g3 = sub3.groupby("简化组")
    print(pd.DataFrame({
        "样本": g3.size(),
        **{f"+{n}日": g3[f"r{n}"].mean() for n in HORIZONS},
        **{f"+{n}日胜率(vs0)": g3[f"r{n}"].apply(lambda s: (s > 0).mean()) for n in HORIZONS},
    }).to_string())

    print("\n【四】逐周 +5 日收益（看时间上的稳定性）")
    df["周"] = pd.to_datetime(df["date"]).dt.to_period("W").astype(str)
    wk = df.pivot_table(index="周", columns="group", values="r5", aggfunc="mean")
    print(wk.to_string())

    out = PROJECT / "experiments" / "llm_sim_diag" / "pool_vs_llm.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
