"""LLM 模拟盘"买入即跌"归因诊断（一次性分析脚本，不修改生产代码）。

要回答的问题
------------
用户观察：LLM（经 8 个指标策略 + LLM 综合研判筛选）选出的股票，
在 T+1 日开盘买入后，经常立刻下跌。

诊断思路——把每笔买入拆成四段，逐段归因：
  ① 信号日 T 涨幅      ：选股时这只票"已经涨了多少"（追高幅度）
  ② 隔夜跳空           ：T 收盘 → T+1 开盘（买入价）
  ③ 买入当日           ：T+1 开盘 → T+1 收盘（"买完就跌"最直观的一段）
  ④ 买入后 1/3/5/10 日 ：T+1 开盘 → 之后 N 日收盘

每段都与**同期沪深300**对照，区分"策略选股差"与"大盘在跌"。

口径
----
- 全库不复权实际成交价（项目铁律，除权日有真实跳空）
- buy_date 取自 sim_closed_trades/sim_positions = 实际成交日 T+1
- 执行价 = stock_daily 的 T+1 日 open（不含滑点，滑点另算）
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
DB = PROJECT / "data" / "sequoia_v2.db"
BENCH = "sh.000300"

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 300)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def load_trades(con: sqlite3.Connection) -> pd.DataFrame:
    """已成交买入：closed_trades（已平仓）+ positions（当前持仓）合并。"""
    closed = pd.read_sql(
        "SELECT symbol, buy_date, buy_price, strategy_from, 'closed' AS state "
        "FROM sim_closed_trades",
        con,
    )
    opened = pd.read_sql(
        "SELECT symbol, buy_date, buy_price, strategy_from, 'holding' AS state "
        "FROM sim_positions",
        con,
    )
    tx = pd.concat([closed, opened], ignore_index=True)
    # 去重：同一 (symbol, buy_date) 只保留一条（closed/position 可能重叠）
    tx = tx.drop_duplicates(subset=["symbol", "buy_date"], keep="first")
    return tx.sort_values("buy_date").reset_index(drop=True)


def load_bars(con: sqlite3.Connection, symbols: list[str], start: str) -> dict[str, pd.DataFrame]:
    """按股票取日线（不复权），返回 {symbol: DataFrame(按日期升序, 索引重置)}。"""
    ph = ",".join("?" * len(symbols))
    df = pd.read_sql(
        f"SELECT symbol, date, open, high, low, close, volume FROM stock_daily "
        f"WHERE symbol IN ({ph}) AND date >= ? ORDER BY symbol, date",
        con,
        params=symbols + [start],
    )
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return {s: g.reset_index(drop=True) for s, g in df.groupby("symbol")}


def load_bench(con: sqlite3.Connection, start: str) -> pd.DataFrame:
    b = pd.read_sql(
        "SELECT date, close FROM index_daily WHERE symbol=? AND date>=? ORDER BY date",
        con,
        params=[BENCH, start],
    )
    b["close"] = pd.to_numeric(b["close"], errors="coerce")
    return b.reset_index(drop=True)


def bench_forward(bench: pd.DataFrame, d0: str, d1: str) -> float | None:
    """基准在 [d0, d1] 两个最新可得交易日之间的收益。"""
    a = bench[bench["date"] <= d0]
    b = bench[bench["date"] <= d1]
    if a.empty or b.empty:
        return None
    return float(b["close"].iloc[-1] / a["close"].iloc[-1] - 1.0)


def main() -> int:
    con = sqlite3.connect(DB)
    tx = load_trades(con)
    if tx.empty:
        print("无成交记录")
        return 1

    start = (pd.to_datetime(tx["buy_date"]).min() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    bars = load_bars(con, sorted(tx["symbol"].unique().tolist()), start)
    bench = load_bench(con, start)

    rows: list[dict] = []
    for r in tx.itertuples(index=False):
        g = bars.get(r.symbol)
        if g is None or g.empty:
            continue
        hit = g.index[g["date"] == r.buy_date]
        if len(hit) == 0:
            continue
        i = int(hit[0])  # 成交日 T+1 在序列中的位置
        if i == 0:
            continue  # 无前一日，无法算信号日涨幅

        T, E = g.iloc[i - 1], g.iloc[i]  # T=信号日, E=成交日
        o = E["open"]
        if not np.isfinite(o) or o <= 0:
            continue

        # 信号日涨幅：选股时该票已经涨了多少
        signal_day_ret = (T["close"] / g.iloc[i - 2]["close"] - 1.0) if i >= 2 else np.nan
        # 隔夜跳空：T 收盘 → T+1 开盘
        gap = (o / T["close"] - 1.0) if T["close"] else np.nan
        # 买入当日：开盘 → 收盘
        day0 = E["close"] / o - 1.0

        def fwd(n: int) -> float:
            j = i + n
            return (g.iloc[j]["close"] / o - 1.0) if j < len(g) else np.nan

        def fwd_excess(n: int) -> float:
            j = i + n
            if j >= len(g):
                return np.nan
            b = bench_forward(bench, r.buy_date, g.iloc[j]["date"])
            return fwd(n) - b if b is not None else np.nan

        rows.append({
            "symbol": r.symbol,
            "signal_day": T["date"],
            "buy_date": r.buy_date,
            "state": r.state,
            "signal_ret": signal_day_ret,   # ① 信号日涨幅（追高幅度）
            "gap": gap,                     # ② 隔夜跳空
            "day0": day0,                   # ③ 买入当日 开盘→收盘
            "fwd1": fwd(1), "fwd3": fwd(3), "fwd5": fwd(5), "fwd10": fwd(10),
            "ex1": fwd_excess(1), "ex3": fwd_excess(3),
            "ex5": fwd_excess(5), "ex10": fwd_excess(10),
        })

    df = pd.DataFrame(rows)
    con.close()

    print("=" * 100)
    print(f"样本：{len(df)} 笔买入 | 区间 {df['buy_date'].min()} ~ {df['buy_date'].max()}")
    print(f"基准 {BENCH} 同区间：", end="")
    b0, b1 = bench["close"].iloc[0], bench["close"].iloc[-1]
    print(f"{b0:.1f} → {b1:.1f}  ({(b1/b0-1)*100:+.2f}%)")
    print("=" * 100)

    print("\n【一】逐段平均收益（相对买入价 T+1 开盘）")
    cols = ["signal_ret", "gap", "day0", "fwd1", "fwd3", "fwd5", "fwd10"]
    names = ["①信号日涨幅", "②隔夜跳空", "③买入当日(开→收)", "④+1日", "④+3日", "④+5日", "④+10日"]
    stat = pd.DataFrame({
        "均值": [df[c].mean() for c in cols],
        "中位数": [df[c].median() for c in cols],
        "下跌占比": [(df[c] < 0).mean() for c in cols],
        "样本": [df[c].notna().sum() for c in cols],
    }, index=names)
    print(stat.to_string())

    print("\n【二】超额收益（减去同期沪深300）")
    ecols = ["ex1", "ex3", "ex5", "ex10"]
    estat = pd.DataFrame({
        "超额均值": [df[c].mean() for c in ecols],
        "超额中位数": [df[c].median() for c in ecols],
        "跑输占比": [(df[c] < 0).mean() for c in ecols],
        "样本": [df[c].notna().sum() for c in ecols],
    }, index=["+1日", "+3日", "+5日", "+10日"])
    print(estat.to_string())

    print("\n【三】按信号日涨幅分档（验证'追高'假设）")
    d = df.dropna(subset=["signal_ret"]).copy()
    d["档位"] = pd.cut(
        d["signal_ret"],
        [-1, 0, 0.03, 0.06, 0.09, 0.20],
        labels=["下跌", "0~3%", "3~6%", "6~9%", ">9%(接近涨停)"],
    )
    pivot = d.groupby("档位", observed=True).agg(
        笔数=("symbol", "size"),
        信号日涨幅=("signal_ret", "mean"),
        买入当日=("day0", "mean"),
        超额5日=("ex5", "mean"),
        超额10日=("ex10", "mean"),
    )
    print(pivot.to_string())

    print("\n【四】按买入日期分周（看是否趋势性恶化）")
    d2 = df.copy()
    d2["周"] = pd.to_datetime(d2["buy_date"]).dt.to_period("W").astype(str)
    wk = d2.groupby("周").agg(
        笔数=("symbol", "size"),
        信号日涨幅=("signal_ret", "mean"),
        买入当日=("day0", "mean"),
        超额5日=("ex5", "mean"),
    )
    print(wk.to_string())

    print("\n【五】买入当日最差的 15 笔")
    show = ["symbol", "signal_day", "buy_date", "signal_ret", "gap", "day0", "ex5"]
    print(df.nsmallest(15, "day0")[show].to_string(index=False))

    out = PROJECT / "experiments" / "llm_sim_diag" / "trades_detail.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
