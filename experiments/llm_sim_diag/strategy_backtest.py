"""8 个规则策略的 70 个月选股能力回测（P1-1 全样本复核）。

为什么需要它
------------
P1-1 用 68 天存档结果检验出「RPS动量突破 / 多周期RPS突破 显著为负」，
但样本只有 3 个月且是**下跌市**，而这 6 个策略全是动量/突破类——
牛市里的表现可能完全不同。本脚本把样本拉长到 70 个月（2020-09 ~ 2026-06）再判一次。

怎么做到"忠实回放"
------------------
不重新实现策略逻辑（那样容易失真），而是让**生产代码原样运行**，只把"时间"卡住：

  1. 内存面板 ReplayEngine：覆盖 `get_local_symbols / get_ohlcv`，
     按 as_of 日期切片返回——服务于走 `engine.get_ohlcv()` 的策略
     （均量线突破/海龟/高紧旗形/涨停洗盘/上涨回调）。
  2. 重放 SQLite（临时文件）：随 as_of 推进增量写入行——
     服务于直接用 `sqlite3.connect(engine.db_path)` 的策略（两个 RPS）。
  3. 逐时点基础池：按 as_of 当日重算
     （板块前缀 + 上市满 250 个交易日 + 收盘价 ≥ 2 元），
     经 `stock_pool` 构造参数注入 —— 避免用"今天的股票池"回测历史（幸存者偏差）。

两个 RPS 策略的特征计算（groupby.shift / 截面 rank）全部是**向后看**的，
所以把 as_of 之后的行情藏掉，结果就等同于当日实盘选股。

口径
----
- 买点：信号日**次日（T+1）开盘价**（与模拟盘 SimEngine 一致）
- 卖点：T+1 之后第 N 个交易日收盘（N = 5/10/20）
- 基准：同日**全市场其余股票**等权均值（逐日配对做差，消除日期效应）
- 样本：每月最后一个交易日，2020-09 ~ 2026-06（70 个）
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from sequoia_x.core.config import Settings, get_settings          # noqa: E402
from sequoia_x.data.engine import DataEngine                       # noqa: E402

DB = PROJECT / "data" / "sequoia_v2.db"
HORIZONS = (5, 10, 20)
START_MONTH, END_MONTH = "2020-09", "2026-06"

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


# ═══════════════════════════════════════════════════════════════
#  内存面板引擎：把"时间"卡在 as_of
# ═══════════════════════════════════════════════════════════════
class ReplayEngine(DataEngine):
    """按 as_of 日期切片的内存行情引擎（只实现策略用到的两个方法）。"""

    def __init__(self, panel: dict[str, pd.DataFrame], as_of: str):
        self._panel = panel          # {symbol: DataFrame(date, open, high, low, close, volume, amount)}
        self.as_of = as_of
        self.db_path = ""            # 走 sqlite 的策略由 replay DB 承担，见 ReplayDb
        # 同一 as_of 下多个策略会对同一只股票重复调用 get_ohlcv，
        # 切片一次缓存住（否则 7 策略 × 70 月 × 数千只 = 数十万次全表过滤）
        self._slice: dict[str, pd.DataFrame] = {}

    def get_local_symbols(self) -> list[str]:
        return list(self._panel)

    def get_ohlcv(self, symbol: str) -> pd.DataFrame:
        hit = self._slice.get(symbol)
        if hit is not None:
            return hit
        df = self._panel.get(symbol)
        out = df[df["date"] <= self.as_of].reset_index(drop=True) if df is not None else pd.DataFrame()
        self._slice[symbol] = out
        return out


# ═══════════════════════════════════════════════════════════════
#  重放 SQLite：随 as_of 推进增量写入
# ═══════════════════════════════════════════════════════════════
class ReplayDb:
    """一个临时 SQLite，永远只包含 date <= as_of 的行情。"""

    def __init__(self, src_conn: sqlite3.Connection, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=MEMORY")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS stock_daily ("
            "symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, "
            "volume REAL, amount REAL, turnover REAL, pctChg REAL, UNIQUE(symbol, date))")
        self.conn.commit()
        self._loaded_until = ""
        self._src = src_conn
        self._cache = pd.DataFrame()

    def advance_to(self, as_of: str) -> None:
        """把 [上次, as_of] 之间的行补进重放库。"""
        if as_of <= self._loaded_until:
            return
        lo = self._loaded_until or "1900-01-01"
        chunk = pd.read_sql(
            "SELECT symbol, date, open, high, low, close, volume, amount, turnover, pctChg "
            "FROM stock_daily WHERE date > ? AND date <= ?",
            self._src, params=(lo, as_of))
        if len(chunk):
            self.conn.executemany(
                "INSERT OR REPLACE INTO stock_daily VALUES (?,?,?,?,?,?,?,?,?,?)",
                chunk.itertuples(index=False, name=None))
            self.conn.commit()
        self._loaded_until = as_of


def point_in_time_pool(panel: dict[str, pd.DataFrame], as_of: str,
                       min_history: int = 250, min_price: float = 2.0) -> list[str]:
    """按 as_of 当日重算基础股票池（对齐 get_base_stock_pool 的三步过滤）。

    1. 板块剔除：科创板(688/689)、创业板(300/301)、北交所(4xx/8xx)
    2. 上市满 250 个交易日（用该股在 as_of 前的行情条数近似）
    3. 收盘价 ≥ 2 元
    """
    out = []
    for sym, df in panel.items():
        if sym[:3] in ("688", "689") or sym[:2] in ("43", "83", "87", "88", "92"):
            continue
        if sym[0] in ("4", "8"):
            continue
        sub = df[df["date"] <= as_of]
        if len(sub) < min_history:
            continue
        px = sub["close"].iloc[-1]
        if not np.isfinite(px) or px < min_price:
            continue
        out.append(sym)
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="8 规则策略的 70 个月选股能力回测")
    ap.add_argument("--start", default=START_MONTH, help="起始月 YYYY-MM")
    ap.add_argument("--end", default=END_MONTH, help="结束月 YYYY-MM")
    ap.add_argument("--only", default="", help="只跑指定策略（类名前缀，逗号分隔），冒烟测试用")
    args = ap.parse_args()

    con = sqlite3.connect(DB)

    # ── 月末交易日 ──
    days = [r[0] for r in con.execute(
        "SELECT DISTINCT date FROM stock_daily WHERE date >= '2019-06-01' ORDER BY date")]
    cal_idx = {d: i for i, d in enumerate(days)}
    month_last: dict[str, str] = {}
    for d in days:
        month_last[d[:7]] = d          # 升序遍历，最后写入的即当月最后交易日
    signal_days = [d for m, d in sorted(month_last.items()) if args.start <= m <= args.end]
    print(f"月末交易日 {len(signal_days)} 个：{signal_days[0]} ~ {signal_days[-1]}")

    # ── 价格面板 ──
    print("载入价格面板 ...")
    t0 = time.time()
    px = pd.read_sql(
        "SELECT symbol, date, open, high, low, close, volume, amount, turnover "
        "FROM stock_daily WHERE date >= '2019-06-01'", con)
    for c in ("open", "high", "low", "close"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    panel = {s: g.reset_index(drop=True) for s, g in px.groupby("symbol")}
    print(f"  {len(panel)} 只股票，耗时 {time.time()-t0:.0f}s")

    # ── 策略实例（每轮 as_of 重建，因为引擎要卡时间）──
    from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
    from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
    from sequoia_x.strategy.ma_volume import MaVolumeStrategy
    from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
    from sequoia_x.strategy.rps_multi_period import RpsMultiPeriodStrategy
    from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
    from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy

    STRATS = [RpsBreakoutStrategy, RpsMultiPeriodStrategy, MaVolumeStrategy,
              HighTightFlagStrategy, LimitUpShakeoutStrategy,
              TurtleTradeStrategy, UptrendLimitDownStrategy]
    if args.only:
        want = [w.strip() for w in args.only.split(",") if w.strip()]
        STRATS = [S for S in STRATS if any(S.__name__.startswith(w) for w in want)]
        print(f"只运行: {[S.__name__ for S in STRATS]}")

    settings = get_settings()
    tmpdir = tempfile.mkdtemp(prefix="replay_")
    rdb = ReplayDb(con, str(Path(tmpdir) / "replay.db"))
    print(f"重放库: {rdb.path}")

    picks: dict[str, dict[str, list[str]]] = {s.__name__: {} for s in STRATS}
    pools: dict[str, list[str]] = {}
    t0 = time.time()
    for k, d in enumerate(signal_days, 1):
        rdb.advance_to(d)
        eng = ReplayEngine(panel, d)
        eng.db_path = rdb.path
        pool = point_in_time_pool(panel, d)
        pools[d] = pool
        for S in STRATS:
            try:
                st = S(engine=eng, settings=settings, stock_pool=pool)
                picks[S.__name__][d] = st.run() or []
            except Exception as e:
                print(f"  ⚠️ {S.__name__} @ {d}: {e}")
                picks[S.__name__][d] = []
        if k % 10 == 0:
            el = time.time() - t0
            print(f"  ... {k}/{len(signal_days)}（{el:.0f}s，预计还需 {el/k*(len(signal_days)-k):.0f}s）")

    # ── 前瞻收益 ──
    def fwd(sym: str, d0: str, n: int) -> float:
        i = cal_idx.get(d0)
        if i is None or i + 1 + n >= len(days):
            return np.nan
        bd, ex = days[i + 1], days[i + 1 + n]
        df = panel.get(sym)
        if df is None:
            return np.nan
        sub = df[df["date"].isin((bd, ex))]
        if len(sub) != 2:
            return np.nan
        o = float(sub[sub["date"] == bd]["open"].iloc[0])
        c = float(sub[sub["date"] == ex]["close"].iloc[0])
        if not (o > 0 and c > 0):
            return np.nan
        return c / o - 1.0

    print("\n计算前瞻收益 ...")
    bench: dict[int, dict[str, float]] = {n: {} for n in HORIZONS}
    for d in signal_days:
        pool = set(pools[d])
        # 基准：全市场非池内股票，每 5 只抽 1
        others = [s for i, s in enumerate(panel) if s not in pool and i % 5 == 0]
        for n in HORIZONS:
            rs = [fwd(s, d, n) for s in others]
            rs = [x for x in rs if not np.isnan(x)]
            if rs:
                bench[n][d] = float(np.mean(rs))

    rows = []
    for name, by_day in picks.items():
        for n in HORIZONS:
            diffs, raws = [], []
            for d, syms in by_day.items():
                if not syms or d not in bench[n]:
                    continue
                rs = [fwd(s, d, n) for s in syms]
                rs = [x for x in rs if not np.isnan(x)]
                if not rs:
                    continue
                raws.append(float(np.mean(rs)))
                diffs.append(float(np.mean(rs)) - bench[n][d])
            if len(diffs) < 12:
                rows.append({"策略": name, "持有期": f"+{n}日", "样本月数": len(diffs),
                             "日均收益": np.nan, "日均超额": np.nan, "t值": np.nan})
                continue
            t, p = stats.ttest_1samp(diffs, 0.0)
            rows.append({"策略": name, "持有期": f"+{n}日", "样本月数": len(diffs),
                         "日均收益": float(np.mean(raws)), "日均超额": float(np.mean(diffs)),
                         "t值": t, "p值": p})

    tbl = pd.DataFrame(rows)
    print("\n" + "=" * 108)
    print("70 个月全样本：各策略相对全市场的月度超额（T+1 开盘买入，逐月配对）")
    print("=" * 108)
    print(tbl.to_string(index=False))

    out = PROJECT / "experiments" / "llm_sim_diag" / "strategy_backtest_70m.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {out}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
