"""P1-1 的 68 天复核：在**修复后**的数据上重算策略选股能力。

为什么必须复核
--------------
原 P1-1（`strategy_ic.py`）用的是 `data/results/*.json` 里的**存档选股结果**。
但那些存档是当天生产跑出来的——而 RPS 类策略要回看 120/250 个交易日，
其回看窗口**跨越了 2026-06-09 的口径边界**，落在深市后复权的那一段里。

后果：算出来的"动量" = 真实收益 ÷ 每股复权因子 K，K 与个股送转史相关、
与动量无关 → 相当于往排序里灌了噪声。**存档选股本身就被污染了。**

本脚本用回放引擎在**修复后**的数据上重跑同一批日期，与 `strategy_ic.py` 的结果对照：
  - 若结论一致（都是负超额）→ 原结论稳健，修复不改变判断
  - 若显著不同 → 原结论是被污染数据误导的伪结论

方法一致性
----------
日期集合、持有期（1/3/5/10 日）、买点（T+1 开盘）、基准（同日全市场其余股票等权）、
逐日配对 t 检验 —— 全部与 `strategy_ic.py` 一致，只有"选股怎么来的"不同
（那边读存档，这边用回放引擎现算）。
"""

from __future__ import annotations

import json
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

from dotenv import load_dotenv                                    # noqa: E402
load_dotenv(PROJECT / ".env")

from sequoia_x.core.config import get_settings                     # noqa: E402
from strategy_backtest import ReplayDb, ReplayEngine, point_in_time_pool  # noqa: E402

DB = PROJECT / "data" / "sequoia_v2.db"
RESULTS_GLOB = str(PROJECT / "data" / "results" / "results_2026*.json")
HORIZONS = (1, 3, 5, 10)
MIN_DAYS = 15

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def main() -> int:
    con = sqlite3.connect(DB)
    settings = get_settings()

    # 与 strategy_ic.py 完全相同的日期集合（存档日期）
    arch_days = sorted({json.load(open(f, encoding="utf-8"))["date"]
                        for f in __import__("glob").glob(RESULTS_GLOB)})
    print(f"存档日期 {len(arch_days)} 天：{arch_days[0]} ~ {arch_days[-1]}")

    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM stock_daily ORDER BY date")]
    cal_idx = {d: i for i, d in enumerate(days)}
    arch_days = [d for d in arch_days if d in cal_idx]

    print("载入价格面板 ...")
    t0 = time.time()
    px = pd.read_sql(
        "SELECT symbol, date, open, high, low, close, volume, amount, turnover "
        "FROM stock_daily WHERE date >= '2024-06-01'", con)
    for c in ("open", "high", "low", "close"):
        px[c] = pd.to_numeric(px[c], errors="coerce")
    panel = {s: g.reset_index(drop=True) for s, g in px.groupby("symbol")}
    print(f"  {len(panel)} 只，耗时 {time.time()-t0:.0f}s")

    from sequoia_x.strategy.high_tight_flag import HighTightFlagStrategy
    from sequoia_x.strategy.limit_up_shakeout import LimitUpShakeoutStrategy
    from sequoia_x.strategy.ma_volume import MaVolumeStrategy
    from sequoia_x.strategy.rps_breakout import RpsBreakoutStrategy
    from sequoia_x.strategy.rps_multi_period import RpsMultiPeriodStrategy
    from sequoia_x.strategy.turtle_trade import TurtleTradeStrategy
    from sequoia_x.strategy.uptrend_limit_down import UptrendLimitDownStrategy

    # 注意：这里**包含海龟**（2026-09-25 已修复），用于回答"修好后它值不值得启用"
    STRATS = [RpsBreakoutStrategy, RpsMultiPeriodStrategy, MaVolumeStrategy,
              HighTightFlagStrategy, LimitUpShakeoutStrategy,
              UptrendLimitDownStrategy, TurtleTradeStrategy]

    tmpdir = tempfile.mkdtemp(prefix="recheck_")
    rdb = ReplayDb(con, str(Path(tmpdir) / "r.db"))
    picks: dict[str, dict[str, list[str]]] = {S.__name__: {} for S in STRATS}
    pools: dict[str, list[str]] = {}
    t0 = time.time()
    for k, d in enumerate(arch_days, 1):
        rdb.advance_to(d)
        eng = ReplayEngine(panel, d)
        eng.db_path = rdb.path
        pool = point_in_time_pool(panel, d)
        pools[d] = pool
        for S in STRATS:
            try:
                picks[S.__name__][d] = S(engine=eng, settings=settings, stock_pool=pool).run() or []
            except Exception as e:
                print(f"  ⚠️ {S.__name__} @ {d}: {e}")
                picks[S.__name__][d] = []
        if k % 20 == 0:
            el = time.time() - t0
            print(f"  ... {k}/{len(arch_days)}（{el:.0f}s）", flush=True)

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
        return c / o - 1.0 if (o > 0 and c > 0) else np.nan

    bench: dict[int, dict[str, float]] = {n: {} for n in HORIZONS}
    for d in arch_days:
        pool = set(pools[d])
        others = [s for i, s in enumerate(panel) if s not in pool and i % 7 == 0]
        for n in HORIZONS:
            rs = [fwd(s, d, n) for s in others]
            rs = [x for x in rs if not np.isnan(x)]
            if rs:
                bench[n][d] = float(np.mean(rs))

    rows = []
    for name, by_day in picks.items():
        row = {"策略": name, "选股日数": sum(1 for v in by_day.values() if v)}
        for n in HORIZONS:
            diffs = []
            for d, syms in by_day.items():
                if not syms or d not in bench[n]:
                    continue
                rs = [fwd(s, d, n) for s in syms]
                rs = [x for x in rs if not np.isnan(x)]
                if rs:
                    diffs.append(float(np.mean(rs)) - bench[n][d])
            if len(diffs) < MIN_DAYS:
                row[f"+{n}日超额"], row[f"+{n}日t"] = np.nan, np.nan
                continue
            t, _ = stats.ttest_1samp(diffs, 0.0)
            row[f"+{n}日超额"], row[f"+{n}日t"] = float(np.mean(diffs)), t
        rows.append(row)

    tbl = pd.DataFrame(rows).sort_values("+10日超额")
    print("\n" + "=" * 110)
    print("复核：修复后数据上重跑 68 天（回放引擎现算选股，非存档）")
    print("=" * 110)
    print(tbl.to_string(index=False))

    print("\n【与 strategy_ic.py（存档选股）对照】")
    old = PROJECT / "experiments" / "llm_sim_diag" / "strategy_ic.csv"
    if old.exists():
        o = pd.read_csv(old)
        m = o.merge(tbl, on="策略", how="inner", suffixes=("_旧", "_新"))
        show = m[["策略", "+10日超额_旧", "+10日t_旧", "+10日超额_新", "+10日t_新"]]
        print(show.to_string(index=False))

    out = PROJECT / "experiments" / "llm_sim_diag" / "recheck_68d.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {out}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
